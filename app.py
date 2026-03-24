"""
Arabidopsis Essential Gene Prediction Web Application
Deployed on Streamlit Cloud (Free Tier)
Features:
- Online sequence input (FASTA format supported)
- Automatic CDS to Protein Translation (3-frame translation with longest ORF selection)
- Real-time prediction with probability scores
- Statistical summary (essential vs non-essential counts)
- Multi-format result download (CSV, JSON, Excel)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Bio import SeqIO
from io import StringIO, BytesIO
import base64
import re

# ==============================================================================
# 1. Genetic Code Configuration (Standard Genetic Code)
# ==============================================================================
GENETIC_CODE = {
    # Phenylalanine
    'TTT': 'F', 'TTC': 'F',
    # Leucine
    'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    # Isoleucine
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    # Methionine (Start)
    'ATG': 'M',
    # Valine
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    # Serine
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    # Proline
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    # Threonine
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    # Alanine
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    # Tyrosine
    'TAT': 'Y', 'TAC': 'Y',
    # Histidine
    'CAT': 'H', 'CAC': 'H',
    # Glutamine
    'CAA': 'Q', 'CAG': 'Q',
    # Asparagine
    'AAT': 'N', 'AAC': 'N',
    # Lysine
    'AAA': 'K', 'AAG': 'K',
    # Aspartic Acid
    'GAT': 'D', 'GAC': 'D',
    # Glutamic Acid
    'GAA': 'E', 'GAG': 'E',
    # Cysteine
    'TGT': 'C', 'TGC': 'C',
    # Tryptophan
    'TGG': 'W',
    # Arginine
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    # Glycine
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    # Stop Codons
    'TAA': '*', 'TAG': '*', 'TGA': '*'
}

START_CODONS = ['ATG']
STOP_CODONS = ['TAA', 'TAG', 'TGA']

# ==============================================================================
# 2. Configuration Class (Simplified for Deployment)
# ==============================================================================
class DeployConfig:
    """Configuration class containing all hyperparameters and settings"""
    K = 3  # K-mer size
    D = 4  # Distance parameter for co-occurrence matrix
    SEQ_MAX_LEN = 1900  # Maximum sequence length
    MAIN_FEAT_DIM = 640  # Reduced dimension for k-mer features
    BIO_FEATURE_DIM = 128  # Reduced dimension for biological features
    RAW_BIO_FEATURE_DIM = 384  # Original dimension (256 CDS + 128 AA)
    CDS_FEATURE_DIM = 256
    AA_FEATURE_DIM = 128
    GCN_HIDDEN = 256
    GCN_LAYERS = 3
    DROPOUT = 0.3
    MIN_KMER_COUNT = 2
    DEFAULT_THRESHOLD = 0.1  # Default classification threshold

    # Device configuration (CPU for deployment)
    DEVICE = torch.device("cpu")

# ==============================================================================
# 3. Sequence Preprocessing Tools
# ==============================================================================
class SequenceProcessor:
    """
    Processor for handling variable-length sequences.
    Implements truncation and padding strategies.
    """
    def __init__(self, max_len: int, strategy: str = "smart"):
        self.max_len = max_len
        self.strategy = strategy

    def process_sequence(self, seq_str: str) -> str:
        """Process input sequence to fixed length"""
        seq_str = seq_str.upper()
        if len(seq_str) > self.max_len:
            return self._truncate_sequence(seq_str)
        elif len(seq_str) < self.max_len:
            return self._pad_sequence(seq_str)
        else:
            return seq_str

    def _truncate_sequence(self, seq_str: str) -> str:
        """Truncate long sequences from center or end"""
        if self.strategy == "smart":
            start = (len(seq_str) - self.max_len) // 2
            return seq_str[start:start + self.max_len]
        else:
            return seq_str[-self.max_len:]

    def _pad_sequence(self, seq_str: str) -> str:
        """Pad short sequences with 'N' characters"""
        if self.strategy == "smart":
            total_pad = self.max_len - len(seq_str)
            left_pad = total_pad // 2
            return "N" * left_pad + seq_str + "N" * (total_pad - left_pad)
        else:
            return seq_str + "N" * (self.max_len - len(seq_str))

# ==============================================================================
# 4. CDS to Protein Translation Module (NEW FEATURE)
# ==============================================================================
class Translator:
    """
    Handles CDS to Protein translation.
    Implements 3-frame translation and selects the longest ORF.
    """

    @staticmethod
    def validate_cds(seq: str) -> bool:
        """Check if sequence contains only valid nucleotides"""
        valid_chars = set('ATCG')
        return all(c in valid_chars for c in seq.upper())

    @staticmethod
    def clean_sequence(seq: str) -> str:
        """Remove non-nucleotide characters and convert to uppercase"""
        return ''.join(c for c in seq.upper() if c in 'ATCG')

    @staticmethod
    def translate_frame(seq: str, frame: int = 0) -> str:
        """
        Translate single reading frame to protein sequence.
        Frame: 0, 1, or 2 (starting position offset)
        """
        seq = Translator.clean_sequence(seq)
        if len(seq) < 3:
            return ""

        # Adjust for frame offset
        seq = seq[frame:]

        # Trim to multiple of 3
        length = len(seq) - (len(seq) % 3)
        seq = seq[:length]

        protein = []
        for i in range(0, length, 3):
            codon = seq[i:i+3]
            aa = GENETIC_CODE.get(codon, '?')  # '?' for unknown codons
            protein.append(aa)

        return ''.join(protein)

    @staticmethod
    def find_orfs(protein_seq: str, dna_seq: str, frame: int) -> list:
        """
        Find all ORFs (Open Reading Frames) in protein sequence.
        Returns list of tuples: (start_pos, end_pos, protein_sequence, length)
        """
        orfs = []
        in_orf = False
        start_pos = 0

        for i, aa in enumerate(protein_seq):
            if aa == 'M' and not in_orf:  # Start codon (Methionine)
                in_orf = True
                start_pos = i
            elif aa == '*' and in_orf:  # Stop codon
                in_orf = False
                orf_seq = protein_seq[start_pos:i]  # Exclude stop codon
                if len(orf_seq) >= 10:  # Minimum ORF length (10 amino acids)
                    orfs.append({
                        'start': start_pos,
                        'end': i,
                        'protein': orf_seq,
                        'length': len(orf_seq),
                        'frame': frame,
                        'dna_start': start_pos * 3 + frame,
                        'dna_end': i * 3 + frame
                    })

        # Handle ORFs without stop codon (incomplete)
        if in_orf:
            orf_seq = protein_seq[start_pos:]
            if len(orf_seq) >= 10:
                orfs.append({
                    'start': start_pos,
                    'end': len(protein_seq),
                    'protein': orf_seq,
                    'length': len(orf_seq),
                    'frame': frame,
                    'dna_start': start_pos * 3 + frame,
                    'dna_end': len(protein_seq) * 3 + frame
                })

        return orfs

    @staticmethod
    def translate_cds_to_protein(cds_seq: str, method: str = "longest_orf") -> str:
        """
        Main translation function.
        Tries all 3 reading frames and returns the best protein sequence.

        Args:
            cds_seq: DNA sequence (CDS)
            method: "longest_orf" (default) or "frame0"

        Returns:
            Protein sequence string (single letter amino acid codes)
        """
        cds_seq = Translator.clean_sequence(cds_seq)

        if len(cds_seq) < 3:
            return ""

        if method == "frame0":
            # Simple frame 0 translation (assume correct frame)
            protein = Translator.translate_frame(cds_seq, 0)
            # Remove stop codon if at end
            if protein.endswith('*'):
                protein = protein[:-1]
            return protein

        elif method == "longest_orf":
            # Find longest ORF across all 3 frames
            all_orfs = []

            for frame in range(3):
                protein = Translator.translate_frame(cds_seq, frame)
                orfs = Translator.find_orfs(protein, cds_seq, frame)
                all_orfs.extend(orfs)

            if not all_orfs:
                # Fallback: return frame 0 translation without stop codons
                protein = Translator.translate_frame(cds_seq, 0)
                return protein.replace('*', '')

            # Select longest ORF
            longest_orf = max(all_orfs, key=lambda x: x['length'])
            return longest_orf['protein']

        else:
            raise ValueError(f"Unknown translation method: {method}")

# ==============================================================================
# 5. K-mer Encoding Functions (Core Feature Extraction)
# ==============================================================================
def _num_transfer(seq: str) -> str:
    """Convert nucleotide sequence to numerical string (A=0, C=1, G=2, T=3)"""
    seq = seq.upper()
    seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
    return "".join(filter(str.isdigit, seq))

def _num_transfer_loc(num_seq: str, K: int) -> list:
    """Convert numerical sequence to k-mer location list"""
    loc = []
    seq_len = len(num_seq)
    if seq_len < K:
        num_seq = num_seq.ljust(K, "0")
        seq_len = K
    for i in range(seq_len - K + 1):
        kmer_val = int(num_seq[i:i + K], base=4)
        loc.append(kmer_val)
    return loc

def _count_kmer_occurrence(loc_list: list, num_kmer: int) -> np.ndarray:
    """Count occurrence of each k-mer"""
    count = np.zeros(num_kmer, dtype=int)
    for loc in loc_list:
        if loc < num_kmer:
            count[loc] += 1
    return count

def _loc_transfer_matrix(loc_list: list, dis: list, K: int, seq_length: int) -> np.ndarray:
    """Build k-mer co-occurrence matrix for given distance"""
    num_kmer = 4 ** K
    matrix = np.zeros((num_kmer, num_kmer), dtype=np.float32)
    num = 0
    dis_val = dis[0]
    valid_len = len(loc_list) - K - dis_val

    if valid_len > 0:
        for i in range(valid_len):
            idx1 = loc_list[i]
            idx2 = loc_list[i + K + dis_val]
            if idx1 < num_kmer and idx2 < num_kmer:
                matrix[idx1][idx2] += 1
        num = max(1, seq_length - 2 * K - dis_val + 1)
    return matrix / num if num != 0 else matrix

def matrix_encoding_no_bio(seq: str, K: int, d: int) -> tuple:
    """
    Encode sequence to matrix features without biological features.
    Returns: (main_feature_vector, kmer_count_vector)
    """
    seq = seq.upper()
    seq_length = len(seq)
    num_seq = _num_transfer(seq) or "0" * K
    loc_list = _num_transfer_loc(num_seq, K)

    d = min(d, 4)
    dis_list = [[0], [1], [3], [5]][:d]

    num_kmer = 4 ** K
    kmer_features = []

    for dis in dis_list:
        matrix = _loc_transfer_matrix(loc_list, dis, K, seq_length)
        flattened_matrix = matrix.flatten()
        kmer_features.append(flattened_matrix)

    if kmer_features:
        main_feat = np.hstack(kmer_features) * 100
    else:
        main_feat = np.zeros(num_kmer * num_kmer * d, dtype=np.float32)

    kmer_count = _count_kmer_occurrence(loc_list, num_kmer)
    return main_feat.astype(np.float32), kmer_count

# ==============================================================================
# 6. Biological Feature Extractor (CDS and Amino Acid Features)
# ==============================================================================
class BioFeatureExtractor:
    """Extract biological features from CDS and protein sequences"""

    def __init__(self):
        # Amino acid property groups
        self.amino_acid_groups = {
            'hydrophobic': ['A', 'V', 'L', 'I', 'P', 'F', 'W', 'M'],
            'hydrophilic': ['R', 'N', 'D', 'Q', 'E', 'K', 'H'],
            'neutral': ['S', 'T', 'Y', 'C', 'G'],
            'aromatic': ['F', 'W', 'Y'],
            'charged_positive': ['R', 'K', 'H'],
            'charged_negative': ['D', 'E'],
            'tiny': ['A', 'G', 'S'],
            'small': ['A', 'G', 'S', 'C', 'T', 'D', 'N', 'V'],
            'large': ['R', 'K', 'E', 'Q', 'H', 'W', 'Y', 'F', 'M', 'I', 'L']
        }

        # Amino acid physicochemical properties
        self.aa_properties = {
            'A': {'mw': 89.09, 'hydrophobicity': 0.62, 'pi': 6.0, 'volume': 88.6},
            'R': {'mw': 174.20, 'hydrophobicity': -2.53, 'pi': 10.76, 'volume': 173.4},
            'N': {'mw': 132.12, 'hydrophobicity': -0.78, 'pi': 5.41, 'volume': 114.1},
            'D': {'mw': 133.10, 'hydrophobicity': -0.90, 'pi': 2.77, 'volume': 111.1},
            'C': {'mw': 121.15, 'hydrophobicity': 0.29, 'pi': 5.07, 'volume': 108.5},
            'Q': {'mw': 146.15, 'hydrophobicity': -0.85, 'pi': 5.65, 'volume': 143.8},
            'E': {'mw': 147.13, 'hydrophobicity': -0.74, 'pi': 3.22, 'volume': 138.4},
            'G': {'mw': 75.07, 'hydrophobicity': 0.48, 'pi': 5.97, 'volume': 60.1},
            'H': {'mw': 155.16, 'hydrophobicity': -0.40, 'pi': 7.59, 'volume': 153.2},
            'I': {'mw': 131.17, 'hydrophobicity': 1.38, 'pi': 6.02, 'volume': 166.7},
            'L': {'mw': 131.17, 'hydrophobicity': 1.06, 'pi': 5.98, 'volume': 166.7},
            'K': {'mw': 146.19, 'hydrophobicity': -1.50, 'pi': 9.74, 'volume': 168.6},
            'M': {'mw': 149.21, 'hydrophobicity': 0.64, 'pi': 5.74, 'volume': 162.9},
            'F': {'mw': 165.19, 'hydrophobicity': 1.19, 'pi': 5.48, 'volume': 189.9},
            'P': {'mw': 115.13, 'hydrophobicity': 0.12, 'pi': 6.30, 'volume': 112.7},
            'S': {'mw': 105.09, 'hydrophobicity': -0.18, 'pi': 5.68, 'volume': 89.0},
            'T': {'mw': 119.12, 'hydrophobicity': -0.05, 'pi': 5.60, 'volume': 116.1},
            'W': {'mw': 204.23, 'hydrophobicity': 0.81, 'pi': 5.89, 'volume': 227.8},
            'Y': {'mw': 181.19, 'hydrophobicity': 0.26, 'pi': 5.66, 'volume': 193.6},
            'V': {'mw': 117.15, 'hydrophobicity': 1.08, 'pi': 5.96, 'volume': 140.0}
        }

    def extract_cds_features(self, cds_seq: str) -> np.ndarray:
        """Extract 256-dimensional features from CDS sequence"""
        features = []
        if not cds_seq:
            return np.zeros(DeployConfig.CDS_FEATURE_DIM, dtype=np.float32)

        seq_len = len(cds_seq)

        # GC content features
        gc_content = (cds_seq.count('G') + cds_seq.count('C')) / seq_len
        at_content = (cds_seq.count('A') + cds_seq.count('T')) / seq_len
        gc_skew = (cds_seq.count('G') - cds_seq.count('C')) / (cds_seq.count('G') + cds_seq.count('C') + 1e-8)
        at_skew = (cds_seq.count('A') - cds_seq.count('T')) / (cds_seq.count('A') + cds_seq.count('T') + 1e-8)
        features.extend([gc_content, at_content, gc_skew, at_skew])

        # Single nucleotide frequencies
        nucleotides = ['A', 'C', 'G', 'T']
        for nt in nucleotides:
            features.append(cds_seq.count(nt) / seq_len)

        # Dinucleotide frequencies
        di_nucleotides = [a + b for a in nucleotides for b in nucleotides]
        total_pairs = max(1, seq_len - 1)
        for di in di_nucleotides:
            features.append(cds_seq.count(di) / total_pairs)

        # Trinucleotide frequencies (first 16)
        tri_nucleotides = [a + b + c for a in nucleotides for b in nucleotides for c in nucleotides][:16]
        total_triplets = max(1, seq_len - 2)
        for tri in tri_nucleotides:
            features.append(cds_seq.count(tri) / total_triplets)

        # Sequence length features
        features.extend([seq_len / 1000.0, np.log(seq_len + 1), seq_len / 3000.0])

        # Codon usage features
        if seq_len >= 3:
            codons = [cds_seq[i:i+3] for i in range(0, seq_len - 2, 3)]
            codon_count = len(codons)
            if codon_count > 0:
                unique_codons = len(set(codons))
                codon_diversity = unique_codons / codon_count
                counts = [codons.count(c) for c in set(codons)]
                effective_codons = 1 / sum((c/codon_count)**2 for c in counts) if codon_count > 0 else 0
                start_count = sum(1 for c in codons if c == 'ATG')
                stop_count = sum(1 for c in codons if c in ['TAA', 'TAG', 'TGA'])
                features.extend([
                    codon_diversity, effective_codons / 61.0,
                    start_count / codon_count, stop_count / codon_count,
                    codon_count / 100.0
                ])
            else:
                features.extend([0.0] * 5)
        else:
            features.extend([0.0] * 5)

        # Padding or truncation to 256 dimensions
        if len(features) < DeployConfig.CDS_FEATURE_DIM:
            features.extend([0.0] * (DeployConfig.CDS_FEATURE_DIM - len(features)))
        elif len(features) > DeployConfig.CDS_FEATURE_DIM:
            features = features[:DeployConfig.CDS_FEATURE_DIM]

        return np.array(features, dtype=np.float32)

    def extract_aa_features(self, aa_seq: str) -> np.ndarray:
        """Extract 128-dimensional features from amino acid sequence"""
        features = []
        if not aa_seq or len(aa_seq) == 0:
            return np.zeros(DeployConfig.AA_FEATURE_DIM, dtype=np.float32)

        total = len(aa_seq)
        aa_list = 'ACDEFGHIKLMNPQRSTVWY'

        # Amino acid composition
        for aa in aa_list:
            features.append(aa_seq.count(aa) / total)

        # Group frequencies
        for group in self.amino_acid_groups.values():
            count = sum(1 for aa in aa_seq if aa in group)
            features.append(count / total)

        # Dipeptide frequencies (limited to 20 for dimension control)
        representative_aa = list(aa_list)
        dipeptides = [a + b for a in representative_aa for b in representative_aa][:20]
        total_pairs = max(1, len(aa_seq) - 1)
        for di in dipeptides:
            count = sum(1 for i in range(len(aa_seq)-1) if aa_seq[i:i+2] == di)
            features.append(count / total_pairs)

        # Physicochemical properties
        valid_count = 0
        mw_sum = hydro_sum = pi_sum = vol_sum = 0
        mw_list = []
        for aa in aa_seq:
            if aa in self.aa_properties:
                p = self.aa_properties[aa]
                mw_sum += p['mw']
                hydro_sum += p['hydrophobicity']
                pi_sum += p['pi']
                vol_sum += p['volume']
                mw_list.append(p['mw'])
                valid_count += 1

        if valid_count > 0:
            features.extend([
                mw_sum / valid_count / 1000.0,
                hydro_sum / valid_count,
                pi_sum / valid_count / 10.0,
                vol_sum / valid_count / 100.0,
                np.std(mw_list) / 100.0 if len(mw_list) > 1 else 0.0
            ])
        else:
            features.extend([0.0] * 5)

        # Secondary structure propensities
        helix = sum(1 for aa in aa_seq if aa in ['E', 'A', 'L', 'M', 'Q', 'K', 'R'])
        sheet = sum(1 for aa in aa_seq if aa in ['V', 'I', 'Y', 'F', 'W', 'T'])
        coil = sum(1 for aa in aa_seq if aa in ['G', 'P', 'S', 'D', 'N'])
        features.extend([helix/total, sheet/total, coil/total])

        # Hydrophobicity and charge
        hydrophobic = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['hydrophobic'])
        hydrophilic = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['hydrophilic'])
        pos_charge = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['charged_positive'])
        neg_charge = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['charged_negative'])
        features.extend([
            hydrophobic/total, hydrophilic/total,
            pos_charge/total, neg_charge/total,
            (pos_charge - neg_charge)/total
        ])

        # Sequence complexity
        features.extend([
            len(aa_seq) / 1000.0,
            len(set(aa_seq)) / len(aa_seq),
            len(aa_seq) / 500.0
        ])

        # Special amino acids
        features.extend([
            aa_seq.count('C') / total,
            aa_seq.count('P') / total,
            aa_seq.count('G') / total
        ])

        # Padding to 128 dimensions
        if len(features) < DeployConfig.AA_FEATURE_DIM:
            features.extend([0.0] * (DeployConfig.AA_FEATURE_DIM - len(features)))
        elif len(features) > DeployConfig.AA_FEATURE_DIM:
            features = features[:DeployConfig.AA_FEATURE_DIM]

        return np.array(features, dtype=np.float32)

# ==============================================================================
# 7. Feature Reduction (PCA Wrapper)
# ==============================================================================
class FeatureReducer:
    """
    Wrapper for PCA transformation.
    Loads pre-fitted PCA and StandardScaler from training.
    """
    def __init__(self, reducer_data: dict):
        self.pca = reducer_data['pca']
        self.scaler = reducer_data['scaler']
        self.target_dim = reducer_data['target_dim']
        self.is_fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler and PCA"""
        if not self.is_fitted:
            raise ValueError("Reducer not fitted!")

        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
        return self.pca.transform(features).flatten().astype(np.float32)

# ==============================================================================
# 8. Graph Construction
# ==============================================================================
def build_sparse_bipartite_graph(main_feat: np.ndarray, kmer_count: np.ndarray,
                                 bio_feat: np.ndarray, K: int, config: DeployConfig) -> Data:
    """
    Build bipartite graph data for GCN model.
    Combines main features (k-mer) and biological features.
    """
    num_kmer = 4 ** K
    main_feat = main_feat.astype(np.float32)

    if bio_feat is not None and len(bio_feat) > 0:
        bio_feat = bio_feat.astype(np.float32)
        combined_feat = np.concatenate([main_feat, bio_feat])
        x_p = torch.tensor(combined_feat, dtype=torch.float32).unsqueeze(0)
    else:
        x_p = torch.tensor(main_feat, dtype=torch.float32).unsqueeze(0)

    x_f = torch.eye(num_kmer, dtype=torch.float32)

    # Filter valid k-mers
    valid_kmer_idx = np.where(kmer_count >= config.MIN_KMER_COUNT)[0]
    if len(valid_kmer_idx) == 0:
        valid_kmer_idx = np.arange(min(5, num_kmer))

    # Build edges
    edge_index_pf = torch.tensor([[0] * len(valid_kmer_idx), valid_kmer_idx.tolist()], dtype=torch.long)
    edge_index_fp = torch.tensor([valid_kmer_idx.tolist(), [0] * len(valid_kmer_idx)], dtype=torch.long)

    return Data(
        x_p=x_p, x_f=x_f,
        edge_index_pf=edge_index_pf, edge_index_fp=edge_index_fp,
        num_nodes_p=1, num_nodes_f=num_kmer, num_nodes=1 + num_kmer,
        y=torch.tensor([0], dtype=torch.long)
    )

# ==============================================================================
# 9. Model Architecture (Same as Training)
# ==============================================================================
class ResidualGCNLayer(nn.Module):
    """Residual GCN layer with BatchNorm and Dropout"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.linear(x)
        if x.size(0) > 1:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + residual

class SimpleModel(nn.Module):
    """
    Main prediction model combining GCN layers and classifier.
    Input: Combined k-mer (640d) + biological (128d) = 768d features.
    """
    def __init__(self, main_feat_dim: int, bio_feat_dim: int, config: DeployConfig):
        super().__init__()
        self.config = config
        total_feat_dim = main_feat_dim + bio_feat_dim

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(total_feat_dim, config.GCN_HIDDEN),
            nn.BatchNorm1d(config.GCN_HIDDEN),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT)
        )

        # GCN layers
        self.gcn_layers = nn.ModuleList([
            ResidualGCNLayer(config.GCN_HIDDEN, config.GCN_HIDDEN, config.DROPOUT)
            for _ in range(config.GCN_LAYERS)
        ])

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.GCN_HIDDEN, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 2)
        )

    def forward(self, batch: Data) -> torch.Tensor:
        x = batch.x_p.squeeze(1)
        if x.dtype != torch.float32:
            x = x.float()

        # Handle single sample BatchNorm (evaluation mode)
        if x.size(0) == 1:
            for module in self.input_projection.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()
            for module in self.classifier.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()

        x = self.input_projection(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # GCN forward
        for layer in self.gcn_layers:
            x = layer(x)

        # Classification
        return self.classifier(x)

# ==============================================================================
# 10. Prediction Pipeline with Auto-Translation
# ==============================================================================
class GenePredictor:
    """
    Main prediction class encapsulating all preprocessing and inference.
    Loads model and reducers on initialization.
    Automatically translates CDS to protein when needed.
    """
    def __init__(self):
        self.config = DeployConfig()
        self.seq_processor = SequenceProcessor(self.config.SEQ_MAX_LEN, "smart")
        self.bio_extractor = BioFeatureExtractor()
        self.translator = Translator()  # Initialize translator
        self.device = self.config.DEVICE

        # Load model and reducers
        self._load_resources()

    def _load_resources(self):
        """Load trained model, PCA reducers, and configuration"""
        try:
            # Load reducers
            main_reducer_data = joblib.load("main_reducer.pkl")
            bio_reducer_data = joblib.load("bio_reducer.pkl")
            self.main_reducer = FeatureReducer(main_reducer_data)
            self.bio_reducer = FeatureReducer(bio_reducer_data)

            # Load model configuration
            with open("model_config.json", "r") as f:
                model_cfg = json.load(f)

            # Initialize model architecture
            self.model = SimpleModel(
                main_feat_dim=model_cfg['main_feat_dim'],
                bio_feat_dim=model_cfg['bio_feat_dim'],
                config=self.config
            ).to(self.device)

            # Load trained weights
            state_dict = torch.load("final_best_model.pth", map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

            st.success("✅ Model loaded successfully with translation support!")

        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            raise e

    def predict_single(self, sequence: str, seq_id: str = "Query") -> dict:
        """
        Predict single sequence.
        Automatically translates CDS to protein.
        Returns dictionary with prediction results.
        """
        try:
            # Clean and validate sequence
            clean_seq = self.translator.clean_sequence(sequence)
            if len(clean_seq) < 50:
                return {
                    "Sequence_ID": seq_id,
                    "Error": "Sequence too short (minimum 50 nucleotides)",
                    "Prediction": "Error"
                }

            # Preprocess DNA sequence
            processed_seq = self.seq_processor.process_sequence(clean_seq)

            # Extract k-mer features
            main_feat, kmer_count = matrix_encoding_no_bio(
                processed_seq, self.config.K, self.config.D
            )

            # Extract CDS features (256-dim)
            cds_features = self.bio_extractor.extract_cds_features(clean_seq)

            # === AUTOMATIC TRANSLATION (NEW FEATURE) ===
            # Translate CDS to protein sequence
            protein_seq = self.translator.translate_cds_to_protein(
                clean_seq, method="longest_orf"
            )

            # Extract AA features from translated protein (128-dim)
            if len(protein_seq) > 0:
                aa_features = self.bio_extractor.extract_aa_features(protein_seq)
            else:
                # Fallback: if translation fails, use zero vector
                aa_features = np.zeros(self.config.AA_FEATURE_DIM, dtype=np.float32)

            # Combine biological features (256 + 128 = 384-dim)
            bio_feat = np.concatenate([cds_features, aa_features])

            # Reduce dimensions using fitted PCA
            reduced_main = self.main_reducer.transform(main_feat)
            reduced_bio = self.bio_reducer.transform(bio_feat)

            # Build graph
            graph = build_sparse_bipartite_graph(
                reduced_main, kmer_count, reduced_bio,
                self.config.K, self.config
            )

            # Inference
            with torch.no_grad():
                batch = graph.to(self.device)
                if batch.x_p.dtype != torch.float32:
                    batch.x_p = batch.x_p.float()

                logits = self.model(batch)
                probs = F.softmax(logits, dim=1)

                essential_prob = probs[0][1].item()
                is_essential = essential_prob >= self.config.DEFAULT_THRESHOLD

                return {
                    "Sequence_ID": seq_id,
                    "DNA_Length": len(clean_seq),
                    "Protein_Length": len(protein_seq),
                    "Translated_Protein": protein_seq[:50] + "..." if len(protein_seq) > 50 else protein_seq,
                    "Essential_Probability": round(essential_prob, 4),
                    "Prediction": "Essential" if is_essential else "Non-Essential",
                    "Confidence": round(max(essential_prob, 1-essential_prob), 4)
                }

        except Exception as e:
            return {
                "Sequence_ID": seq_id,
                "Error": str(e),
                "Prediction": "Error"
            }

    def predict_fasta(self, fasta_content: str) -> pd.DataFrame:
        """
        Predict multiple sequences from FASTA format string.
        Automatically translates all CDS sequences.
        Returns DataFrame with all results.
        """
        results = []
        translation_stats = {"success": 0, "failed": 0}

        try:
            # Parse FASTA
            fasta_io = StringIO(fasta_content)
            for record in SeqIO.parse(fasta_io, "fasta"):
                result = self.predict_single(str(record.seq), record.id)
                results.append(result)

                # Track translation statistics
                if "Error" not in result:
                    if result.get("Protein_Length", 0) > 0:
                        translation_stats["success"] += 1
                    else:
                        translation_stats["failed"] += 1

        except Exception as e:
            st.error(f"Error parsing FASTA: {str(e)}")

        # Store translation stats in session for display
        st.session_state.translation_stats = translation_stats
        return pd.DataFrame(results)

# ==============================================================================
# 11. Web Interface (Streamlit)
# ==============================================================================
def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None
    if 'translation_stats' not in st.session_state:
        st.session_state.translation_stats = None

def get_download_link(df: pd.DataFrame, format_type: str, filename: str):
    """
    Generate download link for different file formats.
    Returns HTML anchor tag for download.
    """
    if format_type == "csv":
        data = df.to_csv(index=False)
        mime = "text/csv"
        b64 = base64.b64encode(data.encode()).decode()
    elif format_type == "json":
        data = df.to_json(orient="records", indent=2)
        mime = "application/json"
        b64 = base64.b64encode(data.encode()).decode()
    elif format_type == "excel":
        output = BytesIO()
        df.to_excel(output, index=False, sheet_name='Predictions')
        data = output.getvalue()
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        b64 = base64.b64encode(data).decode()

    return f'<a href="data:{mime};base64,{b64}" download="{filename}">Download {format_type.upper()}</a>'

def main():
    """Main Streamlit application entry point"""
    st.set_page_config(
        page_title="Arabidopsis Essential Gene Predictor",
        page_icon="🧬",
        layout="wide"
    )

    init_session_state()

    # Header
    st.title("🧬 Arabidopsis Essential Gene Prediction System")
    st.markdown("""
    **Deep Learning Model for Predicting Essential Genes in *Arabidopsis thaliana***
    
    This system uses a Graph Convolutional Network (GCN) with biological feature integration 
    to classify gene sequences as Essential or Non-Essential.
    
    🔬 **New Feature**: Automatic CDS to Protein translation (3-frame ORF detection)
    """)

    # Initialize predictor (singleton pattern)
    if st.session_state.predictor is None:
        with st.spinner("🔄 Loading model and reducers... (One-time initialization)"):
            st.session_state.predictor = GenePredictor()

    predictor = st.session_state.predictor

    # Sidebar controls
    st.sidebar.header("⚙️ Prediction Settings")
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Adjust threshold for essential gene classification (default: 0.1)"
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Model Architecture:**
    - Input: K-mer features (640D) + Biological features (128D)
    - Translation: Automatic 3-frame CDS → Protein
    - GCN Layers: 3 (256 hidden units)
    - Classifier: 768 → 256 → 64 → 2
    """)

    # Main input area
    tab1, tab2 = st.tabs(["📝 Single Sequence", "📁 FASTA File Upload"])

    results = None

    with tab1:
        st.subheader("Input Nucleotide Sequence (CDS)")
        st.info("💡 Input your CDS sequence. The system will automatically translate it to protein using the longest ORF detection across 3 reading frames.")
        seq_input = st.text_area(
            "Paste your DNA sequence here (ATCG only):",
            height=200,
            help="Enter the CDS (Coding Sequence). The system will automatically detect the correct reading frame and translate to protein."
        )

        if st.button("🔍 Predict Single Sequence", type="primary"):
            if len(seq_input) < 50:
                st.warning("⚠️ Sequence too short! Minimum 50 nucleotides required.")
            else:
                with st.spinner("🧬 Translating CDS and predicting..."):
                    result = predictor.predict_single(seq_input, "User_Query")
                    results = pd.DataFrame([result])
                    st.session_state.last_results = results

                    # Show translation details
                    if "Translated_Protein" in result:
                        with st.expander("🔬 View Translation Details"):
                            st.write(f"**Protein Length**: {result.get('Protein_Length', 0)} aa")
                            st.write(f"**Translated Sequence** (first 50 aa):")
                            st.code(result["Translated_Protein"])

    with tab2:
        st.subheader("Batch Prediction")
        st.info("Upload a FASTA file containing multiple CDS sequences. Each sequence will be automatically translated and predicted.")
        fasta_file = st.file_uploader(
            "Upload FASTA file (.fa, .fasta, .txt):",
            type=['fa', 'fasta', 'txt']
        )

        if fasta_file is not None:
            fasta_content = fasta_file.getvalue().decode("utf-8")
            st.text_area("Preview (first 1000 chars):", fasta_content[:1000], height=100)

            if st.button("🔍 Predict All Sequences", type="primary"):
                with st.spinner("🧬 Processing batch prediction with automatic translation... This may take a moment"):
                    results = predictor.predict_fasta(fasta_content)
                    st.session_state.last_results = results

                    # Show translation statistics
                    if st.session_state.translation_stats:
                        stats = st.session_state.translation_stats
                        st.success(f"✅ Translation complete: {stats['success']} successful, {stats['failed']} failed")

    # Results display
    if st.session_state.last_results is not None:
        df = st.session_state.last_results

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        # Statistics
        if "Prediction" in df.columns and "Error" not in df.columns:
            col1, col2, col3, col4 = st.columns(4)

            total = len(df)
            essential_count = len(df[df['Prediction'] == 'Essential'])
            non_essential_count = total - essential_count
            avg_conf = df['Confidence'].mean() if 'Confidence' in df.columns else 0

            with col1:
                st.metric("Total Sequences", total)
            with col2:
                st.metric("🎯 Essential Genes", essential_count,
                         f"{essential_count/total*100:.1f}%" if total > 0 else "0%")
            with col3:
                st.metric("⭕ Non-Essential Genes", non_essential_count,
                         f"{non_essential_count/total*100:.1f}%" if total > 0 else "0%")
            with col4:
                st.metric("Avg Confidence", f"{avg_conf:.2%}")

            # Visualization
            st.markdown("#### Distribution")
            chart_data = pd.DataFrame({
                'Category': ['Essential', 'Non-Essential'],
                'Count': [essential_count, non_essential_count]
            })
            st.bar_chart(chart_data.set_index('Category'))

        # Data table
        st.markdown("#### Detailed Results")
        # Select columns to display (hide raw protein sequence if too long for table)
        display_cols = [col for col in df.columns if col != "Translated_Protein" or len(df) <= 5]
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        # Download section
        st.markdown("#### 💾 Download Results")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            st.markdown(
                get_download_link(df, "csv", "predictions.csv"),
                unsafe_allow_html=True
            )
        with dl_col2:
            st.markdown(
                get_download_link(df, "json", "predictions.json"),
                unsafe_allow_html=True
            )
        with dl_col3:
            st.markdown(
                get_download_link(df, "excel", "predictions.xlsx"),
                unsafe_allow_html=True
            )

        # Summary text
        if "Prediction" in df.columns and "Error" not in df.columns:
            st.markdown(f"""
            **Summary Report:**
            - **Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            - **Model Version:** GCN-v1 with Auto-Translation (5-Fold CV Optimized)
            - **Total Predictions:** {len(df)}
            - **Essential Genes Identified:** {essential_count} ({essential_count/total*100:.2f}%)
            - **Non-Essential Genes:** {non_essential_count} ({non_essential_count/total*100:.2f}%)
            - **Classification Threshold:** {threshold}
            - **Translation Method:** Longest ORF detection (3-frame translation)
            """)

if __name__ == "__main__":
    main()