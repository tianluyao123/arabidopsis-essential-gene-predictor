"""
Arabidopsis Essential Gene Prediction Web Application (ONNX Version)
Lightweight deployment using ONNX Runtime instead of PyTorch.
Features:
- Online sequence input (FASTA format supported)
- Automatic CDS to Protein Translation
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
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
from io import StringIO, BytesIO
import base64
from Bio import SeqIO

# ==============================================================================
# 1. Genetic Code Configuration
# ==============================================================================
GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M', 'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'TAT': 'Y', 'TAC': 'Y',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G', 'TAA': '*', 'TAG': '*', 'TGA': '*'
}


# ==============================================================================
# 2. Configuration
# ==============================================================================
class DeployConfig:
    K = 3
    D = 4
    SEQ_MAX_LEN = 1900
    MAIN_FEAT_DIM = 640
    BIO_FEATURE_DIM = 128
    RAW_BIO_FEATURE_DIM = 384
    CDS_FEATURE_DIM = 256
    AA_FEATURE_DIM = 128
    MIN_KMER_COUNT = 2
    DEFAULT_THRESHOLD = 0.1


# ==============================================================================
# 3. Sequence Processor
# ==============================================================================
class SequenceProcessor:
    def __init__(self, max_len: int, strategy: str = "smart"):
        self.max_len = max_len
        self.strategy = strategy

    def process_sequence(self, seq_str: str) -> str:
        seq_str = seq_str.upper()
        if len(seq_str) > self.max_len:
            start = (len(seq_str) - self.max_len) // 2
            return seq_str[start:start + self.max_len]
        elif len(seq_str) < self.max_len:
            total_pad = self.max_len - len(seq_str)
            left_pad = total_pad // 2
            return "N" * left_pad + seq_str + "N" * (total_pad - left_pad)
        return seq_str


# ==============================================================================
# 4. Translator (CDS to Protein)
# ==============================================================================
class Translator:
    @staticmethod
    def clean_sequence(seq: str) -> str:
        return ''.join(c for c in seq.upper() if c in 'ATCG')

    @staticmethod
    def translate_frame(seq: str, frame: int = 0) -> str:
        seq = ''.join(c for c in seq.upper() if c in 'ATCG')
        if len(seq) < 3:
            return ""
        seq = seq[frame:]
        length = len(seq) - (len(seq) % 3)
        seq = seq[:length]
        protein = []
        for i in range(0, length, 3):
            codon = seq[i:i + 3]
            protein.append(GENETIC_CODE.get(codon, '?'))
        return ''.join(protein)

    @staticmethod
    def translate_cds_to_protein(cds_seq: str) -> str:
        cds_seq = ''.join(c for c in cds_seq.upper() if c in 'ATCG')
        if len(cds_seq) < 3:
            return ""

        all_orfs = []
        for frame in range(3):
            protein = Translator.translate_frame(cds_seq, frame)
            orfs = []
            in_orf = False
            start_pos = 0
            for i, aa in enumerate(protein):
                if aa == 'M' and not in_orf:
                    in_orf = True
                    start_pos = i
                elif aa == '*' and in_orf:
                    in_orf = False
                    orf_seq = protein[start_pos:i]
                    if len(orf_seq) >= 10:
                        orfs.append(orf_seq)
            if in_orf:
                orf_seq = protein[start_pos:]
                if len(orf_seq) >= 10:
                    orfs.append(orf_seq)
            all_orfs.extend(orfs)

        if not all_orfs:
            protein = Translator.translate_frame(cds_seq, 0)
            return protein.replace('*', '')

        return max(all_orfs, key=len)


# ==============================================================================
# 5. K-mer Encoding
# ==============================================================================
def _num_transfer(seq: str) -> str:
    seq = seq.upper()
    seq = seq.replace("A", "0").replace("C", "1").replace("G", "2").replace("T", "3")
    return "".join(filter(str.isdigit, seq))


def _num_transfer_loc(num_seq: str, K: int) -> list:
    loc = []
    seq_len = len(num_seq)
    if seq_len < K:
        num_seq = num_seq.ljust(K, "0")
        seq_len = K
    for i in range(seq_len - K + 1):
        kmer_val = int(num_seq[i:i + K], base=4)
        loc.append(kmer_val)
    return loc


def matrix_encoding_no_bio(seq: str, K: int, d: int) -> np.ndarray:
    seq = seq.upper()
    seq_length = len(seq)
    num_seq = _num_transfer(seq) or "0" * K
    loc_list = _num_transfer_loc(num_seq, K)

    d = min(d, 4)
    dis_list = [[0], [1], [3], [5]][:d]
    num_kmer = 4 ** K
    kmer_features = []

    for dis in dis_list:
        dis_val = dis[0]
        matrix = np.zeros((num_kmer, num_kmer), dtype=np.float32)
        num = 0
        valid_len = len(loc_list) - K - dis_val
        if valid_len > 0:
            for i in range(valid_len):
                idx1 = loc_list[i]
                idx2 = loc_list[i + K + dis_val]
                if idx1 < num_kmer and idx2 < num_kmer:
                    matrix[idx1][idx2] += 1
            num = max(1, seq_length - 2 * K - dis_val + 1)
        matrix = matrix / num if num != 0 else matrix
        kmer_features.append(matrix.flatten())

    if kmer_features:
        main_feat = np.hstack(kmer_features) * 100
    else:
        main_feat = np.zeros(num_kmer * num_kmer * d, dtype=np.float32)

    return main_feat.astype(np.float32)


# ==============================================================================
# 6. BioFeatureExtractor
# ==============================================================================
class BioFeatureExtractor:
    def __init__(self):
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
        features = []
        if not cds_seq:
            return np.zeros(256, dtype=np.float32)

        seq_len = len(cds_seq)
        gc_content = (cds_seq.count('G') + cds_seq.count('C')) / seq_len
        at_content = (cds_seq.count('A') + cds_seq.count('T')) / seq_len
        gc_skew = (cds_seq.count('G') - cds_seq.count('C')) / (cds_seq.count('G') + cds_seq.count('C') + 1e-8)
        at_skew = (cds_seq.count('A') - cds_seq.count('T')) / (cds_seq.count('A') + cds_seq.count('T') + 1e-8)
        features.extend([gc_content, at_content, gc_skew, at_skew])

        nucleotides = ['A', 'C', 'G', 'T']
        for nt in nucleotides:
            features.append(cds_seq.count(nt) / seq_len)

        di_nucleotides = [a + b for a in nucleotides for b in nucleotides]
        total_pairs = max(1, seq_len - 1)
        for di in di_nucleotides:
            features.append(cds_seq.count(di) / total_pairs)

        tri_nucleotides = [a + b + c for a in nucleotides for b in nucleotides for c in nucleotides][:16]
        total_triplets = max(1, seq_len - 2)
        for tri in tri_nucleotides:
            features.append(cds_seq.count(tri) / total_triplets)

        features.extend([seq_len / 1000.0, np.log(seq_len + 1), seq_len / 3000.0])

        if seq_len >= 3:
            codons = [cds_seq[i:i + 3] for i in range(0, seq_len - 2, 3)]
            codon_count = len(codons)
            if codon_count > 0:
                unique_codons = len(set(codons))
                codon_diversity = unique_codons / codon_count
                counts = [codons.count(c) for c in set(codons)]
                effective_codons = 1 / sum((c / codon_count) ** 2 for c in counts) if codon_count > 0 else 0
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

        if len(features) < 256:
            features.extend([0.0] * (256 - len(features)))
        elif len(features) > 256:
            features = features[:256]

        return np.array(features, dtype=np.float32)

    def extract_aa_features(self, aa_seq: str) -> np.ndarray:
        features = []
        if not aa_seq or len(aa_seq) == 0:
            return np.zeros(128, dtype=np.float32)

        total = len(aa_seq)
        aa_list = 'ACDEFGHIKLMNPQRSTVWY'

        for aa in aa_list:
            features.append(aa_seq.count(aa) / total)

        for group in self.amino_acid_groups.values():
            count = sum(1 for aa in aa_seq if aa in group)
            features.append(count / total)

        representative_aa = list(aa_list)
        dipeptides = [a + b for a in representative_aa for b in representative_aa][:20]
        total_pairs = max(1, len(aa_seq) - 1)
        for di in dipeptides:
            count = sum(1 for i in range(len(aa_seq) - 1) if aa_seq[i:i + 2] == di)
            features.append(count / total_pairs)

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

        helix = sum(1 for aa in aa_seq if aa in ['E', 'A', 'L', 'M', 'Q', 'K', 'R'])
        sheet = sum(1 for aa in aa_seq if aa in ['V', 'I', 'Y', 'F', 'W', 'T'])
        coil = sum(1 for aa in aa_seq if aa in ['G', 'P', 'S', 'D', 'N'])
        features.extend([helix / total, sheet / total, coil / total])

        hydrophobic = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['hydrophobic'])
        hydrophilic = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['hydrophilic'])
        pos_charge = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['charged_positive'])
        neg_charge = sum(1 for aa in aa_seq if aa in self.amino_acid_groups['charged_negative'])
        features.extend([
            hydrophobic / total, hydrophilic / total,
            pos_charge / total, neg_charge / total,
            (pos_charge - neg_charge) / total
        ])

        features.extend([
            len(aa_seq) / 1000.0,
            len(set(aa_seq)) / len(aa_seq),
            len(aa_seq) / 500.0
        ])

        features.extend([
            aa_seq.count('C') / total,
            aa_seq.count('P') / total,
            aa_seq.count('G') / total
        ])

        if len(features) < 128:
            features.extend([0.0] * (128 - len(features)))
        elif len(features) > 128:
            features = features[:128]

        return np.array(features, dtype=np.float32)


# ==============================================================================
# 7. Feature Reducer
# ==============================================================================
class FeatureReducer:
    def __init__(self, reducer_data: dict):
        self.pca = reducer_data['pca']
        self.scaler = reducer_data['scaler']
        self.target_dim = reducer_data['target_dim']
        self.is_fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Reducer not fitted!")

        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
        return self.pca.transform(features).flatten().astype(np.float32)


# ==============================================================================
# 8. ONNX Predictor (New Implementation)
# ==============================================================================
class GenePredictorONNX:
    """
    ONNX-based predictor for lightweight deployment.
    Uses ONNX Runtime instead of PyTorch.
    """

    def __init__(self):
        self.config = DeployConfig()
        self.seq_processor = SequenceProcessor(self.config.SEQ_MAX_LEN, "smart")
        self.bio_extractor = BioFeatureExtractor()
        self.translator = Translator()

        self._load_resources()

    def _load_resources(self):
        """Load ONNX model and reducers"""
        try:
            # Load reducers
            main_reducer_data = joblib.load("main_reducer.pkl")
            bio_reducer_data = joblib.load("bio_reducer.pkl")
            self.main_reducer = FeatureReducer(main_reducer_data)
            self.bio_reducer = FeatureReducer(bio_reducer_data)

            # Load ONNX model
            self.ort_session = ort.InferenceSession("gene_predictor.onnx")
            self.input_name = self.ort_session.get_inputs()[0].name

            st.success("✅ ONNX Model loaded successfully!")

        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            raise e

    def predict_single(self, sequence: str, seq_id: str = "Query") -> dict:
        """Predict single sequence using ONNX Runtime"""
        try:
            clean_seq = self.translator.clean_sequence(sequence)
            if len(clean_seq) < 50:
                return {
                    "Sequence_ID": seq_id,
                    "Error": "Sequence too short (minimum 50 nucleotides)",
                    "Prediction": "Error"
                }

            processed_seq = self.seq_processor.process_sequence(clean_seq)
            main_feat = matrix_encoding_no_bio(processed_seq, self.config.K, self.config.D)

            # Extract CDS features
            cds_features = self.bio_extractor.extract_cds_features(clean_seq)

            # Translate to protein
            protein_seq = self.translator.translate_cds_to_protein(clean_seq)

            # Extract AA features
            if len(protein_seq) > 0:
                aa_features = self.bio_extractor.extract_aa_features(protein_seq)
            else:
                aa_features = np.zeros(128, dtype=np.float32)

            # Combine and reduce
            bio_feat = np.concatenate([cds_features, aa_features])
            reduced_main = self.main_reducer.transform(main_feat)
            reduced_bio = self.bio_reducer.transform(bio_feat)

            # Combine features for ONNX input (768-dim)
            combined_input = np.concatenate([reduced_main, reduced_bio]).astype(np.float32).reshape(1, -1)

            # ONNX Inference
            ort_outputs = self.ort_session.run(None, {self.input_name: combined_input})
            logits = ort_outputs[0][0]  # Get first (and only) output, first batch

            # Softmax to get probabilities
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            essential_prob = float(probs[1])
            is_essential = essential_prob >= self.config.DEFAULT_THRESHOLD

            return {
                "Sequence_ID": seq_id,
                "DNA_Length": len(clean_seq),
                "Protein_Length": len(protein_seq),
                "Translated_Protein": protein_seq[:50] + "..." if len(protein_seq) > 50 else protein_seq,
                "Essential_Probability": round(essential_prob, 4),
                "Prediction": "Essential" if is_essential else "Non-Essential",
                "Confidence": round(max(essential_prob, 1 - essential_prob), 4)
            }

        except Exception as e:
            import traceback
            return {
                "Sequence_ID": seq_id,
                "Error": str(e),
                "Traceback": traceback.format_exc(),
                "Prediction": "Error"
            }

    def predict_fasta(self, fasta_content: str) -> pd.DataFrame:
        """Predict multiple sequences from FASTA format"""
        results = []
        translation_stats = {"success": 0, "failed": 0}

        try:
            fasta_io = StringIO(fasta_content)
            for record in SeqIO.parse(fasta_io, "fasta"):
                result = self.predict_single(str(record.seq), record.id)
                results.append(result)

                if "Error" not in result:
                    if result.get("Protein_Length", 0) > 0:
                        translation_stats["success"] += 1
                    else:
                        translation_stats["failed"] += 1

        except Exception as e:
            st.error(f"Error parsing FASTA: {str(e)}")

        st.session_state.translation_stats = translation_stats
        return pd.DataFrame(results)


# ==============================================================================
# 9. Web Interface
# ==============================================================================
def init_session_state():
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'last_results' not in st.session_state:
        st.session_state.last_results = None
    if 'translation_stats' not in st.session_state:
        st.session_state.translation_stats = None


def get_download_link(df: pd.DataFrame, format_type: str, filename: str):
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
    st.set_page_config(
        page_title="Arabidopsis Essential Gene Predictor (ONNX)",
        page_icon="🧬",
        layout="wide"
    )

    init_session_state()

    st.title("🧬 Arabidopsis Essential Gene Prediction System")
    st.markdown("""
    **Deep Learning Model (ONNX Version) for Predicting Essential Genes in *Arabidopsis thaliana***

    ✅ Lightweight deployment | ⚡ Fast inference | 🧬 Auto-translation included
    """)

    if st.session_state.predictor is None:
        with st.spinner("🔄 Loading ONNX model... (One-time initialization)"):
            st.session_state.predictor = GenePredictorONNX()

    predictor = st.session_state.predictor

    st.sidebar.header("⚙️ Prediction Settings")
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Model Info:**
    - Format: ONNX Runtime
    - Input: 768-dim (640 k-mer + 128 bio)
    - Translation: 3-frame ORF detection
    """)

    tab1, tab2 = st.tabs(["📝 Single Sequence", "📁 FASTA File Upload"])

    results = None

    with tab1:
        st.subheader("Input Nucleotide Sequence (CDS)")
        st.info("💡 Input CDS sequence. System will auto-translate to protein using longest ORF detection.")
        seq_input = st.text_area(
            "Paste your DNA sequence here (ATCG only):",
            height=200
        )

        if st.button("🔍 Predict Single Sequence", type="primary"):
            if len(seq_input) < 50:
                st.warning("⚠️ Sequence too short! Minimum 50 nucleotides required.")
            else:
                with st.spinner("🧬 Translating and predicting..."):
                    result = predictor.predict_single(seq_input, "User_Query")
                    results = pd.DataFrame([result])
                    st.session_state.last_results = results

                    if "Translated_Protein" in result:
                        with st.expander("🔬 View Translation Details"):
                            st.write(f"**Protein Length**: {result.get('Protein_Length', 0)} aa")
                            st.code(result["Translated_Protein"])

    with tab2:
        st.subheader("Batch Prediction")
        fasta_file = st.file_uploader(
            "Upload FASTA file (.fa, .fasta, .txt):",
            type=['fa', 'fasta', 'txt']
        )

        if fasta_file is not None:
            fasta_content = fasta_file.getvalue().decode("utf-8")
            st.text_area("Preview (first 1000 chars):", fasta_content[:1000], height=100)

            if st.button("🔍 Predict All Sequences", type="primary"):
                with st.spinner("🧬 Processing batch prediction..."):
                    results = predictor.predict_fasta(fasta_content)
                    st.session_state.last_results = results

                    if st.session_state.translation_stats:
                        stats = st.session_state.translation_stats
                        st.success(f"✅ Translation complete: {stats['success']} successful, {stats['failed']} failed")

    if st.session_state.last_results is not None:
        df = st.session_state.last_results

        st.markdown("---")
        st.subheader("📊 Prediction Results")

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
                          f"{essential_count / total * 100:.1f}%" if total > 0 else "0%")
            with col3:
                st.metric("⭕ Non-Essential Genes", non_essential_count,
                          f"{non_essential_count / total * 100:.1f}%" if total > 0 else "0%")
            with col4:
                st.metric("Avg Confidence", f"{avg_conf:.2%}")

            st.markdown("#### Distribution")
            chart_data = pd.DataFrame({
                'Category': ['Essential', 'Non-Essential'],
                'Count': [essential_count, non_essential_count]
            })
            st.bar_chart(chart_data.set_index('Category'))

        st.markdown("#### Detailed Results")
        display_cols = [col for col in df.columns if col != "Translated_Protein" or len(df) <= 5]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("#### 💾 Download Results")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            st.markdown(get_download_link(df, "csv", "predictions.csv"), unsafe_allow_html=True)
        with dl_col2:
            st.markdown(get_download_link(df, "json", "predictions.json"), unsafe_allow_html=True)
        with dl_col3:
            st.markdown(get_download_link(df, "excel", "predictions.xlsx"), unsafe_allow_html=True)

        if "Prediction" in df.columns and "Error" not in df.columns:
            st.markdown(f"""
            **Summary Report:**
            - **Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
            - **Model Version:** GCN-ONNX (Optimized for Web)
            - **Total Predictions:** {len(df)}
            - **Essential Genes:** {essential_count} ({essential_count / total * 100:.2f}%)
            - **Non-Essential Genes:** {non_essential_count} ({non_essential_count / total * 100:.2f}%)
            """)


if __name__ == "__main__":
    main()