"""
Standalone ProteinMPNN inference script.
No dependency on PyTorch Lightning or maxwell package.
"""
import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import numpy as np
from ..protein_mpnn_utils import parse_PDB as parse_pdb_mpnn, tied_featurize, ProteinMPNN


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def _read_input_names(input_list_txt: str):
    names = []
    with open(input_list_txt, "r") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    return names


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> ProteinMPNN:
    """
    Load ProteinMPNN model from a Lightning checkpoint.
    
    Lightning checkpoint structure:
    {
        'state_dict': {'model.xxx': tensor, ...},  # model weights with "model." prefix
        'hyper_parameters': {'args': Namespace(...)},  # training args
        ...
    }
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model state dict (remove "model." prefix from keys)
    state_dict = checkpoint['state_dict']
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Remove "model." prefix
            model_state_dict[new_key] = value
    
    # Create model with default ProteinMPNN architecture
    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        vocab=21,
        k_neighbors=48,
        augment_eps=0.0,
        dropout=0.1,
        ca_only=False,
    )
    
    # Load weights
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    
    return model


@torch.no_grad()
def score_mutations(model: ProteinMPNN, pdb_path: str, mutant_df: pd.DataFrame, device: torch.device, chain: str = 'A'):
    """Score mutations using ProteinMPNN model."""
    model.eval()
    model.to(device)
    
    # Parse PDB and featurize
    pdb_dict_list = parse_pdb_mpnn(pdb_path, input_chain_list=[chain])
    X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(pdb_dict_list, device, None)
    
    B, L_seq = S.shape
    # chain_M: which positions to predict (all 1s means predict all positions)
    chain_M = torch.ones_like(mask)
    # Use natural order (0, 1, 2, ..., L-1)
    decoding_order = torch.arange(L_seq, device=device).unsqueeze(0).expand(B, -1)

    # Model inference
    log_probs, _ = model(
        X, S, mask, chain_M, residue_idx, chain_encoding_all,
        randn=None, use_input_decoding_order=True, decoding_order=decoding_order
    )
    
    # Compute score: log_prob(mut) - log_prob(wt)
    A = torch.log_softmax(log_probs, dim=-1)  # [B, L, 21]
    L = A - A.gather(-1, S.detach().unsqueeze(-1))  # [B, L, 21]
    
    scores = []
    for _, row in mutant_df.iterrows():
        mutant = row['mutant']
        try:
            wt, pos, mut = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
            if pos < 0 or pos >= L_seq:
                scores.append(float('nan'))
                continue
            
            if mut not in ALPHABET:
                scores.append(float('nan'))
                continue
            
            mut_idx = ALPHABET.index(mut)
            score = L[0, pos, mut_idx].item()
            scores.append(score)
        except Exception:
            scores.append(float('nan'))
            
    return scores


def _resolve_score_column(df: pd.DataFrame, candidates: list[str]):
    """Return (column_name, sign) for the first candidate found in df.
    sign = -1 for ddG (lower is more stable), +1 otherwise.
    Returns (None, 1) if no candidate is found.
    """
    for col in candidates:
        if col in df.columns:
            sign = -1 if col == 'ddG' else 1
            return col, sign
    return None, 1


def main():
    parser = argparse.ArgumentParser(description='ProteinMPNN Mutation Scoring (Standalone)')
    parser.add_argument('--fasta', type=str, help='Path to fasta file (optional)')
    parser.add_argument('--pdb', type=str, help='Path to pdb file')
    parser.add_argument('--mutant', type=str, default=None, help='Path to mutant csv file or directory')
    parser.add_argument('--data_root', type=str, default='dataset/data', help='Root path containing mutant/ and pdb/ folders')
    parser.add_argument('--input_list_txt', type=str, default=None, help='Text file with one protein name per line (without extension)')
    parser.add_argument('--checkpoint_path', type=str, nargs='+', required=True, help='Path to one or more checkpoints')
    parser.add_argument('--chain', type=str, default='A', help='Chain ID to use')
    parser.add_argument('--score_column', type=str, nargs='+', default=['ddG', 'dTm', 'score'],
                        help='Ground truth score column(s) for Spearmanr calculation; first column found in each file is used')
    parser.add_argument('--pred_column', type=str, default='score', help='Column name for predicted scores in output CSV')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--save_to', type=str, default=None, help='Directory to save scored CSV files')
    
    args = parser.parse_args()
    if args.input_list_txt is None and args.mutant is None:
        parser.error("Either --mutant or --input_list_txt must be provided.")
    
    device = torch.device(args.device)
    
    # Load all models
    models = []
    model_names = []
    for cp_path in args.checkpoint_path:
        print(f"Loading model from {cp_path}...")
        model = load_model_from_checkpoint(cp_path, device)
        models.append(model)
        model_names.append(Path(cp_path).stem)
    
    mutant_file_pairs = []
    if args.input_list_txt is not None:
        data_root = Path(args.data_root)
        mutant_folder = data_root / "mutant"
        pdb_folder = data_root / "pdb"
        assert mutant_folder.exists(), f"Mutant folder not found: {mutant_folder}"
        assert pdb_folder.exists(), f"PDB folder not found: {pdb_folder}"
        names = _read_input_names(args.input_list_txt)
        mutant_file_pairs = [
            (mutant_folder / f"{name}.csv", pdb_folder / f"{name}.pdb")
            for name in names
        ]
        for mutant_file, pdb_file in mutant_file_pairs:
            assert mutant_file.exists(), f"Mutant file not found: {mutant_file}"
            assert pdb_file.exists(), f"PDB file not found: {pdb_file}"
    elif Path(args.mutant).is_dir():
        data_path = Path(args.mutant)
        mutant_folder = data_path / "mutant"
        pdb_folder = data_path / "pdb"
        mutant_files = list(mutant_folder.glob("*.csv"))
        mutant_file_pairs = [
            (mutant_file, pdb_folder / mutant_file.with_suffix(".pdb").name)
            for mutant_file in mutant_files
        ]

    if mutant_file_pairs:
        spearman_scores = []
        pearson_scores = []
        for mutant_file, pdb_file in tqdm(mutant_file_pairs, desc="Processing mutants"):
            df = pd.read_csv(mutant_file)
            all_model_scores = []
            for m_idx, model in enumerate(models):
                m_name = model_names[m_idx]
                scores = score_mutations(model, str(pdb_file), df, device, chain=args.chain)
                df[m_name] = scores  # Column name is the checkpoint name
                all_model_scores.append(scores)
            df[args.pred_column] = np.mean(all_model_scores, axis=0)
            gt_col, sign = _resolve_score_column(df, args.score_column)
            if gt_col is not None:
                rho, _ = spearmanr(df[args.pred_column].values, sign * df[gt_col].values)
                spearman_scores.append(rho)
                corr, _ = pearsonr(df[args.pred_column].values, sign * df[gt_col].values)
                pearson_scores.append(corr)
            
            if args.save_to:
                save_dir = Path(args.save_to)
                save_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(save_dir / mutant_file.name, index=False)

        if spearman_scores:
            print(f"\nAverage Ensemble Spearmanr across {len(spearman_scores)} files: {np.mean(spearman_scores):.4f}")
            print(f"Average Ensemble Pearsonr across {len(pearson_scores)} files: {np.mean(pearson_scores):.4f}")
            if args.save_to:
                print(f"Scores saved to {args.save_to}")
    else:
        # Single file mode
        if args.pdb is None:
            parser.error("--pdb is required when --mutant points to a single csv file.")
        df = pd.read_csv(args.mutant)
        all_model_scores = []
        for m_idx, model in enumerate(models):
            m_name = model_names[m_idx]
            scores = score_mutations(model, args.pdb, df, device, chain=args.chain)
            df[m_name] = scores  # Column name is the checkpoint name
            all_model_scores.append(scores)
        df[args.pred_column] = np.mean(all_model_scores, axis=0)
        gt_col, sign = _resolve_score_column(df, args.score_column)
        if gt_col is not None:
            rho, _ = spearmanr(df[args.pred_column].values, sign * df[gt_col].values)
            print(f"Ensemble Spearmanr: {rho:.4f}")
        if args.save_to:
            save_dir = Path(args.save_to)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_name = Path(args.mutant).name if not Path(args.mutant).is_dir() else "output.csv"
            df.to_csv(save_dir / out_name, index=False)
            print(f"Saved results to {save_dir / out_name}")

if __name__ == '__main__':
    main()
