"""
Standalone ProteinMPNN+ESM inference script.
No dependency on PyTorch Lightning runtime.
"""
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer

from ..protein_mpnn_utils import parse_PDB as parse_pdb_mpnn, tied_featurize
from ..protein_mpnnesm import ProteinMPNNESM


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def _read_input_names(input_list_txt: str):
    names = []
    with open(input_list_txt, "r") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    return names


def _get_ckpt_arg(checkpoint: dict, key: str, default: str):
    hparams = checkpoint.get("hyper_parameters", {})
    args = hparams.get("args", None)
    if args is None:
        return default
    return getattr(args, key, default)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    esm_model_path: Optional[str],
    mpnn_model_path: Optional[str],
) -> ProteinMPNNESM:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if esm_model_path is None:
        esm_model_path = _get_ckpt_arg(checkpoint, "esm_model_path", "facebook/esm2_t33_650M_UR50D")
    if mpnn_model_path is None:
        mpnn_model_path = _get_ckpt_arg(checkpoint, "mpnn_model_path", "vendors/ProteinMPNN/vanilla_model_weights/v_48_020.pt")

    state_dict = checkpoint["state_dict"]
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state_dict[key[6:]] = value

    model = ProteinMPNNESM(esm_path=esm_model_path, mpnn_path=mpnn_model_path)
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def score_mutations(
    model: ProteinMPNNESM,
    tokenizer: AutoTokenizer,
    pdb_path: str,
    mutant_df: pd.DataFrame,
    device: torch.device,
    chain: str = "A",
):
    model.eval()
    model.to(device)

    pdb_dict_list = parse_pdb_mpnn(pdb_path, input_chain_list=[chain])
    X, S, mask, _, _, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = tied_featurize(
        pdb_dict_list, device, None
    )

    seq = "".join(ALPHABET[idx] for idx in S[0].tolist())
    esm_tokenized = tokenizer([seq], return_tensors="pt")
    esm_input_ids = esm_tokenized["input_ids"].to(device)
    esm_attention_mask = esm_tokenized["attention_mask"].to(device)

    batch = {
        "X": X,
        "S": S,
        "attention_mask": mask,
        "residue_idx": residue_idx,
        "chain_encoding": chain_encoding_all,
        "esm_input_ids": esm_input_ids,
        "esm_attention_mask": esm_attention_mask,
    }
    landscape, _ = model(batch, no_log_scale=False)  # [B, L, 21]
    seq_len = S.shape[1]

    scores = []
    for _, row in mutant_df.iterrows():
        mutant = row["mutant"]
        try:
            _, pos, mut = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
            if pos < 0 or pos >= seq_len:
                scores.append(float("nan"))
                continue
            if mut not in ALPHABET:
                scores.append(float("nan"))
                continue
            mut_idx = ALPHABET.index(mut)
            scores.append(landscape[0, pos, mut_idx].item())
        except Exception:
            scores.append(float("nan"))
    return scores


def main():
    parser = argparse.ArgumentParser(description="ProteinMPNN+ESM Mutation Scoring (Standalone)")
    parser.add_argument("--pdb", type=str, help="Path to pdb file")
    parser.add_argument("--mutant", type=str, default=None, help="Path to mutant csv file or directory")
    parser.add_argument("--data_root", type=str, default="dataset/data", help="Root path containing mutant/ and pdb/ folders")
    parser.add_argument("--input_list_txt", type=str, default=None, help="Text file with one protein name per line (without extension)")
    parser.add_argument("--output", type=str, default="output.csv", help="Path to output csv file")
    parser.add_argument("--checkpoint_path", type=str, nargs="+", required=True, help="Path to one or more checkpoints")
    parser.add_argument("--chain", type=str, default="A", help="Chain ID to use")
    parser.add_argument("--esm_model_path", type=str, default=None, help="ESM2 model path (optional, can be inferred from checkpoint)")
    parser.add_argument("--mpnn_model_path", type=str, default=None, help="ProteinMPNN base model path (optional, can be inferred from checkpoint)")
    parser.add_argument("--score_column", type=str, default="ddG", help="Ground truth score column for Spearmanr calculation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save", action="store_true", help="Save scores to file (default: only print metrics)")
    args = parser.parse_args()

    if args.input_list_txt is None and args.mutant is None:
        parser.error("Either --mutant or --input_list_txt must be provided.")

    tokenizer_model_path = args.esm_model_path or "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    device = torch.device(args.device)

    models = []
    model_names = []
    for cp_path in args.checkpoint_path:
        print(f"Loading model from {cp_path}...")
        model = load_model_from_checkpoint(
            cp_path,
            device,
            esm_model_path=args.esm_model_path,
            mpnn_model_path=args.mpnn_model_path,
        )
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
                scores = score_mutations(model, tokenizer, str(pdb_file), df, device, chain=args.chain)
                df[m_name] = scores
                all_model_scores.append(scores)
            df["score"] = np.mean(all_model_scores, axis=0)
            target = -df[args.score_column].values if args.score_column == "ddG" else df[args.score_column].values
            rho, _ = spearmanr(df["score"].values, target)
            corr, _ = pearsonr(df["score"].values, target)
            spearman_scores.append(rho)
            pearson_scores.append(corr)

            if args.save:
                df.to_csv(mutant_file, index=False)

        if spearman_scores:
            print(f"\nAverage Ensemble Spearmanr across {len(spearman_scores)} files: {np.mean(spearman_scores):.4f}")
            print(f"Average Ensemble Pearsonr across {len(pearson_scores)} files: {np.mean(pearson_scores):.4f}")
            if args.save:
                print("Scores saved to source files")
    else:
        if args.pdb is None:
            parser.error("--pdb is required when --mutant points to a single csv file.")
        df = pd.read_csv(args.mutant)
        all_model_scores = []
        for m_idx, model in enumerate(models):
            m_name = model_names[m_idx]
            scores = score_mutations(model, tokenizer, args.pdb, df, device, chain=args.chain)
            df[m_name] = scores
            all_model_scores.append(scores)
        df["score"] = np.mean(all_model_scores, axis=0)
        target = -df[args.score_column].values if args.score_column == "ddG" else df[args.score_column].values
        rho, _ = spearmanr(df["score"].values, target)
        print(f"Ensemble Spearmanr: {rho:.4f}")
        if args.save:
            df.to_csv(args.output, index=False)
            print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
