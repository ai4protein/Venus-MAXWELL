"""
Standalone ProSST inference script.
No dependency on PyTorch Lightning runtime.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


def _read_input_names(input_list_txt: str):
    names = []
    with open(input_list_txt, "r") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    return names


def _read_fasta_sequence(fasta_path: str) -> str:
    seq_lines = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    return "".join(seq_lines)


def _read_structure_tokens(prosst_fasta_path: str) -> torch.Tensor:
    raw = _read_fasta_sequence(prosst_fasta_path)
    ids = [int(x) + 3 for x in raw.split(",") if x != ""]
    return torch.tensor([1, *ids, 2], dtype=torch.long)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_path: str,
) -> AutoModelForMaskedLM:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint["state_dict"]
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state_dict[key[6:]] = value

    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def score_mutations(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    seq: str,
    ss_input_ids: torch.Tensor,
    mutant_df: pd.DataFrame,
    device: torch.device,
):
    model.eval()
    model.to(device)

    tokenized = tokenizer([seq], return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    ss_input_ids = ss_input_ids.unsqueeze(0).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
    )

    logits = outputs.logits[:, 1:-1, :]
    wt_ids = input_ids[:, 1:-1]
    A = torch.log_softmax(logits, dim=-1)
    L = A - A.gather(-1, wt_ids.unsqueeze(-1))
    seq_len = wt_ids.shape[1]

    vocab = tokenizer.get_vocab()
    scores = []
    for _, row in mutant_df.iterrows():
        mutant = row["mutant"]
        try:
            _, pos, mut = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
            if pos < 0 or pos >= seq_len:
                scores.append(float("nan"))
                continue
            mut_idx = vocab.get(mut)
            if mut_idx is None:
                scores.append(float("nan"))
                continue
            scores.append(L[0, pos, mut_idx].item())
        except Exception:
            scores.append(float("nan"))
    return scores


def main():
    parser = argparse.ArgumentParser(description="ProSST Mutation Scoring (Standalone)")
    parser.add_argument("--fasta", type=str, help="Path to fasta file (single-file mode)")
    parser.add_argument("--prosst_fasta", type=str, help="Path to ProSST structure token fasta (single-file mode)")
    parser.add_argument("--mutant", type=str, default=None, help="Path to mutant csv file or directory")
    parser.add_argument("--data_root", type=str, default="dataset/data", help="Root path containing mutant/ and fasta/ folders")
    parser.add_argument("--prosst_data_path", type=str, default="dataset/data/prosst", help="Path containing ProSST structure token fasta files")
    parser.add_argument("--input_list_txt", type=str, default=None, help="Text file with one protein name per line (without extension)")
    parser.add_argument("--output", type=str, default="output.csv", help="Path to output csv file")
    parser.add_argument("--checkpoint_path", type=str, nargs="+", required=True, help="Path to one or more checkpoints")
    parser.add_argument("--model_path", type=str, default="AI4Protein/ProSST-2048", help="Base ProSST model path")
    parser.add_argument("--score_column", type=str, default="ddG", help="Ground truth score column for Spearmanr calculation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save", action="store_true", help="Save scores to file (default: only print metrics)")
    args = parser.parse_args()

    if args.input_list_txt is None and args.mutant is None:
        parser.error("Either --mutant or --input_list_txt must be provided.")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    models = []
    model_names = []
    for cp_path in args.checkpoint_path:
        print(f"Loading model from {cp_path}...")
        model = load_model_from_checkpoint(cp_path, device, args.model_path)
        models.append(model)
        model_names.append(Path(cp_path).stem)

    mutant_file_pairs = []
    if args.input_list_txt is not None:
        data_root = Path(args.data_root)
        prosst_root = Path(args.prosst_data_path)
        mutant_folder = data_root / "mutant"
        fasta_folder = data_root / "fasta"
        assert mutant_folder.exists(), f"Mutant folder not found: {mutant_folder}"
        assert fasta_folder.exists(), f"Fasta folder not found: {fasta_folder}"
        assert prosst_root.exists(), f"ProSST folder not found: {prosst_root}"
        names = _read_input_names(args.input_list_txt)
        mutant_file_pairs = [
            (
                mutant_folder / f"{name}.csv",
                fasta_folder / f"{name}.fasta",
                prosst_root / f"{name}.fasta",
            )
            for name in names
        ]
        for mutant_file, fasta_file, prosst_file in mutant_file_pairs:
            assert mutant_file.exists(), f"Mutant file not found: {mutant_file}"
            assert fasta_file.exists(), f"Fasta file not found: {fasta_file}"
            assert prosst_file.exists(), f"ProSST file not found: {prosst_file}"
    elif Path(args.mutant).is_dir():
        data_path = Path(args.mutant)
        prosst_root = Path(args.prosst_data_path)
        mutant_folder = data_path / "mutant"
        fasta_folder = data_path / "fasta"
        mutant_files = list(mutant_folder.glob("*.csv"))
        mutant_file_pairs = [
            (
                mutant_file,
                fasta_folder / mutant_file.with_suffix(".fasta").name,
                prosst_root / mutant_file.with_suffix(".fasta").name,
            )
            for mutant_file in mutant_files
        ]

    if mutant_file_pairs:
        spearman_scores = []
        pearson_scores = []
        for mutant_file, fasta_file, prosst_file in tqdm(mutant_file_pairs, desc="Processing mutants"):
            df = pd.read_csv(mutant_file)
            seq = _read_fasta_sequence(str(fasta_file))
            ss_input_ids = _read_structure_tokens(str(prosst_file))
            if len(ss_input_ids) != len(seq) + 2:
                raise ValueError(
                    f"ProSST structure length mismatch in {prosst_file}: "
                    f"expected {len(seq) + 2}, got {len(ss_input_ids)}"
                )
            all_model_scores = []
            for m_idx, model in enumerate(models):
                m_name = model_names[m_idx]
                scores = score_mutations(model, tokenizer, seq, ss_input_ids, df, device)
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
        if args.fasta is None:
            parser.error("--fasta is required when --mutant points to a single csv file.")
        if args.prosst_fasta is None:
            parser.error("--prosst_fasta is required when --mutant points to a single csv file.")
        df = pd.read_csv(args.mutant)
        seq = _read_fasta_sequence(args.fasta)
        ss_input_ids = _read_structure_tokens(args.prosst_fasta)
        if len(ss_input_ids) != len(seq) + 2:
            raise ValueError(
                f"ProSST structure length mismatch in {args.prosst_fasta}: "
                f"expected {len(seq) + 2}, got {len(ss_input_ids)}"
            )
        all_model_scores = []
        for m_idx, model in enumerate(models):
            m_name = model_names[m_idx]
            scores = score_mutations(model, tokenizer, seq, ss_input_ids, df, device)
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
