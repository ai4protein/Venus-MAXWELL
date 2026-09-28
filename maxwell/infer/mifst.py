"""
Standalone MIF-ST inference script.
No dependency on PyTorch Lightning runtime.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sequence_models.collaters import PROTEIN_ALPHABET, SimpleCollater, StructureCollater
from sequence_models.pdb_utils import parse_PDB, process_coords
from sequence_models.pretrained import load_model_and_alphabet
from tqdm import tqdm


ALPHABET = PROTEIN_ALPHABET


def _read_input_names(input_list_txt: str):
    names = []
    with open(input_list_txt, "r") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    return names


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint["state_dict"]
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            model_state_dict[key[6:]] = value

    model, _ = load_model_and_alphabet("mifst")
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)
    return model


def _build_mifst_inputs(pdb_path: str, structure_collater: StructureCollater, device: torch.device):
    coords, seq, _ = parse_PDB(pdb_path)
    coords_dict = {"N": coords[:, 0], "CA": coords[:, 1], "C": coords[:, 2]}
    dist, omega, theta, phi = process_coords(coords_dict)
    mifst_batch = [[
        seq,
        torch.tensor(dist, dtype=torch.float32),
        torch.tensor(omega, dtype=torch.float32),
        torch.tensor(theta, dtype=torch.float32),
        torch.tensor(phi, dtype=torch.float32),
    ]]
    input_ids, nodes, edges, connections, edge_mask = structure_collater(mifst_batch)
    return {
        "input_ids": input_ids.to(device),
        "nodes": nodes.to(device),
        "edges": edges.to(device),
        "connections": connections.to(device),
        "edge_mask": edge_mask.to(device),
    }


@torch.no_grad()
def score_mutations(model, pdb_path: str, mutant_df: pd.DataFrame, structure_collater: StructureCollater, device: torch.device):
    model.eval()
    model.to(device)

    batch = _build_mifst_inputs(pdb_path, structure_collater, device)
    _, logits = model(
        batch["input_ids"],
        batch["nodes"],
        batch["edges"],
        batch["connections"],
        batch["edge_mask"],
    )

    A = torch.log_softmax(logits, dim=-1)
    L = A - A.gather(-1, batch["input_ids"].unsqueeze(-1))
    seq_len = batch["input_ids"].shape[1]

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
            scores.append(L[0, pos, mut_idx].item())
        except Exception:
            scores.append(float("nan"))
    return scores


def main():
    parser = argparse.ArgumentParser(description="MIF-ST Mutation Scoring (Standalone)")
    parser.add_argument("--pdb", type=str, help="Path to pdb file (single-file mode)")
    parser.add_argument("--mutant", type=str, default=None, help="Path to mutant csv file or directory")
    parser.add_argument("--data_root", type=str, default="dataset/data", help="Root path containing mutant/ and pdb/ folders")
    parser.add_argument("--input_list_txt", type=str, default=None, help="Text file with one protein name per line (without extension)")
    parser.add_argument("--output", type=str, default="output.csv", help="Path to output csv file")
    parser.add_argument("--checkpoint_path", type=str, nargs="+", required=True, help="Path to one or more checkpoints")
    parser.add_argument("--score_column", type=str, default="ddG", help="Ground truth score column for Spearmanr calculation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save", action="store_true", help="Save scores to file (default: only print metrics)")
    args = parser.parse_args()

    if args.input_list_txt is None and args.mutant is None:
        parser.error("Either --mutant or --input_list_txt must be provided.")

    sequence_collater = SimpleCollater(PROTEIN_ALPHABET, pad=True)
    structure_collater = StructureCollater(sequence_collater, n_connections=30)
    device = torch.device(args.device)

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
                scores = score_mutations(model, str(pdb_file), df, structure_collater, device)
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
            scores = score_mutations(model, args.pdb, df, structure_collater, device)
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
