#!/usr/bin/env python3
"""Evaluate one ProteinMPNN-based MAXWELL checkpoint on a labelled dataset.

The script intentionally contains no machine-specific paths. It scores each
protein separately and reports unweighted macro statistics across eligible
proteins. Larger prediction and target values must both indicate greater
stability; use ``--target-sign -1`` for conventional positive-destabilizing
ddG labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maxwell.infer.proteinmpnn import (  # noqa: E402
    load_model_from_checkpoint,
    score_mutations,
)


MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a ProteinMPNN-based MAXWELL checkpoint using per-protein "
            "correlations and optional binary screening metrics."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root containing mutant/ and pdb/ subdirectories.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional frozen CSV with protein and mutation columns. If omitted, "
            "all CSV files under DATA_DIR/mutant are evaluated."
        ),
    )
    parser.add_argument(
        "--protein-column", default="protein", help="Protein-ID column in the manifest."
    )
    parser.add_argument(
        "--mutation-column",
        default="mutation",
        help="Mutation column in the manifest or per-protein CSV files.",
    )
    parser.add_argument(
        "--target-column",
        required=True,
        help="Experimental target column, for example ddG, dTm or score.",
    )
    parser.add_argument(
        "--target-sign",
        type=float,
        choices=(-1.0, 1.0),
        required=True,
        help=(
            "Multiply the target by this value so larger means more stable. "
            "Use -1 for conventional positive-destabilizing ddG and +1 for dTm."
        ),
    )
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--chain", default="A")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda or cuda:N (default: auto).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=None,
        help="Optional target-stability threshold for AUROC/AUPRC/Enrich@10.",
    )
    parser.add_argument(
        "--positive-rule",
        choices=("gt", "ge"),
        default="gt",
        help="Define positives as target > threshold (gt) or >= threshold (ge).",
    )
    parser.add_argument(
        "--include-self-mutations",
        action="store_true",
        help="Keep substitutions such as A42A. They are excluded by default.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def validate_mutation(value: object) -> tuple[str, int, str]:
    match = MUTATION_RE.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"Invalid one-based substitution: {value!r}")
    return match.group(1), int(match.group(2)), match.group(3)


def normalize_columns(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = frame.copy()
    if args.mutation_column not in frame.columns:
        fallback = "mutant" if args.mutation_column == "mutation" else "mutation"
        if fallback in frame.columns:
            frame = frame.rename(columns={fallback: args.mutation_column})
        else:
            raise ValueError(
                f"Missing mutation column {args.mutation_column!r}; columns are "
                f"{frame.columns.tolist()}"
            )
    if args.target_column not in frame.columns:
        raise ValueError(
            f"Missing target column {args.target_column!r}; columns are "
            f"{frame.columns.tolist()}"
        )

    parsed = frame[args.mutation_column].map(validate_mutation)
    frame["wt"] = parsed.map(lambda item: item[0])
    frame["position"] = parsed.map(lambda item: item[1])
    frame["mut"] = parsed.map(lambda item: item[2])
    frame["mutation"] = frame[args.mutation_column].astype(str).str.strip()
    if not args.include_self_mutations:
        frame = frame[frame["wt"] != frame["mut"]].copy()

    frame["target_stability_score"] = (
        pd.to_numeric(frame[args.target_column], errors="raise") * args.target_sign
    )
    finite = np.isfinite(frame["target_stability_score"].to_numpy(dtype=float))
    if not finite.all():
        raise ValueError("The selected target column contains non-finite values.")
    return frame


def load_evaluation_rows(args: argparse.Namespace) -> pd.DataFrame:
    mutant_dir = args.data_dir / "mutant"
    pdb_dir = args.data_dir / "pdb"
    if not mutant_dir.is_dir() or not pdb_dir.is_dir():
        raise FileNotFoundError(
            f"Expected mutant/ and pdb/ under dataset root: {args.data_dir}"
        )

    if args.manifest is not None:
        frame = pd.read_csv(args.manifest)
        if args.protein_column not in frame.columns:
            raise ValueError(
                f"Manifest is missing protein column {args.protein_column!r}."
            )
        frame = normalize_columns(frame, args)
        frame["protein"] = frame[args.protein_column].astype(str)
    else:
        pieces: list[pd.DataFrame] = []
        files = sorted(mutant_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found under {mutant_dir}")
        for csv_path in files:
            part = normalize_columns(pd.read_csv(csv_path), args)
            part["protein"] = csv_path.stem
            pieces.append(part)
        frame = pd.concat(pieces, ignore_index=True)

    if frame.empty:
        raise ValueError("No non-self substitutions remain after input filtering.")
    if frame["protein"].str.contains(r"[\\/]", regex=True).any():
        raise ValueError("Protein IDs must be plain basenames without path separators.")
    duplicates = frame.duplicated(["protein", "mutation"], keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, ["protein", "mutation"]].head().to_dict("records")
        raise ValueError(f"Duplicate protein/mutation rows found, for example: {examples}")

    missing_pdb = [
        protein
        for protein in sorted(frame["protein"].unique())
        if not (pdb_dir / f"{protein}.pdb").is_file()
    ]
    if missing_pdb:
        raise FileNotFoundError(
            f"Missing PDB files for {len(missing_pdb)} proteins; first: {missing_pdb[:5]}"
        )
    return frame.reset_index(drop=True)


def safe_correlation(method, truth: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(truth) < 3 or np.unique(truth).size < 2 or np.unique(prediction).size < 2:
        return None
    value = float(method(truth, prediction).statistic)
    return value if np.isfinite(value) else None


def per_protein_metrics(
    group: pd.DataFrame,
    threshold: float | None,
    positive_rule: str,
) -> dict[str, float | int | str | None]:
    truth = group["target_stability_score"].to_numpy(dtype=float)
    prediction = group["predicted_stability_score"].to_numpy(dtype=float)
    result: dict[str, float | int | str | None] = {
        "protein": str(group["protein"].iloc[0]),
        "n_mutations": int(len(group)),
        "spearman": safe_correlation(spearmanr, truth, prediction),
        "pearson": safe_correlation(pearsonr, truth, prediction),
        "auroc": None,
        "auprc": None,
        "enrich_at_10": None,
    }
    if threshold is None:
        return result

    label = truth > threshold if positive_rule == "gt" else truth >= threshold
    positives = int(label.sum())
    negatives = int(len(label) - positives)
    result["n_positive"] = positives
    result["n_negative"] = negatives
    if positives > 0 and negatives > 0:
        result["auroc"] = float(roc_auc_score(label, prediction))
        result["auprc"] = float(average_precision_score(label, prediction))
    if len(label) >= 10 and positives > 0:
        top_index = np.argsort(prediction)[::-1][:10]
        result["enrich_at_10"] = float(label[top_index].mean() / label.mean())
    return result


def summarize(per_protein: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, float | int | None]] = {}
    for metric in ("spearman", "pearson", "auroc", "auprc", "enrich_at_10"):
        values = pd.to_numeric(per_protein[metric], errors="coerce").dropna()
        summary[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n_proteins": int(len(values)),
        }
    return summary


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if args.manifest is not None and not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    evaluation_rows = load_evaluation_rows(args)
    model = load_model_from_checkpoint(str(checkpoint), device)
    prediction_parts: list[pd.DataFrame] = []
    for protein, group in evaluation_rows.groupby("protein", sort=True):
        pdb_path = data_dir / "pdb" / f"{protein}.pdb"
        query = pd.DataFrame({"mutant": group["mutation"].tolist()})
        scores = score_mutations(model, str(pdb_path), query, device, chain=args.chain)
        scored = group.copy()
        scored["predicted_stability_score"] = scores
        if not np.isfinite(scored["predicted_stability_score"].to_numpy(dtype=float)).all():
            bad = scored.loc[
                ~np.isfinite(scored["predicted_stability_score"].to_numpy(dtype=float)),
                "mutation",
            ].tolist()
            raise ValueError(f"Non-finite predictions for {protein}: {bad[:5]}")
        prediction_parts.append(scored)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_rows = [
        per_protein_metrics(group, args.positive_threshold, args.positive_rule)
        for _, group in predictions.groupby("protein", sort=True)
    ]
    per_protein = pd.DataFrame(metric_rows)
    macro = summarize(per_protein)

    output_dir.mkdir(parents=True, exist_ok=True)
    scored_dir = output_dir / "per_protein_predictions"
    scored_dir.mkdir()
    export_columns = [
        "protein",
        "mutation",
        args.target_column,
        "target_stability_score",
        "predicted_stability_score",
    ]
    export_columns = list(dict.fromkeys(export_columns))
    for protein, group in predictions.groupby("protein", sort=True):
        group[export_columns].to_csv(scored_dir / f"{protein}.csv", index=False)
    predictions[export_columns].to_csv(output_dir / "combined_predictions.csv", index=False)
    per_protein.to_csv(output_dir / "per_protein_metrics.csv", index=False)

    report = {
        "dataset": args.dataset_name,
        "n_proteins": int(predictions["protein"].nunique()),
        "n_mutations": int(len(predictions)),
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": sha256(checkpoint),
        "manifest_file": args.manifest.name if args.manifest is not None else None,
        "manifest_sha256": sha256(args.manifest) if args.manifest is not None else None,
        "target_column": args.target_column,
        "target_sign": args.target_sign,
        "positive_threshold": args.positive_threshold,
        "positive_rule": args.positive_rule if args.positive_threshold is not None else None,
        "chain": args.chain,
        "device": str(device),
        "seed": args.seed,
        "macro_metrics": macro,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (output_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
