"""
ProteinMPNN transfer-learning baseline.

Uses a *frozen* ProteinMPNN as a per-mutation feature extractor and fits a small
supervised head (Ridge by default, or a torch MLP) on top of the extracted
features to predict mutation stability.

Per-mutation embedding (concatenated, see `build_mutation_features`):

    [ h_dec[pos] | W_s(wt) | W_s(mut) | zero_shot_logodds ]

      * h_dec[pos]          decoder hidden state at the mutated residue   (128)
      * W_s(wt)             ProteinMPNN amino-acid embedding of WT aa      (128)
      * W_s(mut)            ProteinMPNN amino-acid embedding of MUT aa     (128)
      * zero_shot_logodds   log p(mut) - log p(wt) at that position        (1)

A single forward pass per protein yields both `log_probs` and `h_dec`, so
embedding extraction is cheap.

The supervised target is the *stabilization direction* value (higher == more
stable):  -ddG for ddG datasets, +value otherwise. The fitted head therefore
produces a `pred_score` where higher means more stable, matching the baseline
prediction CSV convention.

Example
-------
    export PYTHONPATH="./"
    python -m maxwell.proteinmpnn_transfer \
        --train_data stability-dataset/ddG_maxwell/train \
        --test_data  stability-dataset/ddG_maxwell/test \
        --save_to    predictions/proteinmpnn_transfer.csv

    # or k-fold cross-validation over a single dataset (full coverage):
    python -m maxwell.proteinmpnn_transfer \
        --data stability-dataset/ddG_maxwell --cv 5 \
        --save_to predictions/proteinmpnn_transfer_cv.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .protein_mpnn_utils import ProteinMPNN, parse_PDB as parse_pdb_mpnn, tied_featurize

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
# Columns searched (in order) for the experimental label; first match wins.
DEFAULT_SCORE_COLUMNS = ["ddG", "dTm", "score"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _new_proteinmpnn() -> ProteinMPNN:
    return ProteinMPNN(
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


def load_extractor(checkpoint_path: str, device: torch.device) -> ProteinMPNN:
    """Load a frozen ProteinMPNN feature extractor.

    Accepts either the vanilla ProteinMPNN weights (``{"model_state_dict": ...}``)
    or a Lightning checkpoint (``state_dict`` with a ``model.`` prefix).
    """
    model = _new_proteinmpnn()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:  # Lightning checkpoint
        state_dict = {
            k[len("model."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_protein_embeddings(model: ProteinMPNN, pdb_path: str, device: torch.device, chain: str = "A"):
    """Run one forward pass and return per-residue features for a protein.

    Returns
    -------
    h_dec     : [L, hidden_dim] decoder hidden state per residue (numpy)
    log_probs : [L, 21] log-softmax over amino acids per residue (numpy)
    S         : [L] native (structure-derived) amino-acid indices (numpy)
    """
    pdb_dict_list = parse_pdb_mpnn(pdb_path, input_chain_list=[chain])
    X, S, mask, lengths, chain_M, chain_encoding_all, _, _, _, _, _, _, residue_idx, _, _, _, _, _, _, _ = \
        tied_featurize(pdb_dict_list, device, None)

    B, L = S.shape
    chain_M = torch.ones_like(mask)
    decoding_order = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    log_probs, h_dec = model(
        X, S, mask, chain_M, residue_idx, chain_encoding_all,
        randn=None, use_input_decoding_order=True, decoding_order=decoding_order,
    )
    return (
        h_dec[0].cpu().numpy(),
        log_probs[0].cpu().numpy(),
        S[0].cpu().numpy(),
    )


def build_mutation_features(mutant: str, h_dec: np.ndarray, log_probs: np.ndarray, aa_embed: np.ndarray):
    """Build the per-mutation feature vector. Returns (feature_vec, zero_shot) or None if invalid."""
    if ":" in mutant or ";" in mutant:  # multi-site mutations are skipped, matching the datasets
        return None
    wt, mut = mutant[0], mutant[-1]
    try:
        pos = int(mutant[1:-1]) - 1
    except ValueError:
        return None
    L = h_dec.shape[0]
    if pos < 0 or pos >= L or wt not in ALPHABET or mut not in ALPHABET:
        return None

    wt_idx, mut_idx = ALPHABET.index(wt), ALPHABET.index(mut)
    zero_shot = float(log_probs[pos, mut_idx] - log_probs[pos, wt_idx])
    feat = np.concatenate([
        h_dec[pos],          # structural + sequence context at the site
        aa_embed[wt_idx],    # WT amino-acid embedding
        aa_embed[mut_idx],   # MUT amino-acid embedding
        [zero_shot],         # zero-shot ProteinMPNN log-odds score
    ]).astype(np.float32)
    return feat, zero_shot


def _resolve_label_column(df: pd.DataFrame, candidates: list[str]):
    """Return (column_name, sign) where higher (sign * value) == more stable.

    sign = -1 for ddG (lower ddG is more stable), +1 otherwise.
    """
    for col in candidates:
        if col in df.columns:
            return col, (-1 if col == "ddG" else 1)
    return None, 1


def extract_dataset(model, data_dir: str, score_columns: list[str], device, chain: str = "A") -> pd.DataFrame:
    """Extract features for every (valid, single-site) mutation under ``data_dir``.

    Expects ``data_dir/{mutant,pdb}/`` with matching stems. Returns a DataFrame
    with columns: protein, row_index, mutant, label, target, zero_shot, feat.
    """
    data_path = Path(data_dir)
    mutant_folder = data_path / "mutant"
    pdb_folder = data_path / "pdb"
    assert mutant_folder.exists(), f"Mutant folder not found: {mutant_folder}"
    assert pdb_folder.exists(), f"PDB folder not found: {pdb_folder}"

    aa_embed = model.W_s.weight.detach().cpu().numpy()  # [21, hidden_dim]
    records = []
    for mutant_file in sorted(mutant_folder.glob("*.csv")):
        protein = mutant_file.stem
        pdb_file = pdb_folder / f"{protein}.pdb"
        if not pdb_file.exists():
            print(f"[skip] missing PDB for {protein}")
            continue
        df = pd.read_csv(mutant_file)
        label_col, sign = _resolve_label_column(df, score_columns)
        if label_col is None:
            print(f"[skip] no label column {score_columns} in {mutant_file.name}")
            continue

        h_dec, log_probs, _ = extract_protein_embeddings(model, str(pdb_file), device, chain=chain)
        for row_index, row in df.iterrows():
            built = build_mutation_features(str(row["mutant"]), h_dec, log_probs, aa_embed)
            if built is None:
                continue
            feat, zero_shot = built
            label = float(row[label_col])
            records.append({
                "protein": protein,
                "row_index": int(row_index),
                "mutant": str(row["mutant"]),
                "label": label,         # raw experimental value (goes into ddG column)
                "target": sign * label,  # stabilization direction: higher == more stable
                "zero_shot": zero_shot,
                "feat": feat,
            })
    print(f"[{data_dir}] extracted {len(records)} mutations "
          f"from {len({r['protein'] for r in records})} proteins")
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Supervised head
# ---------------------------------------------------------------------------
class _TorchMLP:
    """Minimal standardized MLP regressor with an sklearn-like fit/predict API."""

    def __init__(self, in_dim, hidden=256, epochs=200, lr=1e-3, weight_decay=1e-4, device="cpu", seed=0):
        torch.manual_seed(seed)
        self.device = device
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay
        self.scaler = StandardScaler()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, 1),
        ).to(device)

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32, device=self.device).unsqueeze(-1)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(Xt), yt)
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        X = self.scaler.transform(X)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.net.eval()
        with torch.no_grad():
            return self.net(Xt).squeeze(-1).cpu().numpy()


def make_head(head: str, in_dim: int, alpha: float, device: str):
    if head == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    if head == "mlp":
        return _TorchMLP(in_dim=in_dim, device=device)
    raise ValueError(f"Unknown head: {head}")


def _stack(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["feat"].values)


def fit_predict_split(train_df, test_df, head, alpha, device) -> np.ndarray:
    model = make_head(head, in_dim=_stack(train_df).shape[1], alpha=alpha, device=device)
    model.fit(_stack(train_df), train_df["target"].values)
    return model.predict(_stack(test_df))


def cross_val_predict(df, n_splits, head, alpha, device) -> np.ndarray:
    """Out-of-fold predictions, grouping by protein so a protein is never split across folds."""
    preds = np.full(len(df), np.nan, dtype=np.float64)
    groups = df["protein"].values
    n_splits = min(n_splits, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    X = _stack(df)
    y = df["target"].values
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        model = make_head(head, in_dim=X.shape[1], alpha=alpha, device=device)
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
        print(f"  fold {fold + 1}/{n_splits}: train={len(tr)} test={len(te)}")
    return preds


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def build_baseline_df(df: pd.DataFrame, pred_score: np.ndarray, score_mode: str, checkpoint: str) -> pd.DataFrame:
    """Assemble the baseline prediction CSV (see project baseline format)."""
    out = pd.DataFrame({
        "protein": df["protein"].values,
        "row_index": df["row_index"].values,
        "mutant": df["mutant"].values,
        "ddG": df["label"].values,                 # raw experimental label
        "target_internal_neg_ddG": df["target"].values,  # higher == more stable
        "pred_score": pred_score,                  # higher == more stable
        "score_mode": score_mode,
        "checkpoint": checkpoint,
        "note": "proteinmpnn frozen features + supervised head",
    })
    return out


def report_metrics(target: np.ndarray, pred: np.ndarray):
    ok = ~(np.isnan(target) | np.isnan(pred))
    if ok.sum() < 2:
        print("Not enough valid predictions to compute metrics.")
        return
    rho, _ = spearmanr(pred[ok], target[ok])
    r, _ = pearsonr(pred[ok], target[ok])
    print(f"\nOverall  Spearman={rho:.4f}  Pearson={r:.4f}  (n={int(ok.sum())})")


def main():
    parser = argparse.ArgumentParser(description="ProteinMPNN transfer-learning baseline")
    parser.add_argument("--train_data", type=str, default=None, help="Train dataset dir (mutant/ + pdb/)")
    parser.add_argument("--test_data", type=str, default=None, help="Test dataset dir (mutant/ + pdb/)")
    parser.add_argument("--data", type=str, default=None, help="Single dataset dir for --cv cross-validation")
    parser.add_argument("--cv", type=int, default=0, help="k-fold CV on --data (grouped by protein)")
    parser.add_argument("--checkpoint_path", type=str,
                        default="vendors/ProteinMPNN/vanilla_model_weights/v_48_020.pt",
                        help="Frozen ProteinMPNN weights (vanilla .pt or Lightning .ckpt)")
    parser.add_argument("--head", type=str, default="ridge", choices=["ridge", "mlp"])
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength")
    parser.add_argument("--score_column", type=str, nargs="+", default=DEFAULT_SCORE_COLUMNS,
                        help="Experimental label column(s); first found per file is used")
    parser.add_argument("--chain", type=str, default="A")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_to", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading frozen ProteinMPNN from {args.checkpoint_path} on {device}...")
    model = load_extractor(args.checkpoint_path, device)

    if args.cv and args.data:
        df = extract_dataset(model, args.data, args.score_column, device, chain=args.chain)
        print(f"Running {args.cv}-fold grouped CV with head={args.head}...")
        pred = cross_val_predict(df, args.cv, args.head, args.alpha, device.type)
        score_mode = f"proteinmpnn_transfer_{args.head}_cv{args.cv}"
    elif args.train_data and args.test_data:
        train_df = extract_dataset(model, args.train_data, args.score_column, device, chain=args.chain)
        test_df = extract_dataset(model, args.test_data, args.score_column, device, chain=args.chain)
        print(f"Fitting head={args.head} on {len(train_df)} train mutations...")
        pred = fit_predict_split(train_df, test_df, args.head, args.alpha, device.type)
        df = test_df
        score_mode = f"proteinmpnn_transfer_{args.head}"
    else:
        parser.error("Provide either (--train_data and --test_data) or (--data and --cv).")

    report_metrics(df["target"].values, pred)

    out = build_baseline_df(df, pred, score_mode=score_mode, checkpoint=args.checkpoint_path)
    save_path = Path(args.save_to)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)
    print(f"Wrote {len(out)} rows -> {save_path}")


if __name__ == "__main__":
    main()
