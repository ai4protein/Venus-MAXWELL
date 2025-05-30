import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import esm.inverse_folding
from biotite.structure.io import pdb
from biotite.structure import filter_backbone
from biotite.structure import get_chains
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def read_sequence_from_pdb(pdb_file):
    _, native_seq = esm.inverse_folding.util.load_coords(str(pdb_file), "A")
    return native_seq

def load_pdb(fin, chain=None):
    pdbf = pdb.PDBFile.read(fin)
    structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure

def load_coords(fin, chain="A"):
    structure = load_pdb(fin, chain)
    return esm.inverse_folding.util.extract_coords_from_structure(structure)


class MutantNetIF(nn.Module):
    
    def __init__(self, device="cuda"):
        super(MutantNetIF, self).__init__()
        # 加载ESM-IF模型
        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.model.to(device)
        self.model.eval()
        vocab_size = len(self.alphabet)
        self.extra_head = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.SELU(),
            torch.nn.Linear(512, vocab_size)
        )
        self.aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        self.aa_index = torch.tensor([self.alphabet.get_idx(a) for a in self.aa])
        self.device = device
    
    @torch.no_grad()
    def predict(self, fin):
        coords, native_seq = load_coords(fin)
        batch = [(coords, None, native_seq)]
        batch_converter = esm.inverse_folding.util.CoordBatchConverter(self.alphabet)
        coords, confidence, strs, tokens, padding_mask = batch_converter(
            batch, device=self.device
        )
        prev_output_tokens = tokens[:, :-1].to(self.device)
        logits, _ = self.model.forward(
            coords, padding_mask, confidence, prev_output_tokens
        )
        logits = logits.transpose(1, 2)
        logits = torch.log_softmax(logits, dim=-1)
        one_hot = F.one_hot(tokens[:, 1:], num_classes=len(self.alphabet))
        logits = logits - (logits * one_hot).sum(dim=-1, keepdim=True)
        logits = logits.squeeze()
        return logits
    
    def get_landscape(self, fin):
        logits = self.predict(fin)
        landscape = torch.zeros((len(logits), len(self.aa)))
        for i in range(len(logits)):
            landscape[i, :] = logits[i, self.aa_index]
        return -landscape.cpu().numpy()


def get_landscape_from_pdb(pdb_file, ckpt_path, output_file, device="cuda"):
    model = MutantNetIF(device=device)
    alphabet = model.alphabet
    state = torch.load(ckpt_path, map_location="cpu")
    state = state["state_dict"]
    print(model.load_state_dict(state, strict=False))
    model.eval()
    seq = read_sequence_from_pdb(pdb_file)
    landscape = model.get_landscape(pdb_file)
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    data = {"mutant": [], "relative_ddg": []}
    for i in range(len(landscape)):
        for j in range(len(landscape[i])):
            pos = i + 1
            wt = seq[i]
            mt = amino_acids[j]
            score = landscape[i, j].item()
            data["mutant"].append(f"{wt}{pos}{mt}")
            data["relative_ddg"].append(score)
    df = pd.DataFrame(data)
    df = df.sort_values(by="relative_ddg", ascending=True)
    df.to_csv(output_file, index=False)
    return df

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--pdb_file", type=str, default="example_data/fireprotdb_1AG2_ddG.pdb")
    parser.add_argument("--ckpt_path", type=str, default="weights/esmif-maxwell.ckpt")
    parser.add_argument("--output_file", type=str, default="example_data/fireprotdb_1AG2_ddG.csv")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    df = get_landscape_from_pdb(args.pdb_file, args.ckpt_path, args.output_file, args.device)
    print(df.head(30))
    print(f"Save to {args.output_file}")