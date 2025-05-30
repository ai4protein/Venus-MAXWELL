from torch.utils.data import DataLoader
import torch
from Bio import SeqIO
from pathlib import Path
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import esm
import esm.inverse_folding


def read_single_sequence(fasta_file):
    records = SeqIO.parse(fasta_file, "fasta")
    for each in records:
        return str(each.seq)


def load_mutants(alphabet, fasta_folder, mutant_folder, pdb_folder):
    mutant_files = list(Path(mutant_folder).glob("*.csv"))
    fasta_files = []
    pdb_files = []
    for mutant_file in mutant_files:
        fasta_file = mutant_file.with_suffix(".fasta")
        fasta_file = Path(fasta_folder) / fasta_file.name
        fasta_files.append(fasta_file)
        pdb_file = mutant_file.with_suffix(".pdb")
        pdb_file = Path(pdb_folder) / pdb_file.name
        pdb_files.append(pdb_file)
    dataset = []
    for fasta_file, mutant_file, pdb_file in zip(fasta_files, mutant_files, pdb_files):
        seq = read_single_sequence(fasta_file)
        df = pd.read_csv(mutant_file)
        vocab_size = len(alphabet)
        landscape = torch.zeros((len(seq), vocab_size))
        mask = torch.zeros((len(seq), vocab_size))
        coords, native_seq = esm.inverse_folding.util.load_coords(str(pdb_file), "A")
        assert len(native_seq) == len(seq), "Sequence length mismatch in {} and {}".format(fasta_file, pdb_file)
        for idx, row in df.iterrows():
            if ":" in row["mutant"] or ";" in row["mutant"]:
                continue
            mutant = row["mutant"]
            wt, idx, mut = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
            landscape[idx, alphabet.get_idx(mut)] = -row["ddG"] # bos token
            mask[idx, alphabet.get_idx(mut)] = 1
        dataset.append({
            "seq": seq,
            "coords": coords,
            "landscape": landscape,
            "mask": mask,
            "coords": coords,
            "confidence": None
        })
    return dataset


class MutantCollator:

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.batch_converter = esm.inverse_folding.util.CoordBatchConverter(self.alphabet)
        
    def __call__(self, batch):
        input_sequences = [each["seq"] for each in batch]
        landscapes = [each["landscape"] for each in batch] # B * [L, V]
        masks = [each["mask"] for each in batch] # B * [L, V]
        confidence = [each["confidence"] for each in batch] # B * [L, 1]
        coords = [each["coords"] for each in batch] # B * [L, 3]
        landscapes = pad_sequence(landscapes, batch_first=True, padding_value=0.0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        coords, confidence, _, tokens, padding_mask = self.batch_converter(
            list(zip(coords, confidence, input_sequences)), 
        )
        return {
            "seq": input_sequences,
            "input_ids": tokens,
            "coords": coords,
            "confidence": confidence,
            "attention_mask": padding_mask,
            "landscape": landscapes,
            "mask": masks,
        }
        
def get_mutant_dataloader(alphabet, fasta_folder, mutant_folder, pdb_folder, batch_size=4, shuffle=False):
    collate_fn = MutantCollator(alphabet)
    dataset = load_mutants(alphabet, fasta_folder, mutant_folder, pdb_folder)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dl


if __name__ == "__main__":
    from scipy.stats import spearmanr
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    fasta_f = "custom_datasets/train/fasta"
    mutant_f = "custom_datasets/train/mutant"
    pdb_f = "custom_datasets/train/pdb"
    dl = get_mutant_dataloader(alphabet, fasta_f, mutant_f, pdb_f, batch_size=1, shuffle=False)
    for each in dl:
        print(len(each["seq"][0]))
        print(each["coords"].shape)
        print(each['landscape'].shape)
        print(each['mask'].shape)
        print(each['input_ids'].shape)
        print(each["attention_mask"].shape)
        prev_output_tokens = each["input_ids"][:, :-1]
        extra_, logits = model.forward(
            each["coords"], each["attention_mask"], each["confidence"], prev_output_tokens
        )
        print(each["attention_mask"])
        break