from torch.utils.data import DataLoader
import torch
from Bio import SeqIO
from pathlib import Path
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

def read_single_sequence(fasta_file):
    records = SeqIO.parse(fasta_file, "fasta")
    for each in records:
        return str(each.seq)

def load_mutants(tokenizer, fasta_folder, mutant_folder):
    mutant_files = list(Path(mutant_folder).glob("*.csv"))
    fasta_files = []
    for mutant_file in mutant_files:
        fasta_file = mutant_file.with_suffix(".fasta")
        fasta_file = Path(fasta_folder) / fasta_file.name
        fasta_files.append(fasta_file)
    dataset = []
    for fasta_file, mutant_file in zip(fasta_files, mutant_files):
        seq = read_single_sequence(fasta_file)
        df = pd.read_csv(mutant_file)
        vocab = tokenizer.get_vocab()
        input_ids = tokenizer(seq, return_tensors="pt")["input_ids"][0]
        landscape = torch.zeros((len(input_ids), len(vocab)))
        mask = torch.zeros((len(input_ids), len(vocab)))
        
        for idx, row in df.iterrows():
            if ":" in row["mutant"] or ";" in row["mutant"]:
                continue
            mutant = row["mutant"]
            wt, idx, mut = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
            landscape[idx + 1, vocab[mut]] = -row["ddG"] # bos token
            mask[idx + 1, vocab[mut]] = 1
        dataset.append({
            "seq": seq,
            "landscape": landscape,
            "mask": mask
        })
    return dataset


class MutantCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_sequences = [each["seq"] for each in batch]
        landscapes = [each["landscape"] for each in batch] # B * [L, V]
        masks = [each["mask"] for each in batch] # B * [L, V]
        tokenized_outputs = self.tokenizer(
            input_sequences, return_tensors="pt", padding=True
        )
        landscapes = pad_sequence(landscapes, batch_first=True, padding_value=0.0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return {
            "input_ids": tokenized_outputs["input_ids"],
            "attention_mask": tokenized_outputs["attention_mask"],
            "landscape": landscapes,
            "mask": masks,
        }
        
def get_mutant_dataloader(tokenizer, fasta_folder, mutant_folder, batch_size=4, shuffle=False):
    collate_fn = MutantCollator(tokenizer)
    dataset = load_mutants(tokenizer, fasta_folder, mutant_folder)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dl


if __name__ == "__main__":
    from transformers.models.esm import EsmTokenizer
    
    fasta_f = "custom_datasets/train/fasta"
    mutant_f = "custom_datasets/train/mutant"
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    dl = get_mutant_dataloader(tokenizer, fasta_f, mutant_f, batch_size=4, shuffle=False)
    for each in dl:
        print(each['landscape'].shape)
        print(each['mask'].sum())
        print(each['input_ids'].shape)
        print(each["landscape"].shape)
        break