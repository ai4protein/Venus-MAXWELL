from Bio import SeqIO
import esm.inverse_folding
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
def read_sequence_from_fasta(fasta_file):
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)

def read_sequence_from_pdb(pdb_file):
    _, native_seq = esm.inverse_folding.util.load_coords(str(pdb_file), "A")
    return native_seq

def read_mutants_from_csv(mutant_file):
    df = pd.read_csv(mutant_file)
    return df["mutant"].tolist()

def check_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    mutant_path = dataset_path / "mutant"
    fasta_path = dataset_path / "fasta"
    pdb_path = dataset_path / "pdb"
    mutant_files = list(mutant_path.glob("*.csv"))
    names = [each.stem for each in mutant_files]
    for name in names:
        # Check fasta file
        fasta_file = fasta_path / f"{name}.fasta"
        if not fasta_file.exists():
            raise FileNotFoundError(f"Fasta file {fasta_file} does not exist")
        fasta_seq = read_sequence_from_fasta(fasta_file)
        
        pdb_file = pdb_path / f"{name}.pdb"
        if not pdb_file.exists():
            raise FileNotFoundError(f"PDB file {pdb_file} does not exist")
        pdb_seq = read_sequence_from_pdb(pdb_file)
        
        mutant_file = mutant_path / f"{name}.csv"
        if not mutant_file.exists():
            raise FileNotFoundError(f"Mutant file {mutant_file} does not exist")
        mutants = read_mutants_from_csv(mutant_file)
        
        # Check sequence consistency
        if fasta_seq != pdb_seq:
            raise ValueError(f"Sequence mismatch between fasta and pdb for {name}")

        # Check mutant consistency
        for mutant in mutants:
            if ":" in mutant or ";" in mutant:
                raise ValueError(f"Invalid mutant {mutant} in {mutant_file} (Only single amino acid changes are allowed)")
            wt, idx, mut = mutant[0], int(mutant[1:-1]) - 1, mutant[-1]
            if idx < 0 or idx >= len(fasta_seq):
                raise ValueError(f"Invalid mutant {mutant} in {mutant_file} (Index out of bounds)")
            if fasta_seq[idx] != wt:
                raise ValueError(f"Invalid mutant {mutant} in {mutant_file} (Wildtype mismatch with seq in {fasta_file})")
        
        print(f"Dataset {name} passed all checks")
        

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()
    check_dataset(args.dataset_path)

if __name__ == "__main__":
    main()