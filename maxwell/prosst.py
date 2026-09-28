from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from typing import List
from tqdm import tqdm
from torch.utils.data import DataLoader
from perplexity.models.base import BaseModel
from argparse import ArgumentParser
from Bio import SeqIO


def read_fasta(fasta_file: str) -> str:
    for record in SeqIO.parse(fasta_file, "fasta"):
        return str(record.seq)


class ProSST(BaseModel):

    def __init__(self, model_path, device="cuda"):
        self.name = Path(model_path).name
        super().__init__(self.name)
        self.model_path = model_path
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def compute_perplexity(self, sequence: str, *args, **kwargs) -> float:
        prosst_structure = kwargs["prosst_structure"]
        res = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        ss_input_ids = read_fasta(prosst_structure).split(",")
        ss_input_ids = [int(i) + 3 for i in ss_input_ids]
        ss_input_ids = torch.tensor([[1, *ss_input_ids, 2]], dtype=torch.long, device=self.device)
        outputs = self.model(**res, labels=res.input_ids, ss_input_ids=ss_input_ids)
        return torch.exp(outputs.loss).item()

    @torch.no_grad()
    def score_mutants(
        self, sequence: str, mutants: List[str], offset=1, *args, **kwargs
    ) -> List[float]:
        prosst_structure = kwargs["prosst_structure"]
        res = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        ss_input_ids = read_fasta(prosst_structure).split(",")
        ss_input_ids = [int(i) + 3 for i in ss_input_ids]
        ss_input_ids = torch.tensor([[1, *ss_input_ids, 2]], dtype=torch.long, device=self.device)
        outputs = self.model(**res, labels=res.input_ids, ss_input_ids=ss_input_ids)
        logits = outputs.logits
        logits = torch.log_softmax(logits, dim=-1)
        logits = logits[:, 1:-1, :]
        predictions = []
        for mutant in mutants:
            wt = self.tokenizer.get_vocab()[mutant[0]]
            mt = self.tokenizer.get_vocab()[mutant[-1]]
            index = int(mutant[1:-1]) - offset
            if mutant[0] != sequence[index]:
                raise ValueError(f"Invalid mutant: {mutant}")
            predictions.append((logits[0, index, mt] - logits[0, index, wt]).item())
        return predictions
    
    @staticmethod
    def add_model_args(psr: ArgumentParser):
        psr.add_argument("--model_path", type=str, required=True, help="Path to the model")
        psr.add_argument("--device", type=str, default="cuda", help="Device to run the model")
        return psr

def main():
    psr = ArgumentParser()
    psr = ProSST.add_model_args(psr)
    psr = ProSST.add_data_args(psr)
    args = psr.parse_args()
    model = ProSST(args.model_path, args.device)
    model.score_data_dir(
        data_dir=args.data_dir,
        skip_existsing=args.skip_existsing,
        skip_errors=args.skip_errors,
    )
    model.perplexity_data_dir(
        data_dir=args.data_dir,
        use_cache=args.use_cache,
        skip_errors=args.skip_errors,
    )

if __name__ == "__main__":
    main()