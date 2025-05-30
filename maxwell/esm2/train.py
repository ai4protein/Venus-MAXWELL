from argparse import ArgumentParser
from .dataset import get_mutant_dataloader
from .model import MaxwellWrapperForESM2
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from transformers.models.esm import EsmTokenizer
from pathlib import Path

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--train_folder", type=str, default="ddg_datasets/train")
    parser.add_argument("--valid_folder", type=str, default="ddg_datasets/valid")
    parser.add_argument("--test_folder", type=str, default="ddg_datasets/test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    return parser.parse_args()

def main():
    args = get_args()
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    
    train_dataloader = get_mutant_dataloader(
        tokenizer=tokenizer,
        fasta_folder=Path(args.train_folder) / "fasta",
        mutant_folder=Path(args.train_folder) / "mutant",
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    valid_dataloader = get_mutant_dataloader(
        tokenizer=tokenizer,
        fasta_folder=Path(args.valid_folder) / "fasta",
        mutant_folder=Path(args.valid_folder) / "mutant",
        batch_size=1,
        shuffle=False,
    )
    
    test_dataloader = get_mutant_dataloader(
        tokenizer=tokenizer,
        fasta_folder=Path(args.test_folder) / "fasta",
        mutant_folder=Path(args.test_folder) / "mutant",
        batch_size=1,
        shuffle=False,
    )
    
    model = MaxwellWrapperForESM2(lr=args.lr)
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[
            ModelCheckpoint(
                monitor="val_rho_spearman",
                mode="max",
                save_top_k=1,
                filename="maxwell-esm2-{epoch:02d}-{val_rho_spearman:.4f}",
                dirpath="checkpoints",
                verbose=True,
            ),
            EarlyStopping(
                monitor="val_rho_spearman",
                mode="max",
                patience=args.patience,
                verbose=True,
            )
        ],
    )
    trainer.test(model, test_dataloader)
    trainer.fit(model, train_dataloader, valid_dataloader)
    trainer.test(model, test_dataloader)
    
if __name__ == "__main__":
    main()