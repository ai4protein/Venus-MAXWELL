from argparse import ArgumentParser
from .dataset import get_mutant_dataloader
from .model import MaxwellWrapperForESMIF
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--train_folder", type=str, default="example_datasets/train")
    parser.add_argument("--valid_folder", type=str, default="example_datasets/valid")
    parser.add_argument("--test_folder", type=str, default="example_datasets/test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda_value", type=float, default=0.1)
    return parser.parse_args()

def main():
    args = get_args()
    model = MaxwellWrapperForESMIF(lr=args.lr, lambda_value=args.lambda_value)
    train_dataloader = get_mutant_dataloader(
        alphabet=model.alphabet,
        fasta_folder=Path(args.train_folder) / "fasta",
        mutant_folder=Path(args.train_folder) / "mutant",
        pdb_folder=Path(args.train_folder) / "pdb",
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    valid_dataloader = get_mutant_dataloader(
        alphabet=model.alphabet,
        fasta_folder=Path(args.valid_folder) / "fasta",
        mutant_folder=Path(args.valid_folder) / "mutant", 
        pdb_folder=Path(args.valid_folder) / "pdb",
        batch_size=1,
        shuffle=False
    )
    
    test_dataloader = get_mutant_dataloader(
        alphabet=model.alphabet,
        fasta_folder=Path(args.test_folder) / "fasta",
        mutant_folder=Path(args.test_folder) / "mutant",
        pdb_folder=Path(args.test_folder) / "pdb", 
        batch_size=1,
        shuffle=False
    )
    model_checkpoint_callback = ModelCheckpoint(
        monitor="val_rho_spearman",
        mode="max",
        save_top_k=1,
        filename="maxwell-esmif-{epoch:02d}-{val_rho_spearman:.4f}",
        dirpath="checkpoints",
        verbose=True,
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[
            model_checkpoint_callback,
            EarlyStopping(
                monitor="val_rho_spearman",
                mode="max",
                patience=args.patience,
                verbose=True,
            )
        ],
        val_check_interval=0.5,
    )
    trainer.test(model, test_dataloader)
    trainer.fit(model, train_dataloader, valid_dataloader)
    model = MaxwellWrapperForESMIF.load_from_checkpoint(model_checkpoint_callback.best_model_path)
    trainer.test(model, test_dataloader)
    
if __name__ == "__main__":
    main()