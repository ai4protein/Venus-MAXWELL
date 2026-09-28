from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

def make_callbacks(args):
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,   
        mode=args.monitor_mode,
        save_top_k=args.save_top_k,
        dirpath=args.checkpoint_dir,
        filename=args.experiment_name + "-{epoch:02d}-{val_SC:.4f}",
    )
    early_stopping_callback = EarlyStopping(
        monitor=args.monitor,
        mode=args.monitor_mode,
        patience=args.early_stopping_patience,
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(early_stopping_callback)
    return callbacks

def make_logger(args):
    if args.logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir="tb_logs",
            name=args.experiment_name,
        )
    elif args.logger == "wandb":
        logger = WandbLogger(
            project=args.wandb_project,
        )
        dict_args = vars(args)
        logger.experiment.config.update(dict_args)
    else:
        logger = None
    return logger