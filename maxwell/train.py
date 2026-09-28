import argparse
import lightning as L
from .lit_model import LitModel
from .data_utils import make_train_dataset, make_valid_dataset, make_test_dataset
from .train_utils import make_callbacks, make_logger
import hashlib
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train mutation effect prediction model')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='esm', choices=['esm', 'esmif', 'mifst', 'proteinmpnn', 'proteinmpnn-esm', 'prosst'])
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--esm_model_path', type=str, default='facebook/esm2_t33_650M_UR50D')
    parser.add_argument('--mpnn_model_path', type=str, default='weights/proteinmpnn/v_48_020.pt')
    parser.add_argument('--mpnn_score_mode', type=str, default='autoregressive',
                        choices=['autoregressive', 'conditional', 'unconditional'])
    parser.add_argument('--prosst_data_path', type=str, default='dataset/data/prosst')
    parser.add_argument('--mpnn_fusion_weight_init', type=float, default=2.0)
    parser.add_argument('--esm_fusion_weight_init', type=float, default=1.0)
    parser.add_argument('--loss_type', type=str, default='pearson', choices=['pearson', 'listmle'])
    parser.add_argument('--use_bce', action='store_true')
    parser.add_argument('--bce_weight', type=float, default=0.0)
    parser.add_argument('--use_mse', action='store_true')
    parser.add_argument('--mse_weight', type=float, default=0.0)
    parser.add_argument('--no_log_scale', action='store_true', default=False)

    # LoRA 参数
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)

    # 数据参数
    parser.add_argument('--train_data_path', type=str, default="data/ddG/train")
    parser.add_argument('--valid_data_path', type=str, default="data/ddG/test")
    parser.add_argument('--test_data_path', type=str, default="data/ddG/test")
    parser.add_argument('--data_root', type=str, default="dataset/data")
    parser.add_argument('--train_list_txt', type=str, default=None)
    parser.add_argument('--valid_list_txt', type=str, default=None)
    parser.add_argument('--test_list_txt', type=str, default=None)
    parser.add_argument('--force_pdb_dir', type=str, default=None)
    parser.add_argument('--score_column', type=str, default=None)
    parser.add_argument('--valid_score_column', type=str, default=None)
    parser.add_argument('--test_score_column', type=str, default=None)
    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--train_zscore_norm', action='store_true', default=False)

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--use_scheduler', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=5.0)

    # Checkpoint 参数
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--monitor', type=str, default='val_SC')
    parser.add_argument('--monitor_mode', type=str, default='max')
    parser.add_argument('--save_top_k', type=int, default=0)
    parser.add_argument('--val_check_interval', type=float, default=None)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=None)

    # Early Stopping 参数
    parser.add_argument('--early_stopping_patience', type=int, default=10)

    # Trainer 参数
    parser.add_argument('--accelerator', type=str, default='auto', choices=['auto', 'cpu', 'gpu', 'tpu', 'mps'])
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='32', choices=['16', 'bf16', '32', '16-mixed', 'bf16-mixed'])
    parser.add_argument('--log_every_n_steps', type=int, default=1)

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--logger', type=str, default='tensorboard', choices=['tensorboard', 'wandb', 'None'])
    parser.add_argument('--wandb_project', type=str, default='maxwell')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--quick_log', type=str, default=None)

    args = parser.parse_args()

    if args.model_path is None:
        if args.model_type == 'esm':
            args.model_path = 'facebook/esm2_t33_650M_UR50D'
        elif args.model_type == 'proteinmpnn':
            args.model_path = 'vendors/ProteinMPNN/vanilla_model_weights/v_48_020.pt'
        elif args.model_type == 'prosst':
            args.model_path = 'AI4Protein/ProSST-2048'
        else:
            pass

    if args.experiment_name is None:
        experiment_name = ""
        exp_parts = [
            f"sd{args.seed}",
            f"mt_{args.model_type}",
        ]
        experiment_name = "_".join(exp_parts)
        args_sorted_items = sorted(vars(args).items())
        args_str = repr(args_sorted_items)
        experiment_md5 = hashlib.md5(args_str.encode('utf-8')).hexdigest()[:8]
        experiment_name = f"{experiment_name}_{experiment_md5}"
        args.experiment_name = experiment_name

    if args.use_bce:
        if args.bce_weight <= 0:
            args.bce_weight = 0.1
    if args.use_mse:
        if args.mse_weight <= 0:
            args.mse_weight = 0.1

    if args.score_column is None:
        if "ddg" in args.train_data_path.lower():
            args.score_column = "ddG"
        elif "dtm" in args.train_data_path.lower():
            args.score_column = "dTm"
        elif "activity" in args.train_data_path.lower():
            args.score_column = "score"
        else:
            args.score_column = "score"

    if args.valid_score_column is None:
        if "ddg" in args.valid_data_path.lower():
            args.valid_score_column = "ddG"
        elif "dtm" in args.valid_data_path.lower():
            args.valid_score_column = "dTm"
        elif "activity" in args.valid_data_path.lower():
            args.valid_score_column = "score"
        else:
            args.valid_score_column = "score"

    if args.test_score_column is None:
        if "ddg" in args.test_data_path.lower():
            args.test_score_column = "ddG"
        elif "dtm" in args.test_data_path.lower():
            args.test_score_column = "dTm"
        elif "activity" in args.test_data_path.lower():
            args.test_score_column = "score"
        else:
            args.test_score_column = "score"
    return args

def make_trainer(args):
    return L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=make_callbacks(args),
        logger=make_logger(args),
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )


def main():
    args = parse_args()
    # L.seed_everything(args.seed, workers=True)
    # 固定训练、推理结果
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    train_dataset = make_train_dataset(args)
    valid_dataset = make_valid_dataset(args)
    test_dataset = make_test_dataset(args)
    train_dataloader = train_dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)
    valid_dataloader = valid_dataset.get_dataloader(batch_size=1, shuffle=False)
    test_dataloader = test_dataset.get_dataloader(batch_size=1, shuffle=False)
    model = LitModel(args)
    trainer = make_trainer(args)
    trainer.validate(model, valid_dataloader)
    trainer.fit(model, train_dataloader, valid_dataloader)
    if trainer.checkpoint_callback.best_model_path:
        model = LitModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, weights_only=False)
    trainer.test(model, test_dataloader)
    if trainer.checkpoint_callback.best_model_path:
        print(f"Best model saved to {trainer.checkpoint_callback.best_model_path}")
    best_score = trainer.early_stopping_callback.best_score.item()
    if args.quick_log:
        from pathlib import Path
        Path(args.quick_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.quick_log, 'w+') as f:
            f.write(f"{best_score:.4f}\n")
    if args.logger == 'tensorboard':
        print(f"Tensorboard log saved to tb_logs/{args.experiment_name}")
    return trainer

if __name__ == '__main__':
    main()
