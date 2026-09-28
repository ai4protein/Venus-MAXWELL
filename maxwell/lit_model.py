import torch
import lightning as L
from .model_wrappers import make_model, model_compute, apply_lora, get_dim, get_vocab_size
from .loss_utils import PearsonCorrelationLoss, MaskedBCELoss, MaskedMSELoss, ListMLELoss
from .metrics import evaluate
import torch.nn as nn


class LitModel(L.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = make_model(self.args)
        self.model._maxwell_args = args
        if getattr(args, 'use_lora', False):
            self.model = apply_lora(args, self.model)
            self.model._maxwell_args = args
        self.save_hyperparameters()
        if getattr(args, 'loss_type', 'pearson') == 'pearson':
            self.loss_fn = PearsonCorrelationLoss()
        elif getattr(args, 'loss_type', 'listmle') == 'listmle':
            self.loss_fn = ListMLELoss()
        self.use_bce = getattr(args, 'use_bce', False)
        if self.use_bce:
            self.use_bce = True
            self.bce_head = Head(get_dim(args.model_type), get_vocab_size(args.model_type))
            self.bce_loss = MaskedBCELoss()
        self.use_mse = getattr(args, 'use_mse', False)
        if self.use_mse:
            self.mse_head = Head(get_dim(args.model_type), get_vocab_size(args.model_type))
            self.mse_loss = MaskedMSELoss()

    def training_step(self, batch, batch_idx):
        L, H = model_compute(self.model, batch, self.args.model_type, self.args.no_log_scale)
        M = batch["mask"]
        Y = batch["landscape"]
        loss = self.loss_fn(L, Y, mask=M)
        if self.use_bce:
            bce = self.bce_head(H)
            bce_loss = self.bce_loss(bce, Y, mask=M)
            loss += self.args.bce_weight * bce_loss
            self.log('train_bce_loss', bce_loss, prog_bar=False)
        if self.use_mse:
            mse_L = self.mse_head(H)
            mse_loss = self.mse_loss(mse_L, Y, mask=M)
            loss += self.args.mse_weight * mse_loss
            self.log('train_mse_loss', mse_loss, prog_bar=False)
        self.log('train_loss', loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        L, H = model_compute(self.model, batch, self.args.model_type, self.args.no_log_scale)
        M = batch["mask"]
        Y = batch["landscape"]
        loss = self.loss_fn(L, Y, mask=M)
        metrics = evaluate(L, Y, M)
        metrics = {f"val_{k}": v for k, v in metrics.items()}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        L, H = model_compute(self.model, batch, self.args.model_type, self.args.no_log_scale)
        M = batch["mask"]
        Y = batch["landscape"]
        loss = self.loss_fn(L, Y, mask=M)
        metrics = evaluate(L, Y, M)
        metrics = {f"test_{k}": v for k, v in metrics.items()}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        lr = self.args.learning_rate
        weight_decay = self.args.weight_decay
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay)
        return optimizer


class Head(nn.Module):

    def __init__(self, model_dim, vocab_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, vocab_size),
        )

    def forward(self, H):
        return self.proj(H)
