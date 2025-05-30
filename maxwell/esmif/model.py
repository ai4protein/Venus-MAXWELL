import torch
import torch.nn as nn
from torch.optim import AdamW
from scipy.stats import spearmanr, pearsonr
import lightning as L
import esm
import esm.inverse_folding


class PearsonCorrelationLoss(nn.Module):
    """
    Loss function that computes the negative Pearson correlation coefficient
    using cosine similarity on centered data.
    """
    def __init__(self, dim=1, eps=1e-8):
        super(PearsonCorrelationLoss, self).__init__()
        self.dim = dim  # Dimension along which to compute similarity
        self.eps = eps  # Small epsilon to avoid division by zero
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)
        
    def forward(self, predictions, targets, reduction="sum"):
        """
        Compute the negative Pearson correlation between predictions and targets.
        
        Args:
            predictions: Tensor of shape [B, L]
            targets: Tensor of shape [B, L]
            
        Returns:
            loss: Sum of negative Pearson correlations across the batch
        """
        pred_mean = predictions.mean(dim=self.dim, keepdim=True)
        targ_mean = targets.mean(dim=self.dim, keepdim=True)
        pred_centered = predictions - pred_mean
        targ_centered = targets - targ_mean
        correlation = self.cos(pred_centered, targ_centered)
        if reduction == "sum":
            return 1 - torch.sum(correlation)
        elif reduction == "mean":
            return 1 - torch.mean(correlation)
        elif reduction == "none" or reduction is None:
            return 1 - correlation
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
        

class MaxwellWrapperForESMIF(L.LightningModule):
    
    def __init__(self, lr=5e-5, lambda_value=0.1):
        super().__init__()
        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.auxiliary_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.SELU(),
            nn.Linear(512, 35),
        )
        self.pearson_loss = PearsonCorrelationLoss()
        self.lr = lr
        self.lambda_value = lambda_value
        self.save_hyperparameters()
        
    def training_step(self, batch, *args, **kwargs):
        # Input
        input_ids = batch["input_ids"] # [B, L + 1]    
        one_hot = torch.nn.functional.one_hot(input_ids[:, 1:], num_classes=35) # [B, L, V]
        
        # forward
        logits, extra = self.model.forward(
            batch["coords"], batch["attention_mask"], batch["confidence"], input_ids[:, :-1] # Shifted
        ) # logits [L, B, V]
        
        # Compute predicted landscape
        logits = logits.transpose(1, 2) #  [B, V, L] -> [B, L, V]
        logits = torch.log_softmax(logits, dim=-1) # [B, L, V] -> [B, L, V]
        pred_landscape = logits - (logits * one_hot).sum(dim=-1, keepdim=True)  # [B, L, V]
        
        # Compute corr loss
        masked_pred_landscape = pred_landscape * batch["mask"] # [B, L, V]
        masked_pred_landscape = masked_pred_landscape.reshape(masked_pred_landscape.shape[0], -1) # [B, L * V]
        true_landscape = batch["landscape"].reshape(batch["landscape"].shape[0], -1) # [B, L * V]
        corr_loss = self.pearson_loss(masked_pred_landscape, true_landscape, reduction="mean")
        
        # MSE loss
        hidden_states = extra['inner_states'][-1] # [L, B, 1280]
        hidden_states = hidden_states.transpose(0, 1) # [L, B, 1280] -> [B, L, 1280]
        auxiliary_logits = self.auxiliary_head(hidden_states) # [B, L, 33]
        y_pred = auxiliary_logits[batch["mask"].bool()].flatten()
        y = batch["landscape"][batch["mask"].bool()].flatten()
        mse_loss = nn.MSELoss()(y_pred, y)
        
        # total loss
        loss = corr_loss + self.lambda_value * mse_loss
        self.log("train_corr_loss", corr_loss)
        self.log("train_mse_loss", mse_loss)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, *args, **kwargs):
        # Input
        input_ids = batch["input_ids"] # [B, L + 1]    
        one_hot = torch.nn.functional.one_hot(input_ids[:, 1:], num_classes=35) # [B, L, V]
        
        # forward
        logits, extra = self.model.forward(
            batch["coords"], batch["attention_mask"], batch["confidence"], input_ids[:, :-1] # Shifted
        ) # logits [L, B, V]
        
        # Compute predicted landscape
        logits = logits.transpose(1, 2) # [B, V, L] -> [B, L, V]
        logits = torch.log_softmax(logits, dim=-1) # [B, L, V] -> [B, L, V]
        pred_landscape = logits - (logits * one_hot).sum(dim=-1, keepdim=True)  # [B, L, V]
        y_pred = pred_landscape[batch["mask"].bool()].flatten().cpu().numpy()
        y = batch["landscape"][batch["mask"].bool()].flatten().cpu().numpy()
        assert batch["landscape"].size(0) == 1, "Evaluation on batch size > 1 is not supported"
        rho_spearman = spearmanr(y, y_pred).correlation
        rho_pearson = pearsonr(y, y_pred)[0]
        self.log("val_rho_spearman", rho_spearman, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rho_pearson", rho_pearson, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, *args, **kwargs):
        # Input
        input_ids = batch["input_ids"] # [B, L + 1]    
        one_hot = torch.nn.functional.one_hot(input_ids[:, 1:], num_classes=35) # [B, L, V]
        
        # forward
        logits, extra = self.model.forward(
            batch["coords"], batch["attention_mask"], batch["confidence"], input_ids[:, :-1] # Shifted
        ) # logits [L, B, V]
        
        # Compute predicted landscape
        logits = logits.transpose(1, 2) # [B, V, L] -> [B, L, V]
        logits = torch.log_softmax(logits, dim=-1) # [B, L, V] -> [B, L, V]
        pred_landscape = logits - (logits * one_hot).sum(dim=-1, keepdim=True)  # [B, L, V]
        y_pred = pred_landscape[batch["mask"].bool()].flatten().cpu().numpy()
        y = batch["landscape"][batch["mask"].bool()].flatten().cpu().numpy()
        assert batch["landscape"].size(0) == 1, "Evaluation on batch size > 1 is not supported"
        rho_spearman = spearmanr(y, y_pred).correlation
        rho_pearson = pearsonr(y, y_pred)[0]
        self.log("test_rho_spearman", rho_spearman, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_rho_pearson", rho_pearson, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr)