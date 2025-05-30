import torch
import torch.nn as nn
from torch.optim import AdamW
from scipy.stats import spearmanr, pearsonr
from transformers.models.esm import EsmForMaskedLM
import lightning as L


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
        

class MaxwellWrapperForESM2(L.LightningModule):
    
    def __init__(self, lr):
        super().__init__()
        self.model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.auxiliary_head = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.SELU(),
            nn.Linear(1280, 33),
        )
        self.pearson_loss = PearsonCorrelationLoss()
        self.lr = lr
        
    def training_step(self, batch, *args, **kwargs):
        landscape = batch["landscape"] # [B, L, V]
        input_ids = batch["input_ids"] # [B, L]
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=33) # [B, L, V]
        mask = batch["mask"] # [B, L, V]
        y = landscape[mask.bool()].flatten()
        attention_mask = batch["attention_mask"] # [B, L]
        
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        logits = outputs.logits # [B, L, V]
        logits = torch.log_softmax(logits, dim=-1) # [B, L, V]
        logits = logits - (logits * one_hot).sum(dim=-1, keepdim=True)  # [B, L, V]
        logits = logits * mask # [B, L, V]
        logits = logits.reshape(logits.shape[0], -1) # [B, L * V]
        landscape = landscape.reshape(landscape.shape[0], -1) # [B, L * V]
        corr_loss = self.pearson_loss(logits, landscape, reduction="sum")
        
        hidden_states = outputs.hidden_states[-1] # [B, L, 1280]
        auxiliary_logits = self.auxiliary_head(hidden_states) # [B, L, 33]
        y_pred = auxiliary_logits[mask.bool()].flatten()
        mse_loss = nn.MSELoss()(y_pred, y)
        loss = corr_loss + 0.1 * mse_loss

        self.log("train_corr_loss", corr_loss)
        self.log("train_mse_loss", mse_loss)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, *args, **kwargs):
        landscape = batch["landscape"] # [B, L, V]
        input_ids = batch["input_ids"] # [B, L]
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=33) # [B, L, V]
        mask = batch["mask"] # [B, L, V]
        attention_mask = batch["attention_mask"] # [B, L]
        outputs = self.model(input_ids, attention_mask)
        logits = outputs.logits # [B, L, V]
        logits = torch.log_softmax(logits, dim=-1) # [B, L, V]
        logits = logits - (logits * one_hot).sum(dim=-1, keepdim=True)  # [B, L, V]
        assert landscape.size(0) == 1, "Evaluation on batch size > 1 is not supported"
        y_pred = logits[mask.bool()].flatten().detach().cpu().numpy()
        y = landscape[mask.bool()].flatten().detach().cpu().numpy()
        rho_spearman = spearmanr(y, y_pred).correlation
        rho_pearson = pearsonr(y, y_pred)[0]
        self.log("val_rho_spearman", rho_spearman, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rho_pearson", rho_pearson, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, *args, **kwargs):
        landscape = batch["landscape"] # [B, L, V]
        input_ids = batch["input_ids"] # [B, L]
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=33) # [B, L, V]
        mask = batch["mask"] # [B, L, V]
        attention_mask = batch["attention_mask"] # [B, L]
        outputs = self.model(input_ids, attention_mask)
        logits = outputs.logits # [B, L, V]
        logits = torch.log_softmax(logits, dim=-1) # [B, L, V]
        logits = logits - (logits * one_hot).sum(dim=-1, keepdim=True)  # [B, L, V]
        assert landscape.size(0) == 1, "Evaluation on batch size > 1 is not supported"
        y_pred = logits[mask.bool()].flatten().detach().cpu().numpy()
        y = landscape[mask.bool()].flatten().detach().cpu().numpy()
        rho_spearman = spearmanr(y, y_pred).correlation
        rho_pearson = pearsonr(y, y_pred)[0]
        self.log("test_rho_spearman", rho_spearman, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_rho_pearson", rho_pearson, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr)