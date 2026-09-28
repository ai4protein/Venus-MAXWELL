import torch
import torch.nn as nn


def masked_mean(data_matrix, mask_matrix):
    mask = mask_matrix.float()
    masked_data = data_matrix * mask
    sum_valid = masked_data.sum(dim=1)
    counts_valid = mask.sum(dim=1)
    mean_values = sum_valid / counts_valid
    return mean_values


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, predictions, targets, mask):
        squared_error = (predictions - targets) ** 2 # [B, L, V]
        mask = mask.float() # [B, L, V]
        masked_se = squared_error * mask
        total = masked_se.sum(dim=-1).sum(dim=-1) # [B, ]
        num_valid = mask.sum(dim=-1).sum(dim=-1) # [B, ]
        total = total / num_valid
        return total.mean()


class MaskedBCELoss(nn.Module):

    def __init__(self, eps=1e-7):
        super(MaskedBCELoss, self).__init__()
        self.eps = eps

    def forward(self, predictions, targets, mask):
        targets = (targets > 0).float() # [B, L， V]
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction='none'
        ) # [B, L, V]
        mask = mask.float() # [B, L, V]
        masked_bce = bce * mask
        total = masked_bce.sum(dim=-1).sum(dim=-1) # [B, ]
        num_valid = mask.sum(dim=-1).sum(dim=-1) # [B, ]
        total = total / num_valid
        return total.mean()


class PearsonCorrelationLoss(nn.Module):

    def __init__(self, dim=1, eps=1e-7):
        super(PearsonCorrelationLoss, self).__init__()
        self.dim = dim  # Dimension along which to compute similarity
        self.eps = eps  # Small epsilon to avoid division by zero
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)

    def forward(self, predictions, targets, mask):
        predictions = predictions.reshape(predictions.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        mask = mask.reshape(mask.shape[0], -1)
        pred_mean = masked_mean(predictions, mask).reshape(-1, 1)
        targ_mean = masked_mean(targets, mask).reshape(-1, 1)
        pred_centered = predictions - pred_mean
        pred_centered[~mask] = 0
        targ_centered = targets - targ_mean
        targ_centered[~mask] = 0
        correlation = self.cos(pred_centered, targ_centered)
        icorrelation = 1 - correlation
        return torch.mean(icorrelation)


class ListMLELoss(nn.Module):
    """
    ListMLE loss implementation for ranking.
    It takes inputs of shape [batch_size, length, vocab] and treats the 
    combined (length * vocab) dimensions as a single list to rank per batch sample.
    """
    def __init__(self, eps=1e-7):
        super(ListMLELoss, self).__init__()
        self.eps = eps

    def forward(self, predictions, targets, mask):
        # 1. Reshape from [B, L, V] to [B, L * V]
        # This treats all positions and vocabulary items as a single list per sample
        predictions = predictions.reshape(predictions.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        mask = mask.reshape(mask.shape[0], -1)

        # 2. Prepare targets for sorting: set masked items to -inf
        # so they are pushed to the end of the ranking
        targ_for_sort = targets.masked_fill(~mask, float('-inf'))
        
        # 3. Sort targets in descending order and get indices
        _, indices = targ_for_sort.sort(descending=True, dim=-1)
        
        # 4. Reorder predictions and mask according to target ranking
        pred_sorted = torch.gather(predictions, dim=-1, index=indices)
        mask_sorted = torch.gather(mask, dim=-1, index=indices)
        
        # 5. Masked log-sum-exp for suffixes
        # Set predictions of masked items to -inf for exp calculation
        pred_sorted_masked = pred_sorted.masked_fill(~mask_sorted, float('-inf'))
        
        # To avoid numerical instability, use the max trick for logsumexp
        max_val, _ = pred_sorted_masked.max(dim=-1, keepdim=True)
        # Replace -inf max_val with 0 to avoid NaN during subtraction if all are masked
        max_val = max_val.masked_fill(max_val == float('-inf'), 0)
        
        exp_s = torch.exp(pred_sorted_masked - max_val)
        
        # Reverse cumsum to get suffix sums: sum_{j=i to K} exp(s_j)
        # flip -> cumsum -> flip back
        sum_exp_s = torch.flip(torch.cumsum(torch.flip(exp_s, dims=[-1]), dim=-1), dims=[-1])
        
        # log_sum_exp = max_val + log(sum_exp_s)
        log_sum_exp = max_val + torch.log(sum_exp_s + self.eps)
        
        # 6. Compute ListMLE loss: -s_i + log(sum_{j=i}^K exp(s_j))
        # For masked items, pred_sorted_masked is -inf, we should avoid NaN
        loss = -pred_sorted + log_sum_exp
        
        # 7. Apply mask to loss contributions and average over batch
        loss = loss * mask_sorted.float()
        
        # Normalize by number of valid items in each list
        num_valid = mask.sum(dim=-1)
        sample_loss = loss.sum(dim=-1) / (num_valid + self.eps)
        
        return sample_loss.mean()

