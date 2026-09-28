import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from .protein_mpnn_utils import ProteinMPNN


class ProteinMPNNESM(nn.Module):
    """
    ProteinMPNN + ESM2 集成模型
    
    分别计算两个模型的 landscape，然后做可学习加权融合
    """
    
    PROTEINMPNN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'  # 21 tokens
    
    # ESM -> ProteinMPNN 词表映射
    ESM_AA_INDICES = torch.tensor([5, 23, 13, 9, 18, 6, 21, 12, 15, 4, 20, 17, 14, 16, 10, 8, 11, 7, 22, 19])
    MPNN_AA_INDICES = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    
    def __init__(
        self,
        esm_path="facebook/esm2_t33_650M_UR50D",
        mpnn_path="weights/proteinmpnn/v_48_020.pt",
        mpnn_fusion_weight_init=2.0,
        esm_fusion_weight_init=1.0,
    ):
        super().__init__()
        
        # ProteinMPNN
        self.proteinmpnn = ProteinMPNN(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            vocab=21,
            k_neighbors=48,
            augment_eps=0.0,
            dropout=0.1,
            ca_only=False,
        )
        checkpoint = torch.load(mpnn_path, map_location="cpu", weights_only=True)
        self.proteinmpnn.load_state_dict(checkpoint['model_state_dict'])
        
        # ESM2
        self.esm = AutoModelForMaskedLM.from_pretrained(esm_path)
        for param in self.esm.esm.parameters():
            param.requires_grad = False
        
        self.register_buffer('esm_aa_indices', self.ESM_AA_INDICES)
        self.register_buffer('mpnn_aa_indices', self.MPNN_AA_INDICES)

        # Learnable fusion weights for landscapes.
        # Use logits + softmax so weights are positive and sum to 1.
        init_logits = torch.log(
            torch.tensor([mpnn_fusion_weight_init, esm_fusion_weight_init], dtype=torch.float32)
        )
        self.fusion_weight_logits = nn.Parameter(init_logits)
    
    def _convert_esm_landscape_to_mpnn(self, esm_landscape):
        """将 ESM landscape [B, L, 33] 转换为 MPNN 词表 [B, L, 21]"""
        B, L, _ = esm_landscape.shape
        mpnn_landscape = torch.zeros((B, L, 21), device=esm_landscape.device, dtype=esm_landscape.dtype)
        mpnn_landscape[:, :, self.mpnn_aa_indices] = esm_landscape[:, :, self.esm_aa_indices]
        mpnn_landscape[:, :, 20] = esm_landscape[:, :, 24]  # X token
        return mpnn_landscape
    
    def _get_mpnn_landscape(self, batch, no_log_scale=False):
        """计算 ProteinMPNN 的 landscape"""
        X = batch["X"]
        S = batch["S"]
        mask = batch["attention_mask"]
        residue_idx = batch["residue_idx"]
        chain_encoding = batch["chain_encoding"]
        B, L = S.shape
        device = X.device
        
        chain_M = torch.ones_like(mask)
        decoding_order = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        
        log_probs, hidden = self.proteinmpnn(
            X, S, mask, chain_M, residue_idx, chain_encoding,
            randn=None, use_input_decoding_order=True, decoding_order=decoding_order
        )
        
        S_onehot = torch.nn.functional.one_hot(S.detach(), num_classes=21).float()
        if no_log_scale:
            A = torch.exp(log_probs)
            landscape = A / ((A * S_onehot).sum(dim=-1, keepdim=True))
        else:
            landscape = log_probs - ((log_probs * S_onehot).sum(dim=-1, keepdim=True))
        
        return landscape, hidden
    
    def _get_esm_landscape(self, batch, no_log_scale=False):
        """计算 ESM2 的 landscape (已转换为 MPNN 词表)"""
        esm_input_ids = batch["esm_input_ids"]
        esm_attention_mask = batch["esm_attention_mask"]
        
        outputs = self.esm(
            input_ids=esm_input_ids,
            attention_mask=esm_attention_mask,
            output_hidden_states=True
        )
        
        # 去掉 sos/eos
        logits = outputs.logits[:, 1:-1, :]  # [B, L, 33]
        hidden = outputs.hidden_states[-1][:, 1:-1, :]
        
        # ESM 词表的 one-hot (用于计算 landscape)
        esm_S = esm_input_ids[:, 1:-1]  # [B, L]
        S_onehot = torch.nn.functional.one_hot(esm_S.detach(), num_classes=33).float()
        
        if no_log_scale:
            A = torch.softmax(logits, dim=-1)
            esm_landscape = A / ((A * S_onehot).sum(dim=-1, keepdim=True))
        else:
            A = torch.log_softmax(logits, dim=-1)
            esm_landscape = A - ((A * S_onehot).sum(dim=-1, keepdim=True))
        
        # 转换为 MPNN 词表
        landscape = self._convert_esm_landscape_to_mpnn(esm_landscape)
        
        return landscape, hidden
    
    def forward(self, batch, no_log_scale=False):
        """
        前向传播：分别计算两个模型的 landscape 并进行可学习加权融合
        
        Returns:
            landscape: [B, L, 21] 融合后的 landscape
            hidden_states: dict 包含两个模型的隐藏状态
        """
        mpnn_landscape, mpnn_hidden = self._get_mpnn_landscape(batch, no_log_scale)
        esm_landscape, esm_hidden = self._get_esm_landscape(batch, no_log_scale)
        
        fusion_weights = torch.softmax(self.fusion_weight_logits, dim=0)
        mpnn_weight, esm_weight = fusion_weights[0], fusion_weights[1]
        landscape = (mpnn_weight * mpnn_landscape) + (esm_weight * esm_landscape)
        
        hidden_states = {
            "mpnn": mpnn_hidden,
            "esm": esm_hidden,
        }
        
        return landscape, hidden_states
