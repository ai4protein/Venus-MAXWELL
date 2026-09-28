from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor
from transformers import AutoModelForMaskedLM
from sequence_models.pretrained import load_model_and_alphabet
from peft import LoraConfig, get_peft_model
import esm

from .protein_mpnn_utils import ProteinMPNN
from .protein_mpnnesm import ProteinMPNNESM


class BaseModelWrapper(ABC):
    """模型包装器基类，封装模型配置和前向传播逻辑"""

    dim: int
    vocab_size: int
    lora_targets: list[str]

    @abstractmethod
    def make_model(self, args):
        """创建并返回模型实例"""
        ...

    @abstractmethod
    def compute(self, model, batch: dict, no_log_scale: bool = False) -> Tuple[Tensor, Tensor]:
        """前向传播，返回 (landscape, hidden_states)"""
        ...


class ESMWrapper(BaseModelWrapper):
    dim = 1280
    vocab_size = 33
    lora_targets = ["query", "key", "value", "dense"]

    def make_model(self, args):
        return AutoModelForMaskedLM.from_pretrained(args.model_path)

    def compute(self, model, batch: dict, no_log_scale: bool = False) -> Tuple[Tensor, Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        X = torch.nn.functional.one_hot(input_ids.detach(), num_classes=self.vocab_size)[:, 1:-1].clone()
        if no_log_scale:
            A = outputs.logits.softmax(dim=-1)[:, 1:-1, :]  # Remove sos and eos
            L = A / ((A * X).sum(dim=-1, keepdim=True))
        else:
            A = torch.log_softmax(outputs.logits, dim=-1)[:, 1:-1, :]  # Remove sos and eos
            L = A - ((A * X).sum(dim=-1, keepdim=True))
        H = outputs.hidden_states[-1][:, 1:-1, :]
        return L, H


class ESMIFWrapper(BaseModelWrapper):
    dim = 512
    vocab_size = 35
    lora_targets = ["k_proj", "v_proj", "q_proj", "out_proj", "output_projection"]

    def make_model(self, args):
        model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        return model

    def compute(self, model, batch: dict, no_log_scale: bool = False) -> Tuple[Tensor, Tensor]:
        input_ids = batch["input_ids"]  # [B, L + 1]
        X = torch.nn.functional.one_hot(input_ids.detach(), num_classes=self.vocab_size)[:, 1:].clone()
        coords = batch["coords"]
        attention_mask = batch["attention_mask"]
        confidence = batch["confidence"]
        logits, extra = model(
            coords,
            attention_mask,
            confidence,
            input_ids[:, :-1],  # Shifted
        )  # logits [B, V, L]
        logits = logits.transpose(1, 2)  # [B, L, V]
        if no_log_scale:
            A = torch.softmax(logits, dim=-1)
            L = A / ((A * X).sum(dim=-1, keepdim=True))
        else:
            A = torch.log_softmax(logits, dim=-1)
            L = A - ((A * X).sum(dim=-1, keepdim=True))
        H = extra["inner_states"][-1].transpose(0, 1)
        return L, H


class ProSSTWrapper(BaseModelWrapper):
    dim = 768
    vocab_size = 23
    lora_targets = ["query", "key", "value", "dense"]

    def make_model(self, args):
        return AutoModelForMaskedLM.from_pretrained(args.model_path, trust_remote_code=True)

    def compute(self, model, batch: dict, no_log_scale: bool = False) -> Tuple[Tensor, Tensor]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        ss_input_ids = batch["ss_input_ids"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=ss_input_ids,
            output_hidden_states=True,
        )
        vocab_size = outputs.logits.shape[-1]
        X = torch.nn.functional.one_hot(input_ids.detach(), num_classes=vocab_size)[:, 1:-1].clone()
        if no_log_scale:
            A = outputs.logits.softmax(dim=-1)[:, 1:-1, :]
            L = A / ((A * X).sum(dim=-1, keepdim=True))
        else:
            A = torch.log_softmax(outputs.logits, dim=-1)[:, 1:-1, :]
            L = A - ((A * X).sum(dim=-1, keepdim=True))
        H = outputs.hidden_states[-1][:, 1:-1, :]
        return L, H


class MIFSTWrapper(BaseModelWrapper):
    dim = 256
    vocab_size = 30
    lora_targets = ["W_v", "W_e", "W_out"]

    def make_model(self, args):
        model, _ = load_model_and_alphabet("mifst")
        return model

    def compute(self, model, batch: dict, no_log_scale: bool = False) -> Tuple[Tensor, Tensor]:
        input_ids = batch["input_ids"]  # [B, L]
        X = torch.nn.functional.one_hot(input_ids.detach(), num_classes=self.vocab_size)
        nodes = batch["nodes"]
        edges = batch["edges"]
        connections = batch["connections"]
        edge_mask = batch["edge_mask"]
        hidden_states, logits = model(
            input_ids,
            nodes,
            edges,
            connections,
            edge_mask,
        )  # logits [B, L, V]
        if no_log_scale:
            A = torch.softmax(logits, dim=-1)
            L = A / ((A * X).sum(dim=-1, keepdim=True))
        else:
            A = torch.log_softmax(logits, dim=-1)
            L = A - (A * X).sum(dim=-1, keepdim=True)
        H = hidden_states
        return L, H


class ProteinMPNNWrapper(BaseModelWrapper):
    dim = 128
    vocab_size = 21
    lora_targets = ["W_e", "W_out", "W1", "W2", "W3"]

    def make_model(self, args):
        model = ProteinMPNN(
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
        checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def _score_mode(self, model):
        args = getattr(model, "_maxwell_args", None)
        return getattr(args, "mpnn_score_mode", "autoregressive")

    def _landscape_from_log_probs(self, log_probs: Tensor, S: Tensor, no_log_scale: bool) -> Tensor:
        S_onehot = torch.nn.functional.one_hot(S.detach(), num_classes=self.vocab_size).float()
        if no_log_scale:
            probs = torch.exp(log_probs)
            return probs / ((probs * S_onehot).sum(dim=-1, keepdim=True))
        return log_probs - ((log_probs * S_onehot).sum(dim=-1, keepdim=True))

    def compute(self, model, batch: dict, no_log_scale: bool = False) -> Tuple[Tensor, Tensor]:
        X = batch["X"]  # [B, L, 4, 3]
        S = batch["input_ids"]  # [B, L]
        mask = batch["attention_mask"]  # [B, L]
        residue_idx = batch["residue_idx"]  # [B, L]
        chain_encoding = batch["chain_encoding"]  # [B, L]
        B, L = S.shape
        score_mode = self._score_mode(model)
        # chain_M: 哪些位置需要预测 (全1表示预测所有位置)
        chain_M = mask

        if score_mode == "autoregressive":
            # 使用自然顺序 (0, 1, 2, ..., L-1)
            decoding_order = torch.arange(L, device=S.device).unsqueeze(0).expand(B, -1)
            log_probs, H = model(
                X,
                S,
                mask,
                chain_M,
                residue_idx,
                chain_encoding,
                randn=None,
                use_input_decoding_order=True,
                decoding_order=decoding_order,
            )
        elif score_mode == "conditional":
            if B != 1:
                raise ValueError("ProteinMPNN conditional scoring currently requires batch_size=1")
            randn = torch.ones_like(mask)
            log_probs = model.conditional_probs(
                X,
                S,
                mask,
                chain_M,
                residue_idx,
                chain_encoding,
                randn=randn,
                backbone_only=False,
            )
            H = torch.empty((*S.shape, self.dim), device=S.device, dtype=X.dtype)
        elif score_mode == "unconditional":
            log_probs = model.unconditional_probs(X, mask, residue_idx, chain_encoding)
            H = torch.empty((*S.shape, self.dim), device=S.device, dtype=X.dtype)
        else:
            raise ValueError(f"Invalid ProteinMPNN score mode: {score_mode}")

        # log_probs: [B, L, 21], H: [B, L, hidden_dim]
        L = self._landscape_from_log_probs(log_probs, S, no_log_scale)
        return L, H


class ProteinMPNNESMWrapper(BaseModelWrapper):
    """ProteinMPNN + ESM2 混合模型"""

    dim = 128  # 使用 MPNN 的 hidden dim
    vocab_size = 21  # 使用 MPNN 词表
    # 包含 ESM 和 MPNN 两个模型的 LoRA targets
    lora_targets = [
        # ESM targets
        "esm.esm.encoder.layer.*.attention.self.query",
        "esm.esm.encoder.layer.*.attention.self.key",
        "esm.esm.encoder.layer.*.attention.self.value",
        "esm.esm.encoder.layer.*.attention.output.dense",
        # ProteinMPNN targets
        "proteinmpnn.W_e",
        "proteinmpnn.W_out",
        "proteinmpnn.encoder_layers.*.W1",
        "proteinmpnn.encoder_layers.*.W2",
        "proteinmpnn.encoder_layers.*.W3",
        "proteinmpnn.decoder_layers.*.W1",
        "proteinmpnn.decoder_layers.*.W2",
        "proteinmpnn.decoder_layers.*.W3",
    ]

    def make_model(self, args):
        return ProteinMPNNESM(
            esm_path=getattr(args, "esm_model_path", "facebook/esm2_t33_650M_UR50D"),
            mpnn_path=getattr(args, "mpnn_model_path", "weights/proteinmpnn/v_48_020.pt"),
            mpnn_fusion_weight_init=getattr(args, "mpnn_fusion_weight_init", 2.0),
            esm_fusion_weight_init=getattr(args, "esm_fusion_weight_init", 1.0),
        )

    def compute(self, model, batch: dict, no_log_scale: bool = False) -> Tuple[Tensor, Tensor]:
        landscape, hidden_states = model(batch, no_log_scale=no_log_scale)
        # hidden_states 是 dict{"mpnn": ..., "esm": ...}，这里返回 mpnn 的
        H = hidden_states["mpnn"]
        return landscape, H


# =============================================================================
# 模型注册表和便捷函数
# =============================================================================

WRAPPERS: dict[str, type[BaseModelWrapper]] = {
    "esm": ESMWrapper,
    "esmif": ESMIFWrapper,
    "prosst": ProSSTWrapper,
    "mifst": MIFSTWrapper,
    "proteinmpnn": ProteinMPNNWrapper,
    "proteinmpnn-esm": ProteinMPNNESMWrapper,
}


def get_wrapper(model_type: str) -> BaseModelWrapper:
    """根据 model_type 返回对应的模型包装器实例"""
    if model_type not in WRAPPERS:
        raise ValueError(f"Invalid model type: {model_type}")
    return WRAPPERS[model_type]()


def apply_lora(args, model):
    """对模型应用 LoRA 微调"""
    wrapper = get_wrapper(args.model_type)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=wrapper.lora_targets,
    )
    return get_peft_model(model, config)


def get_dim(model_type: str) -> int:
    """获取模型隐藏层维度"""
    return get_wrapper(model_type).dim


def get_vocab_size(model_type: str) -> int:
    """获取模型词表大小"""
    return get_wrapper(model_type).vocab_size


def make_model(args):
    """创建模型实例"""
    return get_wrapper(args.model_type).make_model(args)


def model_compute(model, batch: dict, model_type: str, no_log_scale: bool = False):
    """模型前向传播，返回 (landscape, hidden_states)"""
    return get_wrapper(model_type).compute(model, batch, no_log_scale)
