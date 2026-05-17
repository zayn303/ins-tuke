from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_IDS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "wavlm": "microsoft/wavlm-base",
}


class SpeechBackbone(nn.Module):
    def __init__(self, model_name: str, freeze_backbone: bool = True, unfreeze_top_n_layers: int = 0):
        super().__init__()
        model_id = MODEL_IDS[model_name]
        self.model = AutoModel.from_pretrained(model_id)
        self.hidden_dim = 768

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            if unfreeze_top_n_layers > 0:
                layers = self.model.encoder.layers
                n_layers = len(layers)
                assert n_layers >= unfreeze_top_n_layers, (
                    f"{model_name} has only {n_layers} encoder layers, "
                    f"cannot unfreeze {unfreeze_top_n_layers}"
                )
                print(
                    f"[backbone] {model_name}: {n_layers} encoder layers, "
                    f"unfreezing top {unfreeze_top_n_layers}",
                    flush=True,
                )
                for layer in layers[-unfreeze_top_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def _compute_token_mask(
        self,
        attention_mask: torch.Tensor,
        encoder_seq_len: int,
    ) -> torch.Tensor:
        """Map sample-resolution attention_mask [B, T_audio] to token-resolution [B, T_encoded]."""
        valid_lengths = attention_mask.sum(dim=-1)
        # HF method exists on Wav2Vec2Model, HubertModel, WavLMModel
        get_lens = getattr(self.model, "_get_feat_extract_output_lengths", None)
        if get_lens is not None:
            token_lengths = get_lens(valid_lengths)
            if isinstance(token_lengths, torch.Tensor):
                token_lengths = token_lengths.long()
            else:
                token_lengths = torch.tensor(token_lengths, dtype=torch.long, device=attention_mask.device)
        else:
            ratio = encoder_seq_len / max(attention_mask.shape[1], 1)
            token_lengths = (valid_lengths.float() * ratio).long()

        token_lengths = token_lengths.clamp(min=1, max=encoder_seq_len)
        arange = torch.arange(encoder_seq_len, device=attention_mask.device).unsqueeze(0)
        mask = (arange < token_lengths.unsqueeze(1)).float()
        return mask  # [B, T_encoded]

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state  # [B, T_encoded, H]

        if attention_mask is None:
            return hidden.mean(dim=1)

        token_mask = self._compute_token_mask(attention_mask, hidden.shape[1])  # [B, T_encoded]
        mask_3d = token_mask.unsqueeze(-1)
        masked_sum = (hidden * mask_3d).sum(dim=1)
        valid_count = token_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return masked_sum / valid_count
