from pathlib import Path
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
                for layer in layers[-unfreeze_top_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_values=input_values)
        hidden = outputs.last_hidden_state
        return hidden.mean(dim=1)
