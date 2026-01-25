import torch
import torch.nn as nn


class MiniPatchTST(nn.Module):
    def __init__(
        self,
        input_length,
        forecast_horizon,
        patch_len=24,
        d_model=64,
        nhead=4,
        num_layers=4,
        num_stations=264,
    ):
        super(MiniPatchTST, self).__init__()
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.patch_len = patch_len
        self.n_patches = input_length // patch_len
        self.d_model = d_model
        self.patch_embed = nn.Linear(patch_len * 4, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model * self.n_patches, d_model)
        self.connect_res = nn.Linear(d_model + forecast_horizon * d_model, d_model + forecast_horizon * d_model)
        self.final = nn.Linear(d_model, forecast_horizon)

    def forward(self, x):
        B = x.size(0)
        # Patch and positional embedding
        F = 4
        x = x.view(B, self.n_patches, self.patch_len, F)
        x = x.view(B, self.n_patches, self.patch_len * F)
        x = self.patch_embed(x) + self.pos_embed
        # Transformer
        x = self.transformer_encoder(x)
        x = x.flatten(1)
        x = self.head(x)

        return self.final(x)
