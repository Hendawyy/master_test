"""
MultimodalTransformer architecture -- extracted from NeuroDT_GPU_Lab.ipynb
Cell 6 so dashboard.py's `from model import MultimodalTransformer` and
`model.load_state_dict(ckpt['model_state_dict'])` resolve correctly.
Must stay in sync with the notebook's Cell 6 definition -- the checkpoint's
state_dict keys are tied to this exact class shape.
"""
import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121


class MultimodalTransformer(nn.Module):
    """
    Hybrid Neuro-DT diagnostic engine.
    Inputs:
        image   : (B, 1, 128, 128, 128) -- preprocessed 3D MRI volume
        tabular : (B, tabular_dim)      -- normalised clinical features
    Output:
        logits  : (B, num_classes)      -- raw scores for CN / MCI / Dementia
    """

    def __init__(self,
                 tabular_dim       = 4,
                 num_classes       = 3,
                 transformer_heads = 8,
                 transformer_dim   = 512,
                 transformer_layers= 2,
                 dropout           = 0.1):
        super().__init__()

        # 3D-CNN backbone (feature extractor only, no classification head).
        # DenseNet121 with out_channels=1024 acts as a 1024-d embedding network.
        self.cnn_backbone = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1024          # embedding dimension
        )
        image_embedding_dim = 1024

        # Multimodal fusion & projection.
        total_dim    = image_embedding_dim + tabular_dim
        projected_dim = ((total_dim + transformer_heads - 1) // transformer_heads) * transformer_heads

        self.projection = nn.Sequential(
            nn.Linear(total_dim, projected_dim),
            nn.LayerNorm(projected_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = projected_dim,
            nhead          = transformer_heads,
            dim_feedforward= transformer_dim,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True      # Pre-LN (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Classification head.
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(projected_dim, num_classes)
        )

    def forward(self, image, tabular):
        image_embedding = self.cnn_backbone(image)
        image_embedding = image_embedding.flatten(1)

        fused = torch.cat([image_embedding, tabular], dim=1)

        proj = self.projection(fused).unsqueeze(1)
        enc  = self.transformer_encoder(proj).squeeze(1)

        return self.classifier_head(enc)
