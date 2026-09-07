import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoTokenizer, AutoModel, logging


logging.set_verbosity_error()


class MiniVLA(nn.Module):
    """
    Lightweight Vision-Language-Action model.

    Inputs:
        image: [B, 3, 224, 224]
        language_instruction: List[str]
        state: [B, 8]

    Output:
        action: [B, 7]

    Fine-tuning strategy:
        - ResNet-18: only the final FC layer is trainable
        - DistilBERT: only the last 2 Transformer layers are trainable
        - State encoder: trainable
        - Fusion: trainable
        - Action head: trainable
    """

    def __init__(self, language_model_name="pretrained/distilbert-base-uncased"):
        super().__init__()

        # =========================================================
        # Vision Encoder - ResNet-18
        # =========================================================

        weights = ResNet18_Weights.DEFAULT
        self.vision_encoder = resnet18(weights=weights)
        vision_dim = self.vision_encoder.fc.in_features

        # Freeze the entire pretrained ResNet
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

       # Replace the ImageNet classification head with a trainable 512D projection.
        self.vision_encoder.fc = nn.Linear(vision_dim, vision_dim)

        # Only the final FC layer is trainable
        for param in self.vision_encoder.fc.parameters():
            param.requires_grad = True

        # =========================================================
        # Language Encoder - DistilBERT
        # =========================================================

        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.language_encoder = AutoModel.from_pretrained(language_model_name)
        language_dim = self.language_encoder.config.hidden_size

        # Freeze the entire DistilBERT
        for param in self.language_encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 Transformer layers
        for layer in self.language_encoder.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # =========================================================
        # State Encoder
        # =========================================================

        self.state_encoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        state_dim = 128

        # =========================================================
        # Multimodal Fusion
        # =========================================================

        fusion_dim = (vision_dim + language_dim + state_dim)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # =========================================================
        # Action Head
        # =========================================================

        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    # =============================================================
    # Forward
    # =============================================================

    def forward(self, image, language_instruction, state):

        # Vision
        visual_features = self.vision_encoder(image)

        # ---------------------------------------------------------
        # Language
        # ---------------------------------------------------------

        tokens = self.tokenizer(language_instruction, padding=True, truncation=True, mmax_length=64, return_tensors="pt")
        tokens = {
            key: value.to(image.device)
            for key, value in tokens.items()
        }

        language_output = self.language_encoder(**tokens)

        # DistilBERT does not have pooler_output. Use the [CLS] token representation.
        language_features = (language_output.last_hidden_state[:, 0])

        # State
        state_features = self.state_encoder(state)

        # ---------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------

        combined_features = torch.cat([visual_features, language_features, state_features], dim=1)
        fused_features = self.fusion(combined_features)

        # Action
        action = self.action_head(fused_features)

        return action