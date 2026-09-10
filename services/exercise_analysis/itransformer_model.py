"""
iTransformer RehabNet Model Architecture
=========================================
An alternative deep-learning model architecture for the PS2 Exercise Analysis Engine.
Transplanted from the `FULL_INTEGRATED_iTransplant (2).ipynb` notebook.

Replaces the traditional temporal Transformer & BiLSTM sequence modeling in RehabNet
with an iTransformer approach (each joint serves as a token, and self-attention
runs across joints rather than time-steps, coupled with Temporal Attention Pooling).

To switch the active model in the backend:
1. Open `backend/services/exercise_analysis/model_analyzer.py`
2. Change the import at the top of load_model() from:
       from .rehabnet_model import load_rehabnet as load_model_fn
   to:
       from .itransformer_model import load_itransformer as load_model_fn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Adjacency matrix for 6 lower-body joints ──────────────────────────────────
# Joints: 0=LEFT_HIP, 1=RIGHT_HIP, 2=LEFT_KNEE, 3=RIGHT_KNEE, 4=LEFT_ANKLE, 5=RIGHT_ANKLE
def _build_adj(num_nodes=6):
    edges = [
        (0, 1),  # left_hip – right_hip
        (0, 2),  # left_hip – left_knee
        (1, 3),  # right_hip – right_knee
        (2, 4),  # left_knee – left_ankle
        (3, 5),  # right_knee – right_ankle
    ]
    A = np.eye(num_nodes, dtype=np.float32)  # self-loops
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Normalize: D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


# ── F0. Spatial-Temporal Graph Convolution (ST-GCN) ───────────────────────────

class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution block using einsum aggregation."""

    def __init__(self, c_in, c_out, ks=9, stride=1):
        super().__init__()
        self.gcn_conv = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.gcn_bn   = nn.BatchNorm2d(c_out)
        pad = (ks - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(c_out, c_out, (ks, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(c_out),
        )
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.res = (nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)),
                                   nn.BatchNorm2d(c_out))
                    if c_in != c_out or stride != 1 else nn.Identity())

    def forward(self, x, A):
        """
        x: (B, C, T, V)
        A: (V, V) adjacency matrix
        """
        if A.dim() == 3:
            A = A[0]
        res = self.res(x)
        xs  = torch.einsum('bctv,vw->bctw', x, A)
        xs  = self.relu(self.gcn_bn(self.gcn_conv(xs)))
        out = self.relu(self.tcn(xs) + res)
        return self.dropout(out)


class STGCN(nn.Module):
    """3-layer Spatial-Temporal Graph Convolution network."""

    def __init__(self, num_joints=6, in_channels=3):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_channels * num_joints)
        self.b1 = STGCNBlock(in_channels, 32)
        self.b2 = STGCNBlock(32, 64)
        self.b3 = STGCNBlock(64, 128)

    def forward(self, x, A):
        B, C, T, V = x.shape
        x  = torch.clamp(x, -5.0, 5.0)
        xf = x.permute(0, 2, 1, 3).contiguous().view(B * T, C * V)
        xf = self.input_norm(xf).view(B, T, C, V).permute(0, 2, 1, 3)
        return self.b3(self.b2(self.b1(xf, A), A), A)   # [B, 128, T, V]


# ── F1. iTransformer Building Blocks ──────────────────────────────────────────

class TemporalAttentionPooling(nn.Module):
    """Learnable soft-attention pooling over the temporal (T) axis."""

    def forward(self, x):
        # x: [B, C, T, V]
        w = F.softmax(x.mean(dim=1, keepdim=True), dim=2)  # [B, 1, T, V]
        return (x * w).sum(dim=2)                           # [B, C, V]


class iTransformerBlock(nn.Module):
    """
    Inverted Transformer Block where each joint is treated as a token.
    Attention is calculated across joints rather than time-steps.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, V, d_model]
        q = k = v = self.norm1(x)
        attn_out, attn_w = self.attn(q, k, v)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_w


class iTransformerEncoder(nn.Module):
    """
    Encoder processing spatial-temporal feature maps into a joint-relation representation.
    Pools along time -> projects features -> applies joint-wise self-attention.
    """

    def __init__(self, in_ch=128, d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1):
        super().__init__()
        self.temp_pool  = TemporalAttentionPooling()
        self.input_norm = nn.LayerNorm(in_ch)
        self.proj       = nn.Linear(in_ch, d_model)
        self.blocks     = nn.ModuleList([
            iTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self._last_attn = []

    def forward(self, x):
        # x: [B, C, T, V]
        x = self.temp_pool(x)            # [B, C, V]
        x = x.permute(0, 2, 1)          # [B, V, C]
        x = self.input_norm(x)
        x = torch.clamp(x, -10.0, 10.0)
        x = self.proj(x)                 # [B, V, d_model]
        self._last_attn = []
        for blk in self.blocks:
            x, w = blk(x)
            self._last_attn.append(w.detach().cpu())
        return x.mean(dim=1)             # [B, d_model]


# ── F2. Fusion Layer with Dataset Correction Embedding ────────────────────────

class FusionLayer(nn.Module):
    """
    Fuses deep features, hand-crafted scalar stats, exercise class, and support modes.
    Includes a dataset embedding vector to correct for domain shift across clinics.
    """

    def __init__(self, deep_dim=128, scalar_dim=10, n_modes=2, n_exercises=10,
                 n_datasets=4, dataset_embed_dim=16, out_dim=128, dropout=0.2):
        super().__init__()
        self.mode_emb    = nn.Embedding(n_modes,    16)
        self.ex_emb      = nn.Embedding(n_exercises, 8)
        self.dataset_emb = nn.Embedding(n_datasets, dataset_embed_dim)
        in_dim = deep_dim + scalar_dim + 16 + 8 + dataset_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )

    def forward(self, deep, scalars, mode, exercise_idx, dataset_id):
        cat = torch.cat([
            deep, scalars,
            self.mode_emb(mode),
            self.ex_emb(exercise_idx),
            self.dataset_emb(dataset_id),
        ], dim=-1)
        return self.net(cat)


# ── F3. Full iTransformer-based RehabNet Model ────────────────────────────────

class iTransformerRehabNet(nn.Module):
    """
    iTransformer-RehabNet Module.
    Exposes an identical interface contract to the original RehabNet model
    allowing drop-in compatibility in the ModelExerciseAnalyzer backend.
    """

    NUM_JOINTS = 6
    NUM_MODES = 2
    NUM_EXERCISES = 10
    NUM_DATASETS = 4

    def __init__(self):
        super().__init__()
        A = _build_adj(self.NUM_JOINTS)
        self.register_buffer("A", A)

        self.stgcn  = STGCN(num_joints=self.NUM_JOINTS, in_channels=3)
        self.itrans = iTransformerEncoder(in_ch=128)
        self.fusion = FusionLayer(n_modes=self.NUM_MODES, n_exercises=self.NUM_EXERCISES, n_datasets=self.NUM_DATASETS)

        self.classify_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
        self.quality_head  = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.ex_head       = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, self.NUM_EXERCISES))

        self._init_weights()
        self._last_attn = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mode_id=None, ex_id=None, hand_features=None, dataset_id=None):
        """
        Drop-in forward execution mapping to the API caller contract.
        
        Args:
            x: (B, 3, T, 6) - joint angle coordinates time series
            mode_id: (B,) - mode index (0 or 1), default zeros
            ex_id: (B,) - exercise type index (0-9), default zeros
            hand_features: (B, 10) - hand-crafted stats, default zeros
            dataset_id: (B,) - dataset source index, default zeros
        """
        B, device = x.shape[0], x.device

        # Handle defaults to ensure downstream compatibility with the standard forward loop
        if mode_id is None:
            mode_id = torch.zeros(B, dtype=torch.long, device=device)
        if ex_id is None:
            ex_id = torch.zeros(B, dtype=torch.long, device=device)
        if hand_features is None:
            hand_features = torch.zeros(B, 10, device=device)
        if dataset_id is None:
            dataset_id = torch.zeros(B, dtype=torch.long, device=device)

        # Clamp indexes to valid ranges to prevent indexing faults
        mode_id      = torch.clamp(mode_id,      0, self.NUM_MODES - 1)
        ex_id        = torch.clamp(ex_id,        0, self.NUM_EXERCISES - 1)
        dataset_id   = torch.clamp(dataset_id,   0, self.NUM_DATASETS - 1)

        # Clear nan values
        x             = torch.nan_to_num(x,             nan=0., posinf=1., neginf=-1.)
        hand_features = torch.nan_to_num(hand_features, nan=0., posinf=1., neginf=-1.)

        # 1. Spatial-Temporal Graph Convolution
        feat_map = self.stgcn(x, self.A)                      # [B, 128, T, V]
        
        # 2. Joint-Token Attention & Temporal Attention Pooling
        global_rep = self.itrans(feat_map)                    # [B, 128]
        self._last_attn = self.itrans._last_attn
        
        # 3. Dense Feature Fusion
        fused = self.fusion(global_rep, hand_features, mode_id, ex_id, dataset_id)  # [B, 128]

        # Return identical dict payload contract to match backend evaluation
        return {
            "classify_logits": self.classify_head(fused),            # [B, 2]
            "quality_score": torch.sigmoid(self.quality_head(fused)).squeeze(-1),  # [B]
            "ex_logits": self.ex_head(global_rep),                   # [B, 10]
        }


def load_itransformer(model_path: str, device: str = "cpu") -> iTransformerRehabNet:
    """Load an iTransformer-RehabNet model from a state_dict checkpoint."""
    model = iTransformerRehabNet()
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model
