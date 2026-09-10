import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReverse.apply(x, lambd)


class STGCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.15):
        super().__init__()
        self.spatial = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.temporal = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.res = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, adj):
        # x: (B, C, T, V), adj: (V, V) or (B, V, V)
        if adj.dim() == 2:
            x_sp = torch.einsum("bctv,vw->bctw", x, adj)
        else:
            x_sp = torch.einsum("bctv,bvw->bctw", x, adj)
        y = self.spatial(x_sp)
        y = self.temporal(y)
        return y + self.res(x)


class RehabRuleEncoder(nn.Module):
    def __init__(self, scalar_dim=10, rule_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scalar_dim + rule_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, scalars, rule_flags):
        return self.net(torch.cat([scalars, rule_flags], dim=1))


class RGDARRehabNet(nn.Module):
    """
    Rule-Guided Domain-Adaptive RehabNet.

    Inputs:
      keypoints:    (B, 3, 150, 6)
      adj:          (6, 6) or (B, 6, 6)
      scalars:      (B, 10)
      rule_flags:   (B, 4)  [insufficient_rom, asymmetric, jerky, too_fast]
      mode:         (B,)
      exercise_idx: (B,)

    Outputs:
      correctness_logits: (B, 3)  incorrect / correct / review
      quality:            (B,)
      fault_logits:       (B, 4)
      exercise_logits:    (B, n_exercises)
      hardware_logits:    (B, n_modes)
      torque:             (B,)
      domain_logits:      (B, n_domains)
    """

    def __init__(
        self,
        n_exercises=10,
        n_modes=5,
        n_domains=4,
        scalar_dim=10,
        rule_dim=4,
        n_joints=6,
        d_model=128,
    ):
        super().__init__()
        self.n_exercises = n_exercises
        self.n_modes = n_modes
        self.n_domains = n_domains

        self.stgcn = nn.Sequential(
            STGCNBlock(3, 64),
            STGCNBlock(64, 96),
            STGCNBlock(96, d_model),
        )

        self.joint_pool = nn.AdaptiveAvgPool2d((150, 1))
        self.pos_emb = nn.Parameter(torch.zeros(1, 150, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            dropout=0.15,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.scalar_rule = RehabRuleEncoder(scalar_dim=scalar_dim, rule_dim=rule_dim, hidden=64)
        self.exercise_emb = nn.Embedding(n_exercises, 16)
        self.mode_emb = nn.Embedding(n_modes, 16)

        fusion_dim = d_model + d_model + 64 + 16 + 16
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 192),
            nn.LayerNorm(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.correctness_head = nn.Linear(128, 3)
        self.quality_head = nn.Linear(128, 1)
        self.fault_head = nn.Linear(128, rule_dim)
        self.exercise_head = nn.Linear(d_model, n_exercises)
        self.hardware_head = nn.Linear(128, n_modes)
        self.torque_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.domain_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_domains),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_emb, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_temporal(self, keypoints, adj):
        x = self.stgcn[0](keypoints, adj)
        x = self.stgcn[1](x, adj)
        x = self.stgcn[2](x, adj)
        x = self.joint_pool(x).squeeze(-1).transpose(1, 2)  # (B, T, D)
        x = x + self.pos_emb[:, : x.shape[1], :]
        tr = self.transformer(x)
        lstm_out, _ = self.bilstm(x)
        transformer_ctx = tr.mean(dim=1)
        lstm_ctx = lstm_out.mean(dim=1)
        return transformer_ctx, lstm_ctx, tr

    def forward(self, keypoints, adj, scalars, rule_flags, mode, exercise_idx, grl_lambda=1.0):
        mode = torch.clamp(mode.long(), 0, self.n_modes - 1)
        exercise_idx = torch.clamp(exercise_idx.long(), 0, self.n_exercises - 1)

        transformer_ctx, lstm_ctx, seq = self.encode_temporal(keypoints, adj)
        scalar_ctx = self.scalar_rule(scalars, rule_flags)
        ex_ctx = self.exercise_emb(exercise_idx)
        mode_ctx = self.mode_emb(mode)

        fused = self.fusion(torch.cat([transformer_ctx, lstm_ctx, scalar_ctx, ex_ctx, mode_ctx], dim=1))
        domain_logits = self.domain_head(grad_reverse(fused, grl_lambda))

        return {
            "correctness_logits": self.correctness_head(fused),
            "quality": self.quality_head(fused).squeeze(1),
            "fault_logits": self.fault_head(fused),
            "exercise_logits": self.exercise_head(transformer_ctx),
            "hardware_logits": self.hardware_head(fused),
            "torque": self.torque_head(fused).squeeze(1),
            "domain_logits": domain_logits,
            "embedding": fused,
        }


def rule_flags_from_scalars(scalars, exercise="squat"):
    """
    scalars shape: (B, 10) or (10,)
    returns flags: insufficient_rom, asymmetric, jerky, too_fast
    """
    single = False
    if scalars.dim() == 1:
        scalars = scalars.unsqueeze(0)
        single = True

    rom_targets = {
        "inline_lunge": 90.0, "side_lunge": 90.0, "deep_squat": 95.0,
        "sit_to_stand": 80.0, "straight_leg_raise": 55.0, "knee_bend": 80.0,
        "squat": 90.0, "ctk_squat": 90.0, "hurdle_step": 65.0,
        "unknown": 90.0, "default": 90.0,
    }
    unilateral = {"inline_lunge", "side_lunge", "straight_leg_raise", "hurdle_step"}
    target = rom_targets.get(exercise, rom_targets["default"])

    l_rom = scalars[:, 0] * 180.0
    r_rom = scalars[:, 1] * 180.0
    sym = scalars[:, 4] * 100.0
    vel = scalars[:, 5] * 200.0
    jerk = scalars[:, 6] * 10.0
    active_rom = torch.maximum(l_rom, r_rom)

    insufficient_rom = active_rom < (0.70 * target)
    asymmetric = (sym > 35.0) & (exercise not in unilateral)
    jerky = jerk > 27.66
    too_fast = vel > 150.0

    flags = torch.stack([insufficient_rom, asymmetric, jerky, too_fast], dim=1).float()
    return flags[0] if single else flags


def rgda_loss(
    outputs,
    labels,
    quality_targets,
    fault_targets,
    exercise_targets,
    hardware_targets,
    torque_targets,
    domain_targets,
    weights=None,
):
    """
    Multi-task objective.

    labels are 3-class:
      0 incorrect
      1 correct
      2 review
    fault_targets shape: (B, 4), multi-label binary flags.
    """
    weights = weights or {}
    loss_correct = F.cross_entropy(outputs["correctness_logits"], labels)
    loss_quality = F.mse_loss(torch.sigmoid(outputs["quality"]), quality_targets.float())
    loss_fault = F.binary_cross_entropy_with_logits(outputs["fault_logits"], fault_targets.float())
    loss_ex = F.cross_entropy(outputs["exercise_logits"], exercise_targets.long())
    loss_hw = F.cross_entropy(outputs["hardware_logits"], hardware_targets.long())
    loss_torque = F.mse_loss(outputs["torque"], torque_targets.float())
    loss_domain = F.cross_entropy(outputs["domain_logits"], domain_targets.long())

    total = (
        weights.get("correct", 1.0) * loss_correct
        + weights.get("quality", 0.30) * loss_quality
        + weights.get("fault", 0.60) * loss_fault
        + weights.get("exercise", 0.20) * loss_ex
        + weights.get("hardware", 0.20) * loss_hw
        + weights.get("torque", 0.20) * loss_torque
        + weights.get("domain", 0.20) * loss_domain
    )
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_correct": float(loss_correct.detach().cpu()),
        "loss_quality": float(loss_quality.detach().cpu()),
        "loss_fault": float(loss_fault.detach().cpu()),
        "loss_exercise": float(loss_ex.detach().cpu()),
        "loss_hardware": float(loss_hw.detach().cpu()),
        "loss_torque": float(loss_torque.detach().cpu()),
        "loss_domain": float(loss_domain.detach().cpu()),
    }


@torch.no_grad()
def hybrid_decision_from_outputs(outputs, rule_flags, correct_threshold=0.60, wrong_threshold=0.60):
    """
    Patient-facing decision:
      - if rules find safety issue: incorrect
      - else if model confident correct: correct
      - else if model confident incorrect but rules clean: review
      - else review
    """
    probs = torch.softmax(outputs["correctness_logits"], dim=1)
    p_incorrect = probs[:, 0]
    p_correct = probs[:, 1]
    rule_bad = rule_flags.sum(dim=1) > 0

    final = torch.full((probs.shape[0],), 2, dtype=torch.long, device=probs.device)
    final[rule_bad] = 0
    final[(~rule_bad) & (p_correct >= correct_threshold)] = 1
    final[(~rule_bad) & (p_incorrect >= wrong_threshold)] = 2

    return {
        "p_incorrect": p_incorrect,
        "p_correct": p_correct,
        "p_review": probs[:, 2],
        "final_label": final,
    }


if __name__ == "__main__":
    batch = 2
    model = RGDARRehabNet(n_exercises=10, n_modes=5, n_domains=4)
    keypoints = torch.randn(batch, 3, 150, 6)
    adj = torch.eye(6)
    scalars = torch.rand(batch, 10)
    rule_flags = rule_flags_from_scalars(scalars, exercise="squat")
    mode = torch.zeros(batch, dtype=torch.long)
    exercise_idx = torch.full((batch,), 6, dtype=torch.long)
    out = model(keypoints, adj, scalars, rule_flags, mode, exercise_idx)
    print({k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
