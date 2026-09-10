from .data_pipeline import *

# F. MODEL â€” STGCNBlock â†’ STGCN â†’ ClinicalTransformer â†’ RehabNet
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, ks=9, stride=1):
        super().__init__()
        self.gcn = nn.Linear(c_in, c_out)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(),
            nn.Conv2d(c_out, c_out, (ks, 1), (stride, 1), ((ks - 1) // 2, 0)),
            nn.BatchNorm2d(c_out), nn.Dropout(0.1),
        )
        self.res = (nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)),
                                   nn.BatchNorm2d(c_out))
                    if c_in != c_out or stride != 1 else nn.Identity())
        self.relu = nn.ReLU()

    def forward(self, x, A):
        B, C, T, V = x.shape
        if A.dim() == 3:
            A = A[0]
        xs = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, C)
        xs = torch.bmm(A.unsqueeze(0).expand(B * T, -1, -1), xs)
        xs = self.gcn(xs).view(B, T, V, -1).permute(0, 3, 1, 2)
        return self.relu(
            torch.nan_to_num(self.tcn(xs), 0.0) +
            torch.nan_to_num(self.res(x),  0.0)
        )


class STGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = nn.LayerNorm(IN_CHANNELS * N_JOINTS)
        self.b1 = STGCNBlock(3, 32)
        self.b2 = STGCNBlock(32, 64)
        self.b3 = STGCNBlock(64, 128)

    def forward(self, x, A):
        B, C, T, V = x.shape
        x  = torch.clamp(x, -5.0, 5.0)
        xf = x.permute(0, 2, 1, 3).contiguous().view(B * T, C * V)
        xf = self.input_norm(xf).view(B, T, C, V).permute(0, 2, 1, 3)
        return self.b3(self.b2(self.b1(xf, A), A), A).mean(3).permute(0, 2, 1)


class ClinicalTransformer(nn.Module):
    def __init__(self, d=128, heads=4, d_ff=256, drop=0.1):
        super().__init__()
        pe  = torch.zeros(TARGET_LEN + 1, d)
        pos = torch.arange(TARGET_LEN + 1).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.cls  = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.drop = nn.Dropout(drop)
        mk = lambda: nn.TransformerEncoderLayer(
            d, heads, d_ff, drop, batch_first=True, norm_first=True)
        self.l1, self.l2 = mk(), mk()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x, return_attn=False):
        B, T, _ = x.shape
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        x = self.drop(x + self.pe[:, :T + 1, :])
        x = self.n1(self.l1(x))
        if return_attn:
            sa  = self.l2.self_attn
            xn  = self.l2.norm1(x)
            out, attn = sa(xn, xn, xn, need_weights=True,
                           average_attn_weights=True)
            x2  = x + self.l2.dropout1(out)
            xn2 = self.l2.norm2(x2)
            ff  = self.l2.linear2(
                self.l2.dropout(self.l2.activation(self.l2.linear1(xn2)))
            )
            return self.n2(x2 + self.l2.dropout2(ff))[:, 0, :], attn
        return self.n2(self.l2(x))[:, 0, :], None


class RehabNet(nn.Module):
    # These must match the constants in Cell 2.
    # If you change N_EXERCISES or N_MODES there, update these too.
    _N_MODES     = 2    # â† must equal N_MODES
    _N_EXERCISES = 10   # â† must equal N_EXERCISES  (FIX: was 9)

    def __init__(self):
        super().__init__()
        self.stgcn       = STGCN()
        self.trans       = ClinicalTransformer()
        self.bilstm      = nn.LSTM(128, 64, num_layers=2, batch_first=True,
                                   bidirectional=True, dropout=0.1)
        self.bilstm_proj = nn.Sequential(
            nn.Linear(128, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1))
        self.mode_emb    = nn.Embedding(self._N_MODES,      16)
        self.ex_emb      = nn.Embedding(self._N_EXERCISES,   8)
        # 128 (cls) + 128 (bilstm) + 10 (scalars) + 16 (mode) + 8 (ex) = 290
        self.shared      = nn.Sequential(
            nn.Linear(290, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2))
        self.classify_head = nn.Linear(128, 2)
        self.quality_head  = nn.Linear(128, 1)
        self.ex_head       = nn.Linear(128, self._N_EXERCISES)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, kps, adj, return_attn=False):
        if adj.dim() == 3:
            adj = adj[0]
        feats        = self.stgcn(kps, adj)
        cls, attn    = self.trans(feats, return_attn)
        lstm_out, _  = self.bilstm(feats)
        bilstm_ctx   = self.bilstm_proj(lstm_out.mean(1))
        return cls, bilstm_ctx, attn

    def forward(self, kps, adj, scalars, mode=None, exercise_idx=None):
        B, device = kps.shape[0], kps.device
        if mode         is None: mode         = torch.zeros(B, dtype=torch.long, device=device)
        if exercise_idx is None: exercise_idx = torch.zeros(B, dtype=torch.long, device=device)
        mode         = torch.clamp(mode,         0, self._N_MODES     - 1)
        exercise_idx = torch.clamp(exercise_idx, 0, self._N_EXERCISES - 1)
        cls, bilstm_ctx, _ = self.encode(kps, adj)
        x = self.shared(torch.cat([
            cls, bilstm_ctx, scalars,
            self.mode_emb(mode),
            self.ex_emb(exercise_idx),
        ], dim=1))
        # Returns 3 values â€” always unpack with *_ if you only need 2
        return (self.classify_head(x),
                self.quality_head(x).squeeze(1),
                self.ex_head(cls))

    def predict_exercise(self, kps, adj):
        self.eval()
        with torch.no_grad():
            if adj.dim() == 3:
                adj = adj[0]
            cls, _, _ = self.encode(kps, adj)
            idx = int(self.ex_head(cls).argmax(1).item())
        return EXERCISE_LIST[idx], idx


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# G. PYTORCH DATASET + DATALOADER HELPERS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
