import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ============================================================
# Phase Resonance Core
# ============================================================

class ResonanceTuner(nn.Module):
    def __init__(self, L: int):
        super().__init__()
        self.L = L
        self.phase_weights = nn.Parameter(torch.randn(L) * 0.02)

    def forward(self, x: torch.Tensor, invert: bool = False) -> torch.Tensor:
        w = F.softmax(self.phase_weights, dim=-1)
        if invert:
            w = torch.flip(w, dims=[0]).roll(1, dims=0)

        x_fft = torch.fft.fft(x, dim=1)
        w_fft = torch.fft.fft(w.view(1, -1, 1), n=self.L, dim=1)
        return torch.fft.ifft(x_fft * w_fft, dim=1).real


# ============================================================
# Strand Encoder (lightweight, stable)
# ============================================================

class StrandEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ff(x))


# ============================================================
# Local Relational Utilities
# ============================================================

def gather_local(x: torch.Tensor, radius: int) -> torch.Tensor:
    """
    x: (B, L, d)
    returns: (B, L, 2r+1, d)
    """
    B, L, d = x.shape
    idx = torch.arange(L, device=x.device)
    windows = [(idx + i) % L for i in range(-radius, radius + 1)]
    return torch.stack([x[:, w] for w in windows], dim=2)


class LocalRelationProbe(nn.Module):
    """
    Fixed-window, phase-conditioned local relation probe.
    No global attention. No topology storage.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        # x: (B,L,d)
        # neighbors: (B,L,K,d)
        q = self.q(x).unsqueeze(2)        # (B,L,1,d)
        k = self.k(neighbors)             # (B,L,K,d)
        v = self.v(neighbors)

        attn = (q * k).sum(-1) / (x.size(-1) ** 0.5)
        attn = attn.softmax(dim=-1)

        out = (attn.unsqueeze(-1) * v).sum(dim=2)
        return self.out(out)


# ============================================================
# Confidence Gate (mathematical, cheap)
# ============================================================

class ConfidenceGate(nn.Module):
    """
    Decides whether local relational reasoning is needed.
    True  -> fire local probe
    False -> skip
    """
    def __init__(self, threshold: float = 0.15):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,d)
        pooled = x.mean(dim=1)             # (B,d)
        sims = pooled @ pooled.T           # (B,B)

        # remove self-similarity
        sims = sims - torch.eye(sims.size(0), device=sims.device) * 1e9

        top2 = sims.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]

        probs = F.softmax(sims, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

        confidence = margin - 0.5 * entropy
        return confidence < self.threshold


# ============================================================
# Cross Resonance Layer (Phase-first, relations optional)
# ============================================================

class CrossResonanceLayer(nn.Module):
    def __init__(self, d_model: int, L: int, radius: int = 2):
        super().__init__()
        self.tuner = ResonanceTuner(L)
        self.local_probe = LocalRelationProbe(d_model)
        self.gate = ConfidenceGate()
        self.norm = nn.LayerNorm(d_model)
        self.radius = radius

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Phase-align B into A's coordinate system
        B_aligned = self.tuner(B)

        # Decide if local relational reasoning is needed
        fire = self.gate(A)

        if fire.any():
            neighbors = gather_local(B_aligned, self.radius)
            rel = self.local_probe(A, neighbors)
            A = self.norm(A + rel)

        # Restore B back to original coordinate space
        B = self.tuner(A, invert=True)
        return A, B


# ============================================================
# Global Convergence (cheap stabilizer)
# ============================================================

class GlobalConvergence(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        fused = self.fuse(torch.cat([A.mean(dim=1), B.mean(dim=1)], dim=-1))
        bc = fused.unsqueeze(1)
        return A + bc, B + bc, fused


# ============================================================
# Final Operator (deploy this)
# ============================================================

class PhaseRelationalOperator(nn.Module):
    """
    Phase-first operator with confidence-gated local relational reasoning.
    """
    def __init__(self, d_model: int, L: int, n_layers: int = 6):
        super().__init__()
        self.encoders = nn.ModuleList(
            [StrandEncoder(d_model) for _ in range(n_layers)]
        )
        self.cross_layers = nn.ModuleList(
            [CrossResonanceLayer(d_model, L) for _ in range(n_layers)]
        )
        self.convergence = GlobalConvergence(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        global_state = None

        for i in range(len(self.encoders)):
            A = self.encoders[i](A)
            B = self.encoders[i](B)
            A, B = self.cross_layers[i](A, B)

            if i % 2 == 0:
                A, B, global_state = self.convergence(A, B)

        return self.final_norm(A), self.final_norm(B), global_state