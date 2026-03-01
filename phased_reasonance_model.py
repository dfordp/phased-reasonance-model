import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ---------------------------
# Core Engine: Differentiable Phase Shift
# ---------------------------
class ResonanceTuner(nn.Module):
    """
    Replaces 'roll' with a learnable Fourier-domain phase shift.
    This allows the model to 'tune' the alignment between strands 
    mathematically rather than hard-coding it.
    """
    def __init__(self, L: int):
        super().__init__()
        self.L = L
        # The 'Secret Sauce': Learnable weights for the circular alignment
        self.phase_weights = nn.Parameter(torch.randn(L) * 0.02)

    def forward(self, x: torch.Tensor, invert: bool = False) -> torch.Tensor:
        weights = F.softmax(self.phase_weights, dim=-1)
        if invert: # Used to 'unshift' the strand back to original alignment
            weights = torch.flip(weights, dims=[0]).roll(1, dims=0)
            
        x_fft = torch.fft.fft(x, dim=1)
        # Circular convolution via FFT allows for 'fluid' rotation
        w_fft = torch.fft.fft(weights.view(1, -1, 1), n=self.L, dim=1)
        return torch.fft.ifft(x_fft * w_fft, dim=1).real

# ---------------------------
# Strand Encoder
# ---------------------------
class StrandEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))

# ---------------------------
# Cross-Resonance Layer (Updated Cross-Pairing)
# ---------------------------
class CrossResonanceLayer(nn.Module):
    def __init__(self, d_model: int, L: int, nhead: int = 4):
        super().__init__()
        self.tuner = ResonanceTuner(L)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.msg_lin = nn.Linear(d_model, d_model)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # B 'tunes' its rotation to find resonance with A
        B_aligned = self.tuner(B)
        
        # Cross-communication
        attn_A, _ = self.cross_attn(A, B_aligned, B_aligned)
        attn_B, _ = self.cross_attn(B_aligned, A, A)
        
        # A updates based on B, B updates based on A (then un-tunes back)
        A_out = self.norm(A + attn_A)
        B_out_aligned = self.norm(B_aligned + attn_B)
        
        # Restore B to original coordinate space
        B_out = self.tuner(B_out_aligned, invert=True)
        
        return A_out, B_out

# ---------------------------
# Convergence Layer (The Global Brain)
# ---------------------------
class GlobalConvergence(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.broadcast = nn.Linear(d_model, d_model)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary = torch.cat([A.mean(dim=1), B.mean(dim=1)], dim=-1)
        fused = self.fuse(summary)
        # Broadcast the global 'vibe' back to every token
        bc = self.broadcast(fused).unsqueeze(1)
        return A + bc, B + bc, fused

# ---------------------------
# MAIN OPERATOR
# ---------------------------
class PhasedResonanceOperator(nn.Module):
    def __init__(self, d_model: int, L: int, n_layers: int = 6):
        super().__init__()
        self.strand_layers = nn.ModuleList([StrandEncoderLayer(d_model) for _ in range(n_layers)])
        self.res_layers = nn.ModuleList([CrossResonanceLayer(d_model, L) for _ in range(n_layers)])
        self.convergence = GlobalConvergence(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for li in range(len(self.strand_layers)):
            A = self.strand_layers[li](A)
            B = self.strand_layers[li](B)
            
            # The Learnable Resonance Step
            A, B = self.res_layers[li](A, B)
            
            # Global Synchronization every 2 layers
            if li % 2 == 0:
                A, B, global_state = self.convergence(A, B)

        return self.final_norm(A), self.final_norm(B), global_state
