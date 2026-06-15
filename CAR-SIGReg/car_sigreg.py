"""
CAR-SIGReg: Controllability-Aware Rank-Adaptive SIGReg

Standalone module — no le-wm imports required.
Requires Python >= 3.10 (uses X | Y union type hints).

Architecture
------------
CARSIGReg wraps SketchSIGReg and adds:
  1. EMA covariance of the full latent space.
  2. Action-conditioned controllability score via finite-difference on act_emb.
  3. Adaptive active-subspace selection: top-k directions ranked by
     q_i = var_i^alpha * ctrl_i^beta.
  4. SketchSIGReg applied only on the active subspace.
  5. Inactive-subspace compression via L2 penalty.
  6. Optional controllability loss (gated by use_ctrl_loss).

Shapes throughout
-----------------
  emb      : (B, T, D)     — full latent sequence from encode()
  ctx_emb  : (B, T_ctx, D) — context slice passed to predictor
  ctx_act  : (B, T_ctx, A) — action embeddings for context
  z_A      : (B, T, r_A)   — active-subspace projection
  z_N      : (B, T, r_N)   — inactive-subspace projection
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# SketchSIGReg
# ---------------------------------------------------------------------------

class SketchSIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer, compatible with (T, B, D) input.

    Mirrors the original LeWM SIGReg exactly — same Epps-Pulley statistic,
    same integration weights — but accepts any latent dimension D so it can
    be applied to a projected subspace.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = int(num_proj)

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z_tbd: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_tbd: (T, B, D) — latent tensor, batch dimension in the middle.

        Returns:
            Scalar SIGReg loss.
        """
        if z_tbd.dim() != 3:
            raise ValueError(f"SketchSIGReg expects (T, B, D), got {tuple(z_tbd.shape)}")

        d = z_tbd.size(-1)
        A = torch.randn(d, self.num_proj, device=z_tbd.device, dtype=z_tbd.dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-8)

        # (T, B, num_proj, 1) * (knots,)  →  (T, B, num_proj, knots)
        x_t = (z_tbd @ A).unsqueeze(-1) * self.t.to(z_tbd.dtype)

        # mean over B (dim=1)
        phi = self.phi.to(z_tbd.dtype)
        err = (
            (x_t.cos().mean(dim=1) - phi).square()
            + x_t.sin().mean(dim=1).square()
        )

        # integrate over knots, average over projections and T
        statistic = (err @ self.weights.to(z_tbd.dtype)) * z_tbd.size(1)
        return statistic.mean()


# ---------------------------------------------------------------------------
# CARSIGReg
# ---------------------------------------------------------------------------

class CARSIGReg(nn.Module):
    """
    Controllability-Aware Rank-Adaptive SIGReg (V1).

    Parameters
    ----------
    embed_dim : int
        Latent dimension D (must match the model's output embedding size).
    knots, num_proj : int
        Passed through to SketchSIGReg.
    ema : float
        Exponential moving-average decay for the covariance estimate.
    update_basis_every : int
        How many training steps between basis re-computations.
    warmup_steps : int
        Number of steps before the first basis update; during warmup the
        active mask covers the top r_max//2 directions (conservative default
        to avoid premature collapse).
    tau : float
        Cumulative-score threshold in [0, 1] for selecting active dims.
    r_min, r_max : int
        Hard bounds on the number of active dimensions.
    alpha, beta : float
        Exponents for variance-score and ctrl-score in q_i = var^alpha * ctrl^beta.
        Set beta=0 for PCA-Rank ablation; set alpha=0 for Ctrl-Rank ablation.
    action_eps : float
        Finite-difference step size applied to the action embedding.
    ctrl_margin : float
        Minimum active-energy margin for the controllability loss.
    ctrl_gamma : float
        Weight of inactive-energy term inside ctrl_loss.
    eps : float
        Numerical stability constant.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        knots: int = 17,
        num_proj: int = 1024,
        ema: float = 0.99,
        update_basis_every: int = 50,
        warmup_steps: int = 100,
        tau: float = 0.95,
        r_min: int = 4,
        r_max: int = 64,
        alpha: float = 1.0,
        beta: float = 1.0,
        action_eps: float = 0.05,
        ctrl_margin: float = 0.01,
        ctrl_gamma: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.ema = float(ema)
        self.update_basis_every = int(update_basis_every)
        self.warmup_steps = int(warmup_steps)
        self.tau = float(tau)
        self.r_min = int(r_min)
        self.r_max = int(r_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.action_eps = float(action_eps)
        self.ctrl_margin = float(ctrl_margin)
        self.ctrl_gamma = float(ctrl_gamma)
        self.eps = float(eps)

        self.sigreg = SketchSIGReg(knots=knots, num_proj=num_proj)

        # --- persistent state (not optimized) ---
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("C_ema", torch.eye(self.embed_dim, dtype=torch.float32))
        self.register_buffer("U", torch.eye(self.embed_dim, dtype=torch.float32))

        # Warmup mask: use top r_max//2 dims so SIGReg covers a reasonable
        # subspace before the first real basis update.
        warmup_active = max(self.r_min, min(self.r_max // 2, self.embed_dim))
        active = torch.zeros(self.embed_dim, dtype=torch.bool)
        active[:warmup_active] = True
        self.register_buffer("active_mask", active)

        # --- diagnostic metrics (detached scalars logged to WandB) ---
        self.register_buffer("last_eff_rank", torch.zeros((), dtype=torch.float32))
        self.register_buffer(
            "last_active_rank",
            torch.tensor(float(warmup_active), dtype=torch.float32),
        )
        self.register_buffer("last_ctrl_align", torch.zeros((), dtype=torch.float32))
        self.register_buffer("last_inactive_energy", torch.zeros((), dtype=torch.float32))

    # ------------------------------------------------------------------
    # Internal helpers (all @no_grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _covariance(self, emb: torch.Tensor) -> torch.Tensor:
        """Batch covariance of emb (B, T, D) → (D, D) in float32."""
        z = emb.detach().reshape(-1, emb.size(-1)).float()
        z = z - z.mean(dim=0, keepdim=True)
        n = max(z.size(0) - 1, 1)
        return z.T @ z / n

    @torch.no_grad()
    def _effective_rank(self, eigvals: torch.Tensor) -> torch.Tensor:
        """Entropy-based effective rank from eigenvalue spectrum."""
        vals = eigvals.clamp_min(0)
        p = vals / (vals.sum() + self.eps)
        entropy = -(p * (p + self.eps).log()).sum()
        return entropy.exp()

    @torch.no_grad()
    def _action_sensitivity(
        self,
        predictor: Callable | None,
        ctx_emb: torch.Tensor | None,
        ctx_act: torch.Tensor | None,
        U: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate per-basis-direction action sensitivity via finite difference
        on the *action embedding* (not raw action), because the predictor
        consumes act_emb directly.

        Returns c_i ≥ 0 for each of the D basis directions.
        Falls back to uniform scores when predictor/inputs are unavailable.
        """
        if predictor is None or ctx_emb is None or ctx_act is None:
            return torch.ones(self.embed_dim, device=U.device, dtype=torch.float32)

        z0 = ctx_emb.detach()
        a0 = ctx_act.detach()

        noise = torch.randn_like(a0)
        z_plus = predictor(z0, a0 + self.action_eps * noise)
        z_minus = predictor(z0, a0 - self.action_eps * noise)

        # finite-difference sensitivity in latent space
        dz = (z_plus - z_minus) / (2.0 * self.action_eps)
        dz_flat = dz.reshape(-1, dz.size(-1)).float()

        # project onto current basis and measure per-direction energy
        dz_u = dz_flat @ U.float()
        c = dz_u.square().mean(dim=0)
        return c.clamp_min(0)

    @torch.no_grad()
    def _update_basis(
        self,
        emb: torch.Tensor,
        predictor: Callable | None = None,
        ctx_emb: torch.Tensor | None = None,
        ctx_act: torch.Tensor | None = None,
    ) -> None:
        """Recompute eigenbasis and active mask."""
        C_batch = self._covariance(emb)
        self.C_ema.mul_(self.ema).add_(C_batch.to(self.C_ema.device), alpha=1.0 - self.ema)

        # eigh returns ascending eigenvalues; flip to descending
        eigvals, U = torch.linalg.eigh(self.C_ema.float())
        eigvals = eigvals.flip(0)
        U = U.flip(1)

        eff_rank = self._effective_rank(eigvals)

        c = self._action_sensitivity(
            predictor=predictor,
            ctx_emb=ctx_emb,
            ctx_act=ctx_act,
            U=U.to(emb.device),
        ).to(eigvals.device)

        # Normalise both scores to a probability simplex
        var_score = eigvals.clamp_min(0)
        var_score = var_score / (var_score.sum() + self.eps)

        ctrl_score = c.clamp_min(0)
        ctrl_score = ctrl_score / (ctrl_score.sum() + self.eps)

        # Combined score: q_i = var_i^alpha * ctrl_i^beta
        q = (var_score + self.eps).pow(self.alpha) * (ctrl_score + self.eps).pow(self.beta)

        # Greedy threshold: keep fewest dims whose cumulative q exceeds tau
        order = torch.argsort(q, descending=True)
        q_sorted = q[order]
        cum = torch.cumsum(q_sorted, dim=0)
        threshold = self.tau * cum[-1]
        k = int(torch.searchsorted(cum, threshold).item()) + 1
        k = max(self.r_min, min(k, self.r_max, self.embed_dim))

        active_idx = order[:k]
        mask = torch.zeros_like(self.active_mask)
        mask[active_idx] = True

        self.U.copy_(U.to(self.U.device))
        self.active_mask.copy_(mask.to(self.active_mask.device))
        self.last_eff_rank.copy_(eff_rank.to(self.last_eff_rank.device))
        self.last_active_rank.copy_(
            torch.tensor(float(k), device=self.last_active_rank.device)
        )

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _project_active_inactive(
        self, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split emb (B, T, D) into active and inactive subspace projections.

        Returns
        -------
        z_A  : (B, T, r_A)
        z_N  : (B, T, r_N)   — empty last dim if all dims are active
        U_A  : (D, r_A)
        U_N  : (D, r_N)
        """
        U = self.U.to(device=emb.device, dtype=emb.dtype)
        mask = self.active_mask.to(device=emb.device)

        U_A = U[:, mask]
        U_N = U[:, ~mask]

        z_A = emb @ U_A

        if U_N.numel() == 0:
            z_N = emb.new_zeros(*emb.shape[:-1], 1)
        else:
            z_N = emb @ U_N

        return z_A, z_N, U_A, U_N

    # ------------------------------------------------------------------
    # Controllability loss (optional)
    # ------------------------------------------------------------------

    def _ctrl_loss(
        self,
        predictor: Callable | None,
        ctx_emb: torch.Tensor | None,
        ctx_act: torch.Tensor | None,
        U_A: torch.Tensor,
        U_N: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Controllability alignment loss (with gradient).

        Encourages the active subspace to absorb action effects and the
        inactive subspace to be action-free.

        Returns (ctrl_loss scalar, ctrl_align scalar detached).
        """
        if predictor is None or ctx_emb is None or ctx_act is None:
            zero = ctx_emb.new_zeros(()) if ctx_emb is not None else torch.zeros(())
            return zero, zero

        noise = torch.randn_like(ctx_act)
        z_plus = predictor(ctx_emb, ctx_act + self.action_eps * noise)
        z_minus = predictor(ctx_emb, ctx_act - self.action_eps * noise)

        dz = (z_plus - z_minus) / (2.0 * self.action_eps)

        dz_A = dz @ U_A
        active_energy = dz_A.square().sum(dim=-1).mean()

        if U_N.numel() == 0:
            inactive_energy = dz.new_zeros(())
        else:
            dz_N = dz @ U_N
            inactive_energy = dz_N.square().sum(dim=-1).mean()

        ctrl_loss = (
            F.relu(self.ctrl_margin - active_energy)
            + self.ctrl_gamma * inactive_energy
        )

        align = active_energy / (active_energy + inactive_energy + self.eps)
        return ctrl_loss, align.detach()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        emb: torch.Tensor,
        predictor: Callable | None = None,
        ctx_emb: torch.Tensor | None = None,
        ctx_act: torch.Tensor | None = None,
        use_ctrl_loss: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all CAR-SIGReg loss terms and diagnostic metrics.

        Args:
            emb          : (B, T, D) — full encoded latent sequence.
            predictor    : callable(ctx_emb, ctx_act) → pred_emb, e.g. model.predict.
            ctx_emb      : (B, T_ctx, D) — context embeddings for basis update / ctrl.
            ctx_act      : (B, T_ctx, A) — context action embeddings.
            use_ctrl_loss: whether to compute the controllability loss.

        Returns dict with keys:
            sigreg_loss, inactive_loss, ctrl_loss   — loss terms (unweighted)
            eff_rank, active_rank, ctrl_align,
            inactive_energy                          — diagnostic scalars
        """
        if emb.dim() != 3:
            raise ValueError(f"Expected emb (B, T, D), got {tuple(emb.shape)}")
        if emb.size(-1) != self.embed_dim:
            raise ValueError(
                f"embed_dim mismatch: expected {self.embed_dim}, got {emb.size(-1)}"
            )

        if self.training:
            self.step.add_(1)
            step = int(self.step.item())
            if step >= self.warmup_steps and step % self.update_basis_every == 0:
                self._update_basis(
                    emb=emb,
                    predictor=predictor,
                    ctx_emb=ctx_emb,
                    ctx_act=ctx_act,
                )

        z_A, z_N, U_A, U_N = self._project_active_inactive(emb)

        # SIGReg on active subspace: (B, T, r_A) → (T, B, r_A) for SketchSIGReg
        sigreg_loss = self.sigreg(z_A.transpose(0, 1).contiguous())

        # Inactive subspace compression
        inactive_loss = z_N.square().mean()

        if use_ctrl_loss:
            ctrl_loss, ctrl_align = self._ctrl_loss(predictor, ctx_emb, ctx_act, U_A, U_N)
        else:
            ctrl_loss = emb.new_zeros(())
            ctrl_align = emb.new_zeros(())

        with torch.no_grad():
            self.last_inactive_energy.copy_(
                inactive_loss.detach().float().to(self.last_inactive_energy.device)
            )
            self.last_ctrl_align.copy_(
                ctrl_align.detach().float().to(self.last_ctrl_align.device)
            )

        return {
            "sigreg_loss": sigreg_loss,
            "inactive_loss": inactive_loss,
            "ctrl_loss": ctrl_loss,
            "eff_rank": self.last_eff_rank.to(emb.device),
            "active_rank": self.last_active_rank.to(emb.device),
            "ctrl_align": self.last_ctrl_align.to(emb.device),
            "inactive_energy": self.last_inactive_energy.to(emb.device),
        }
