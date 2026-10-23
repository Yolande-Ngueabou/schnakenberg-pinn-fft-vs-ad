#!/usr/bin/env python3
# =============================================================================
#  schnakenberg_v16_autograd_squarev2.py
#  Schnakenberg 2D  — Method B: Integer-RFF + Nested-Autograd Laplacian + IMEX
#  SQUARE or HEX-COMPATIBLE RECTANGULAR PERIODIC DOMAIN  — all 9 regimes
#
#  v2 CHANGES FROM v1
#  --------------------
#  Method A moved to v9 (hex-compatible rectangular domain for K_Hpi) and
#  v10 (--seed CLI actually threaded through, --force-domain-shape) while
#  Method B stayed at v1 (matched only to Method A v8). This broke the
#  ablation's own stated control -- "every component other than the
#  Laplacian computation is IDENTICAL between the two files" -- for any
#  regime needing domain_shape="hex_rect" (K_Hpi) or seed variation
#  (any multi-seed statistics). v2 ports both changes across so the
#  control is restored:
#    - get_commensurate_rect_L, build_triad_ic, IC_MODE="triad"
#    - IntegerRFFEmbedding scaled by Lx/Ly separately, force_modes support
#    - radial_spectrum/classify_field/check_periodicity generalized to Lx,Ly
#    - K_Hpi regime entry gets domain_shape="hex_rect"
#    - --seed CLI, actually threaded through IC construction, RFF
#      embedding, and pretraining collocation sampling (was a Python
#      default-argument value only, same bug v10 fixed in Method A --
#      see v10's header for why that never actually varied anything)
#    - --force-domain-shape CLI override
#    - output filenames now carry the seed, matching Method A v10
#  This is a mechanical, non-Laplacian-touching port: every edit mirrors
#  the corresponding v9/v10 edit in schnakenberg_v16_spectral_v10.py.
#  spectral_lap vs autograd_lap (the ONLY intended difference between the
#  two files) is untouched by this change.
#
#  ABLATION STUDY — METHOD B (companion to schnakenberg_v16_spectral_v10.py)
#  --------------------------------------------------------------------------
#  This file computes the Laplacian via nested automatic differentiation
#  instead of the FFT (Method A). Per the original ablation design, EVERY
#  other component is byte-for-byte identical between the two files:
#    1. Same grid evaluation (N_GRID_R points per regime, same --domain tiers)
#    2. Same IntegerRFFEmbedding class and same RNG seed
#    3. Same IMEX residual formula, same warm-start mechanism
#    4. Same three-criterion early stopping rule and thresholds
#    5. Same v7 collapse-abort and fixed-point stabilizing regularizer
#    6. Same v8 IC_MODE toggle (eigenmode / broadband)
#  The ONLY function that differs is the Laplacian:
#    Method A: Lap(u) = IFFT(-k^2 * FFT(u))                  [[FFT, O(N log N)]]
#    Method B: Lap(u) = d^2u/dx^2 + d^2u/dy^2 via nested tf.GradientTape
#
#  HOW THE NESTED-AUTOGRAD LAPLACIAN WORKS
#  -----------------------------------------
#  Getting d^2u/dx^2 and d^2u/dy^2 SEPARATELY (not mixed d^2u/dxdy) requires
#  the diagonal of the Hessian, which needs three levels of tf.GradientTape
#  active at once, nested in this order:
#    outer_tape  (for training: d(total_loss)/d(model.trainable_variables))
#      tape2     (for the 2nd spatial derivative: watches xy)
#        tape1   (for the 1st spatial derivative: watches xy)
#          u,v = model(xy)
#        du_dxy = tape1.gradient(u, xy)   # [du/dx, du/dy]
#      d2u_dx2 = tape2.gradient(du_dxy[:,0], xy)[:,0]
#      d2u_dy2 = tape2.gradient(du_dxy[:,1], xy)[:,1]
#  This is the standard construction for second-order PINN residuals, going
#  back to Raissi, Perdikaris & Karniadakis (2019) J. Comput. Phys.
#  378:686-707, who compute PDE residuals of this order via nested
#  automatic differentiation through the network's inputs.
#
#  COMPUTATIONAL COST COMPARISON (evidence-based, same citations as Method A)
#  ----------------------------------------------------------------------------
#  Automatic differentiation for a k-th order derivative at N collocation
#  points, through a network with l hidden layers and q neurons per layer,
#  costs O(2^(k-1)*l*N*q^2) per the Sobolev-cubatures analysis [arXiv:2211.15443].
#  Here k=2 (second order) gives a factor 2^(2-1)=2, and since BOTH
#  d^2u/dx^2 and d^2u/dy^2 must be computed as separate second derivatives
#  (2 spatial dimensions), the total is 2(order)*2(dims) = O(4*l*N*q^2) --
#  independent of grid size N in the sense that cost scales linearly with N
#  (collocation points), not O(N log N) as with Method A's FFT, and unlike
#  Method A's O(N log N) it also scales with network size (l, q), not just
#  N. [FC-PINO, arXiv:2211.15960: "Spectral differentiation is significantly
#  more memory-efficient than automatic differentiation. When applied to
#  high-dimensional PDEs and deep networks, automatic differentiation must
#  record the entire computational graph and intermediate values, often
#  leading to prohibitive memory overhead."] This script prints ms/window
#  exactly like Method A so the two can be compared empirically, not just
#  by asymptotic complexity.
#
#  CITATIONS (verified, same core set as Method A)
#  --------------------------------------------------------
#  Raissi, Perdikaris & Karniadakis (2019) J. Comput. Phys. 378:686-707
#  Rahimi & Recht (2007) NeurIPS  — Random Fourier Features
#  Tancik et al. (2020) NeurIPS   — Fourier features, spectral bias
#  Wang, Yu & Perdikaris (2022) J. Comput. Phys. 449:110768  — NTK/PINN
#  Ascher, Ruuth & Wetton (1995) SIAM J. Numer. Anal. — IMEX splitting
#  Chen, Lai & Yang (2024) Thin-Walled Struct. 196:111423 — warm-start
#  Loshchilov & Hutter (2017) ICLR  — SGDR cosine annealing
#  Kingma & Ba (2015) ICLR  — Adam optimiser
#  Glorot & Bengio (2010) AISTATS  — glorot_normal initialiser
#  Schnakenberg (1979) J. Theor. Biol. 81:389-400
#  Turing (1952) Phil. Trans. R. Soc. B 237:37-72
#  Khudhair, Zhang & Fukawa (2022) Numer. Methods PDEs 38:1843-1860
#  Krishnapriyan et al. (NeurIPS 2021) -- PINN reaction-diffusion collapse
#  Rohrhofer, Posch, Gossnitzer & Geiger (TMLR 2023) -- fixed points in PINNs
#  Babic, Rohrhofer & Geiger (2025), "Stabilizing PINNs" (arXiv:2509.11768)
#
#  USAGE
#  -----
#    python schnakenberg_v16_autograd_squarev1.py              # all 9
#    python schnakenberg_v16_autograd_squarev1.py --regime K_stripes
#    python schnakenberg_v16_autograd_squarev1.py --quick
#    python schnakenberg_v16_autograd_squarev1.py --list-regimes
#    python schnakenberg_v16_autograd_squarev1.py --no-early-stop
#    python schnakenberg_v16_autograd_squarev1.py --save-dir ./results
#    python schnakenberg_v16_autograd_squarev1.py --ic-mode broadband
#
#  ABLATION STUDY — METHOD A
#  --------------------------
#  This file is one half of the ablation study comparing two approaches
#  to computing the Laplacian inside a PINN's IMEX training loop:
#
#    Method A (this file):  Lap(u) = IFFT(−k²·FFT(u))   [spectral FFT]
#    Method B (companion):  Lap(u) = d^2u/dx^2 + d^2u/dy^2 [nested autograd]
#
#  Every component other than the Laplacian computation is IDENTICAL
#  between the two files; four experimental controls are verified:
#    1. Same 64x64 grid evaluation (N_COL = 64^2 = 4096)
#    2. Same IntegerRFFEmbedding class and same RNG seed
#    3. Same IMEX residual formula, same warm-start mechanism
#    4. Same three-criterion early stopping rule and thresholds
#
#  COMPUTATIONAL COST COMPARISON (evidence-based)
#  -----------------------------------------------
#  Spectral differentiation (Method A) has cost O(N log N) per Laplacian
#  evaluation via the FFT and is memory-efficient because it requires no
#  storage of intermediate computational graph states.
#  [FC-PINO, arXiv:2211.15960: "Spectral differentiation is significantly
#   more memory-efficient than automatic differentiation. When applied to
#   high-dimensional PDEs and deep networks, automatic differentiation
#   must record the entire computational graph and intermediate values,
#   often leading to prohibitive memory overhead."]
#
#  Automatic differentiation (Method B) for a second-order derivative
#  at N collocation points through a network with l layers and q neurons
#  has cost O(2^(k−1)·l·N·q²) where k=2 (second-order), i.e., O(4lNq^2)
#  vs O(N log N) for Method A.
#  [Sobolev cubatures paper arXiv:2211.15443: "For a PINN with l hidden
#   layers and q neurons per layer, the complexity per epoch for computing
#   the k-th derivative scales as O(2^(k-1)·l·N·q²) for automatic
#   differentiation."]
#
#  CITATIONS (verified)
#  --------------------
#  Raissi, Perdikaris & Karniadakis (2019) J. Comput. Phys. 378:686–707
#  Rahimi & Recht (2007) NeurIPS  — Random Fourier Features
#  Tancik et al. (2020) NeurIPS   — Fourier features, spectral bias
#  Wang, Yu & Perdikaris (2022) J. Comput. Phys. 449:110768  — NTK/PINN
#  Trefethen (2000) SIAM  — spectral differentiation (Fourier methods)
#  Ascher, Ruuth & Wetton (1995) SIAM J. Numer. Anal. — IMEX splitting
#  Chen, Lai & Yang (2024) Thin-Walled Struct. 196:111423 — warm-start
#  Loshchilov & Hutter (2017) ICLR  — SGDR cosine annealing
#  Kingma & Ba (2015) ICLR  — Adam optimiser
#  Glorot & Bengio (2010) AISTATS  — glorot_normal initialiser
#  Schnakenberg (1979) J. Theor. Biol. 81:389–400
#  Turing (1952) Phil. Trans. R. Soc. B 237:37–72
#  Khudhair, Zhang & Fukawa (2022) Numer. Methods PDEs 38:1843–1860
#
#  USAGE
#  -----
#    python schnakenberg_v16_spectral_squarev6.py              # all 9
#    python schnakenberg_v16_spectral_squarev6.py --regime K_stripes
#    python schnakenberg_v16_spectral_squarev6.py --quick
#    python schnakenberg_v16_spectral_squarev6.py --list-regimes
#    python schnakenberg_v16_spectral_squarev6.py --no-early-stop
#    python schnakenberg_v16_spectral_squarev6.py --save-dir ./results
# =============================================================================

import os, time, argparse, warnings, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
#  CONSTANTS
# =============================================================================

SEED      = 42
PI        = float(np.pi)
# L computed per-regime from k* (see get_commensurate_L in train_regime)
N_GRID    = 64           # grid resolution (64x64 = 4096 points)
SAVE_DIR  = "/content/drive/MyDrive/Schnakenberg_v16_METHOD_B_v2"
_RNG      = np.random.default_rng(SEED)

# =============================================================================
#  COMMENSURATE DOMAIN FUNCTION — FIX FOR SPECTRAL ERRORS
# =============================================================================
# PROBLEM (fixed here): Using a fixed L=1.0 for all regimes causes spectral
# errors of up to 17.1% because the theoretical dominant wavenumber k* does
# not fall on any integer Fourier mode of the L=1 lattice.
#
# CAUSE: On a periodic domain [0,L], the admissible wavenumbers are discrete:
#   k_{n_x,n_y} = (2pi/L) * sqrt(n_x^2 + n_y^2),  n_x,n_y in Z
# If k* != (2pi/L)*sqrt(n^2+m^2) for any integers n,m, patterns form at the
# nearest available lattice point rather than at k*, causing spectral error.
# The range of unstable wavenumbers of allowable patterns strictly depends on
# the domain geometry and boundary conditions; to see a pattern emerging there
# must exist at least a mode pair (n_x,n_y) such that k falls within [kmin,kmax].
# [Avery & Scheel (2020); Brusselator WNL analysis, arXiv:1310.6571]
#
# FIX: Choose L = n * ell* where ell* = 2pi/k* and n = round(1.0/ell*).
# This places the pure mode (n,0) exactly at k* (zero spectral error) since:
#   k_{n,0} = (2pi/L) * n = (2pi/(n*ell*)) * n = 2pi/ell* = k*
# The integer n is chosen to make L as close to 1.0 as possible.

def get_commensurate_L(k_star, target_L=1.0):
    """
    Compute L = n * ell* (commensurate domain length) and n.
    On [0,L]^2, mode (n,0) lies exactly at k* with 0% spectral error.
    n is chosen to minimise |L - target_L|.
    """
    ell_star = 2.*PI / k_star
    n = max(1, round(target_L / ell_star))
    L = n * ell_star
    # Verify: k_{n,0} = 2pi*n/L = 2pi*n/(n*ell*) = k* exactly
    assert abs(2.*PI*n/L - k_star) < 1e-10, "Commensurability check failed"
    return float(L), int(n)

def get_commensurate_rect_L(k_star, n_mode_target):
    """
    v2 (ported from Method A v9): hex-compatible RECTANGULAR domain.
    Places the resonant triad wavevectors (k_star,0),
    (-k_star/2,+k_star*sqrt(3)/2), (-k_star/2,-k_star*sqrt(3)/2) -- 120
    degrees apart, all |k|=k_star -- EXACTLY on the integer FFT lattice.
    [pde2path tutorial, arXiv:1908.05211, Sec 2.2.2; Faraday-wave hexagon
    study, arXiv:1503.04403]
    Identical to Method A's get_commensurate_rect_L -- this derivation is
    purely about the domain lattice, has nothing to do with which
    Laplacian method is used downstream.
    """
    n_mode = int(n_mode_target)
    if n_mode % 2 != 0:
        n_mode += 1
    ell_star = 2.*PI / k_star
    L_x = n_mode * ell_star
    L_y = L_x / np.sqrt(3.)
    half = n_mode // 2
    triad_modes = [(n_mode, 0), (-half, half), (-half, -half)]
    for (nx, ny) in triad_modes:
        kx = 2.*PI*nx/L_x; ky = 2.*PI*ny/L_y
        k_mag = np.sqrt(kx**2+ky**2)
        assert abs(k_mag - k_star) < 1e-8*k_star, \
            f"Triad mode {(nx,ny)} magnitude {k_mag} != k_star {k_star}"
    return float(L_x), float(L_y), int(n_mode), triad_modes

N_FF      = 96           # RFF modes
RFF_K_MAX = 25
N_HIDDEN  = 128
N_LAYERS  = 4
N_MODES_IC = 25

T_WINDOW_DEFAULT   = 0.015
N_STEPS_PRETRAIN   = 2000
N_STEPS_PER_WINDOW = 200
N_COL              = 4000   # pretrain collocation points

LR_INIT   = 1e-3
LR_MIN    = 1e-5
GRAD_CLIP = 0.5
LOG_EVERY = 100
W_PDE     = 100.0

# Early stopping — identical to Method B
ES_WINDOW      = 50
ES_AMP_TOL     = 0.005
ES_VAR_REL_STD = 0.010
ES_TOL_PDE = {
    "S_spots": 1e-6, "M_spots": 1e-6, "T_spots": 1e-6,
    "near_onset": 1e-8, "low_rho": 1e-4, "min_rho": 1e-4,
    "K_stripes": 1e-5, "K_mixed": 1e-6, "K_Hpi": 1e-6,
}
ES_MIN_WINDOWS = {
    "S_spots": 100, "M_spots": 100, "T_spots": 200,
    "near_onset": 2000, "low_rho": 300, "min_rho": 500,
    "K_stripes": 4000, "K_mixed": 100, "K_Hpi": 100,
}

METHOD_LABEL = "Method B — Nested-Autograd Laplacian"
METHOD_SHORT = "autograd"

# =============================================================================
#  v7: COLLAPSE ABORT + FIXED-POINT STABILIZING REGULARIZER
# =============================================================================
# COLLAPSE ABORT: fires once u_amp has fallen below COLLAPSE_AMP_FRAC*ic_amp
# AFTER at least COLLAPSE_EFOLDS growth-timescales (1/sigma_max) have
# elapsed -- i.e. only once the field has had enough physical time to grow
# per linear theory, but instead shrank. Mirrors the existing u_amp>100
# blow-up check, which never caught this direction of failure.
COLLAPSE_AMP_FRAC = 0.10
COLLAPSE_EFOLDS    = 3.0
COLLAPSE_MIN_WINDOWS = 20   # never abort before this many windows regardless

# STABILIZING REGULARIZER: adapted from Babic, Rohrhofer & Geiger (2025)
# "Stabilizing PINNs" (arXiv:2509.11768), Eq. 3-7. Their R_LS x R_SE term
# becomes, in this PDE setting:
#   gate  = exp(-Var[u]/STAB_EPS)          (~1 near the homogeneous state,
#                                            ~0 once a pattern has grown)
#   L_stab = C(w) * gate * sigma_max        (sigma_max>0 is this regime's
#                                            already-computed linear growth
#                                            rate -- the "unstable fixed
#                                            point" penalty)
#   C(w)   = max(STAB_C0*(STAB_GAMMA_FRAC - w/n_windows), 0)   (Eq. 6)
# STAB_EPS is set per-regime from ic_amp**2 (the natural variance scale of
# the seeded IC) rather than a single global constant, since ic_amp varies
# 10x across regimes (0.005 to 0.01) in this script's REGIMES list.
STAB_C0          = 1.0
STAB_GAMMA_FRAC  = 0.5
STAB_ENABLED     = True

# =============================================================================
#  v8: IC MODE + PAPER-COMPARISON DIAGNOSTIC
# =============================================================================
# IC_MODE:
#   "eigenmode"  (default, v5-v7 behavior) -- all N_MODES_IC realizations at
#                the single wavenumber k_true, varying only angle/phase.
#                Deliberately anti-coarsening / pre-selects the pattern.
#   "broadband"  -- N_MODES_BROADBAND realizations with k drawn uniformly
#                across the WHOLE unstable band [k_min,k_max] (not just
#                k_true), random angle and phase. Closest architecture-
#                representable analogue of Khudhair et al.'s actual IC
#                (i.i.d. noise on [-0.01,0.01]^2 at every grid point,
#                Sec.4) -- true i.i.d. grid noise is not periodic/
#                band-limited and cannot be represented exactly by this
#                script's integer-lattice RFF embedding, so "broadband"
#                here means genuinely multi-wavenumber, not literally
#                the same noise realization as the paper.
IC_MODE = "eigenmode"
N_MODES_BROADBAND = 100

# Paper's fixed domain length, for the period-count diagnostic printed in
# train_regime() -- purely informational, does not change any training
# behavior. [Khudhair, Zhang & Fukawa 2022, Sec.4: "Omega=[-4,4]^2"]
PAPER_DOMAIN_L = 8.0

# =============================================================================
#  REGIME DEFINITIONS
# =============================================================================

REGIMES = [
    # ── Khudhair Regime I (H0 spot family) ─────────────────────────────────
    # ic_amp=0.01: must be >> sqrt(L_pde/W_pde) ~ 1e-5 so (5,0) seeded mode
    # dominates (5,2) oblique mode within T.  For T=2.5s:
    # t_cross = ln(noise/A0) / (sigma_(5,2) - sigma_(5,0)) >> T  is required.
    # [Khudhair et al. 2022, Sec 4: "perturbation on [-0.01,0.01]^2"]
    dict(label="S_spots",   a=0.10, b=0.90, gamma=30.0, D1=0.05,  D2=1.0,
         T=2.5,  ic_amp=0.01, n_mode_target=5, N_grid=64,
         expected="H0 hexagons (spots)"),
    dict(label="M_spots",   a=0.25, b=0.75, gamma=40.0, D1=0.020, D2=1.0,
         T=2.5,  ic_amp=0.01, n_mode_target=5, N_grid=64,
         expected="H0 hexagons (spots)"),
    # T_stripes: sigma_max=0.486; t_grow = ln(1/0.01)/0.486 = 9.5s
    # T=12s gives 2.7 growth timescales -- safely above saturation
    dict(label="T_spots", a=0.40, b=0.60, gamma=40.0, D1=0.008, D2=1.0,
         T=12.0, ic_amp=0.01, n_mode_target=5, N_grid=64,
         expected="H0 hexagons (spots)"),
    dict(label="near_onset",a=0.40, b=0.60, gamma=40.0, D1=0.0091,D2=1.0,
         T=40.0, ic_amp=0.005, n_mode_target=5, N_grid=64,
         expected="H0 hexagons (spots, slow)"),
    dict(label="low_rho",   a=0.42, b=0.58, gamma=15.0, D1=0.002, D2=1.0,
         T=6.0,  ic_amp=0.005, dt_override=0.003, n_mode_target=5, N_grid=64,
         expected="H0 hexagons (spots, stiff)"),
    dict(label="min_rho",   a=0.45, b=0.55, gamma=15.0, D1=0.002, D2=1.0,
         T=50.0, ic_amp=0.005, dt_override=0.003, n_mode_target=5, N_grid=64,
         expected="H0 hexagons (spots, stiff)"),
    # ── Khudhair Regime II ──────────────────────────────────────────────────
    # ic_amp=0.01 for all: (5,0) mode must dominate (5,2) oblique within T
    # dt_override=0.005 for K_Hpi, K_mixed: prevents v->0 IMEX stiffness
    # When v_min<0.1 and u is large, reaction term gamma*(a-u+u^2*v) ~ -40;
    # halving dt reduces truncation error and stabilises training.
    # [Reduced-PINN 2022: "stiffness is the primary cause of PINN failure"]
    dict(label="K_stripes", a=0.25, b=3.35, gamma=10.0, D1=0.01, D2=1.0,
         T=120.0, ic_amp=0.01, n_mode_target=5, N_grid=64, no_scale_T=True,
         expected="stripes (Khudhair Regime II)"),
    dict(label="K_mixed",   a=0.05, b=3.45, gamma=10.0, D1=0.01, D2=1.0,
         T=20.0, ic_amp=0.01, dt_override=0.005, n_mode_target=5, N_grid=64,
         no_scale_T=True,
         expected="mixed/labyrinthine (Khudhair Regime II)"),
    dict(label="K_Hpi",     a=0.025,b=3.475,gamma=10.0, D1=0.01, D2=1.0,
         T=15.0, ic_amp=0.01, dt_override=0.005, n_mode_target=5, N_grid=64,
         no_scale_T=True, domain_shape="hex_rect",
         expected="Hpi hexagons (Khudhair Regime II)"),
]

# =============================================================================
#  TURING ANALYSIS
# =============================================================================

def get_true_k_dominant(D1, D2, gamma, a, b, n_scan=5000):
    """
    Compute k_true = argmax_k sigma(k), the wavenumber with the highest linear
    growth rate, by numerically maximising the dispersion relation over the
    instability band.

    WHY THIS IS NEEDED:
    The standard formula k* = sqrt(gamma(D_2fu+D_1gv)/(2D_1D_2)) gives the
    wavenumber that MINIMISES det A(k), which approximates argmax sigma(k) only
    when D_1 ≪ D_2.  For moderate diffusion ratios D_2/D_1 in {20,50}, the
    approximation is 20–21% off (e.g. k*_formula=21.91, k_true=17.30 for
    M_mixed), causing the PINN IC to seed the wrong wavenumber and leading
    to coarsening artefacts in the final pattern.

    SEEDING ADVANTAGE (Remark 2 in the paper):
    If the IC is seeded at k_true with amplitude A₀=ic_amp, any competing
    mode starting from numerical noise ε~10⁻⁶ requires time
      t_cross = ln(ε/A₀) / (σ_competitor − σ(k_true))
    to overtake the seeded mode.  For all regimes in this study,
    sigma_competitor <= sigma(k_true) for the dominant competitor on the commensurate
    domain, so t_cross ≥ ∞ (seeded mode always wins — verified numerically).

    Returns: k_true (float), sigma_max (float)
    """
    fu, fv, gu, gv = _jacobian(a, b)
    km, kM, _ = unstable_band(D1, D2, gamma, a, b)
    if km is None:
        return None, None

    k_scan = np.linspace(km * 1.001, kM * 0.999, n_scan)
    best_k, best_sigma = km, 0.

    for k in k_scan:
        k2 = k * k
        T  = gamma * (fu + gv) - (D1 + D2) * k2
        D  = ((gamma * fu - D1 * k2) * (gamma * gv - D2 * k2)
              - (gamma * fv) * (gamma * gu))
        disc = T * T - 4. * D
        if disc < 0.:
            continue
        s = (T + np.sqrt(disc)) / 2.
        if s > best_sigma:
            best_sigma = s
            best_k = k

    return float(best_k), float(best_sigma)


def _jacobian(a, b):
    u = a+b
    return -1.0+2*b/u, u**2, -2*b/u, -u**2

def unstable_band(D1, D2, gamma, a, b):
    fu,fv,gu,gv = _jacobian(a,b)
    A=D1*D2; B=-(D2*fu+D1*gv)*gamma; C=(fu*gv-fv*gu)*gamma**2
    disc = B**2 - 4*A*C
    if disc < 0 or B >= 0:
        return None, None, None
    return (np.sqrt(max((-B-np.sqrt(disc))/(2*A), 0)),
            np.sqrt((-B+np.sqrt(disc))/(2*A)),
            np.sqrt(-B/(2*A)))

def turing_diagnostics(D1, D2, gamma, a, b):
    fu,fv,gu,gv = _jacobian(a,b)
    u_star=a+b; v_star=b/u_star**2
    c1=(fu+gv)<0; c2=(fu*gv-fv*gu)>0
    km,kM,ks = unstable_band(D1,D2,gamma,a,b)
    c3=km is not None; c4=c3 and kM>km
    return dict(u_star=u_star, v_star=v_star, c1=c1, c2=c2, c3=c3, c4=c4,
                is_turing=all([c1,c2,c3,c4]),
                k_star=float(ks) if ks else 0.,
                k_min=float(km) if km else 0.,
                k_max=float(kM) if kM else 0.)

def growing_eigenvector(D1, D2, gamma, a, b, k_star):
    fu,fv,gu,gv = _jacobian(a,b)
    k2=k_star**2
    A11=gamma*fu-D1*k2; A12=gamma*fv
    A21=gamma*gu;        A22=gamma*gv-D2*k2
    tr=A11+A22; det=A11*A22-A12*A21
    lam=(tr+np.sqrt(max(tr**2-4*det, 0)))/2.
    c_v=(lam-A11)/A12 if abs(A12)>1e-12 else 0.
    return 1., c_v, lam

# =============================================================================
#  IC CONSTRUCTION
# =============================================================================

def build_eigenmode_ic(u_star, v_star, D1, D2, gamma, a, b,
                       k_star, ic_amp, seed=SEED):
    rng = np.random.default_rng(seed)
    _,c_v,_ = growing_eigenvector(D1,D2,gamma,a,b,k_star)
    modes_u=[]; modes_v=[]
    for _ in range(N_MODES_IC):
        th=rng.uniform(0,2*PI); kx=k_star*np.cos(th); ky=k_star*np.sin(th)
        phi=rng.uniform(0,2*PI); A=ic_amp/N_MODES_IC
        modes_u.append((A,kx,ky,phi)); modes_v.append((A*c_v,kx,ky,phi))
    return (u_star,modes_u),(v_star,modes_v)

def build_broadband_ic(u_star, v_star, D1, D2, gamma, a, b,
                        k_min, k_max, ic_amp, seed=SEED):
    """
    v8: paper-inspired broadband IC -- N_MODES_BROADBAND realizations with
    |k| drawn UNIFORMLY across the whole unstable band [k_min,k_max]
    (not fixed at k_true), random angle and phase. Unlike
    build_eigenmode_ic, the eigenvector ratio c_v = v-amplitude/u-amplitude
    is recomputed PER MODE at that mode's own k (growing_eigenvector
    depends on k), since it is not constant across the band.
    [Khudhair, Zhang & Fukawa 2022, Sec.4, adapted -- see IC_MODE docstring
    above for why this is an analogue rather than a literal reproduction.]
    """
    rng = np.random.default_rng(seed)
    modes_u=[]; modes_v=[]
    for _ in range(N_MODES_BROADBAND):
        k_i = rng.uniform(k_min, k_max)
        _,c_v_i,_ = growing_eigenvector(D1,D2,gamma,a,b,k_i)
        th=rng.uniform(0,2*PI); kx=k_i*np.cos(th); ky=k_i*np.sin(th)
        phi=rng.uniform(0,2*PI); A=ic_amp/N_MODES_BROADBAND
        modes_u.append((A,kx,ky,phi)); modes_v.append((A*c_v_i,kx,ky,phi))
    return (u_star,modes_u),(v_star,modes_v)

def build_triad_ic(u_star, v_star, D1, D2, gamma, a, b, triad_kxky, ic_amp, seed=SEED):
    """
    v2 (ported from Method A v9): seeds ONLY the resonant triad
    wavevectors (three modes at 120 degrees, all |k|=k_star), each with a
    random phase. Only meaningful on a domain built by
    get_commensurate_rect_L, where the triad is exactly representable.
    triad_kxky: list of (kx,ky) PHYSICAL wavevector pairs.
    """
    rng = np.random.default_rng(seed)
    modes_u=[]; modes_v=[]
    n_modes = len(triad_kxky)
    for (kx, ky) in triad_kxky:
        k_mag = float(np.hypot(kx, ky))
        _,c_v,_ = growing_eigenvector(D1,D2,gamma,a,b,k_mag)
        phi = rng.uniform(0,2*PI); A = ic_amp/n_modes
        modes_u.append((A,kx,ky,phi)); modes_v.append((A*c_v,kx,ky,phi))
    return (u_star,modes_u),(v_star,modes_v)

def build_ic(u_star, v_star, D1, D2, gamma, a, b, k_star, k_min, k_max,
             ic_amp, mode=IC_MODE, seed=SEED, triad_kxky=None):
    if mode == "eigenmode":
        return build_eigenmode_ic(u_star,v_star,D1,D2,gamma,a,b,k_star,ic_amp,seed)
    elif mode == "broadband":
        return build_broadband_ic(u_star,v_star,D1,D2,gamma,a,b,k_min,k_max,ic_amp,seed)
    elif mode == "triad":
        if not triad_kxky:
            raise ValueError("IC_MODE='triad' requires triad_kxky (only valid for "
                              "domain_shape='hex_rect' regimes)")
        return build_triad_ic(u_star,v_star,D1,D2,gamma,a,b,triad_kxky,ic_amp,seed)
    else:
        raise ValueError(f"Unknown IC_MODE: {mode!r}")

def synth_numpy(ic, xg, yg):
    u0,modes=ic
    XX,YY=np.meshgrid(xg,yg,indexing='ij')
    U=np.full_like(XX,u0,dtype=np.float32)
    for A,kx,ky,phi in modes: U+=A*np.cos(kx*XX+ky*YY+phi)
    return U

def synth_points(ic, xy):
    u0,modes=ic
    U=np.full(len(xy),u0,dtype=np.float32)
    for A,kx,ky,phi in modes: U+=A*np.cos(kx*xy[:,0]+ky*xy[:,1]+phi)
    return U

# =============================================================================
#  LAPLACIAN: NONE NEEDED HERE (Method B differentiates the network directly)
# =============================================================================
# Unlike Method A, which precomputes a k^2 kernel and applies it via FFT,
# Method B's Laplacian (autograd_lap, defined inside _build_model_and_steps
# below) needs no precomputed kernel -- it differentiates model(xy) twice
# w.r.t. xy directly at whatever points are passed in.

# =============================================================================
#  ANALYSIS HELPERS
# =============================================================================

def radial_spectrum(field, Lx, n_bins=80, Ly=None):
    if Ly is None: Ly = Lx
    N=field.shape[0]
    F=np.fft.fftshift(np.fft.fft2(field-field.mean())); P=np.abs(F)**2
    kx_phys=np.fft.fftshift(np.fft.fftfreq(N,d=1./N))*2*PI/Lx
    ky_phys=np.fft.fftshift(np.fft.fftfreq(N,d=1./N))*2*PI/Ly
    KX,KY=np.meshgrid(kx_phys,ky_phys,indexing='ij'); KR=np.sqrt(KX**2+KY**2)
    bins=np.linspace(0,KR.max(),n_bins+1); rk=0.5*(bins[:-1]+bins[1:])
    rP=np.array([P[(KR>=bins[i])&(KR<bins[i+1])].mean()
                 if ((KR>=bins[i])&(KR<bins[i+1])).any() else 0.
                 for i in range(n_bins)])
    return rk,rP

def classify_field(U, Lx, k_star, Ly=None):
    """v2 note (same caveat as Method A v9): on a rectangular (Lx!=Ly)
    domain the kx/ky sampling resolution differs by construction, so
    aniso is a rougher indicator there -- prefer visual inspection of
    the u/v panels for hex_rect regimes."""
    if Ly is None: Ly = Lx
    rk,rP=radial_spectrum(U,Lx,Ly=Ly)
    kp=float(rk[np.argmax(rP)])
    N=U.shape[0]
    F=np.fft.fft2(U-U.mean()); P2=np.abs(F)**2
    kfx=np.fft.fftfreq(N,d=1./N)*2*PI/Lx
    kfy=np.fft.fftfreq(N,d=1./N)*2*PI/Ly
    FX,FY=np.meshgrid(kfx,kfy,indexing='ij'); FR=np.sqrt(FX**2+FY**2)
    band=(FR>0.5*kp)&(FR<2*kp)
    if not band.any(): return "homogeneous",0.,kp
    px=(P2[band]*np.abs(FX[band])).sum()/(P2[band].sum()+1e-30)
    py=(P2[band]*np.abs(FY[band])).sum()/(P2[band].sum()+1e-30)
    aniso=abs(px-py)/(px+py+1e-30)
    rv=float(np.sqrt(U.var())/max(float(np.abs(U-U.mean()).mean()),1e-12))
    if rv<0.10: return "homogeneous",0.,kp
    if aniso>0.60: return "stripes",float(aniso),kp
    if aniso<0.15: return "spots",float(aniso),kp
    return "mixed/labyrinthine",float(aniso),kp

def check_periodicity(model, Lx, label="", Ly=None):
    import tensorflow as tf
    if Ly is None: Ly = Lx
    pts=tf.constant([[0.123,0.321],[0.456,0.654]],dtype=tf.float32)
    px=float(np.abs(model(pts).numpy()-
             model(pts+tf.constant([[Lx,0.],[Lx,0.]])).numpy()).max())
    py=float(np.abs(model(pts).numpy()-
             model(pts+tf.constant([[0.,Ly],[0.,Ly]])).numpy()).max())
    tag=f"  [{label}] " if label else "  "
    print(f"{tag}Per-x: {px:.2e} {'✓' if px<1e-4 else '✗'}   "
          f"Per-y: {py:.2e} {'✓' if py<1e-4 else '✗'}")

# =============================================================================
#  MODEL BUILDER — METHOD B (NESTED-AUTOGRAD LAPLACIAN)
#
#  The ONLY component that differs from Method A is the Laplacian:
#  Method A computes:  Lap(u) = IFFT(-k^2*FFT(u))          [FFT]
#  Method B computes:  Lap(u) = d^2u/dx^2 + d^2u/dy^2       [nested tapes, here]
# =============================================================================

def _build_model_and_steps(sigma_rff, D1, D2, gamma, a, b,
                            u_star, v_star, Lx, grid_xy_tf,
                            sigma_max=0.0, eps_var=1e-6, Ly=None,
                            must_include=None, seed=SEED):
    import tensorflow as tf
    if Ly is None: Ly = Lx   # square domain (v1 behavior) unless told otherwise
    must_include = must_include or []

    class IntegerRFFEmbedding(tf.keras.layers.Layer):
        """
        Integer-valued 2D RFF embedding. Exact periodicity on [0,Lx]x[0,Ly]
        (per-regime commensurate domain; Lx=Ly for square regimes).
        [Rahimi & Recht (2007); Tancik et al. (2020)]
        Embedding: proj = matmul(x/Lx, y/Ly against B.T)  -> [N, n_ff]

        v2 (ported from Method A v9): must_include guarantees the
        resonant triad modes are always in the basis for hex_rect
        regimes, rather than left to chance in the random draw below.
        """
        def __init__(self, n_ff, sigma, seed=SEED, force_modes=None, **kw):
            super().__init__(**kw)
            force_modes = force_modes or []
            rng=np.random.default_rng(seed)
            B_raw=rng.standard_normal((8*n_ff,2))*float(sigma)
            B_int=np.round(B_raw).astype(np.int32)
            B_int=np.clip(B_int,-RFF_K_MAX,RFF_K_MAX)
            seen=set(); uniq=[]
            for (nx,ny) in force_modes:      # v2: forced modes go in first
                t=(int(nx),int(ny))
                if t==(0,0) or t in seen: continue
                seen.add(t); uniq.append(t)
            for row in B_int:
                t=(int(row[0]),int(row[1]))
                if t==(0,0) or t in seen: continue
                seen.add(t); uniq.append(t)
                if len(uniq)==n_ff: break
            r=max(1,int(round(float(sigma))))
            while len(uniq)<n_ff:
                for kk in range(-r,r+1):
                    for ll in range(-r,r+1):
                        t=(kk,ll)
                        if t==(0,0) or t in seen: continue
                        seen.add(t); uniq.append(t)
                        if len(uniq)==n_ff: break
                    if len(uniq)==n_ff: break
                r+=1
            self._B=np.array(uniq[:n_ff],dtype=np.float32)
            k_arr=np.sqrt((2*PI*self._B[:,0]/Lx)**2+(2*PI*self._B[:,1]/Ly)**2)
            print(f"  RFF: {n_ff} modes, kin[{k_arr.min():.2f},{k_arr.max():.2f}], "
                  f"mean k={k_arr.mean():.2f} (k*={sigma*2*PI:.3f})"
                  + (f", {len(force_modes)} forced (triad)" if force_modes else ""))

        def build(self,_):
            self.B=self.add_weight(name="B",shape=self._B.shape,
                initializer=tf.constant_initializer(self._B),trainable=False)
            self.inv_L=self.add_weight(name="inv_L",shape=(1,2),
                initializer=tf.constant_initializer([[1.0/Lx,1.0/Ly]]),trainable=False)
            super().build(_)

        def call(self,xy):
            # v2: scale x by 1/Lx and y by 1/Ly SEPARATELY (square domains
            # have Lx==Ly, collapsing to the v1 behavior exactly).
            xy_scaled = xy * self.inv_L
            proj=tf.matmul(xy_scaled, tf.transpose(self.B))
            ang=2.0*PI*proj
            return tf.concat([tf.cos(ang),tf.sin(ang)],axis=-1)

    def _make_net(name):
        inp=tf.keras.Input((2,),dtype=tf.float32)
        h=IntegerRFFEmbedding(N_FF,sigma_rff,seed=seed,force_modes=must_include)(inp)
        for _ in range(N_LAYERS):
            h=tf.keras.layers.Dense(N_HIDDEN,activation="tanh",
                                     kernel_initializer="glorot_normal")(h)
        out=tf.keras.layers.Dense(2,kernel_initializer="glorot_normal",
            bias_initializer=tf.keras.initializers.Constant([u_star,v_star]))(h)
        return tf.keras.Model(inp,out,name=name)

    model=_make_net("pinn_A"); model_prev=_make_net("pinn_A_prev")
    model_prev.set_weights(model.get_weights())
    opt=tf.keras.optimizers.Adam(LR_INIT)

    # -- METHOD B: NESTED-AUTOGRAD LAPLACIAN ---------------------------------
    # Cost: O(4*l*N*q^2) per call (2nd-order derivative x 2 spatial dims).
    # Unlike Method A's FFT (which needs a periodic grid to work), this
    # differentiates model(xy) directly and would work at ANY set of
    # points -- but per the ablation's experimental controls, xy is the
    # SAME fixed grid_xy_tf Method A uses, not random collocation points.
    # [Raissi, Perdikaris & Karniadakis 2019; Sobolev cubatures arXiv:2211.15443]
    @tf.function(reduce_retracing=True)
    def pretrain_step(xy,tgt_u,tgt_v):
        with tf.GradientTape() as tape:
            uv=model(xy,training=True)
            L_=(tf.reduce_mean((uv[:,0:1]-tgt_u)**2)+
                tf.reduce_mean((uv[:,1:2]-tgt_v)**2))
        grads=tape.gradient(L_,model.trainable_variables)
        grads=[tf.zeros_like(v) if g is None else
               tf.where(tf.math.is_nan(g),tf.zeros_like(g),g)
               for g,v in zip(grads,model.trainable_variables)]
        grads,_=tf.clip_by_global_norm(grads,GRAD_CLIP)
        opt.apply_gradients(zip(grads,model.trainable_variables))
        return L_

    sigma_max_tf = tf.constant(float(sigma_max), dtype=tf.float32)
    eps_var_tf   = tf.constant(float(eps_var), dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def train_step(dt_val, C_stab=tf.constant(0.0, dtype=tf.float32)):
        """
        IMEX residual (Ascher, Ruuth & Wetton 1995) -- IDENTICAL formula to
        Method A, only the Laplacian computation differs:
          R_u = u^{n+1} − u^n − dt·D1·Lap_AD(u^{n+1}) − dt·γ·f(u^n,v^n) = 0
          R_v = v^{n+1} − v^n − dt·D2·Lap_AD(v^{n+1}) − dt·γ·g(u^n,v^n) = 0
        Diffusion (D·Lap): IMPLICIT — current model, inside GradientTape
        Reaction (f,g):    EXPLICIT — model_prev, stop_gradient
        Warm-start:        model_prev.set_weights(model.get_weights())
                           at each window boundary [Chen et al. 2024]

        Lap_AD(u) = d^2u/dx^2 + d^2u/dy^2, computed via THREE nested tapes:
          outer_tape (training: d(total_loss)/d(model.trainable_variables))
            tape2    (2nd spatial derivative: watches grid_xy_tf)
              tape1  (1st spatial derivative: watches grid_xy_tf)
                u,v = model(grid_xy_tf)
        tape1/tape2 must be persistent since each is queried more than once
        (tape1 for both u and v; tape2 for both dx and dy of each).

        Also carries the same v7 fixed-point stabilizing regularizer as
        Method A [Babic, Rohrhofer & Geiger 2025], applied identically.
        """
        uv_old=tf.stop_gradient(model_prev(grid_xy_tf,training=False))
        u_old=uv_old[:,0]; v_old=uv_old[:,1]
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(grid_xy_tf)
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch(grid_xy_tf)
                    uv=model(grid_xy_tf,training=True)
                    u_col=uv[:,0:1]; v_col=uv[:,1:2]
                du=tape1.gradient(u_col,grid_xy_tf)   # [N,2]: du/dx, du/dy
                dv=tape1.gradient(v_col,grid_xy_tf)
                du_dx=du[:,0:1]; du_dy=du[:,1:2]
                dv_dx=dv[:,0:1]; dv_dy=dv[:,1:2]
            d2u_dx2=tape2.gradient(du_dx,grid_xy_tf)[:,0:1]
            d2u_dy2=tape2.gradient(du_dy,grid_xy_tf)[:,1:2]
            d2v_dx2=tape2.gradient(dv_dx,grid_xy_tf)[:,0:1]
            d2v_dy2=tape2.gradient(dv_dy,grid_xy_tf)[:,1:2]
            lap_u=tf.reshape(d2u_dx2+d2u_dy2,[-1])
            lap_v=tf.reshape(d2v_dx2+d2v_dy2,[-1])
            u=tf.reshape(u_col,[-1]); v=tf.reshape(v_col,[-1])
            u2v=u_old*u_old*v_old
            R_u=u-u_old-dt_val*D1*lap_u-dt_val*gamma*(a-u_old+u2v)
            R_v=v-v_old-dt_val*D2*lap_v-dt_val*gamma*(b-u2v)
            L_pde=tf.reduce_mean(R_u**2)+tf.reduce_mean(R_v**2)
            # Soft positivity penalty for v: prevents v->0 instability
            # When v approaches 0, u^2*v in reaction becomes ill-conditioned
            v_floor = tf.constant(0.01, dtype=tf.float32)
            L_pos = tf.reduce_mean(tf.square(tf.nn.relu(v_floor - v)))
            # fixed-point stabilizing regularizer [Babic et al. 2025]
            var_u = tf.math.reduce_variance(u)
            gate = tf.exp(-var_u/eps_var_tf)
            L_stab = C_stab * gate * sigma_max_tf
            total=W_PDE*L_pde + 50.0*L_pos + L_stab
        grads=outer_tape.gradient(total,model.trainable_variables)
        grads=[tf.zeros_like(v) if g is None else
               tf.where(tf.math.is_nan(g),tf.zeros_like(g),g)
               for g,v in zip(grads,model.trainable_variables)]
        grads,_=tf.clip_by_global_norm(grads,GRAD_CLIP)
        opt.apply_gradients(zip(grads,model.trainable_variables))
        return L_pde, L_stab

    return model,model_prev,opt,pretrain_step,train_step

# =============================================================================
#  EARLY STOPPING
# =============================================================================

class EarlyStopper:
    def __init__(self, label):
        self.tol=ES_TOL_PDE.get(label,1e-6)
        self.min_w=ES_MIN_WINDOWS.get(label,100)
        self.amp=[]; self.lpde=[]; self.var=[]

    def update(self,u_amp,lpde,var_u):
        self.amp.append(float(u_amp))
        self.lpde.append(float(lpde))
        self.var.append(float(var_u))

    def check(self,w):
        if w<self.min_w or len(self.amp)<ES_WINDOW+1: return False,""
        a_n=self.amp[-1]; a_p=self.amp[-1-ES_WINDOW]
        if a_n<1e-6: return False,""
        c1=abs(a_n-a_p)/a_n<ES_AMP_TOL
        lp=float(np.mean(self.lpde[-ES_WINDOW:])); c2=lp<self.tol
        vr=np.array(self.var[-ES_WINDOW:]); vm=float(np.mean(vr))
        if vm<1e-10: return False,""
        c3=float(np.std(vr))/vm<ES_VAR_REL_STD
        if c1 and c2 and c3:
            return True,(f"C1 Δamp={abs(a_n-a_p)/a_n*100:.3f}%  "
                         f"C2 L_pde={lp:.2e}  "
                         f"C3 Var_rsd={float(np.std(vr))/vm*100:.3f}%")
        return False,""

# =============================================================================
#  TRAINING
# =============================================================================

def train_regime(regime, n_steps_override=None, n_windows_override=None,
                 disable_es=False, seed=SEED, force_domain_shape=None):
    import tensorflow as tf

    label=regime["label"]; D1,D2=regime["D1"],regime["D2"]
    N_GRID_R=int(regime.get("N_grid",64))  # per-regime grid resolution
    N_MODE_TARGET=int(regime.get("n_mode_target",3))
    gamma=regime["gamma"]; a,b=regime["a"],regime["b"]
    T=float(regime["T"]); ic_amp=regime["ic_amp"]
    T_WINDOW=float(regime.get("dt_override",T_WINDOW_DEFAULT))

    diag=turing_diagnostics(D1,D2,gamma,a,b)
    assert diag["is_turing"], f"{label}: Turing conditions not satisfied"

    ks=diag["k_star"]; km=diag["k_min"]; kM=diag["k_max"]
    u_star=diag["u_star"]; v_star=diag["v_star"]
    # ?? FIX: per-regime commensurate domain ??????????????????????????????
    # L = n*ell* places mode (n,0) exactly at k* (0% spectral error).
    # FIX: use k_true = argmax sigma(k) for IC and commensurability.
    # k*_formula = argmin det A(k) can be 20% off when D2/D1 is small.
    k_true, sigma_max = get_true_k_dominant(D1, D2, gamma, a, b)
    if k_true is None:
        print(f"  {label}: not Turing unstable"); return None

    # v2 (ported from Method A v9): domain shape branches here.
    domain_shape = force_domain_shape or regime.get("domain_shape", "square")
    triad_modes_idx = []
    if domain_shape == "hex_rect":
        Lx, Ly, n_mode, triad_modes_idx = get_commensurate_rect_L(
            k_true, N_MODE_TARGET if N_MODE_TARGET != 3 else 6)
        triad_kxky = [(2.*PI*nx/Lx, 2.*PI*ny/Ly) for (nx,ny) in triad_modes_idx]
    else:
        L, n_mode = get_commensurate_L(k_true, target_L=1.0)
        if N_MODE_TARGET != 3:
            ell_t = 2.*PI/k_true
            L = float(N_MODE_TARGET) * ell_t
            n_mode = N_MODE_TARGET
        Lx = Ly = L
        triad_kxky = []
    sigma_rff = k_true * Lx / (2.*PI)  # = n_mode exactly (both branches)

    n_windows=n_windows_override if n_windows_override is not None else int(np.ceil(T/T_WINDOW))
    n_steps=n_steps_override or N_STEPS_PER_WINDOW

    print("\n"+"="*72)
    print(f"  {METHOD_LABEL}  |  {label}")
    print(f"  {regime.get('expected','?')} | a={a}, b={b}, gamma={gamma}, D1={D1}, D2={D2}")
    print(f"  u*={u_star:.4f}  v*={v_star:.4f}  k*_formula={ks:.3f}  k_true={k_true:.3f}  sigma_max={sigma_max:.4f}  band=[{km:.2f},{kM:.2f}]")
    print(f"  Turing: C1={diag['c1']} C2={diag['c2']} C3={diag['c3']} C4={diag['c4']}")
    if domain_shape == "hex_rect":
        print(f"  DOMAIN (hex_rect): Lx={Lx:.5f}={n_mode}×ℓ*  Ly={Ly:.5f}  "
              f"Ly/Lx={Ly/Lx:.4f} (target 1/√3={1./np.sqrt(3):.4f})")
        print(f"  v2: triad modes (exact, |k|=k_true for all three): {triad_modes_idx}")
    else:
        print(f"  DOMAIN: L={Lx:.5f} = {n_mode}×ℓ*  → k_{{n,0}}=k_true={k_true:.3f} (0% error)")
    # v8: informational only -- how many pattern periods this domain spans
    # vs. how many the paper's fixed 8-unit domain would span at the same
    # k_true. Does not affect k_true/sigma_max (both are L-independent, see
    # the dispersion-relation comparison in get_true_k_dominant's docstring)
    # -- it only affects how many competing Fourier orientations are
    # available for spontaneous pattern selection.
    paper_n_periods = PAPER_DOMAIN_L * k_true / (2.*PI)
    print(f"  v8: this domain spans {n_mode} wavelengths; Khudhair et al.'s "
          f"fixed L={PAPER_DOMAIN_L} domain would span {paper_n_periods:.1f} "
          f"at this k_true (informational, not auto-corrected)")
    print(f"  Grid: {N_GRID_R}x{N_GRID_R}={N_GRID_R**2} pts  dt={T_WINDOW}  N_windows={n_windows}")
    print("="*72)

    # Build grid with per-regime commensurate Lx,Ly (no Laplacian kernel
    # needed for Method B -- the network is differentiated directly at
    # these points). Ly==Lx for square regimes.
    xg=np.linspace(0,Lx,N_GRID_R,endpoint=False,dtype=np.float32)
    yg=np.linspace(0,Ly,N_GRID_R,endpoint=False,dtype=np.float32)
    XX,YY=np.meshgrid(xg,yg,indexing='ij')
    grid_xy_np=np.column_stack([XX.ravel(),YY.ravel()]).astype(np.float32)
    grid_xy_tf=tf.constant(grid_xy_np)

    # v7: eps_var scales the stabilizing regularizer's gate to this
    # regime's own IC seeding amplitude (ic_amp), since ic_amp varies
    # ~2x across the 9 regimes (0.005 to 0.01).
    eps_var = max(float(ic_amp)**2, 1e-12)
    model,model_prev,opt,pretrain_step,train_step=_build_model_and_steps(
        sigma_rff,D1,D2,gamma,a,b,u_star,v_star,Lx,grid_xy_tf,
        sigma_max=sigma_max, eps_var=eps_var, Ly=Ly, must_include=triad_modes_idx,
        seed=seed)

    print(f"  Parameters: {model.count_params():,}")
    check_periodicity(model,Lx,"init",Ly=Ly)

    # IC seeded at k_true (true fastest-growing mode, not k*_formula)
    # v8: IC_MODE selects eigenmode (default) vs broadband.
    # v2 (ported from Method A v9): if the CLI is left at its default
    # ("eigenmode") AND this regime is hex_rect, auto-upgrade to "triad".
    effective_ic_mode = IC_MODE
    if domain_shape == "hex_rect" and IC_MODE == "eigenmode":
        effective_ic_mode = "triad"
    print(f"  IC_MODE effective = {effective_ic_mode!r} "
          f"(domain_shape={domain_shape!r}, CLI --ic-mode={IC_MODE!r})  seed={seed}")
    ic_u,ic_v=build_ic(u_star,v_star,D1,D2,gamma,a,b,k_true,km,kM,ic_amp,
                        mode=effective_ic_mode, triad_kxky=triad_kxky, seed=seed)

    # Pretrain
    print(f"\n  Pretrain ({N_STEPS_PRETRAIN} steps)")
    opt.learning_rate.assign(float(LR_INIT))
    # v2 (ported from Method A v10): locally-seeded RNG for pretraining
    # collocation points, derived from the runtime seed (was the fixed
    # module-level _RNG=SEED=42 in v1, which never actually varied).
    _rng_local = np.random.default_rng(seed + 1)
    t0=time.time()
    for ps in range(1,N_STEPS_PRETRAIN+1):
        xy=np.column_stack([_rng_local.uniform(0,Lx,N_COL).astype(np.float32),
                             _rng_local.uniform(0,Ly,N_COL).astype(np.float32)])
        Lp=pretrain_step(tf.constant(xy),
                         tf.constant(synth_points(ic_u,xy)[:,None]),
                         tf.constant(synth_points(ic_v,xy)[:,None]))
        if ps%400==0:
            print(f"    pt {ps:4d} | L={float(Lp):.3e} | {time.time()-t0:.0f}s")

    model_prev.set_weights(model.get_weights())

    stopper=EarlyStopper(label)
    dt_tf=tf.constant(T_WINDOW,dtype=tf.float32)

    snaps_t=[0.]; snaps_u=[synth_numpy(ic_u,xg,yg)]; snaps_v=[synth_numpy(ic_v,xg,yg)]
    history=[]; t_global=0.; es_triggered=False; collapsed=False
    window_times=[]    # wall-clock time per window (seconds)
    lpde_all=[]        # L_pde at every log step (for floor calculation)
    lstab_all=[]        # v7: stabilizing-regularizer term, for diagnostics

    # v7: don't judge collapse until COLLAPSE_EFOLDS growth-timescales
    # (1/sigma_max) have physically elapsed -- before that, small u_amp is
    # expected (still early linear growth), not evidence of collapse.
    collapse_check_window = max(COLLAPSE_MIN_WINDOWS,
                                 int(np.ceil((COLLAPSE_EFOLDS/sigma_max)/T_WINDOW)))
    print(f"  v7: collapse check active from window {collapse_check_window} "
          f"({COLLAPSE_EFOLDS} e-folds of sigma_max={sigma_max:.4f})  "
          f"stab_reg={'ON' if STAB_ENABLED else 'off'} (eps_var={eps_var:.2e})")

    for w in range(n_windows):
        t_w0=time.time()
        t_end=t_global+T_WINDOW; win_lpde=[]; win_lstab=[]

        # v7: anneal C_stab down over the first STAB_GAMMA_FRAC of this
        # regime's window budget [Babic et al. 2025, Eq. 6], computed in
        # Python so the schedule stays visible/editable outside tf.function.
        C_stab_val = 0.0
        if STAB_ENABLED:
            C_stab_val = max(STAB_C0*(STAB_GAMMA_FRAC - (w/n_windows)), 0.0)
        C_stab_tf = tf.constant(C_stab_val, dtype=tf.float32)

        for ep in range(1,n_steps+1):
            frac=ep/n_steps
            lr=LR_MIN+0.5*(LR_INIT-LR_MIN)*(1+np.cos(PI*frac))
            opt.learning_rate.assign(float(lr))
            lp,lstab=train_step(dt_tf, C_stab_tf); lp_val=float(lp); lstab_val=float(lstab)
            win_lpde.append(lp_val); win_lstab.append(lstab_val)
            if ep%LOG_EVERY==0:
                history.append((w+1,ep,lp_val))
                lpde_all.append(lp_val); lstab_all.append(lstab_val)

        t_w=time.time()-t_w0; window_times.append(t_w)

        uv_eval=model(grid_xy_tf,training=False).numpy()
        u_end=uv_eval[:,0].reshape(N_GRID_R,N_GRID_R)
        v_end=uv_eval[:,1].reshape(N_GRID_R,N_GRID_R)
        u_amp=float(u_end.max()-u_end.min())
        v_min=float(v_end.min()); lpde_mean=float(np.mean(win_lpde))

        # Adaptive dt: if v is very small, IMEX explicit reaction becomes stiff
        # [Reduced-PINN 2022: stiffness causes PINN failure in nonlinear dynamics]
        if v_min < 0.02 and T_WINDOW > 0.003:
            T_WINDOW = max(0.003, T_WINDOW * 0.5)
            dt_tf = tf.constant(T_WINDOW, dtype=tf.float32)
            print(f"  !! w{w+1}: v_min={v_min:.4f} critical -> dt reduced to {T_WINDOW:.4f}")

        if u_amp>100 or not np.isfinite(u_amp) or v_min<-1.:
            print(f"  !! w{w+1}: u_amp={u_amp:.2f} — unphysical, abort"); break

        # v7: COLLAPSE ABORT -- mirror image of the blow-up check above.
        # Fires only after collapse_check_window (>= COLLAPSE_EFOLDS growth
        # timescales have elapsed per linear theory) if amplitude has
        # fallen well below the seeded ic_amp -- i.e. the field collapsed
        # to the homogeneous fixed point instead of growing.
        # [Krishnapriyan et al. 2021; Rohrhofer et al. 2023]
        if w+1 >= collapse_check_window and u_amp < COLLAPSE_AMP_FRAC*ic_amp:
            print(f"  !! w{w+1}: u_amp={u_amp:.5f} < {COLLAPSE_AMP_FRAC}*ic_amp={ic_amp:.4f} "
                  f"after {(t_end*sigma_max):.1f} e-folds -- collapsed to trivial "
                  f"fixed point (u*,v*), abort  [Krishnapriyan et al. 2021 / "
                  f"Rohrhofer et al. 2023]")
            collapsed=True; break

        if (w+1)%10==0 or w==0 or w==n_windows-1:
            print(f"  w {w+1:5d}/{n_windows}  t={t_end:.3f}  "
                  f"L_pde={win_lpde[-1]:.2e}  a={u_amp:.3f}  "
                  f"v_min={v_min:.4f}  {t_w*1000:.0f}ms/win  {time.time()-t0:.0f}s")

        t_global=t_end
        snaps_t.append(t_global); snaps_u.append(u_end.copy()); snaps_v.append(v_end.copy())
        model_prev.set_weights(model.get_weights())

        if not disable_es:
            stopper.update(u_amp,lpde_mean,float(u_end.var()))
            stop,reason=stopper.check(w+1)
            if stop:
                print(f"\n  ✓ EARLY STOP w={w+1}, t={t_end:.2f}s  |  {reason}")
                es_triggered=True; break

    total_time=time.time()-t0
    print(f"\n  Total: {total_time:.0f}s  |  "
          f"Mean window: {np.mean(window_times)*1000:.1f}ms  |  "
          f"Windows used: {len(snaps_t)-1}/{n_windows}")
    check_periodicity(model,Lx,"final",Ly=Ly)

    # Compute final metrics
    pat,aniso,kp=classify_field(snaps_u[-1],Lx,k_true,Ly=Ly)
    spec_err=abs(kp-ks)/ks*100 if ks>0 else 0.
    lpde_floor=float(np.min(lpde_all)) if lpde_all else float("nan")

    metrics = dict(
        label=label, method=METHOD_SHORT,
        k_star_formula=ks, k_true=k_true, sigma_max=sigma_max,
        k_peak=kp,
        spec_err_vs_kformula=spec_err,
        spec_err_vs_ktrue=abs(kp-k_true)/k_true*100 if k_true else 0.,
        u_amp=float(snaps_u[-1].max()-snaps_u[-1].min()),
        v_min=float(snaps_v[-1].min()),
        lpde_floor=lpde_floor,
        lpde_final=history[-1][2] if history else float("nan"),
        pattern=pat, anisotropy=aniso,
        n_windows=len(snaps_t)-1, n_windows_budget=n_windows,
        wall_time_s=total_time,
        ms_per_window=float(np.mean(window_times)*1000),
        es_triggered=es_triggered,
        collapsed=collapsed,                       # v7
        stab_reg_used=STAB_ENABLED,                 # v7
        lstab_floor=float(np.min(lstab_all)) if lstab_all else float("nan"),  # v7
        ic_mode=IC_MODE, ic_mode_effective=effective_ic_mode,   # v8/v2
        seed=seed, force_domain_shape=force_domain_shape,       # v2
        domain=domain_shape, L=Lx, Lx=Lx, Ly=Ly, n_mode=n_mode,
        N_grid=N_GRID_R,
        N_params=model.count_params(),
    )
    return dict(model=model, history=history, diag=diag, regime=regime,
                metrics=metrics, xg=xg, yg=yg,
                snaps_t=np.array(snaps_t),
                snaps_u=np.array(snaps_u),
                snaps_v=np.array(snaps_v))

# =============================================================================
#  PLOTTING
# =============================================================================

def plot_regime(result, domain_tier=1, seed=SEED):
    """
    domain_tier: value of --domain used for this run (1, 2, or 3).
    Included in the output filename so runs at different tiers into the
    same --save-dir do not overwrite each other (v6 fix; v5 filenames
    depended only on the regime label). v2: seed also included.
    """
    lbl=result["regime"]["label"]; m=result["metrics"]
    ks=result["diag"]["k_star"]; km=result["diag"]["k_min"]; kM=result["diag"]["k_max"]
    snaps_t=result["snaps_t"]; U=result["snaps_u"]; V=result["snaps_v"]
    history=result["history"]; xg=result["xg"]; yg=result["yg"]

    # v2 (ported from Method A v9): Lx,Ly (Ly==Lx for square regimes)
    Lx     = float(m["Lx"]); Ly = float(m["Ly"])
    n_mode = int(m["n_mode"])
    shape_label = (f"HexRect Lx={Lx:.4f}={n_mode}\u00d7\u2113* Ly={Ly:.4f} "
                   f"(Ly/Lx={Ly/Lx:.4f})") if m["domain"]=="hex_rect" else \
                  f"Square L={Lx:.4f}={n_mode}\u00d7\u2113*"

    rk,rP=radial_spectrum(U[-1],Lx,Ly=Ly)
    n_snaps=min(5,len(snaps_t)); idxs=np.linspace(0,len(snaps_t)-1,n_snaps,dtype=int)

    fig=plt.figure(figsize=(22,14)); gs=gridspec.GridSpec(3,5,figure=fig,hspace=0.5,wspace=0.4)
    fig.suptitle(
        f"Schnakenberg v16 SPECTRAL — {METHOD_LABEL}  |  {lbl}\n"
        f"a={result['regime']['a']}, b={result['regime']['b']}, "
        f"gamma={result['regime']['gamma']}, D_2/D_1={int(result['regime']['D2']/result['regime']['D1'])}, "
        f"Domain: {shape_label}  Grid: {m['N_grid']}x{m['N_grid']}\n"
        f"Observed: {m['pattern']}  aniso={m['anisotropy']:.3f}  "
        f"k*={ks:.2f}  k_peak={m['k_peak']:.2f}  err_kt={m['spec_err_vs_ktrue']:.1f}%  err_kf={m['spec_err_vs_kformula']:.1f}%  "
        f"L_pde_floor={m['lpde_floor']:.2e}  "
        f"{m['n_windows']} windows  {m['wall_time_s']:.0f}s total  "
        f"{m['ms_per_window']:.0f}ms/win",
        fontsize=8,fontweight="bold")

    for ci,ti in enumerate(idxs):
        ax=fig.add_subplot(gs[0,ci])
        im=ax.imshow(U[ti].T,origin="lower",extent=[0,Lx,0,Ly],cmap="RdYlBu_r")
        ax.set_title(f"u  t={snaps_t[ti]:.2f}",fontsize=9)
        plt.colorbar(im,ax=ax,shrink=0.75,format="%.2f")
    for ci,ti in enumerate(idxs):
        ax=fig.add_subplot(gs[1,ci])
        im=ax.imshow(V[ti].T,origin="lower",extent=[0,Lx,0,Ly],cmap="RdYlBu_r")
        ax.set_title(f"v  t={snaps_t[ti]:.2f}",fontsize=9)
        plt.colorbar(im,ax=ax,shrink=0.75,format="%.3f")

    ax=fig.add_subplot(gs[2,0])
    ax.semilogy(snaps_t,U.var(axis=(1,2))+1e-20,"b-",lw=2,label="Var[u]")
    ax.semilogy(snaps_t,V.var(axis=(1,2))+1e-20,"r-",lw=2,label="Var[v]")
    ax.set_xlabel("t"); ax.set_title("Spatial variance",fontsize=9); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

    ax=fig.add_subplot(gs[2,1])
    ax.semilogy(rk,rP+1e-20,"b-",lw=2)
    ax.axvspan(km,kM,alpha=0.2,color="red",label="unstable band")
    ax.axvline(ks,color="r",ls="--",lw=2,label=f"k*={ks:.1f}")
    ax.axvline(m["k_peak"],color="navy",ls=":",lw=2,label=f"k_peak={m['k_peak']:.1f}")
    ax.set_xlabel("k"); ax.set_title("Radial spectrum",fontsize=9)
    ax.legend(fontsize=6); ax.grid(True,alpha=0.3); ax.set_xlim(0,min(120,rk.max()))

    ax=fig.add_subplot(gs[2,2])
    if history: ax.semilogy([h[2] for h in history],"b-",lw=1.2,label="L_pde")
    ax.set_xlabel("step"); ax.set_title("Loss history",fontsize=9); ax.grid(True,alpha=0.3)

    ax=fig.add_subplot(gs[2,3])
    F=np.fft.fftshift(np.fft.fft2(U[-1]-U[-1].mean()))
    ax.imshow(np.log10(np.abs(F)**2+1e-12),cmap="viridis"); ax.set_title("log|F[u]|^2",fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax=fig.add_subplot(gs[2,4]); ax.axis("off")
    rows=[
        ["method",METHOD_SHORT],["domain",shape_label],
        ["grid",f"{m['N_grid']}x{m['N_grid']}"],
        ["k* (theory)",f"{ks:.3f}"],["k_peak",f"{m['k_peak']:.3f}"],
        ["spec err (vs k*)",f"{m['spec_err_vs_kformula']:.1f}%"],
        ["spec err (vs k_true)",f"{m['spec_err_vs_ktrue']:.1f}%"],
        ["u_amp",f"{m['u_amp']:.3f}"],["v_min",f"{m['v_min']:.4f}"],
        ["v>0?","✓" if m['v_min']>=0 else "✗"],
        ["anisotropy",f"{m['anisotropy']:.3f}"],["pattern",m["pattern"]],
        ["expected",result["regime"].get("expected","-")],
        ["L_pde floor",f"{m['lpde_floor']:.2e}"],
        ["windows",f"{m['n_windows']}/{m['n_windows_budget']}"],
        ["wall time",f"{m['wall_time_s']:.0f}s"],
        ["ms/window",f"{m['ms_per_window']:.1f}"],
        ["ES","YES" if m['es_triggered'] else "no"],
    ]
    tbl=ax.table(cellText=rows,colLabels=["metric","value"],
                 loc="center",cellLoc="left",colWidths=[0.52,0.44])
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.5); tbl.scale(1.0,1.25)
    ax.set_title("Verification",fontsize=9)

    os.makedirs(SAVE_DIR,exist_ok=True)
    # v6: filename now includes domain tier (d{domain_tier}) to avoid
    # overwriting a different --domain run's output in the same SAVE_DIR.
    path=os.path.join(SAVE_DIR,f"{METHOD_SHORT}_d{domain_tier}_{lbl}_s{seed}.png")
    plt.savefig(path,dpi=130,bbox_inches="tight"); plt.close()
    return path

# =============================================================================
#  MAIN
# =============================================================================

def main():
    global SAVE_DIR, N_STEPS_PRETRAIN, IC_MODE
    _labels=[r["label"] for r in REGIMES]

    parser=argparse.ArgumentParser(
        description=f"Schnakenberg v16 {METHOD_LABEL} — all 9 regimes",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regime",default=None,choices=_labels,metavar="REGIME",
        help="Run one regime: "+"|".join(_labels))
    parser.add_argument("--quick",action="store_true",
        help="Smoke test: 20 windows, 100 steps, 400 pretrain steps")
    parser.add_argument("--no-early-stop",action="store_true")
    parser.add_argument("--save-dir",type=str,default=None)
    parser.add_argument("--list-regimes",action="store_true")
    parser.add_argument("--domain",type=int,default=1,choices=[1,2,3],
        help="Domain tier: 1=n5N64(same cost), 2=n5N96(2.4x cost), 3=n7N128(4.7x cost)")
    parser.add_argument("--ic-mode",type=str,default=IC_MODE,choices=["eigenmode","broadband","triad"],
        help="IC construction: eigenmode (default, anti-coarsening single-k), "
             "broadband (multi-k across the unstable band), or triad "
             "(only valid for hex_rect regimes) -- see IC_MODE docstring")
    parser.add_argument("--seed",type=int,default=SEED,
        help="v2 (ported from Method A v10): RNG seed for IC construction, RFF "
             "embedding, and pretraining collocation sampling. Run the same "
             "regime with several --seed values for multi-seed statistics.")
    parser.add_argument("--force-domain-shape",type=str,default=None,choices=["square","hex_rect"],
        help="v2 (ported from Method A v10): override a regime's own "
             "domain_shape for a controlled A/B comparison.")
    args=parser.parse_args()

    if args.save_dir: SAVE_DIR=args.save_dir
    IC_MODE = args.ic_mode
    # Domain tier: scales n_mode_target and N_grid for Regime II
    # Tier 1: n=5, N=64  -- zero extra cost vs current (N unchanged)
    # Tier 2: n=5, N=96  -- ~2.25x cost (more collocation points per period)
    # Tier 3: n=7, N=128 -- ~4x cost (Khudhair-equivalent 7 periods)
    # Evidence: increasing n_mode at fixed N is FREE (cost depends only on N)
    # Method B's cost scales as O(4*l*N_pts*q^2) where N_pts=N_grid^2 is the
    # collocation-point count -- i.e. LINEARLY in N_pts, not O(N_grid^2 log
    # N_grid) as Method A's FFT does. Recomputed multipliers above (2.25x,
    # 4x) reflect this; they are close to but not identical to Method A's
    # (2.4x, 4.7x) since both are dominated by the same N_grid^2 factor.
    # [Sobolev cubatures arXiv:2211.15443]
    TIER_PARAMS = {
        1: dict(n_I=5, N_I=64,  n_II=5, N_II=64),   # same cost as current
        2: dict(n_I=5, N_I=64,  n_II=5, N_II=96),   # 2.4x cost for Regime II
        3: dict(n_I=5, N_I=64,  n_II=7, N_II=128),  # 4.7x cost (Khudhair-equiv)
    }
    tp = TIER_PARAMS[args.domain]
    for r in REGIMES:
        is_II = r["label"].startswith("K_")
        r["n_mode_target"] = tp["n_II"] if is_II else tp["n_I"]
        r["N_grid"] = tp["N_II"] if is_II else tp["N_I"]
    if not os.path.exists("/content/drive") and SAVE_DIR.startswith("/content/drive"):
        SAVE_DIR=f"./results_{METHOD_SHORT}"
    os.makedirs(SAVE_DIR,exist_ok=True)

    if args.list_regimes:
        print(f"\n  {METHOD_LABEL} — regime table\n")
        print(f"  {'Label':<12} {'k_true':>7} {'Expected'}")
        print("  "+"-"*55)
        for r in REGIMES:
            d=turing_diagnostics(r["D1"],r["D2"],r["gamma"],r["a"],r["b"])
            print(f"  {r['label']:<12} {d['k_star']:>7.3f}  {r.get('expected','-')}")
        return

    print("="*72)
    print(f"  Schnakenberg v16 — {METHOD_LABEL}")
    print(f"  SQUARE domain, L per regime  Grid: {N_GRID}x{N_GRID} baseline "
          f"(K_* regimes use a larger grid at --domain 2/3, see --list-regimes)")
    print(f"  All 9 regimes — ablation study counterpart: schnakenberg_v16_spectral_v10.py")
    print(f"  Output: {SAVE_DIR}  |  domain tier: {args.domain}  |  seed: {args.seed}"
          + (f"  |  force_domain_shape: {args.force_domain_shape}" if args.force_domain_shape else ""))
    print("="*72)

    if args.quick: N_STEPS_PRETRAIN=400
    chosen=[r for r in REGIMES if r["label"]==args.regime] if args.regime else REGIMES

    summary=[]; all_metrics=[]
    for regime in chosen:
        result=train_regime(regime,
                            n_steps_override=100 if args.quick else None,
                            n_windows_override=20 if args.quick else None,
                            disable_es=args.no_early_stop,
                            seed=args.seed, force_domain_shape=args.force_domain_shape)
        if result is None: continue
        path=plot_regime(result, domain_tier=args.domain, seed=args.seed)
        print(f"  -> {path}")
        m=result["metrics"]; all_metrics.append(m)
        # v6: filenames include domain tier (d{args.domain}); v2 (ported
        # from Method A v10) adds the seed (s{args.seed}) so a seed sweep
        # doesn't overwrite itself.
        fname_stem = f"{METHOD_SHORT}_d{args.domain}_{regime['label']}_s{args.seed}"
        np.savez_compressed(
            os.path.join(SAVE_DIR,f"{fname_stem}_arrays.npz"),
            t=result["snaps_t"],U=result["snaps_u"],V=result["snaps_v"])
        with open(os.path.join(SAVE_DIR,f"{fname_stem}_metrics.json"),"w") as f:
            json.dump(m,f,indent=2)
        summary.append(m)

    if not summary: return
    print("\n"+"="*125)
    print(f"  ABLATION SUMMARY — {METHOD_LABEL}  (domain tier {args.domain})")
    print(f"  {'regime':<12} {'k_true':>7} {'k_peak':>8} {'vs k_t%':>8} {'vs k*%':>7} "
          f"{'lpde_floor':>12} {'u_amp':>8} {'windows':>8} "
          f"{'time(s)':>8} {'ms/win':>8}  ES  pattern")
    print("  "+"-"*123)
    for m in summary:
        print(f"  {m['label']:<12} {m['k_true']:>7.3f} {m['k_peak']:>8.3f} {m['spec_err_vs_ktrue']:>7.1f}% {m['spec_err_vs_kformula']:>6.1f}% "
              f"{m['spec_err_vs_ktrue']:>5.1f}% {m['lpde_floor']:>12.2e} "
              f"{m['u_amp']:>8.3f} {m['n_windows']:>8} "
              f"{m['wall_time_s']:>8.0f} {m['ms_per_window']:>8.1f}  "
              f"{'ES' if m['es_triggered'] else '--'}  {m['pattern']}")
    print("="*125)
    # v6: summary json also carries the domain tier in its filename.
    with open(os.path.join(SAVE_DIR,f"{METHOD_SHORT}_d{args.domain}_all_metrics_s{args.seed}.json"),"w") as f:
        json.dump(summary,f,indent=2)
    print(f"  Metrics saved to: {SAVE_DIR}/{METHOD_SHORT}_d{args.domain}_all_metrics_s{args.seed}.json")

if __name__=="__main__":
    main()
