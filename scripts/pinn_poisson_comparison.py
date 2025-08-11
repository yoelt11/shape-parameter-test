#!/usr/bin/env python3
"""
Poisson PINN loss comparison between Standard (full covariance) and Advanced Shape Transform RBF models.
Uses analytical Laplacian for Gaussian bases: for phi(x)=exp(-0.5 d^T A d),
Δphi = phi * (||A d||^2 - trace(A)). Then Δu = sum_k w_k Δphi_k.
"""

import os
import time
from datetime import datetime
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

jax.config.update('jax_enable_x64', True)

# Experiment toggles
# Recommended clean config (keeps performance, well-behaved kernels)
RUN_SWEEP: bool = False              # False = single clean pass; True = run A/B sweeps
EVAL_SMOOTH_STANDARD: bool = False   # Old-like: raw Standard
EVAL_SMOOTH_ADVANCED: bool = False   # Old-like: raw Advanced
PROJECTION_EVERY_N: int = 1          # Old-like: project every step
WIDEN_BOUNDS: bool = False           # Old-like: narrower bounds


def make_domain(n: int = 32, low: float = -1.0, high: float = 1.0) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.linspace(low, high, n)
    y = jnp.linspace(low, high, n)
    X, Y = jnp.meshgrid(x, y)
    pts = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    return X, Y, pts
# ------------------------ Projection helpers ------------------------ #

def apply_projection_standard(lambdas_0: jnp.ndarray, X_grid: jnp.ndarray, Y_grid: jnp.ndarray) -> jnp.ndarray:
    """Project Standard model params [mu_x, mu_y, log_sigma_x, log_sigma_y, angle, weight]
    to dynamic domain bounds and reasonable sigmas as provided by the user spec."""
    x_min, x_max = X_grid.min(), X_grid.max()
    y_min, y_max = Y_grid.min(), Y_grid.max()
    n_points = X_grid.size
    lambdas_0 = lambdas_0.at[:, 0:2].set(jnp.clip(
        lambdas_0[:, 0:2], jnp.array([x_min, y_min]), jnp.array([x_max, y_max])
    ))
    domain_width_x = x_max - x_min
    domain_width_y = y_max - y_min
    domain_width = jnp.maximum(domain_width_x, domain_width_y)
    avg_point_spacing = domain_width / jnp.sqrt(n_points)
    if WIDEN_BOUNDS:
        min_sigma = avg_point_spacing / 4.0
        max_sigma = domain_width
    else:
        min_sigma = avg_point_spacing / 2.0
        max_sigma = domain_width / 2.0
    lambdas_0 = lambdas_0.at[:, 2:4].set(jnp.clip(
        lambdas_0[:, 2:4], jnp.log(min_sigma), jnp.log(max_sigma)
    ))
    return lambdas_0


def apply_projection_advanced(params: jnp.ndarray, X_grid: jnp.ndarray, Y_grid: jnp.ndarray) -> jnp.ndarray:
    """Project Advanced params [mu_x, mu_y, epsilon, scale, weight] to domain and scale bounds.
    We clamp centers to [x_min,x_max]x[y_min,y_max], and scale to [min_scale,max_scale]."""
    x_min, x_max = X_grid.min(), X_grid.max()
    y_min, y_max = Y_grid.min(), Y_grid.max()
    n_points = X_grid.size
    out = params
    out = out.at[:, 0:2].set(jnp.clip(out[:, 0:2], jnp.array([x_min, y_min]), jnp.array([x_max, y_max])))
    domain_width_x = x_max - x_min
    domain_width_y = y_max - y_min
    domain_width = jnp.maximum(domain_width_x, domain_width_y)
    avg_point_spacing = domain_width / jnp.sqrt(n_points)
    if WIDEN_BOUNDS:
        min_scale = (avg_point_spacing / 4.0)
        max_scale = (domain_width)
    else:
        min_scale = (avg_point_spacing / 2.0)
        max_scale = (domain_width / 2.0)
    out = out.at[:, 3].set(jnp.clip(out[:, 3], min_scale, max_scale))
    return out



def poisson_gt_u_and_f(X: jnp.ndarray, Y: jnp.ndarray, kx: int = 1, ky: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Ground truth u and forcing f for Poisson Δu = f on [-1,1]^2.
    u = sin(kx π x) sin(ky π y) ⇒ Δu = - (kx^2 + ky^2) π^2 u.
    """
    u = jnp.sin(kx * jnp.pi * X) * jnp.sin(ky * jnp.pi * Y)
    f = - (kx**2 + ky**2) * (jnp.pi ** 2) * u
    return u, f


# ------------------------ Models ------------------------ #

def standard_init(n_kernels: int, key: jax.Array) -> jnp.ndarray:
    params = jnp.zeros((n_kernels, 6))
    key, k1 = jax.random.split(key)
    params = params.at[:, 0:2].set(jax.random.uniform(k1, (n_kernels, 2), minval=-0.8, maxval=0.8))
    params = params.at[:, 2].set(jnp.log(0.3) * jnp.ones(n_kernels))  # log_sigma_x
    params = params.at[:, 3].set(jnp.log(0.3) * jnp.ones(n_kernels))  # log_sigma_y
    params = params.at[:, 4].set(jnp.zeros(n_kernels))  # angle
    key, k2 = jax.random.split(key)
    params = params.at[:, 5].set(jax.random.normal(k2, (n_kernels,)) * 0.1)  # weights
    return params


def standard_eval_and_laplacian(X_pts: jnp.ndarray, params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Domain-derived helpers (used by smooth path)
    x_min, x_max = jnp.min(X_pts[:, 0]), jnp.max(X_pts[:, 0])
    y_min, y_max = jnp.min(X_pts[:, 1]), jnp.max(X_pts[:, 1])
    n_points = X_pts.shape[0]
    domain_width = jnp.maximum(x_max - x_min, y_max - y_min)
    avg_point_spacing = domain_width / jnp.sqrt(n_points)
    sigma_min = avg_point_spacing / 2.0
    sigma_max = domain_width / 2.0

    mu_raw = params[:, 0:2]
    sigx_raw = params[:, 2]
    sigy_raw = params[:, 3]
    theta = params[:, 4]
    weights = params[:, 5]

    if EVAL_SMOOTH_STANDARD:
        # Smooth: keep centers/sigmas bounded to domain-aware range
        mus = jnp.array([x_min, y_min]) + jax.nn.sigmoid(mu_raw) * jnp.array([x_max - x_min, y_max - y_min])
        sigma_x = sigma_min + jax.nn.sigmoid(sigx_raw) * (sigma_max - sigma_min)
        sigma_y = sigma_min + jax.nn.sigmoid(sigy_raw) * (sigma_max - sigma_min)
    else:
        # Raw: previous working behavior
        mus = mu_raw
        sigma_x = jnp.exp(sigx_raw)
        sigma_y = jnp.exp(sigy_raw)

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    R = jnp.stack([jnp.stack([cos_t, -sin_t], axis=1), jnp.stack([sin_t, cos_t], axis=1)], axis=1)  # (k,2,2)
    S = jnp.stack([jnp.stack([sigma_x ** 2, jnp.zeros_like(sigma_x)], axis=1),
                   jnp.stack([jnp.zeros_like(sigma_y), sigma_y ** 2], axis=1)], axis=1)
    C = jnp.matmul(jnp.matmul(R, S), jnp.swapaxes(R, 1, 2))  # cov
    A = jnp.linalg.inv(C)  # inv_cov

    d = X_pts[:, None, :] - mus[None, :, :]  # (n,k,2)
    Ad = jnp.einsum('kij,nkj->nki', A, d)  # (n,k,2)
    quad = jnp.einsum('nki,nki->nk', d, Ad)
    phi = jnp.exp(-0.5 * quad)  # (n,k)

    u_pred = jnp.dot(phi, weights)
    # Laplacian per kernel: Δphi = phi * (||A d||^2 - trace(A))
    norm_Ad_sq = jnp.sum(Ad ** 2, axis=2)
    trace_A = A[:, 0, 0] + A[:, 1, 1]  # (k,)
    lap_phi = phi * (norm_Ad_sq - trace_A[None, :])  # (n,k)
    lap_u = jnp.dot(lap_phi, weights)
    return u_pred, lap_u


def advanced_init(n_kernels: int, key: jax.Array) -> jnp.ndarray:
    # Param layout: [mu_x, mu_y, epsilon, scale, weight]; inv_cov constructed from epsilon/scale
    params = jnp.zeros((n_kernels, 5))
    key, k1 = jax.random.split(key)
    params = params.at[:, 0:2].set(jax.random.uniform(k1, (n_kernels, 2), minval=-0.8, maxval=0.8))
    eps = jnp.linspace(0, 2 * jnp.pi, n_kernels, endpoint=False)
    params = params.at[:, 2].set(eps)
    params = params.at[:, 3].set(0.1 * jnp.ones(n_kernels))
    key, k2 = jax.random.split(key)
    params = params.at[:, 4].set(jax.random.normal(k2, (n_kernels,)) * 0.1)
    return params


def advanced_eval_and_laplacian(X_pts: jnp.ndarray, params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Domain-derived helpers (used by smooth path)
    x_min, x_max = jnp.min(X_pts[:, 0]), jnp.max(X_pts[:, 0])
    y_min, y_max = jnp.min(X_pts[:, 1]), jnp.max(X_pts[:, 1])
    n_points = X_pts.shape[0]
    domain_width = jnp.maximum(x_max - x_min, y_max - y_min)
    avg_point_spacing = domain_width / jnp.sqrt(n_points)
    sigma_min = avg_point_spacing / 2.0
    sigma_max = domain_width / 2.0
    lam_min = 1.0 / (sigma_max ** 2)
    lam_max = 1.0 / (sigma_min ** 2)
    lam_mid = jnp.sqrt(lam_min * lam_max)

    mu_raw = params[:, 0:2]
    eps = params[:, 2]
    scale_raw = params[:, 3]
    weights = params[:, 4]

    if EVAL_SMOOTH_ADVANCED:
        # Smooth centers and gentle eigenvalue squash
        mus = jnp.array([x_min, y_min]) + jax.nn.sigmoid(mu_raw) * jnp.array([x_max - x_min, y_max - y_min])
        r = (1.0 / (domain_width ** 2 + 1e-12)) * (1.0 + jax.nn.softplus(scale_raw))
        inv11 = jnp.abs(r * (1.0 + jnp.sin(eps))) + 1e-6
        inv22 = jnp.abs(r * (1.0 + jnp.cos(eps))) + 1e-6
        inv12 = (0.05 * (1.0 + jax.nn.softplus(scale_raw))) * jnp.sin(2 * eps)

        A = jnp.zeros((params.shape[0], 2, 2))
        A = A.at[:, 0, 0].set(inv11)
        A = A.at[:, 1, 1].set(inv22)
        A = A.at[:, 0, 1].set(inv12)
        A = A.at[:, 1, 0].set(inv12)

        w, V = jnp.linalg.eigh(A)
        beta = 0.5
        w_smooth = lam_min + (lam_max - lam_min) * jax.nn.sigmoid(beta * (w - lam_mid))
        A = jnp.einsum('kij,kj,kjl->kil', V, w_smooth, jnp.swapaxes(V, 1, 2))
    else:
        # Raw: previous working behavior with PD correction on off-diagonal
        mus = mu_raw
        r = 100.0 * scale_raw
        inv11 = jnp.clip(jnp.abs(r * (1.0 + jnp.sin(eps))) + 1e-6, 1e-6, 1e6)
        inv22 = jnp.clip(jnp.abs(r * (1.0 + jnp.cos(eps))) + 1e-6, 1e-6, 1e6)
        inv12_raw = 10.0 * scale_raw * jnp.sin(2 * eps)
        max_b = jnp.sqrt(jnp.maximum(inv11 * inv22 - 1e-12, 0.0))
        inv12 = jnp.clip(inv12_raw, -max_b, max_b)
        A = jnp.zeros((params.shape[0], 2, 2))
        A = A.at[:, 0, 0].set(inv11)
        A = A.at[:, 1, 1].set(inv22)
        A = A.at[:, 0, 1].set(inv12)
        A = A.at[:, 1, 0].set(inv12)

    d = X_pts[:, None, :] - mus[None, :, :]
    Ad = jnp.einsum('kij,nkj->nki', A, d)
    quad = jnp.einsum('nki,nki->nk', d, Ad)
    phi = jnp.exp(-0.5 * quad)

    u_pred = jnp.dot(phi, weights)
    norm_Ad_sq = jnp.sum(Ad ** 2, axis=2)
    trace_A = A[:, 0, 0] + A[:, 1, 1]
    lap_phi = phi * (norm_Ad_sq - trace_A[None, :])
    lap_u = jnp.dot(lap_phi, weights)
    return u_pred, lap_u


def train_pinn(model_name: str,
               init_fn,
               eval_fn,
               X_pts: jnp.ndarray,
               f_rhs: jnp.ndarray,
               n_kernels: int = 64,
               epochs: int = 500,
               seed: int = 42,
               lr: float = 1e-2,
               use_projection: bool = False,
               X_grid: jnp.ndarray | None = None,
               Y_grid: jnp.ndarray | None = None) -> Dict:
    key = jax.random.PRNGKey(seed)
    params = init_fn(n_kernels, key)

    def loss_fn(p):
        _, lap_u = eval_fn(X_pts, p)
        res = lap_u - f_rhs
        return jnp.mean(res ** 2)

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss_fn)(p)
        up, s = opt.update(g, s)
        p = optax.apply_updates(p, up)
        return p, s, l

    losses = []
    t0 = time.time()
    grad_t0 = time.time(); _ = jax.grad(loss_fn)(params); grad_t = time.time() - grad_t0
    eval_t0 = time.time(); _ = loss_fn(params); eval_t = time.time() - eval_t0

    best = 1e9
    pat = 25
    patience = 0
    for e in range(epochs):
        params, opt_state, l = step(params, opt_state)
        # Post-update projection only every N steps (outside jit)
        if use_projection and PROJECTION_EVERY_N and (e % PROJECTION_EVERY_N == 0):
            if model_name == 'Standard (Full)':
                params = apply_projection_standard(params, X_grid, Y_grid)
            elif model_name == 'Advanced Shape Transform':
                params = apply_projection_advanced(params, X_grid, Y_grid)
        lv = float(l)
        losses.append(lv)
        if lv < best - 1e-8:
            best = lv
            patience = 0
        else:
            patience += 1
        if patience >= pat:
            break

    train_time = time.time() - t0
    iters = len(losses)
    it_per_s = iters / train_time if train_time > 0 else float('inf')
    return {
        'model': model_name,
        'final_loss': float(losses[-1]),
        'loss_history': losses,
        'epochs_run': iters,
        'train_time_s': train_time,
        'iters_per_s': it_per_s,
        'grad_time_s': grad_t,
        'eval_time_s': eval_t,
        'params': params,
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'pinn_poisson', timestamp)
    os.makedirs(outdir, exist_ok=True)

    X, Y, P = make_domain(n=32)
    n_k = 64
    epochs = 500
    wave_numbers = [(1,1), (2,1), (2,2)]

    def draw_ellipses_standard(ax, params, color='k', alpha=0.25):
        mus = np.array(params[:, 0:2])
        sigma_x = np.array(jnp.exp(params[:, 2]))
        sigma_y = np.array(jnp.exp(params[:, 3]))
        theta = np.array(params[:, 4])
        for (mx, my), sx, sy, th in zip(mus, sigma_x, sigma_y, theta):
            e = Ellipse(xy=(mx, my), width=2*sx, height=2*sy, angle=np.degrees(th),
                        edgecolor=color, facecolor='none', lw=0.8, alpha=alpha)
            ax.add_patch(e)

    def draw_ellipses_advanced(ax, params, color='k', alpha=0.25):
        mus = np.array(params[:, 0:2])
        eps = params[:, 2]
        scales = params[:, 3]
        # Build A (inv cov)
        r = 100.0 * scales
        inv11 = np.clip(np.abs(r * (1.0 + np.sin(eps))) + 1e-6, 1e-6, 1e6)
        inv22 = np.clip(np.abs(r * (1.0 + np.cos(eps))) + 1e-6, 1e-6, 1e6)
        inv12_raw = 10.0 * scales * np.sin(2 * eps)
        max_b = np.sqrt(np.maximum(inv11 * inv22 - 1e-12, 0.0))
        inv12 = np.clip(inv12_raw, -max_b, max_b)
        for i, (mx, my) in enumerate(mus):
            A = np.array([[inv11[i], inv12[i]], [inv12[i], inv22[i]]], dtype=float)
            # Covariance = inv(A)
            w, V = np.linalg.eigh(A)
            w = np.clip(w, 1e-12, None)
            cov_eigs = 1.0 / w
            axes = 2 * np.sqrt(cov_eigs)  # full width/height for 1-sigma ellipse
            angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))  # angle of largest axis
            e = Ellipse(xy=(mx, my), width=axes[1], height=axes[0], angle=angle,
                        edgecolor=color, facecolor='none', lw=0.8, alpha=alpha)
            ax.add_patch(e)

    # Loop over wave numbers and produce contourf comparison + losses
    for (kx, ky) in wave_numbers:
        if RUN_SWEEP:
            # A/B sweeps
            for eval_mode_name, std_smooth, adv_smooth in [
                ('std_smooth', True, False),
                ('adv_smooth', False, True),
            ]:
                for projN in [0, 5, 10, 20]:
                    for widen in [False, True]:
                        global EVAL_SMOOTH_STANDARD, EVAL_SMOOTH_ADVANCED, PROJECTION_EVERY_N, WIDEN_BOUNDS
                        EVAL_SMOOTH_STANDARD = std_smooth
                        EVAL_SMOOTH_ADVANCED = adv_smooth
                        PROJECTION_EVERY_N = projN
                        WIDEN_BOUNDS = widen

                        case_dir = os.path.join(outdir, f'{eval_mode_name}_proj{projN}_wide{int(widen)}', f'kx{kx}_ky{ky}')
                        os.makedirs(case_dir, exist_ok=True)
                        U_true, f = poisson_gt_u_and_f(X, Y, kx=kx, ky=ky)
                        f_rhs = f.flatten()

                        std_res = train_pinn('Standard (Full)', standard_init, standard_eval_and_laplacian, P, f_rhs,
                                             n_kernels=n_k, epochs=epochs, seed=42, lr=1e-2,
                                             use_projection=True, X_grid=X, Y_grid=Y)
                        adv_res = train_pinn('Advanced Shape Transform', advanced_init, advanced_eval_and_laplacian, P, f_rhs,
                                             n_kernels=n_k, epochs=epochs, seed=42, lr=1e-2,
                                             use_projection=True, X_grid=X, Y_grid=Y)

                        # Loss curves per case
                        plt.figure(figsize=(7,4))
                        plt.plot(std_res['loss_history'], label='Standard (Full)')
                        plt.plot(adv_res['loss_history'], label='Advanced Shape Transform')
                        plt.yscale('log')
                        plt.xlabel('Epoch'); plt.ylabel('Poisson residual MSE')
                        plt.title(f'PINN Poisson Loss (kx={kx}, ky={ky})')
                        plt.grid(True, alpha=0.3)
                        plt.legend()
                        plt.savefig(os.path.join(case_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
                        plt.close()

                        # Predictions and errors with contourf
                        def compute_prediction(eval_fn, params):
                            u_pred, _ = eval_fn(P, params)
                            return u_pred.reshape(X.shape)

                        U_std = compute_prediction(standard_eval_and_laplacian, std_res['params'])
                        U_adv = compute_prediction(advanced_eval_and_laplacian, adv_res['params'])
                        E_std = jnp.abs(U_std - U_true)
                        E_adv = jnp.abs(U_adv - U_true)

                        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
                        # Row 1: predictions
                        c0 = axes[0, 0].contourf(X, Y, U_std, levels=40, cmap='viridis')
                        axes[0, 0].set_title('Standard: Prediction')
                        fig.colorbar(c0, ax=axes[0, 0], fraction=0.046, pad=0.04)
                        c1 = axes[0, 1].contourf(X, Y, U_adv, levels=40, cmap='viridis')
                        axes[0, 1].set_title('Advanced: Prediction')
                        fig.colorbar(c1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                        # Row 2: absolute error
                        c2 = axes[1, 0].contourf(X, Y, E_std, levels=40, cmap='magma')
                        axes[1, 0].set_title('|Error| Standard')
                        fig.colorbar(c2, ax=axes[1, 0], fraction=0.046, pad=0.04)
                        c3 = axes[1, 1].contourf(X, Y, E_adv, levels=40, cmap='magma')
                        axes[1, 1].set_title('|Error| Advanced')
                        fig.colorbar(c3, ax=axes[1, 1], fraction=0.046, pad=0.04)
                        # Row 3: kernel ellipses
                        axes[2, 0].contourf(X, Y, U_true, levels=20, cmap='Greys', alpha=0.3)
                        draw_ellipses_standard(axes[2, 0], std_res['params'], color='cyan', alpha=0.6)
                        axes[2, 0].set_title('Standard: Kernel ellipses (1σ)')
                        axes[2, 0].set_aspect('equal', adjustable='box')
                        axes[2, 1].contourf(X, Y, U_true, levels=20, cmap='Greys', alpha=0.3)
                        draw_ellipses_advanced(axes[2, 1], adv_res['params'], color='lime', alpha=0.6)
                        axes[2, 1].set_title('Advanced: Kernel ellipses (1σ)')
                        axes[2, 1].set_aspect('equal', adjustable='box')
                        for ax in axes.ravel():
                            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(False)
                        fig.suptitle(f'Solution comparison (mode={eval_mode_name}, projN={projN}, wide={widen}) (kx={kx}, ky={ky})', y=0.95)
                        plt.tight_layout()
                        plt.savefig(os.path.join(case_dir, 'solution_comparison_contourf.png'), dpi=300, bbox_inches='tight')
                        plt.close()
        else:
            # Clean pass with recommended defaults
            case_dir = os.path.join(outdir, f'clean_default', f'kx{kx}_ky{ky}')
            os.makedirs(case_dir, exist_ok=True)
            U_true, f = poisson_gt_u_and_f(X, Y, kx=kx, ky=ky)
            f_rhs = f.flatten()

            std_res = train_pinn('Standard (Full)', standard_init, standard_eval_and_laplacian, P, f_rhs,
                                 n_kernels=n_k, epochs=epochs, seed=42, lr=1e-2,
                                 use_projection=True, X_grid=X, Y_grid=Y)
            adv_res = train_pinn('Advanced Shape Transform', advanced_init, advanced_eval_and_laplacian, P, f_rhs,
                                 n_kernels=n_k, epochs=epochs, seed=42, lr=1e-2,
                                 use_projection=True, X_grid=X, Y_grid=Y)

            # Loss curves per case
            plt.figure(figsize=(7,4))
            plt.plot(std_res['loss_history'], label='Standard (Full)')
            plt.plot(adv_res['loss_history'], label='Advanced Shape Transform')
            plt.yscale('log')
            plt.xlabel('Epoch'); plt.ylabel('Poisson residual MSE')
            plt.title(f'PINN Poisson Loss (kx={kx}, ky={ky})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(case_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Predictions and errors with contourf
            def compute_prediction(eval_fn, params):
                u_pred, _ = eval_fn(P, params)
                return u_pred.reshape(X.shape)

            U_std = compute_prediction(standard_eval_and_laplacian, std_res['params'])
            U_adv = compute_prediction(advanced_eval_and_laplacian, adv_res['params'])
            E_std = jnp.abs(U_std - U_true)
            E_adv = jnp.abs(U_adv - U_true)

            fig, axes = plt.subplots(3, 2, figsize=(10, 12))
            # Row 1: predictions
            c0 = axes[0, 0].contourf(X, Y, U_std, levels=40, cmap='viridis')
            axes[0, 0].set_title('Standard: Prediction')
            fig.colorbar(c0, ax=axes[0, 0], fraction=0.046, pad=0.04)
            c1 = axes[0, 1].contourf(X, Y, U_adv, levels=40, cmap='viridis')
            axes[0, 1].set_title('Advanced: Prediction')
            fig.colorbar(c1, ax=axes[0, 1], fraction=0.046, pad=0.04)
            # Row 2: absolute error
            c2 = axes[1, 0].contourf(X, Y, E_std, levels=40, cmap='magma')
            axes[1, 0].set_title('|Error| Standard')
            fig.colorbar(c2, ax=axes[1, 0], fraction=0.046, pad=0.04)
            c3 = axes[1, 1].contourf(X, Y, E_adv, levels=40, cmap='magma')
            axes[1, 1].set_title('|Error| Advanced')
            fig.colorbar(c3, ax=axes[1, 1], fraction=0.046, pad=0.04)
            # Row 3: kernel ellipses
            axes[2, 0].contourf(X, Y, U_true, levels=20, cmap='Greys', alpha=0.3)
            draw_ellipses_standard(axes[2, 0], std_res['params'], color='cyan', alpha=0.6)
            axes[2, 0].set_title('Standard: Kernel ellipses (1σ)')
            axes[2, 0].set_aspect('equal', adjustable='box')
            axes[2, 1].contourf(X, Y, U_true, levels=20, cmap='Greys', alpha=0.3)
            draw_ellipses_advanced(axes[2, 1], adv_res['params'], color='lime', alpha=0.6)
            axes[2, 1].set_title('Advanced: Kernel ellipses (1σ)')
            axes[2, 1].set_aspect('equal', adjustable='box')
            for ax in axes.ravel():
                ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(False)
            fig.suptitle(f'Solution comparison (clean defaults) (kx={kx}, ky={ky})', y=0.95)
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, 'solution_comparison_contourf.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Save brief report
    md = []
    md.append('# Poisson PINN Comparison\n')
    md.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    md.append('## Setup\n')
    md.append('- Domain: [-1,1]^2, grid 32x32\n')
    md.append('- PDE: Δu = f with u=sin(pi x) sin(pi y), f=-2 pi^2 u\n')
    md.append(f'- Kernels: {n_k}, Epochs: {epochs}, Optimizer: Adam(lr=1e-2)\n')
    md.append('## Results\n')
    for res in [std_res, adv_res]:
        md.append(f"### {res['model']}\n")
        md.append(f"- Final residual MSE: {res['final_loss']:.3e}\n")
        md.append(f"- Epochs run: {res['epochs_run']}\n")
        md.append(f"- Train time (s): {res['train_time_s']:.3f} (it/s: {res['iters_per_s']:.2f})\n")
        md.append(f"- First grad time (s): {res['grad_time_s']:.3f}, first eval time (s): {res['eval_time_s']:.3f}\n")

    with open(os.path.join(outdir, 'RESULTS.md'), 'w', encoding='utf-8') as fmd:
        fmd.write('\n'.join(md))

    # Save raw metrics
    import json
    # Note: per-case metrics are not aggregated here; see each case_dir for loss curves and visualizations

    print('Saved results to', outdir)


if __name__ == '__main__':
    main()


