#!/usr/bin/env python3
"""
1D Wave equation PINN comparison using 2D RBFs over (t, x).

PDE: u_tt = c^2 u_xx on (t,x) in [0,1]x[0,1], c=1
ICs: u(0,x)=sin(pi x), u_t(0,x)=0
BCs: u(t,0)=0, u(t,1)=0

Analytical: u(t,x)=sin(pi x) cos(pi t)

We compare Standard (full covariance) vs Advanced Shape Transform kernels
following the structure of previous Poisson tests.
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

# Old-like config similar to good-performing Poisson run
EVAL_SMOOTH_STANDARD: bool = False
EVAL_SMOOTH_ADVANCED: bool = False
PROJECTION_EVERY_N: int = 1
WIDEN_BOUNDS: bool = False


def make_domain(n_t: int = 64, n_x: int = 64) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t = jnp.linspace(0.0, 1.0, n_t)
    x = jnp.linspace(0.0, 1.0, n_x)
    T, X = jnp.meshgrid(t, x, indexing='ij')
    pts = jnp.stack([T.flatten(), X.flatten()], axis=1)  # (N, 2) with columns [t, x]
    return T, X, pts


def analytical_u(T: jnp.ndarray, X: jnp.ndarray, c: float = 1.0) -> jnp.ndarray:
    return jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * c * T)


# ------------------------ Projection helpers ------------------------ #
def apply_projection_standard(params: jnp.ndarray, T_grid: jnp.ndarray, X_grid: jnp.ndarray) -> jnp.ndarray:
    # params: [mu_t, mu_x, log_sigma_t, log_sigma_x, angle, weight]
    t_min, t_max = 0.0, 1.0
    x_min, x_max = 0.0, 1.0
    out = params
    out = out.at[:, 0].set(jnp.clip(out[:, 0], t_min, t_max))
    out = out.at[:, 1].set(jnp.clip(out[:, 1], x_min, x_max))
    # bounds based on grid spacing
    dt = (t_max - t_min) / max(T_grid.shape[0]-1, 1)
    dx = (x_max - x_min) / max(X_grid.shape[1]-1, 1)
    if WIDEN_BOUNDS:
        min_sig_t, max_sig_t = dt / 4.0, (t_max - t_min)
        min_sig_x, max_sig_x = dx / 4.0, (x_max - x_min)
    else:
        min_sig_t, max_sig_t = dt / 2.0, (t_max - t_min) / 2.0
        min_sig_x, max_sig_x = dx / 2.0, (x_max - x_min) / 2.0
    out = out.at[:, 2].set(jnp.clip(out[:, 2], jnp.log(min_sig_t), jnp.log(max_sig_t)))
    out = out.at[:, 3].set(jnp.clip(out[:, 3], jnp.log(min_sig_x), jnp.log(max_sig_x)))
    return out


def apply_projection_advanced(params: jnp.ndarray, T_grid: jnp.ndarray, X_grid: jnp.ndarray) -> jnp.ndarray:
    # params: [mu_t, mu_x, epsilon, scale, weight]
    t_min, t_max = 0.0, 1.0
    x_min, x_max = 0.0, 1.0
    out = params
    out = out.at[:, 0].set(jnp.clip(out[:, 0], t_min, t_max))
    out = out.at[:, 1].set(jnp.clip(out[:, 1], x_min, x_max))
    dt = (t_max - t_min) / max(T_grid.shape[0]-1, 1)
    dx = (x_max - x_min) / max(X_grid.shape[1]-1, 1)
    dom = jnp.maximum(t_max - t_min, x_max - x_min)
    if WIDEN_BOUNDS:
        min_scale, max_scale = (min(dt, dx) / 4.0), dom
    else:
        min_scale, max_scale = (min(dt, dx) / 2.0), dom / 2.0
    out = out.at[:, 3].set(jnp.clip(out[:, 3], min_scale, max_scale))
    return out


# ------------------------ Models ------------------------ #
def standard_init(n_kernels: int, key: jax.Array) -> jnp.ndarray:
    params = jnp.zeros((n_kernels, 6))
    key, k1 = jax.random.split(key)
    # centers in [0,1] to speed up
    params = params.at[:, 0:2].set(jax.random.uniform(k1, (n_kernels, 2), minval=0.0, maxval=1.0))
    # log sigmas
    params = params.at[:, 2].set(jnp.log(0.2) * jnp.ones(n_kernels))  # log_sigma_t
    params = params.at[:, 3].set(jnp.log(0.2) * jnp.ones(n_kernels))  # log_sigma_x
    params = params.at[:, 4].set(jnp.zeros(n_kernels))  # angle
    key, k2 = jax.random.split(key)
    params = params.at[:, 5].set(jax.random.normal(k2, (n_kernels,)) * 0.1)  # weights
    return params


def standard_eval_with_derivs(P: jnp.ndarray, params: jnp.ndarray):
    # P: (N,2) [t,x]
    mu = params[:, 0:2]
    log_sig_t = params[:, 2]
    log_sig_x = params[:, 3]
    theta = params[:, 4]
    weights = params[:, 5]

    if EVAL_SMOOTH_STANDARD:
        t_min, t_max = 0.0, 1.0
        x_min, x_max = 0.0, 1.0
        n_points = P.shape[0]
        # rough bounds
        sigma_min_t = 1.0 / (64.0)
        sigma_max_t = 0.5
        sigma_min_x = 1.0 / (64.0)
        sigma_max_x = 0.5
        mu = jnp.array([t_min, x_min]) + jax.nn.sigmoid(mu) * jnp.array([t_max - t_min, x_max - x_min])
        sig_t = sigma_min_t + jax.nn.sigmoid(log_sig_t) * (sigma_max_t - sigma_min_t)
        sig_x = sigma_min_x + jax.nn.sigmoid(log_sig_x) * (sigma_max_x - sigma_min_x)
    else:
        sig_t = jnp.exp(log_sig_t)
        sig_x = jnp.exp(log_sig_x)

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    R = jnp.stack([jnp.stack([cos_t, -sin_t], axis=1), jnp.stack([sin_t, cos_t], axis=1)], axis=1)  # (k,2,2)
    S = jnp.stack([jnp.stack([sig_t ** 2, jnp.zeros_like(sig_t)], axis=1),
                   jnp.stack([jnp.zeros_like(sig_x), sig_x ** 2], axis=1)], axis=1)
    C = jnp.matmul(jnp.matmul(R, S), jnp.swapaxes(R, 1, 2))  # cov
    A = jnp.linalg.inv(C)  # inv_cov

    d = P[:, None, :] - mu[None, :, :]  # (N,k,2)
    Ad = jnp.einsum('kij,nkj->nki', A, d)
    quad = jnp.einsum('nki,nki->nk', d, Ad)
    phi = jnp.exp(-0.5 * quad)

    # u = sum w phi
    u = jnp.dot(phi, weights)
    # grad phi = -phi * (A d)
    grad_phi = -phi[:, :, None] * Ad
    du_dt = jnp.dot(grad_phi[:, :, 0], weights)
    du_dx = jnp.dot(grad_phi[:, :, 1], weights)
    # Hessian diag: H = phi * ((Ad)(Ad)^T - A)
    # d2/dt2 = sum w * phi * ((Ad_t)^2 - A_tt)
    A_tt = A[:, 0, 0]
    A_xx = A[:, 1, 1]
    d2u_dt2 = jnp.dot(phi * (Ad[:, :, 0] ** 2 - A_tt[None, :]), weights)
    d2u_dx2 = jnp.dot(phi * (Ad[:, :, 1] ** 2 - A_xx[None, :]), weights)
    return u, du_dt, du_dx, d2u_dt2, d2u_dx2


def advanced_init(n_kernels: int, key: jax.Array) -> jnp.ndarray:
    params = jnp.zeros((n_kernels, 5))
    key, k1 = jax.random.split(key)
    params = params.at[:, 0:2].set(jax.random.uniform(k1, (n_kernels, 2), minval=0.0, maxval=1.0))
    eps = jnp.linspace(0, 2 * jnp.pi, n_kernels, endpoint=False)
    params = params.at[:, 2].set(eps)
    params = params.at[:, 3].set(0.1 * jnp.ones(n_kernels))
    key, k2 = jax.random.split(key)
    params = params.at[:, 4].set(jax.random.normal(k2, (n_kernels,)) * 0.1)
    return params


def advanced_eval_with_derivs(P: jnp.ndarray, params: jnp.ndarray):
    mu = params[:, 0:2]
    epsv = params[:, 2]
    scale_raw = params[:, 3]
    weights = params[:, 4]

    if EVAL_SMOOTH_ADVANCED:
        # smooth centers to [0,1]^2
        mu = jax.nn.sigmoid(mu)
        # gentle eigenvalue squash via scale
        r = (1.0) * (1.0 + jax.nn.softplus(scale_raw))
        inv11 = jnp.abs(r * (1.0 + jnp.sin(epsv))) + 1e-6
        inv22 = jnp.abs(r * (1.0 + jnp.cos(epsv))) + 1e-6
        inv12 = (0.05 * (1.0 + jax.nn.softplus(scale_raw))) * jnp.sin(2 * epsv)
        A = jnp.zeros((params.shape[0], 2, 2))
        A = A.at[:, 0, 0].set(inv11)
        A = A.at[:, 1, 1].set(inv22)
        A = A.at[:, 0, 1].set(inv12)
        A = A.at[:, 1, 0].set(inv12)
    else:
        r = 100.0 * scale_raw
        inv11 = jnp.clip(jnp.abs(r * (1.0 + jnp.sin(epsv))) + 1e-6, 1e-6, 1e6)
        inv22 = jnp.clip(jnp.abs(r * (1.0 + jnp.cos(epsv))) + 1e-6, 1e-6, 1e6)
        inv12_raw = 10.0 * scale_raw * jnp.sin(2 * epsv)
        max_b = jnp.sqrt(jnp.maximum(inv11 * inv22 - 1e-12, 0.0))
        inv12 = jnp.clip(inv12_raw, -max_b, max_b)
        A = jnp.zeros((params.shape[0], 2, 2))
        A = A.at[:, 0, 0].set(inv11)
        A = A.at[:, 1, 1].set(inv22)
        A = A.at[:, 0, 1].set(inv12)
        A = A.at[:, 1, 0].set(inv12)

    d = P[:, None, :] - mu[None, :, :]
    Ad = jnp.einsum('kij,nkj->nki', A, d)
    quad = jnp.einsum('nki,nki->nk', d, Ad)
    phi = jnp.exp(-0.5 * quad)

    u = jnp.dot(phi, weights)
    grad_phi = -phi[:, :, None] * Ad
    du_dt = jnp.dot(grad_phi[:, :, 0], weights)
    du_dx = jnp.dot(grad_phi[:, :, 1], weights)
    A_tt = A[:, 0, 0]
    A_xx = A[:, 1, 1]
    d2u_dt2 = jnp.dot(phi * (Ad[:, :, 0] ** 2 - A_tt[None, :]), weights)
    d2u_dx2 = jnp.dot(phi * (Ad[:, :, 1] ** 2 - A_xx[None, :]), weights)
    return u, du_dt, du_dx, d2u_dt2, d2u_dx2


def train_wave(model_name: str,
               init_fn,
               eval_fn,
               P: jnp.ndarray,
               T: jnp.ndarray,
               X: jnp.ndarray,
               c: float = 1.0,
               n_kernels: int = 64,
               epochs: int = 400,
               seed: int = 42,
               lr: float = 1e-2) -> Dict:
    key = jax.random.PRNGKey(seed)
    params = init_fn(n_kernels, key)

    # Collocation sets
    # Residual on full grid
    P_res = P
    # IC at t=0
    x_ic = jnp.linspace(0.0, 1.0, X.shape[1])
    P_ic = jnp.stack([jnp.zeros_like(x_ic), x_ic], axis=1)
    u0 = jnp.sin(jnp.pi * x_ic)
    ut0 = jnp.zeros_like(x_ic)
    # BC on x=0 and x=1 across t
    t_bc = jnp.linspace(0.0, 1.0, T.shape[0])
    P_bc0 = jnp.stack([t_bc, jnp.zeros_like(t_bc)], axis=1)
    P_bc1 = jnp.stack([t_bc, jnp.ones_like(t_bc)], axis=1)

    def loss_fn(p):
        u, du_dt, du_dx, d2u_dt2, d2u_dx2 = eval_fn(P_res, p)
        res = d2u_dt2 - (c ** 2) * d2u_dx2
        loss_res = jnp.mean(res ** 2)

        u_ic, du_dt_ic, *_ = eval_fn(P_ic, p)
        loss_ic_u = jnp.mean((u_ic - u0) ** 2)
        loss_ic_ut = jnp.mean((du_dt_ic - ut0) ** 2)

        u_bc0, *_ = eval_fn(P_bc0, p)
        u_bc1, *_ = eval_fn(P_bc1, p)
        loss_bc = jnp.mean(u_bc0 ** 2) + jnp.mean(u_bc1 ** 2)

        return loss_res + loss_ic_u + loss_ic_ut + loss_bc

    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss_fn)(p)
        up, s = opt.update(g, s)
        p = optax.apply_updates(p, up)
        return p, s, l

    losses = []
    best = 1e9
    pat = 30
    patience = 0
    for e in range(epochs):
        params, opt_state, l = step(params, opt_state)
        if PROJECTION_EVERY_N and (e % PROJECTION_EVERY_N == 0):
            if model_name == 'Standard (Full)':
                params = apply_projection_standard(params, T, X)
            else:
                params = apply_projection_advanced(params, T, X)
        lv = float(l)
        losses.append(lv)
        if lv < best - 1e-8:
            best = lv
            patience = 0
        else:
            patience += 1
        if patience >= pat:
            break

    return {
        'model': model_name,
        'final_loss': float(losses[-1]),
        'loss_history': losses,
        'epochs_run': len(losses),
        'params': params,
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'wave1d', timestamp)
    os.makedirs(outdir, exist_ok=True)

    T, X, P = make_domain(n_t=64, n_x=64)
    c_values = [0.5, 0.8, 1.0, 1.2, 1.5]
    n_k = 64
    epochs = 400
    lr = 1e-2

    for c in c_values:
        case_dir = os.path.join(outdir, f'c{c:.2f}')
        os.makedirs(case_dir, exist_ok=True)

        std_res = train_wave('Standard (Full)', standard_init, standard_eval_with_derivs, P, T, X,
                             c=c, n_kernels=n_k, epochs=epochs, seed=42, lr=lr)
        adv_res = train_wave('Advanced Shape Transform', advanced_init, advanced_eval_with_derivs, P, T, X,
                             c=c, n_kernels=n_k, epochs=epochs, seed=42, lr=lr)

        # Loss curves
        plt.figure(figsize=(7,4))
        plt.plot(std_res['loss_history'], label='Standard (Full)')
        plt.plot(adv_res['loss_history'], label='Advanced Shape Transform')
        plt.yscale('log')
        plt.xlabel('Epoch'); plt.ylabel('Wave residual + IC/BC MSE')
        plt.title(f'1D Wave PINN Loss (c={c:.2f})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(case_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Predictions heatmaps
        def eval_grid(eval_fn, params):
            u, *_ = eval_fn(P, params)
            return u.reshape(T.shape)

        U_true = analytical_u(T, X, c=c)
        U_std = eval_grid(standard_eval_with_derivs, std_res['params'])
        U_adv = eval_grid(advanced_eval_with_derivs, adv_res['params'])
        E_std = jnp.abs(U_std - U_true)
        E_adv = jnp.abs(U_adv - U_true)

        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        im0 = axes[0, 0].imshow(U_std, origin='lower', aspect='auto', extent=[0,1,0,1], cmap='viridis')
        axes[0, 0].set_title('Standard: u(t,x)')
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        im1 = axes[0, 1].imshow(U_adv, origin='lower', aspect='auto', extent=[0,1,0,1], cmap='viridis')
        axes[0, 1].set_title('Advanced: u(t,x)')
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        im2 = axes[1, 0].imshow(E_std, origin='lower', aspect='auto', extent=[0,1,0,1], cmap='magma')
        axes[1, 0].set_title('|Error| Standard')
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        im3 = axes[1, 1].imshow(E_adv, origin='lower', aspect='auto', extent=[0,1,0,1], cmap='magma')
        axes[1, 1].set_title('|Error| Advanced')
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # Simple kernel ellipses in (t,x) for visualization
        def draw_ellipses_standard(ax, params, color='cyan', alpha=0.5):
            mus = np.array(params[:, 0:2])
            sig_t = np.array(np.exp(np.array(params[:, 2])))
            sig_x = np.array(np.exp(np.array(params[:, 3])))
            theta = np.array(params[:, 4])
            for (mt, mx), st, sx, th in zip(mus, sig_t, sig_x, theta):
                e = Ellipse((mx, mt), width=2*sx, height=2*st, angle=np.degrees(th),
                            edgecolor=color, facecolor='none', lw=0.6, alpha=alpha)
                ax.add_patch(e)

        def draw_ellipses_advanced(ax, params, color='lime', alpha=0.5):
            mus = np.array(params[:, 0:2])
            epsv = np.array(params[:, 2])
            scales = np.array(params[:, 3])
            r = 100.0 * scales
            inv11 = np.clip(np.abs(r * (1.0 + np.sin(epsv))) + 1e-6, 1e-6, 1e6)
            inv22 = np.clip(np.abs(r * (1.0 + np.cos(epsv))) + 1e-6, 1e-6, 1e6)
            inv12_raw = 10.0 * scales * np.sin(2 * epsv)
            max_b = np.sqrt(np.maximum(inv11 * inv22 - 1e-12, 0.0))
            inv12 = np.clip(inv12_raw, -max_b, max_b)
            for i, (mt, mx) in enumerate(mus):
                A = np.array([[inv11[i], inv12[i]], [inv12[i], inv22[i]]], dtype=float)
                w, V = np.linalg.eigh(A)
                w = np.clip(w, 1e-12, None)
                cov_eigs = 1.0 / w
                axes_len = 2 * np.sqrt(cov_eigs)
                angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
                e = Ellipse((mx, mt), width=axes_len[1], height=axes_len[0], angle=angle,
                            edgecolor=color, facecolor='none', lw=0.6, alpha=alpha)
                ax.add_patch(e)

        axes[2, 0].imshow(U_true, origin='lower', aspect='auto', extent=[0,1,0,1], cmap='Greys', alpha=0.3)
        draw_ellipses_standard(axes[2, 0], std_res['params'])
        axes[2, 0].set_title('Standard: kernel ellipses on (t,x)')
        axes[2, 1].imshow(U_true, origin='lower', aspect='auto', extent=[0,1,0,1], cmap='Greys', alpha=0.3)
        draw_ellipses_advanced(axes[2, 1], adv_res['params'])
        axes[2, 1].set_title('Advanced: kernel ellipses on (t,x)')
        for ax in axes.ravel():
            ax.set_xlabel('x'); ax.set_ylabel('t')
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, 'solution_comparison_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Report
    md = []
    md.append('# 1D Wave PINN Comparison\n')
    md.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    md.append('- Domain: t,x in [0,1], grid 64x64\n')
    md.append('- PDE: u_tt = u_xx, ICs u(0,x)=sin(pi x), u_t(0,x)=0, BCs u(t,0)=u(t,1)=0\n')
    md.append(f'- Wave speeds tested: {c_values}\n')
    with open(os.path.join(outdir, 'RESULTS.md'), 'w', encoding='utf-8') as fmd:
        fmd.write('\n'.join(md))

    print('Saved results to', outdir)


if __name__ == '__main__':
    main()


