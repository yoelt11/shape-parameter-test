#!/usr/bin/env python3
"""
    Steady 2D incompressible Navier–Stokes (flow over cylinder) PINN comparison.

    Unknowns: u(x,y), v(x,y), p(x,y)
    Equations (steady):
      u*u_x + v*u_y + p_x - nu * (u_xx + u_yy) = 0
      u*v_x + v*v_y + p_y - nu * (v_xx + v_yy) = 0
      u_x + v_y = 0

    Domain: [0, Lx] x [0, Ly] \\ cylinder((xc, yc), R)
    Defaults: Lx=2.2, Ly=0.41, (xc,yc)=(0.2,0.2), R=0.05, U_in=1.0

    Boundary conditions:
      Inlet x=0: u(y)=4 U_in y (Ly - y) / Ly^2, v=0
      Top/Bottom y in {0,Ly}: u=0, v=0
      Cylinder r=R around (xc,yc): u=0, v=0
      Outlet x=Lx: du/dx=0, dv/dx=0, p=0 (fix gauge at outlet)

    Models: Standard (full covariance) vs Advanced Shape Transform basis; both share basis across u,v,p with separate weights.
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
from matplotlib.patches import Circle

jax.config.update('jax_enable_x64', True)

# Config tuned similar to previous strong settings
EVAL_SMOOTH_STANDARD: bool = False
EVAL_SMOOTH_ADVANCED: bool = False
PROJECTION_EVERY_N: int = 1
WIDEN_BOUNDS: bool = False

# Capacity and collocation controls
NX: int = 256  # Increased from 192 for better resolution
NY: int = 128  # Increased from 96 for better resolution
N_KERNELS: int = 256  # Increased from 128 for better flow resolution
N_CYL_SAMPLES: int = 512


def make_grid(nx: int = 96, ny: int = 48, Lx: float = 2.2, Ly: float = 0.41):
    x = jnp.linspace(0.0, Lx, nx)
    y = jnp.linspace(0.0, Ly, ny)
    X, Y = jnp.meshgrid(x, y)
    P = jnp.stack([X.flatten(), Y.flatten()], axis=1)
    return X, Y, P


def cylinder_levelset(x, y, xc=0.2, yc=0.2, R=0.05):
    return (x - xc) ** 2 + (y - yc) ** 2 - (R ** 2)


# ------------------------ Projection helpers ------------------------ #
def project_standard(params: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray,
                     Lx: float, Ly: float) -> jnp.ndarray:
    # params per kernel: [mu_x, mu_y, log_sig_x, log_sig_y, theta, w_u, w_v, w_p]
    out = params
    out = out.at[:, 0].set(jnp.clip(out[:, 0], 0.0, Lx))
    out = out.at[:, 1].set(jnp.clip(out[:, 1], 0.0, Ly))
    dx = Lx / max(X.shape[1] - 1, 1)
    dy = Ly / max(Y.shape[0] - 1, 1)
    if WIDEN_BOUNDS:
        min_sig_x, max_sig_x = dx / 4.0, Lx
        min_sig_y, max_sig_y = dy / 4.0, Ly
    else:
        min_sig_x, max_sig_x = dx / 2.0, Lx / 2.0
        min_sig_y, max_sig_y = dy / 2.0, Ly / 2.0
    out = out.at[:, 2].set(jnp.clip(out[:, 2], jnp.log(min_sig_x), jnp.log(max_sig_x)))
    out = out.at[:, 3].set(jnp.clip(out[:, 3], jnp.log(min_sig_y), jnp.log(max_sig_y)))
    return out


def project_advanced(params: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray,
                     Lx: float, Ly: float) -> jnp.ndarray:
    # params per kernel: [mu_x, mu_y, epsilon, scale, w_u, w_v, w_p]
    out = params
    out = out.at[:, 0].set(jnp.clip(out[:, 0], 0.0, Lx))
    out = out.at[:, 1].set(jnp.clip(out[:, 1], 0.0, Ly))
    dx = Lx / max(X.shape[1] - 1, 1)
    dy = Ly / max(Y.shape[0] - 1, 1)
    dom = jnp.maximum(Lx, Ly)
    if WIDEN_BOUNDS:
        min_scale, max_scale = (jnp.minimum(dx, dy) / 4.0), dom
    else:
        min_scale, max_scale = (jnp.minimum(dx, dy) / 2.0), dom / 2.0
    out = out.at[:, 3].set(jnp.clip(out[:, 3], min_scale, max_scale))
    return out


# ------------------------ Basis models ------------------------ #
def standard_init(n_kernels: int, key: jax.Array, Lx: float, Ly: float) -> jnp.ndarray:
    # [mu_x, mu_y, log_sig_x, log_sig_y, theta, w_u, w_v, w_p]
    params = jnp.zeros((n_kernels, 8))
    key, k1 = jax.random.split(key)
    params = params.at[:, 0].set(jax.random.uniform(k1, (n_kernels,), minval=0.0, maxval=Lx))
    key, k2 = jax.random.split(key)
    params = params.at[:, 1].set(jax.random.uniform(k2, (n_kernels,), minval=0.0, maxval=Ly))
    params = params.at[:, 2].set(jnp.log(0.1 * Lx) * jnp.ones(n_kernels))
    params = params.at[:, 3].set(jnp.log(0.05 * Ly) * jnp.ones(n_kernels))
    params = params.at[:, 4].set(jnp.zeros(n_kernels))
    key, k3 = jax.random.split(key)
    params = params.at[:, 5].set(jax.random.normal(k3, (n_kernels,)) * 0.1)
    key, k4 = jax.random.split(key)
    params = params.at[:, 6].set(jax.random.normal(k4, (n_kernels,)) * 0.1)
    key, k5 = jax.random.split(key)
    params = params.at[:, 7].set(jax.random.normal(k5, (n_kernels,)) * 0.1)
    return params


def standard_basis(P: jnp.ndarray, params: jnp.ndarray):
    # Returns phi, grad_phi (N,K,2), lap_phi (N,K)
    mu = params[:, 0:2]
    log_sig_x = params[:, 2]
    log_sig_y = params[:, 3]
    theta = params[:, 4]
    sig_x = jnp.exp(log_sig_x)
    sig_y = jnp.exp(log_sig_y)
    c, s = jnp.cos(theta), jnp.sin(theta)
    R = jnp.stack([jnp.stack([c, -s], axis=1), jnp.stack([s, c], axis=1)], axis=1)
    S = jnp.stack([jnp.stack([sig_x ** 2, jnp.zeros_like(sig_x)], axis=1),
                   jnp.stack([jnp.zeros_like(sig_y), sig_y ** 2], axis=1)], axis=1)
    C = jnp.matmul(jnp.matmul(R, S), jnp.swapaxes(R, 1, 2))
    A = jnp.linalg.inv(C)
    d = P[:, None, :] - mu[None, :, :]  # (N,K,2)
    Ad = jnp.einsum('kij,nkj->nki', A, d)
    quad = jnp.einsum('nki,nki->nk', d, Ad)
    phi = jnp.exp(-0.5 * quad)
    grad_phi = -phi[:, :, None] * Ad
    # Laplacian of Gaussian with general A: Δphi = phi * (||A d||^2 - trace(A))
    norm_Ad_sq = jnp.sum(Ad ** 2, axis=2)
    trace_A = A[:, 0, 0] + A[:, 1, 1]
    lap_phi = phi * (norm_Ad_sq - trace_A[None, :])
    return phi, grad_phi, lap_phi


def advanced_init(n_kernels: int, key: jax.Array, Lx: float, Ly: float) -> jnp.ndarray:
    # [mu_x, mu_y, epsilon, scale, w_u, w_v, w_p]
    params = jnp.zeros((n_kernels, 7))
    key, k1 = jax.random.split(key)
    params = params.at[:, 0].set(jax.random.uniform(k1, (n_kernels,), minval=0.0, maxval=Lx))
    key, k2 = jax.random.split(key)
    params = params.at[:, 1].set(jax.random.uniform(k2, (n_kernels,), minval=0.0, maxval=Ly))
    eps = jnp.linspace(0, 2 * jnp.pi, n_kernels, endpoint=False)
    params = params.at[:, 2].set(eps)
    params = params.at[:, 3].set(0.1 * jnp.ones(n_kernels))
    key, k3 = jax.random.split(key)
    params = params.at[:, 4].set(jax.random.normal(k3, (n_kernels,)) * 0.1)
    key, k4 = jax.random.split(key)
    params = params.at[:, 5].set(jax.random.normal(k4, (n_kernels,)) * 0.1)
    key, k5 = jax.random.split(key)
    params = params.at[:, 6].set(jax.random.normal(k5, (n_kernels,)) * 0.1)
    return params


def advanced_basis(P: jnp.ndarray, params: jnp.ndarray, Lx: float, Ly: float):
    mu = params[:, 0:2]
    epsv = params[:, 2]
    scale_raw = params[:, 3]
    if EVAL_SMOOTH_ADVANCED:
        r = (1.0 / (jnp.maximum(Lx, Ly) ** 2 + 1e-12)) * (1.0 + jax.nn.softplus(scale_raw))
        inv11 = jnp.abs(r * (1.0 + jnp.sin(epsv))) + 1e-6
        inv22 = jnp.abs(r * (1.0 + jnp.cos(epsv))) + 1e-6
        inv12 = (0.05 * (1.0 + jax.nn.softplus(scale_raw))) * jnp.sin(2 * epsv)
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
    grad_phi = -phi[:, :, None] * Ad
    norm_Ad_sq = jnp.sum(Ad ** 2, axis=2)
    trace_A = A[:, 0, 0] + A[:, 1, 1]
    lap_phi = phi * (norm_Ad_sq - trace_A[None, :])
    return phi, grad_phi, lap_phi


# ------------------------ Training ------------------------ #
def train_ns(model_name: str,
             init_fn,
             basis_fn,
             X: jnp.ndarray,
             Y: jnp.ndarray,
             P: jnp.ndarray,
             Re: float,
             n_kernels: int = 64,
             epochs: int = 1000,
             seed: int = 42,
             lr: float = 3e-3,
             U_in: float = 1.0,
             Lx: float = 2.2,
             Ly: float = 0.41,
             xc: float = 0.2,
             yc: float = 0.2,
             R: float = 0.05) -> Dict:
    key = jax.random.PRNGKey(seed)
    params = init_fn(n_kernels, key, Lx, Ly)

    # viscosity from Reynolds number: nu = U_in * D / Re, D=0.1
    D = 0.1
    nu = U_in * D / Re

    # Masks and collocation subsets
    phi_cyl = cylinder_levelset(P[:, 0], P[:, 1], xc=xc, yc=yc, R=R)
    interior_mask = (phi_cyl > 0.0) & (P[:, 0] > 0.0) & (P[:, 0] < Lx) & (P[:, 1] > 0.0) & (P[:, 1] < Ly)
    P_int = P[interior_mask]

    # Inlet (x=0), Outlet (x=Lx), Walls (y=0,y=Ly)
    tol = min(float(Lx) / X.shape[1], float(Ly) / Y.shape[0]) * 0.75
    inlet_mask = (jnp.abs(P[:, 0] - 0.0) < tol)
    outlet_mask = (jnp.abs(P[:, 0] - Lx) < tol)
    wallb_mask = (jnp.abs(P[:, 1] - 0.0) < tol)
    wallt_mask = (jnp.abs(P[:, 1] - Ly) < tol)
    # Precompute boundary point sets (static sizes)
    P_inlet = P[inlet_mask]
    P_outlet = P[outlet_mask]
    P_wallb = P[wallb_mask]
    P_wallt = P[wallt_mask]
    n_in = int(P_inlet.shape[0])
    n_out = int(P_outlet.shape[0])
    n_wb = int(P_wallb.shape[0])
    n_wt = int(P_wallt.shape[0])
    # Cylinder boundary parametric samples
    thetas = jnp.linspace(0, 2 * jnp.pi, N_CYL_SAMPLES, endpoint=False)
    P_cyl = jnp.stack([xc + R * jnp.cos(thetas), yc + R * jnp.sin(thetas)], axis=1)

    # Inlet target profile
    y_in = P_inlet[:, 1] if n_in > 0 else jnp.zeros((1,))
    u_inlet_target = 4.0 * U_in * y_in * (Ly - y_in) / (Ly ** 2)

    def forward_fields(p, P_any):
        phi, gphi, lphi = basis_fn(P_any, p)
        w_u = p[:, -3]
        w_v = p[:, -2]
        w_p = p[:, -1]
        u = jnp.dot(phi, w_u)
        v = jnp.dot(phi, w_v)
        pfield = jnp.dot(phi, w_p)
        du_dx = jnp.dot(gphi[:, :, 0], w_u)
        du_dy = jnp.dot(gphi[:, :, 1], w_u)
        dv_dx = jnp.dot(gphi[:, :, 0], w_v)
        dv_dy = jnp.dot(gphi[:, :, 1], w_v)
        dp_dx = jnp.dot(gphi[:, :, 0], w_p)
        dp_dy = jnp.dot(gphi[:, :, 1], w_p)
        lap_u = jnp.dot(lphi, w_u)
        lap_v = jnp.dot(lphi, w_v)
        return u, v, pfield, du_dx, du_dy, dv_dx, dv_dy, dp_dx, dp_dy, lap_u, lap_v

    def loss_fn(p):
        # Interior residuals
        u, v, pf, du_dx, du_dy, dv_dx, dv_dy, dp_dx, dp_dy, lap_u, lap_v = forward_fields(p, P_int)
        mom_x = u * du_dx + v * du_dy + dp_dx - nu * lap_u
        mom_y = u * dv_dx + v * dv_dy + dp_dy - nu * lap_v
        cont = du_dx + dv_dy
        loss_int = jnp.mean(mom_x ** 2) + jnp.mean(mom_y ** 2) + jnp.mean(cont ** 2)

        # Inlet BC
        if n_in > 0:
            u_in, v_in, *_ = forward_fields(p, P_inlet)
            l_in = jnp.mean((u_in - u_inlet_target) ** 2) + jnp.mean(v_in ** 2)
        else:
            l_in = 0.0

        # Walls BC
        if n_wb > 0:
            u_b, v_b, *_ = forward_fields(p, P_wallb)
            l_wb = jnp.mean(u_b ** 2) + jnp.mean(v_b ** 2)
        else:
            l_wb = 0.0
        if n_wt > 0:
            u_t, v_t, *_ = forward_fields(p, P_wallt)
            l_wt = jnp.mean(u_t ** 2) + jnp.mean(v_t ** 2)
        else:
            l_wt = 0.0

        # Cylinder BC (no-slip)
        u_c, v_c, *_ = forward_fields(p, P_cyl)
        l_cyl = jnp.mean(u_c ** 2) + jnp.mean(v_c ** 2)

        # Outlet BC (Neumann for velocity, p=0)
        if n_out > 0:
            u_o, v_o, pf_o, du_dx_o, _, dv_dx_o, _, _, _, _, _ = forward_fields(p, P_outlet)
            l_out = jnp.mean(du_dx_o ** 2) + jnp.mean(dv_dx_o ** 2) + jnp.mean(pf_o ** 2)
        else:
            l_out = 0.0

        return loss_int + l_in + l_wb + l_wt + l_cyl + l_out

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
    pat = 50
    patience = 0
    for e in range(epochs):
        params, opt_state, l = step(params, opt_state)
        if PROJECTION_EVERY_N and (e % PROJECTION_EVERY_N == 0):
            if model_name == 'Standard (Full)':
                params = project_standard(params, X, Y, Lx, Ly)
            else:
                params = project_advanced(params, X, Y, Lx, Ly)
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
        'final_loss': float(losses[-1]) if losses else float('inf'),
        'loss_history': losses,
        'epochs_run': len(losses),
        'params': params,
        'nu': float(nu),
        'Re': float(Re),
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'navier_stokes_steady', timestamp)
    os.makedirs(outdir, exist_ok=True)

    Lx, Ly = 2.2, 0.41
    X, Y, P = make_grid(nx=NX, ny=NY, Lx=Lx, Ly=Ly)

    U_in = 1.0
    # Reynolds numbers to test - higher values for better vortex visualization
    Re_values = [20.0, 40.0, 100.0, 200.0, 500.0, 1000.0]
    n_k = N_KERNELS
    epochs = 1200
    lr = 3e-3

    # Cylinder geometry for overlays
    xc, yc, R = 0.2, 0.2, 0.05

    for Re in Re_values:
        case_dir = os.path.join(outdir, f'Re{int(Re)}')
        os.makedirs(case_dir, exist_ok=True)

        std_res = train_ns('Standard (Full)', standard_init,
                           lambda P_, p: standard_basis(P_, p),
                           X, Y, P, Re,
                           n_kernels=n_k, epochs=epochs, seed=42, lr=lr,
                           U_in=U_in, Lx=Lx, Ly=Ly)
        adv_res = train_ns('Advanced Shape Transform', advanced_init,
                           lambda P_, p: advanced_basis(P_, p, Lx, Ly),
                           X, Y, P, Re,
                           n_kernels=n_k, epochs=epochs, seed=42, lr=lr,
                           U_in=U_in, Lx=Lx, Ly=Ly)

        # Loss curves
        plt.figure(figsize=(7,4))
        plt.plot(std_res['loss_history'], label='Standard (Full)')
        plt.plot(adv_res['loss_history'], label='Advanced Shape Transform')
        plt.yscale('log')
        plt.xlabel('Epoch'); plt.ylabel('NS residual + BC MSE')
        plt.title(f'Steady NS PINN Loss (Re={int(Re)})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(case_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Visualize u-velocity and pressure
        def eval_fields(basis_fn, params):
            phi, gphi, lphi = basis_fn(P)
            w_u = params[:, -3]; w_v = params[:, -2]; w_p = params[:, -1]
            u = np.array(jnp.dot(phi, w_u)).reshape(X.shape)
            v = np.array(jnp.dot(phi, w_v)).reshape(X.shape)
            pfield = np.array(jnp.dot(phi, w_p)).reshape(X.shape)
            return u, v, pfield

        U_std, V_std, P_std = eval_fields(lambda P_: standard_basis(P_, std_res['params']), std_res['params'])
        U_adv, V_adv, P_adv = eval_fields(lambda P_: advanced_basis(P_, adv_res['params'], Lx, Ly), adv_res['params'])

        # Compute velocity gradients for v_x and v_y
        def compute_velocity_gradients(u, v, dx, dy):
            # Compute gradients using finite differences
            u_x = np.gradient(u, dx, axis=1)
            u_y = np.gradient(u, dy, axis=0)
            v_x = np.gradient(v, dx, axis=1)
            v_y = np.gradient(v, dy, axis=0)
            return u_x, u_y, v_x, v_y
        
        dx = Lx / (X.shape[1] - 1)
        dy = Ly / (Y.shape[0] - 1)
        
        U_x_std, U_y_std, V_x_std, V_y_std = compute_velocity_gradients(U_std, V_std, dx, dy)
        U_x_adv, U_y_adv, V_x_adv, V_y_adv = compute_velocity_gradients(U_adv, V_adv, dx, dy)

        # Preserve aspect ratio so the cylinder appears circular
        fig_aspect = Ly / Lx
        fig, axes = plt.subplots(4, 4, figsize=(16, 16 * fig_aspect))
        
        # Row 1: u-velocity components
        im0 = axes[0, 0].imshow(U_std, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Standard: u(x,y)')
        axes[0, 0].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        im1 = axes[0, 1].imshow(U_adv, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='viridis')
        axes[0, 1].set_title('Advanced: u(x,y)')
        axes[0, 1].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        im2 = axes[0, 2].imshow(U_x_std, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='coolwarm')
        axes[0, 2].set_title('Standard: ∂u/∂x')
        axes[0, 2].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        im3 = axes[0, 3].imshow(U_x_adv, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='coolwarm')
        axes[0, 3].set_title('Advanced: ∂u/∂x')
        axes[0, 3].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # Row 2: v-velocity components
        im4 = axes[1, 0].imshow(V_std, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Standard: v(x,y)')
        axes[1, 0].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        im5 = axes[1, 1].imshow(V_adv, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='viridis')
        axes[1, 1].set_title('Advanced: v(x,y)')
        axes[1, 1].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        im6 = axes[1, 2].imshow(V_x_std, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='coolwarm')
        axes[1, 2].set_title('Standard: ∂v/∂x')
        axes[1, 2].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        im7 = axes[1, 3].imshow(V_x_adv, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='coolwarm')
        axes[1, 3].set_title('Advanced: ∂v/∂x')
        axes[1, 3].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04)
        
        # Row 3: Additional velocity gradients
        im8 = axes[2, 0].imshow(U_y_std, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='coolwarm')
        axes[2, 0].set_title('Standard: ∂u/∂y')
        axes[2, 0].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im8, ax=axes[2, 0], fraction=0.046, pad=0.04)
        
        im9 = axes[2, 1].imshow(U_y_adv, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='coolwarm')
        axes[2, 1].set_title('Advanced: ∂u/∂y')
        axes[2, 1].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im9, ax=axes[2, 1], fraction=0.046, pad=0.04)
        
        im10 = axes[2, 2].imshow(V_y_std, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='coolwarm')
        axes[2, 2].set_title('Standard: ∂v/∂y')
        axes[2, 2].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im10, ax=axes[2, 2], fraction=0.046, pad=0.04)
        
        im11 = axes[2, 3].imshow(V_y_adv, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='coolwarm')
        axes[2, 3].set_title('Advanced: ∂v/∂y')
        axes[2, 3].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im11, ax=axes[2, 3], fraction=0.046, pad=0.04)
        
        # Row 4: Pressure and velocity fields
        im12 = axes[3, 0].imshow(P_std, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='coolwarm')
        axes[3, 0].set_title('Standard: p(x,y)')
        axes[3, 0].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im12, ax=axes[3, 0], fraction=0.046, pad=0.04)
        
        im13 = axes[3, 1].imshow(P_adv, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='coolwarm')
        axes[3, 1].set_title('Advanced: p(x,y)')
        axes[3, 1].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        fig.colorbar(im13, ax=axes[3, 1], fraction=0.046, pad=0.04)
        
        # Velocity fields with quiver plots
        step = 4
        axes[3, 2].quiver(np.array(X[::step, ::step]), np.array(Y[::step, ::step]),
                          U_std[::step, ::step], V_std[::step, ::step], scale=20)
        axes[3, 2].set_title('Standard: velocity field')
        axes[3, 2].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        
        axes[3, 3].quiver(np.array(X[::step, ::step]), np.array(Y[::step, ::step]),
                          U_adv[::step, ::step], V_adv[::step, ::step], scale=20)
        axes[3, 3].set_title('Advanced: velocity field')
        axes[3, 3].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
        
        for ax in axes.ravel():
            ax.set_xlabel('x'); ax.set_ylabel('y')
            ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
            ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, 'fields.png'), dpi=300, bbox_inches='tight')
        plt.close()

    with open(os.path.join(outdir, 'RESULTS.md'), 'w') as f:
        f.write('# Steady Navier–Stokes (Cylinder) PINN Comparison\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('- Domain: [0,2.2] x [0,0.41], cylinder at (0.2,0.2), R=0.05\n')
        f.write('- BCs: inlet parabolic, walls no-slip, cylinder no-slip, outlet Neumann + p=0\n')
        f.write('- Models: Standard (Full), Advanced Shape Transform; shared basis, separate weights for u,v,p\n')

    print('Saved results to', outdir)


if __name__ == '__main__':
    main()


