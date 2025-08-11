#!/usr/bin/env python3
"""
    Transient 2D incompressible Navier–Stokes (flow over cylinder) PINN comparison.

    Unknowns: u(x,y,t), v(x,y,t), p(x,y,t)
    Equations (transient):
      u_t + u*u_x + v*u_y + p_x - nu * (u_xx + u_yy) = 0
      v_t + u*v_x + v*v_y + p_y - nu * (v_xx + v_yy) = 0
      u_x + v_y = 0

    Domain: [0, Lx] x [0, Ly] x [0, T] \\ cylinder((xc, yc), R)
    Defaults: Lx=2.2, Ly=0.41, T=10.0, (xc,yc)=(0.2,0.2), R=0.05, U_in=1.0

    Boundary conditions:
      Inlet x=0: u(y,t)=4 U_in y (Ly - y) / Ly^2, v=0
      Top/Bottom y in {0,Ly}: u=0, v=0
      Cylinder r=R around (xc,yc): u=0, v=0
      Outlet x=Lx: du/dx=0, dv/dx=0, p=0 (fix gauge at outlet)
      Initial t=0: u(x,y,0)=0, v(x,y,0)=0 (except inlet)

    Models: Standard (full covariance) vs Advanced Shape Transform basis; 
    both share basis across u,v,p with separate weights and now include time dimension.
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
from mpl_toolkits.mplot3d import Axes3D

jax.config.update('jax_enable_x64', True)

# Config tuned similar to previous strong settings
EVAL_SMOOTH_STANDARD: bool = False
EVAL_SMOOTH_ADVANCED: bool = False
PROJECTION_EVERY_N: int = 1
WIDEN_BOUNDS: bool = False

# Capacity and collocation controls
NX: int = 128  # Reduced for 3D transient computation
NY: int = 64
NT: int = 50   # Time steps
N_KERNELS: int = 256
N_CYL_SAMPLES: int = 512


def make_grid_3d(nx: int = 128, ny: int = 64, nt: int = 50, 
                 Lx: float = 2.2, Ly: float = 0.41, T: float = 10.0):
    """Create 3D grid including time dimension."""
    x = jnp.linspace(0.0, Lx, nx)
    y = jnp.linspace(0.0, Ly, ny)
    t = jnp.linspace(0.0, T, nt)
    X, Y, T = jnp.meshgrid(x, y, t, indexing='ij')
    P = jnp.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
    return X, Y, T, P


def cylinder_levelset(x, y, xc=0.2, yc=0.2, R=0.05):
    return (x - xc) ** 2 + (y - yc) ** 2 - (R ** 2)


# ------------------------ Projection helpers ------------------------ #
def project_standard_3d(params: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, T: jnp.ndarray,
                         Lx: float, Ly: float, T_max: float) -> jnp.ndarray:
    # params per kernel: [mu_x, mu_y, mu_t, log_sig_x, log_sig_y, log_sig_t, theta, w_u, w_v, w_p]
    out = params
    out = out.at[:, 0].set(jnp.clip(out[:, 0], 0.0, Lx))
    out = out.at[:, 1].set(jnp.clip(out[:, 1], 0.0, Ly))
    out = out.at[:, 2].set(jnp.clip(out[:, 2], 0.0, T_max))
    dx = Lx / max(X.shape[0] - 1, 1)
    dy = Ly / max(Y.shape[1] - 1, 1)
    dt = T_max / max(T.shape[2] - 1, 1)
    if WIDEN_BOUNDS:
        min_sig_x, max_sig_x = dx / 4.0, Lx
        min_sig_y, max_sig_y = dy / 4.0, Ly
        min_sig_t, max_sig_t = dt / 4.0, T_max
    else:
        min_sig_x, max_sig_x = dx / 2.0, Lx / 2.0
        min_sig_y, max_sig_y = dy / 2.0, Ly / 2.0
        min_sig_t, max_sig_t = dt / 2.0, T_max / 2.0
    out = out.at[:, 3].set(jnp.clip(out[:, 3], jnp.log(min_sig_x), jnp.log(max_sig_x)))
    out = out.at[:, 4].set(jnp.clip(out[:, 4], jnp.log(min_sig_y), jnp.log(max_sig_y)))
    out = out.at[:, 5].set(jnp.clip(out[:, 5], jnp.log(min_sig_t), jnp.log(max_sig_t)))
    return out


def project_advanced_3d(params: jnp.ndarray, X: jnp.ndarray, Y: jnp.ndarray, T: jnp.ndarray,
                        Lx: float, Ly: float, T_max: float) -> jnp.ndarray:
    # params per kernel: [mu_x, mu_y, mu_t, epsilon, scale, w_u, w_v, w_p]
    out = params
    out = out.at[:, 0].set(jnp.clip(out[:, 0], 0.0, Lx))
    out = out.at[:, 1].set(jnp.clip(out[:, 1], 0.0, Ly))
    out = out.at[:, 2].set(jnp.clip(out[:, 2], 0.0, T_max))
    dx = Lx / max(X.shape[0] - 1, 1)
    dy = Ly / max(Y.shape[1] - 1, 1)
    dt = T_max / max(T.shape[2] - 1, 1)
    dom = jnp.maximum(jnp.maximum(Lx, Ly), T_max)
    if WIDEN_BOUNDS:
        min_scale, max_scale = (jnp.minimum(jnp.minimum(dx, dy), dt) / 4.0), dom
    else:
        min_scale, max_scale = (jnp.minimum(jnp.minimum(dx, dy), dt) / 2.0), dom / 2.0
    out = out.at[:, 4].set(jnp.clip(out[:, 4], min_scale, max_scale))
    return out


# ------------------------ Basis models ------------------------ #
def standard_init_3d(n_kernels: int, key: jax.Array, Lx: float, Ly: float, T_max: float) -> jnp.ndarray:
    """Initialize 3D standard RBF parameters."""
    key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)
    
    # [mu_x, mu_y, mu_t, log_sig_x, log_sig_y, log_sig_t, theta, w_u, w_v, w_p]
    mu_x = jax.random.uniform(key1, (n_kernels,), minval=0.0, maxval=Lx, dtype=jnp.float64)
    mu_y = jax.random.uniform(key2, (n_kernels,), minval=0.0, maxval=Ly, dtype=jnp.float64)
    mu_t = jax.random.uniform(key3, (n_kernels,), minval=0.0, maxval=T_max, dtype=jnp.float64)
    
    # Initialize sigmas based on domain size
    log_sig_x = jnp.log(jax.random.uniform(key4, (n_kernels,), minval=Lx/100, maxval=Lx/10, dtype=jnp.float64))
    log_sig_y = jnp.log(jax.random.uniform(key5, (n_kernels,), minval=Ly/100, maxval=Ly/10, dtype=jnp.float64))
    log_sig_t = jnp.log(jax.random.uniform(key6, (n_kernels,), minval=T_max/100, maxval=T_max/10, dtype=jnp.float64))
    
    theta = jax.random.uniform(key7, (n_kernels,), minval=0.0, maxval=2*jnp.pi, dtype=jnp.float64)
    
    # Initialize weights small
    w_u = jax.random.normal(key1, (n_kernels,), dtype=jnp.float64) * 0.01
    w_v = jax.random.normal(key2, (n_kernels,), dtype=jnp.float64) * 0.01
    w_p = jax.random.normal(key3, (n_kernels,), dtype=jnp.float64) * 0.01
    
    return jnp.stack([mu_x, mu_y, mu_t, log_sig_x, log_sig_y, log_sig_t, theta, w_u, w_v, w_p], axis=1)


def advanced_init_3d(n_kernels: int, key: jax.Array, Lx: float, Ly: float, T_max: float) -> jnp.ndarray:
    """Initialize 3D advanced shape transform RBF parameters."""
    key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)
    
    # [mu_x, mu_y, mu_t, epsilon, scale, w_u, w_v, w_p]
    mu_x = jax.random.uniform(key1, (n_kernels,), minval=0.0, maxval=Lx, dtype=jnp.float64)
    mu_y = jax.random.uniform(key2, (n_kernels,), minval=0.0, maxval=Ly, dtype=jnp.float64)
    mu_t = jax.random.uniform(key3, (n_kernels,), minval=0.0, maxval=T_max, dtype=jnp.float64)
    
    # Shape parameter and scale
    epsilon = jax.random.uniform(key4, (n_kernels,), minval=-2.0, maxval=2.0, dtype=jnp.float64)
    scale = jax.random.uniform(key5, (n_kernels,), minval=jnp.minimum(jnp.minimum(Lx, Ly), T_max)/100, 
                              maxval=jnp.maximum(jnp.maximum(Lx, Ly), T_max)/10, dtype=jnp.float64)
    
    # Initialize weights small
    w_u = jax.random.normal(key6, (n_kernels,), dtype=jnp.float64) * 0.01
    w_v = jax.random.normal(key7, (n_kernels,), dtype=jnp.float64) * 0.01
    w_p = jax.random.normal(key1, (n_kernels,), dtype=jnp.float64) * 0.01
    
    return jnp.stack([mu_x, mu_y, mu_t, epsilon, scale, w_u, w_v, w_p], axis=1)


def standard_basis_3d(P: jnp.ndarray, params: jnp.ndarray):
    """3D standard RBF basis with time dimension."""
    # P: (N, 3) - [x, y, t]
    # params: (K, 10) - [mu_x, mu_y, mu_t, log_sig_x, log_sig_y, log_sig_t, theta, w_u, w_v, w_p]
    
    mu_x, mu_y, mu_t = params[:, 0], params[:, 1], params[:, 2]
    log_sig_x, log_sig_y, log_sig_t = params[:, 3], params[:, 4], params[:, 5]
    theta = params[:, 6]
    
    # Extract coordinates
    x, y, t = P[:, 0], P[:, 1], P[:, 2]
    
    # Compute distances
    dx = x[:, None] - mu_x[None, :]
    dy = y[:, None] - mu_y[None, :]
    dt = t[:, None] - mu_t[None, :]
    
    # Rotate coordinates
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    dx_rot = dx * cos_theta[None, :] + dy * sin_theta[None, :]
    dy_rot = -dx * sin_theta[None, :] + dy * cos_theta[None, :]
    
    # Compute RBF values
    sig_x = jnp.exp(log_sig_x)
    sig_y = jnp.exp(log_sig_y)
    sig_t = jnp.exp(log_sig_t)
    
    r2 = (dx_rot / sig_x[None, :])**2 + (dy_rot / sig_y[None, :])**2 + (dt / sig_t[None, :])**2
    phi = jnp.exp(-r2)
    
    # Compute gradients
    dphi_dx = -2 * phi * dx_rot / (sig_x[None, :]**2)
    dphi_dy = -2 * phi * dy_rot / (sig_y[None, :]**2)
    dphi_dt = -2 * phi * dt / (sig_t[None, :]**2)
    
    # Compute Laplacians
    d2phi_dx2 = 2 * phi * (2 * (dx_rot / sig_x[None, :])**2 - 1) / (sig_x[None, :]**2)
    d2phi_dy2 = 2 * phi * (2 * (dy_rot / sig_y[None, :])**2 - 1) / (sig_y[None, :]**2)
    d2phi_dt2 = 2 * phi * (2 * (dt / sig_t[None, :])**2 - 1) / (sig_t[None, :]**2)
    
    return phi, jnp.stack([dphi_dx, dphi_dy, dphi_dt], axis=1), jnp.stack([d2phi_dx2, d2phi_dy2, d2phi_dt2], axis=1)


def advanced_basis_3d(P: jnp.ndarray, params: jnp.ndarray, Lx: float, Ly: float, T_max: float):
    """3D advanced shape transform RBF basis with time dimension."""
    # P: (N, 3) - [x, y, t]
    # params: (K, 8) - [mu_x, mu_y, mu_t, epsilon, scale, w_u, w_v, w_p]
    
    mu_x, mu_y, mu_t = params[:, 0], params[:, 1], params[:, 2]
    epsilon = params[:, 3]
    scale = params[:, 4]
    
    # Extract coordinates
    x, y, t = P[:, 0], P[:, 1], P[:, 2]
    
    # Compute distances
    dx = x[:, None] - mu_x[None, :]
    dy = y[:, None] - mu_y[None, :]
    dt = t[:, None] - mu_t[None, :]
    
    # Shape parameter transform
    epsilon_tensor = jnp.sin(epsilon) * 3
    sx = jnp.exp(epsilon_tensor)
    sy = 1.0 / (1.0 + jnp.exp(-epsilon_tensor))
    st = scale  # Time scale
    
    # Compute RBF values
    r2 = (dx / (sx[None, :] * scale[None, :]))**2 + (dy / (sy[None, :] * scale[None, :]))**2 + (dt / st[None, :])**2
    phi = jnp.exp(-r2)
    
    # Compute gradients
    dphi_dx = -2 * phi * dx / ((sx[None, :] * scale[None, :])**2)
    dphi_dy = -2 * phi * dy / ((sy[None, :] * scale[None, :])**2)
    dphi_dt = -2 * phi * dt / (st[None, :]**2)
    
    # Compute Laplacians
    d2phi_dx2 = 2 * phi * (2 * (dx / (sx[None, :] * scale[None, :]))**2 - 1) / ((sx[None, :] * scale[None, :])**2)
    d2phi_dy2 = 2 * phi * (2 * (dy / (sy[None, :] * scale[None, :]))**2 - 1) / ((sy[None, :] * scale[None, :])**2)
    d2phi_dt2 = 2 * phi * (2 * (dt / st[None, :])**2 - 1) / (st[None, :]**2)
    
    return phi, jnp.stack([dphi_dx, dphi_dy, dphi_dt], axis=1), jnp.stack([d2phi_dx2, d2phi_dy2, d2phi_dt2], axis=1)


def train_ns_transient(model_name: str,
                       init_fn,
                       basis_fn,
                       X: jnp.ndarray,
                       Y: jnp.ndarray,
                       T: jnp.ndarray,
                       P: jnp.ndarray,
                       Re: float,
                       n_kernels: int = 64,
                       epochs: int = 1000,
                       seed: int = 42,
                       lr: float = 3e-3,
                       U_in: float = 1.0,
                       Lx: float = 2.2,
                       Ly: float = 0.41,
                       T_max: float = 10.0,
                       xc: float = 0.2,
                       yc: float = 0.2,
                       R: float = 0.05) -> Dict:
    """Train transient Navier-Stokes PINN."""
    
    key = jax.random.PRNGKey(seed)
    
    # Initialize parameters
    params = init_fn(n_kernels, key, Lx, Ly, T_max)
    
    # Create boundary masks
    def create_boundary_masks():
        x_flat = X.flatten()
        y_flat = Y.flatten()
        t_flat = T.flatten()
        
        # Inlet (x=0)
        inlet_mask = x_flat == 0.0
        
        # Walls (y=0 or y=Ly)
        wall_mask = (y_flat == 0.0) | (y_flat == Ly)
        
        # Cylinder
        cylinder_mask = (x_flat - xc)**2 + (y_flat - yc)**2 <= R**2
        
        # Outlet (x=Lx)
        outlet_mask = x_flat == Lx
        
        # Initial condition (t=0)
        initial_mask = t_flat == 0.0
        
        return inlet_mask, wall_mask, cylinder_mask, outlet_mask, initial_mask
    
    inlet_mask, wall_mask, cylinder_mask, outlet_mask, initial_mask = create_boundary_masks()
    
    # Kinematic viscosity
    nu = U_in * 2 * R / Re
    
    def forward_fields(p, P_any):
        """Forward pass to compute u, v, p fields."""
        phi, gphi, lphi = basis_fn(P_any, p)
        w_u = p[:, -3]
        w_v = p[:, -2]
        w_p = p[:, -1]
        
        u = jnp.dot(phi, w_u)
        v = jnp.dot(phi, w_v)
        p_field = jnp.dot(phi, w_p)
        
        return u, v, p_field
    
    def loss_fn(p):
        """Transient Navier-Stokes loss function."""
        # Interior residuals
        phi, gphi, lphi = basis_fn(P, p)
        w_u = p[:, -3]
        w_v = p[:, -2]
        w_p = p[:, -1]
        
        u = jnp.dot(phi, w_u)
        v = jnp.dot(phi, w_v)
        p_field = jnp.dot(phi, w_p)
        
        # Gradients
        du_dx = jnp.dot(gphi[:, 0], w_u)
        du_dy = jnp.dot(gphi[:, 1], w_u)
        du_dt = jnp.dot(gphi[:, 2], w_u)
        
        dv_dx = jnp.dot(gphi[:, 0], w_v)
        dv_dy = jnp.dot(gphi[:, 1], w_v)
        dv_dt = jnp.dot(gphi[:, 2], w_v)
        
        dp_dx = jnp.dot(gphi[:, 0], w_p)
        dp_dy = jnp.dot(gphi[:, 1], w_p)
        
        # Laplacians
        d2u_dx2 = jnp.dot(lphi[:, 0], w_u)
        d2u_dy2 = jnp.dot(lphi[:, 1], w_u)
        
        d2v_dx2 = jnp.dot(lphi[:, 0], w_v)
        d2v_dy2 = jnp.dot(lphi[:, 1], w_v)
        
        # Momentum equations
        momentum_u = du_dt + u * du_dx + v * du_dy + dp_dx - nu * (d2u_dx2 + d2u_dy2)
        momentum_v = dv_dt + u * dv_dx + v * dv_dy + dp_dy - nu * (d2v_dx2 + d2v_dy2)
        
        # Continuity equation
        continuity = du_dx + dv_dy
        
        # Interior residual loss
        interior_loss = jnp.mean(momentum_u**2 + momentum_v**2 + continuity**2)
        
        # Boundary conditions
        u_inlet = u[inlet_mask]
        v_inlet = v[inlet_mask]
        u_wall = u[wall_mask]
        v_wall = v[wall_mask]
        u_cylinder = u[cylinder_mask]
        v_cylinder = v[cylinder_mask]
        p_outlet = p_field[outlet_mask]
        
        # Inlet BC: parabolic profile
        y_inlet = P[inlet_mask, 1]
        u_inlet_target = 4 * U_in * y_inlet * (Ly - y_inlet) / (Ly**2)
        inlet_loss = jnp.mean((u_inlet - u_inlet_target)**2 + v_inlet**2)
        
        # Wall BC: no-slip
        wall_loss = jnp.mean(u_wall**2 + v_wall**2)
        
        # Cylinder BC: no-slip
        cylinder_loss = jnp.mean(u_cylinder**2 + v_cylinder**2)
        
        # Outlet BC: pressure = 0
        outlet_loss = jnp.mean(p_outlet**2)
        
        # Initial condition: u=v=0 at t=0
        u_initial = u[initial_mask]
        v_initial = v[initial_mask]
        initial_loss = jnp.mean(u_initial**2 + v_initial**2)
        
        # Total loss
        total_loss = (interior_loss + 
                      inlet_loss + wall_loss + cylinder_loss + 
                      outlet_loss + initial_loss)
        
        return total_loss
    
    # Optimizer
    optimizer = optax.adam(lr)
    
    @jax.jit
    def step(p, s):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, s = optimizer.update(grads, s)
        p = optax.apply_updates(p, updates)
        return p, s, loss
    
    # Training loop
    opt_state = optimizer.init(params)
    loss_history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        params, opt_state, loss = step(params, opt_state)
        loss_history.append(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
    
    training_time = time.time() - start_time
    
    return {
        'params': params,
        'loss_history': loss_history,
        'training_time': training_time,
        'final_loss': loss_history[-1]
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'navier_stokes_transient', timestamp)
    os.makedirs(outdir, exist_ok=True)

    Lx, Ly, T_max = 2.2, 0.41, 10.0
    X, Y, T, P = make_grid_3d(nx=NX, ny=NY, nt=NT, Lx=Lx, Ly=Ly, T=T_max)

    U_in = 1.0
    Re_values = [20.0, 40.0, 100.0, 200.0]  # Focus on lower Re for transient stability
    n_k = N_KERNELS
    epochs = 800  # Reduced for 3D computation
    lr = 3e-3

    # Cylinder geometry for overlays
    xc, yc, R = 0.2, 0.2, 0.05

    for Re in Re_values:
        case_dir = os.path.join(outdir, f'Re{int(Re)}')
        os.makedirs(case_dir, exist_ok=True)

        print(f"Training Re={Re}...")
        
        std_res = train_ns_transient('Standard (Full)', standard_init_3d,
                                     lambda P_, p: standard_basis_3d(P_, p),
                                     X, Y, T, P, Re,
                                     n_kernels=n_k, epochs=epochs, seed=42, lr=lr,
                                     U_in=U_in, Lx=Lx, Ly=Ly, T_max=T_max)
        adv_res = train_ns_transient('Advanced Shape Transform', advanced_init_3d,
                                     lambda P_, p: advanced_basis_3d(P_, p, Lx, Ly, T_max),
                                     X, Y, T, P, Re,
                                     n_kernels=n_k, epochs=epochs, seed=42, lr=lr,
                                     U_in=U_in, Lx=Lx, Ly=Ly, T_max=T_max)

        # Loss curves
        plt.figure(figsize=(7,4))
        plt.plot(std_res['loss_history'], label='Standard (Full)')
        plt.plot(adv_res['loss_history'], label='Advanced Shape Transform')
        plt.yscale('log')
        plt.xlabel('Epoch'); plt.ylabel('NS residual + BC MSE')
        plt.title(f'Transient NS PINN Loss (Re={int(Re)})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(case_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Visualize fields at different time slices
        def eval_fields_3d(basis_fn, params, t_idx):
            """Evaluate fields at a specific time slice."""
            P_slice = P.reshape(-1, 3)
            mask = P_slice[:, 2] == T[0, 0, t_idx]
            P_masked = P_slice[mask]
            
            phi, gphi, lphi = basis_fn(P_masked, params)
            w_u = params[:, -3]
            w_v = params[:, -2]
            w_p = params[:, -1]
            
            u = np.array(jnp.dot(phi, w_u))
            v = np.array(jnp.dot(phi, w_v))
            p_field = np.array(jnp.dot(phi, w_p))
            
            # Reshape to 2D
            u_2d = u.reshape(X.shape[0], X.shape[1])
            v_2d = v.reshape(X.shape[0], X.shape[1])
            p_2d = p_field.reshape(X.shape[0], X.shape[1])
            
            return u_2d, v_2d, p_2d

        # Plot at different time slices
        time_indices = [0, NT//4, NT//2, 3*NT//4, NT-1]
        time_values = [T[0, 0, i] for i in time_indices]
        
        fig, axes = plt.subplots(2, len(time_indices), figsize=(4*len(time_indices), 8))
        
        for i, (t_idx, t_val) in enumerate(zip(time_indices, time_values)):
            U_std, V_std, P_std = eval_fields_3d(lambda P_: standard_basis_3d(P_, std_res['params']), std_res['params'], t_idx)
            U_adv, V_adv, P_adv = eval_fields_3d(lambda P_: advanced_basis_3d(P_, adv_res['params'], Lx, Ly, T_max), adv_res['params'], t_idx)
            
            # Standard model
            im1 = axes[0, i].imshow(U_std, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='viridis')
            axes[0, i].set_title(f'Standard: u(x,y,t={t_val:.1f})')
            axes[0, i].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            fig.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # Advanced model
            im2 = axes[1, i].imshow(U_adv, origin='lower', extent=[0,Lx,0,Ly], aspect='auto', cmap='viridis')
            axes[1, i].set_title(f'Advanced: u(x,y,t={t_val:.1f})')
            axes[1, i].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            fig.colorbar(im2, ax=axes[1, i], fraction=0.046, pad=0.04)
            
            for ax in [axes[0, i], axes[1, i]]:
                ax.set_xlabel('x'); ax.set_ylabel('y')
                ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
                ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(os.path.join(case_dir, 'time_evolution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    with open(os.path.join(outdir, 'RESULTS.md'), 'w') as f:
        f.write('# Transient Navier–Stokes (Cylinder) PINN Comparison\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('- Domain: [0,2.2] x [0,0.41] x [0,10.0], cylinder at (0.2,0.2), R=0.05\n')
        f.write('- BCs: inlet parabolic, walls no-slip, cylinder no-slip, outlet Neumann + p=0\n')
        f.write('- IC: u=v=0 at t=0 (except inlet)\n')
        f.write('- Models: Standard (Full), Advanced Shape Transform; 3D basis with time dimension\n')

    print('Saved results to', outdir)


if __name__ == '__main__':
    main()
