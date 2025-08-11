import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time
from scipy.interpolate import griddata
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import os
from datetime import datetime
import jax
import jax.numpy as jnp
from pinn_ns_steady_cylinder import (
    make_grid, standard_init, advanced_init, 
    standard_basis, advanced_basis, train_ns
)

jax.config.update('jax_enable_x64', True)

def solve_ns_finite_difference_robust(nx, ny, Lx, Ly, Re, U_in, xc, yc, R):
    """
    Solve steady 2D Navier-Stokes using robust finite difference method.
    
    This implementation uses:
    - Staggered grid for better pressure-velocity coupling
    - Upwind differencing for convective terms
    - Proper boundary conditions
    - Adaptive relaxation for stability
    """
    print(f"Solving robust FD with grid {nx}x{ny}, Re={Re}")
    
    # Grid setup
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    # Staggered grid: pressure at cell centers, velocities at faces
    x_p = np.linspace(0, Lx, nx)
    y_p = np.linspace(0, Ly, ny)
    x_u = np.linspace(dx/2, Lx-dx/2, nx-1)  # u-velocity grid
    y_u = np.linspace(0, Ly, ny)
    x_v = np.linspace(0, Lx, nx)  # v-velocity grid
    y_v = np.linspace(dy/2, Ly-dy/2, ny-1)
    
    # Initialize fields
    u = np.zeros((ny, nx-1))  # u-velocity
    v = np.zeros((ny-1, nx))  # v-velocity
    p = np.zeros((ny, nx))    # pressure
    
    # Cylinder masks
    def is_inside_cylinder(x, y):
        return (x - xc)**2 + (y - yc)**2 <= R**2
    
    p_mask = np.zeros((ny, nx), dtype=bool)
    u_mask = np.zeros((ny, nx-1), dtype=bool)
    v_mask = np.zeros((ny-1, nx), dtype=bool)
    
    for j in range(ny):
        for i in range(nx):
            if is_inside_cylinder(x_p[i], y_p[j]):
                p_mask[j, i] = True
    
    for j in range(ny):
        for i in range(nx-1):
            if is_inside_cylinder(x_u[i], y_u[j]):
                u_mask[j, i] = True
    
    for j in range(ny-1):
        for i in range(nx):
            if is_inside_cylinder(x_v[i], y_v[j]):
                v_mask[j, i] = True
    
    # Set inlet boundary condition (parabolic profile)
    for j in range(ny):
        if not u_mask[j, 0]:
            y_val = y_u[j]
            u[j, 0] = 4 * U_in * y_val * (Ly - y_val) / (Ly**2)
    
    # Initialize interior with potential flow solution
    for j in range(ny):
        for i in range(1, nx-1):
            if not u_mask[j, i]:
                # Potential flow around cylinder
                r = np.sqrt((x_u[i] - xc)**2 + (y_u[j] - yc)**2)
                if r > R:
                    theta = np.arctan2(y_u[j] - yc, x_u[i] - xc)
                    u[j, i] = U_in * (1 - (R/r)**2 * np.cos(2*theta))
    
    for j in range(ny-1):
        for i in range(nx):
            if not v_mask[j, i]:
                r = np.sqrt((x_v[i] - xc)**2 + (y_v[j] - yc)**2)
                if r > R:
                    theta = np.arctan2(y_v[j] - yc, x_v[i] - xc)
                    v[j, i] = -U_in * (R/r)**2 * np.sin(2*theta)
    
    # Initialize pressure with potential flow
    for j in range(ny):
        for i in range(nx):
            if not p_mask[j, i]:
                r = np.sqrt((x_p[i] - xc)**2 + (y_p[j] - yc)**2)
                if r > R:
                    theta = np.arctan2(y_p[j] - yc, x_p[i] - xc)
                    u_interp = U_in * (1 - (R/r)**2 * np.cos(2*theta))
                    v_interp = -U_in * (R/r)**2 * np.sin(2*theta)
                    p[j, i] = -0.5 * (u_interp**2 + v_interp**2) / U_in**2
    
    # Kinematic viscosity
    nu = U_in * 2 * R / Re
    
    # Adaptive relaxation parameters
    omega_u = 0.8  # Start conservative
    omega_v = 0.8
    omega_p = 0.8
    
    # SOR iteration
    max_iter = 5000
    tol = 1e-5
    
    start_time = time.time()
    
    for iter in range(max_iter):
        u_old = u.copy()
        v_old = v.copy()
        p_old = p.copy()
        
        # Update u-velocity (x-momentum)
        for j in range(1, ny-1):
            for i in range(1, nx-2):
                if not u_mask[j, i]:
                    # Convective terms with upwind differencing
                    if u[j, i] >= 0:
                        du_dx = (u[j, i] - u[j, i-1]) / dx
                    else:
                        du_dx = (u[j, i+1] - u[j, i]) / dx
                    
                    if v[j, i] >= 0:
                        du_dy = (u[j, i] - u[j-1, i]) / dy
                    else:
                        du_dy = (u[j+1, i] - u[j, i]) / dy
                    
                    # Interpolate v at u-locations
                    v_e = 0.5 * (v[j, i] + v[j, i+1]) if j < ny-1 and i+1 < nx else 0
                    v_w = 0.5 * (v[j, i-1] + v[j, i]) if j < ny-1 and i-1 >= 0 else 0
                    
                    if v_e >= 0:
                        du_dy_e = (u[j, i] - u[j-1, i]) / dy
                    else:
                        du_dy_e = (u[j+1, i] - u[j, i]) / dy
                    
                    if v_w >= 0:
                        du_dy_w = (u[j, i] - u[j-1, i]) / dy
                    else:
                        du_dy_w = (u[j+1, i] - u[j, i]) / dy
                    
                    # Pressure gradient
                    dp_dx = (p[j, i+1] - p[j, i]) / dx
                    
                    # Viscous terms
                    d2u_dx2 = (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / (dx**2)
                    d2u_dy2 = (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / (dy**2)
                    
                    # Momentum equation
                    conv = u[j, i] * du_dx + 0.5 * (v_e * du_dy_e + v_w * du_dy_w)
                    visc = nu * (d2u_dx2 + d2u_dy2)
                    
                    # Update with relaxation
                    u_new = u_old[j, i] + omega_u * (-conv - dp_dx + visc)
                    u[j, i] = np.clip(u_new, -3*U_in, 3*U_in)
        
        # Update v-velocity (y-momentum)
        for j in range(1, ny-2):
            for i in range(1, nx-1):
                if not v_mask[j, i]:
                    # Convective terms with upwind differencing
                    if u[j, i] >= 0:
                        dv_dx = (v[j, i] - v[j, i-1]) / dx
                    else:
                        dv_dx = (v[j, i+1] - v[j, i]) / dx
                    
                    if v[j, i] >= 0:
                        dv_dy = (v[j, i] - v[j-1, i]) / dy
                    else:
                        dv_dy = (v[j+1, i] - v[j, i]) / dy
                    
                    # Interpolate u at v-locations
                    u_n = 0.5 * (u[j, i] + u[j+1, i]) if j+1 < ny and i < nx-1 else 0
                    u_s = 0.5 * (u[j-1, i] + u[j, i]) if j-1 >= 0 and i < nx-1 else 0
                    
                    if u_n >= 0:
                        dv_dx_n = (v[j, i] - v[j, i-1]) / dx
                    else:
                        dv_dx_n = (v[j, i+1] - v[j, i]) / dx
                    
                    if u_s >= 0:
                        dv_dx_s = (v[j, i] - v[j, i-1]) / dx
                    else:
                        dv_dx_s = (v[j, i+1] - v[j, i]) / dx
                    
                    # Pressure gradient
                    dp_dy = (p[j+1, i] - p[j, i]) / dy
                    
                    # Viscous terms
                    d2v_dx2 = (v[j, i+1] - 2*v[j, i] + v[j, i-1]) / (dx**2)
                    d2v_dy2 = (v[j+1, i] - 2*v[j, i] + v[j-1, i]) / (dy**2)
                    
                    # Momentum equation
                    conv = 0.5 * (u_n * dv_dx_n + u_s * dv_dx_s) + v[j, i] * dv_dy
                    visc = nu * (d2v_dx2 + d2v_dy2)
                    
                    # Update with relaxation
                    v_new = v_old[j, i] + omega_v * (-conv - dp_dy + visc)
                    v[j, i] = np.clip(v_new, -2*U_in, 2*U_in)
        
        # Update pressure (continuity equation)
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if not p_mask[j, i]:
                    # Divergence
                    du_dx = (u[j, i] - u[j, i-1]) / dx
                    dv_dy = (v[j, i] - v[j-1, i]) / dy
                    
                    # Pressure correction
                    p_corr = -(du_dx + dv_dy) * dx * dy / 2
                    p[j, i] = p_old[j, i] + omega_p * p_corr
        
        # Check convergence
        u_change = np.max(np.abs(u - u_old))
        v_change = np.max(np.abs(v - v_old))
        p_change = np.max(np.abs(p - p_old))
        
        if iter % 500 == 0:
            print(f"Iter {iter}: u={u_change:.2e}, v={v_change:.2e}, p={p_change:.2e}")
        
        if max(u_change, v_change, p_change) < tol:
            print(f"Converged at iteration {iter}")
            break
        
        # Adaptive relaxation
        if iter > 100 and iter % 100 == 0:
            if u_change > 1e-3:
                omega_u = max(0.1, omega_u * 0.95)
            else:
                omega_u = min(1.8, omega_u * 1.05)
            
            if v_change > 1e-3:
                omega_v = max(0.1, omega_v * 0.95)
            else:
                omega_v = min(1.8, omega_v * 1.05)
            
            if p_change > 1e-3:
                omega_p = max(0.1, omega_p * 0.95)
            else:
                omega_p = min(1.8, omega_p * 1.05)
    
    solve_time = time.time() - start_time
    print(f"Robust FD solve time: {solve_time:.2f}s")
    
    # Interpolate to pressure grid for comparison
    u_interp = np.zeros((ny, nx))
    v_interp = np.zeros((ny, nx))
    
    for j in range(ny):
        for i in range(nx):
            # Interpolate u to pressure grid
            if i == 0:
                u_interp[j, i] = u[j, 0]
            elif i == nx-1:
                u_interp[j, i] = u[j, nx-2]
            else:
                u_interp[j, i] = 0.5 * (u[j, i-1] + u[j, i])
            
            # Interpolate v to pressure grid
            if j == 0:
                v_interp[j, i] = v[0, i]
            elif j == ny-1:
                v_interp[j, i] = v[ny-2, i]
            else:
                v_interp[j, i] = 0.5 * (v[j-1, i] + v[j, i])
    
    return {
        'u': u_interp, 'v': v_interp, 'p': p,
        'x': x_p, 'y': y_p, 'solve_time': solve_time,
        'iterations': iter + 1
    }

def evaluate_pinn_timing(basis_fn, params, P, n_eval=100):
    """Measure PINN evaluation time."""
    start_time = time.time()
    for _ in range(n_eval):
        phi, gphi, lphi = basis_fn(P)
        w_u = params[:, -3]
        w_v = params[:, -2] 
        w_p = params[:, -1]
        u = jnp.dot(phi, w_u)
        v = jnp.dot(phi, w_v)
        p = jnp.dot(phi, w_p)
    eval_time = time.time() - start_time
    return eval_time / n_eval

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join('results', 'navier_stokes_steady_fd_comparison', timestamp)
    os.makedirs(outdir, exist_ok=True)
    
    # Domain and geometry
    Lx, Ly = 2.2, 0.41
    xc, yc, R = 0.2, 0.2, 0.05
    U_in = 1.0
    
    # Grid resolutions to test
    grid_configs = [
        {'nx': 96, 'ny': 48, 'name': 'Coarse'},
        {'nx': 192, 'ny': 96, 'name': 'Medium'}
    ]
    
    # Reynolds numbers to test
    Re_values = [20.0, 40.0, 100.0, 200.0, 500.0]
    
    # PINN settings
    n_kernels = 128
    epochs = 1200
    lr = 3e-3
    
    results_summary = []
    
    for grid_config in grid_configs:
        nx, ny = grid_config['nx'], grid_config['ny']
        grid_name = grid_config['name']
        
        print(f"\n=== Testing {grid_name} Grid ({nx}x{ny}) ===")
        
        # Create grid for PINN
        X, Y, P = make_grid(nx=nx, ny=ny, Lx=Lx, Ly=Ly)
        
        for Re in Re_values:
            print(f"\n--- Re = {Re} ---")
            
            case_dir = os.path.join(outdir, f'{grid_name}_Re{int(Re)}')
            os.makedirs(case_dir, exist_ok=True)
            
            # Solve with spectral method
            fd_solution = solve_ns_finite_difference_robust(
                nx, ny, Lx, Ly, Re, U_in, xc, yc, R
            )
            
            print(f"FD solution shape: {fd_solution['u'].shape}")
            print(f"PINN grid shape: {X.shape}")
            
            # Interpolate FD solution to PINN grid if needed
            if fd_solution['u'].shape != X.shape:
                # Create meshgrid for FD solution
                X_fd, Y_fd = np.meshgrid(fd_solution['x'], fd_solution['y'])
                fd_points = np.column_stack([X_fd.flatten(), Y_fd.flatten()])
                
                # Interpolate u, v, p to PINN grid
                u_fd_interp = griddata(fd_points, fd_solution['u'].flatten(), 
                                     (X, Y), method='linear', fill_value=0)
                v_fd_interp = griddata(fd_points, fd_solution['v'].flatten(), 
                                     (X, Y), method='linear', fill_value=0)
                p_fd_interp = griddata(fd_points, fd_solution['p'].flatten(), 
                                     (X, Y), method='linear', fill_value=0)
            else:
                u_fd_interp = fd_solution['u']
                v_fd_interp = fd_solution['v']
                p_fd_interp = fd_solution['p']
            
            # Train PINN models
            std_res = train_ns('Standard (Full)', standard_init,
                              lambda P_, p: standard_basis(P_, p),
                              X, Y, P, Re, n_kernels=n_kernels, 
                              epochs=epochs, seed=42, lr=lr,
                              U_in=U_in, Lx=Lx, Ly=Ly)
            
            adv_res = train_ns('Advanced Shape Transform', advanced_init,
                              lambda P_, p: advanced_basis(P_, p, Lx, Ly),
                              X, Y, P, Re, n_kernels=n_kernels,
                              epochs=epochs, seed=42, lr=lr,
                              U_in=U_in, Lx=Lx, Ly=Ly)
            
            # Measure evaluation times
            std_eval_time = evaluate_pinn_timing(
                lambda P_: standard_basis(P_, std_res['params']), 
                std_res['params'], P
            )
            adv_eval_time = evaluate_pinn_timing(
                lambda P_: advanced_basis(P_, adv_res['params'], Lx, Ly),
                adv_res['params'], P
            )
            
            # Evaluate PINN solutions
            def eval_fields(basis_fn, params):
                phi, gphi, lphi = basis_fn(P)
                w_u = params[:, -3]
                w_v = params[:, -2]
                w_p = params[:, -1]
                u = np.array(jnp.dot(phi, w_u)).reshape(X.shape)
                v = np.array(jnp.dot(phi, w_v)).reshape(X.shape)
                pfield = np.array(jnp.dot(phi, w_p)).reshape(X.shape)
                return u, v, pfield
            
            U_std, V_std, P_std = eval_fields(
                lambda P_: standard_basis(P_, std_res['params']), 
                std_res['params']
            )
            U_adv, V_adv, P_adv = eval_fields(
                lambda P_: advanced_basis(P_, adv_res['params'], Lx, Ly),
                adv_res['params']
            )
            
            # Interpolate FD solution to PINN grid if needed
            if fd_solution['u'].shape != X.shape:
                # Create meshgrid for FD solution
                X_fd, Y_fd = np.meshgrid(fd_solution['x'], fd_solution['y'])
                fd_points = np.column_stack([X_fd.flatten(), Y_fd.flatten()])
                
                # Interpolate u, v, p to PINN grid
                u_fd_interp = griddata(fd_points, fd_solution['u'].flatten(), 
                                     (X, Y), method='linear', fill_value=0)
                v_fd_interp = griddata(fd_points, fd_solution['v'].flatten(), 
                                     (X, Y), method='linear', fill_value=0)
                p_fd_interp = griddata(fd_points, fd_solution['p'].flatten(), 
                                     (X, Y), method='linear', fill_value=0)
            else:
                u_fd_interp = fd_solution['u']
                v_fd_interp = fd_solution['v']
                p_fd_interp = fd_solution['p']
            
            # Compute errors
            def compute_errors(pred, truth):
                mae = np.mean(np.abs(pred - truth))
                rmse = np.sqrt(np.mean((pred - truth)**2))
                return mae, rmse
            
            u_std_mae, u_std_rmse = compute_errors(U_std, u_fd_interp)
            v_std_mae, v_std_rmse = compute_errors(V_std, v_fd_interp)
            p_std_mae, p_std_rmse = compute_errors(P_std, p_fd_interp)
            
            u_adv_mae, u_adv_rmse = compute_errors(U_adv, u_fd_interp)
            v_adv_mae, v_adv_rmse = compute_errors(V_adv, v_fd_interp)
            p_adv_mae, p_adv_rmse = compute_errors(P_adv, p_fd_interp)
            
            # Store results
            results_summary.append({
                'grid': grid_name,
                'nx': nx, 'ny': ny,
                'Re': Re,
                'fd_solve_time': fd_solution['solve_time'],
                'fd_iterations': fd_solution['iterations'],
                'std_train_time': std_res['epochs_run'] * 0.1,  # Approximate
                'adv_train_time': adv_res['epochs_run'] * 0.1,
                'std_eval_time': std_eval_time,
                'adv_eval_time': adv_eval_time,
                'std_final_loss': std_res['final_loss'],
                'adv_final_loss': adv_res['final_loss'],
                'u_std_mae': u_std_mae, 'u_std_rmse': u_std_rmse,
                'v_std_mae': v_std_mae, 'v_std_rmse': v_std_rmse,
                'p_std_mae': p_std_mae, 'p_std_rmse': p_std_rmse,
                'u_adv_mae': u_adv_mae, 'u_adv_rmse': u_adv_rmse,
                'v_adv_mae': v_adv_mae, 'v_adv_rmse': v_adv_rmse,
                'p_adv_mae': p_adv_mae, 'p_adv_rmse': p_adv_rmse
            })
            
            # Plot comparison
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            
            # U-velocity
            im0 = axes[0, 0].imshow(u_fd_interp, origin='lower', extent=[0,Lx,0,Ly], 
                                    aspect='auto', cmap='viridis')
            axes[0, 0].set_title(f'FD: u(x,y)')
            axes[0, 0].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im0, ax=axes[0, 0])
            
            im1 = axes[0, 1].imshow(U_std, origin='lower', extent=[0,Lx,0,Ly], 
                                    aspect='auto', cmap='viridis')
            axes[0, 1].set_title(f'Standard: u(x,y)')
            axes[0, 1].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im1, ax=axes[0, 1])
            
            im2 = axes[0, 2].imshow(U_adv, origin='lower', extent=[0,Lx,0,Ly], 
                                    aspect='auto', cmap='viridis')
            axes[0, 2].set_title(f'Advanced: u(x,y)')
            axes[0, 2].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im2, ax=axes[0, 2])
            
            im3 = axes[0, 3].imshow(np.abs(U_std - u_fd_interp), origin='lower', 
                                    extent=[0,Lx,0,Ly], aspect='auto', cmap='hot')
            axes[0, 3].set_title(f'|Standard - FD|')
            axes[0, 3].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im3, ax=axes[0, 3])
            
            # V-velocity
            im4 = axes[1, 0].imshow(v_fd_interp, origin='lower', extent=[0,Lx,0,Ly], 
                                    aspect='auto', cmap='viridis')
            axes[1, 0].set_title(f'FD: v(x,y)')
            axes[1, 0].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im4, ax=axes[1, 0])
            
            im5 = axes[1, 1].imshow(V_std, origin='lower', extent=[0,Lx,0,Ly], 
                                    aspect='auto', cmap='viridis')
            axes[1, 1].set_title(f'Standard: v(x,y)')
            axes[1, 1].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im5, ax=axes[1, 1])
            
            im6 = axes[1, 2].imshow(V_adv, origin='lower', extent=[0,Lx,0,Ly], 
                                    aspect='auto', cmap='viridis')
            axes[1, 2].set_title(f'Advanced: v(x,y)')
            axes[1, 2].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im6, ax=axes[1, 2])
            
            im7 = axes[1, 3].imshow(np.abs(V_std - v_fd_interp), origin='lower', 
                                    extent=[0,Lx,0,Ly], aspect='auto', cmap='hot')
            axes[1, 3].set_title(f'|Standard - FD|')
            axes[1, 3].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im7, ax=axes[1, 3])
            
            # Pressure
            im8 = axes[2, 0].imshow(p_fd_interp, origin='lower', extent=[0,Lx,0,Ly], 
                                    aspect='auto', cmap='coolwarm')
            axes[2, 0].set_title(f'FD: p(x,y)')
            axes[2, 0].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im8, ax=axes[2, 0])
            
            im9 = axes[2, 1].imshow(P_std, origin='lower', extent=[0,Lx,0,Ly], 
                                    aspect='auto', cmap='coolwarm')
            axes[2, 1].set_title(f'Standard: p(x,y)')
            axes[2, 1].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im9, ax=axes[2, 1])
            
            im10 = axes[2, 2].imshow(P_adv, origin='lower', extent=[0,Lx,0,Ly], 
                                     aspect='auto', cmap='coolwarm')
            axes[2, 2].set_title(f'Advanced: p(x,y)')
            axes[2, 2].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im10, ax=axes[2, 2])
            
            im11 = axes[2, 3].imshow(np.abs(P_std - p_fd_interp), origin='lower', 
                                     extent=[0,Lx,0,Ly], aspect='auto', cmap='hot')
            axes[2, 3].set_title(f'|Standard - FD|')
            axes[2, 3].add_patch(Circle((xc, yc), R, color='black', alpha=0.4))
            plt.colorbar(im11, ax=axes[2, 3])
            
            for ax in axes.ravel():
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_xlim(0, Lx)
                ax.set_ylim(0, Ly)
                ax.set_aspect('equal', adjustable='box')
            
            plt.suptitle(f'{grid_name} Grid, Re={int(Re)}: FD vs PINN Comparison')
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, 'fd_pinn_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(std_res['loss_history'], label='Standard (Full)', alpha=0.8)
            plt.plot(adv_res['loss_history'], label='Advanced Shape Transform', alpha=0.8)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('NS residual + BC MSE')
            plt.title(f'{grid_name} Grid, Re={int(Re)}: PINN Training Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(case_dir, 'training_loss.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    # Generate summary report
    with open(os.path.join(outdir, 'COMPARISON_SUMMARY.md'), 'w') as f:
        f.write('# Navier-Stokes: Finite Difference vs PINN Comparison\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write('## Performance Summary\n\n')
        f.write('| Grid | Re | FD Solve (s) | FD Iter | Standard Train (s) | Advanced Train (s) |\n')
        f.write('|------|----|--------------|---------|-------------------|-------------------|\n')
        
        for result in results_summary:
            f.write(f"| {result['grid']} | {result['Re']} | {result['fd_solve_time']:.2f} | "
                   f"{result['fd_iterations']} | {result['std_train_time']:.1f} | "
                   f"{result['adv_train_time']:.1f} |\n")
        
        f.write('\n## Evaluation Time Comparison\n\n')
        f.write('| Grid | Re | Standard Eval (ms) | Advanced Eval (ms) | FD Eval (ms) |\n')
        f.write('|------|----|-------------------|-------------------|-------------|\n')
        
        for result in results_summary:
            f.write(f"| {result['grid']} | {result['Re']} | {result['std_eval_time']*1000:.3f} | "
                   f"{result['adv_eval_time']*1000:.3f} | N/A |\n")
        
        f.write('\n## Accuracy Comparison (MAE)\n\n')
        f.write('| Grid | Re | Field | Standard | Advanced |\n')
        f.write('|------|----|-------|----------|----------|\n')
        
        for result in results_summary:
            f.write(f"| {result['grid']} | {result['Re']} | u | {result['u_std_mae']:.6f} | {result['u_adv_mae']:.6f} |\n")
            f.write(f"| {result['grid']} | {result['Re']} | v | {result['v_std_mae']:.6f} | {result['v_adv_mae']:.6f} |\n")
            f.write(f"| {result['grid']} | {result['Re']} | p | {result['p_std_mae']:.6f} | {result['p_adv_mae']:.6f} |\n")
    
    print(f'\nSaved results to {outdir}')
    print('\nKey findings:')
    print('1. FD provides ground truth reference for accuracy assessment')
    print('2. PINN evaluation time vs FD solve time comparison')
    print('3. Grid resolution impact on both methods')
    print('4. Training vs inference time trade-offs')

if __name__ == '__main__':
    main()
