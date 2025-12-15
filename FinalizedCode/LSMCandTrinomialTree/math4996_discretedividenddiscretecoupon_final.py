import numpy as np
import time
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# ==================== MONTE CARLO WITH DISCRETE DIVIDENDS ====================

def rational_basis(S):
    """Basis function for regression"""
    eps = 1e-10
    return np.vstack([
        np.ones(len(S)),
        S,
        S**2,
        1 / (1 + S + eps),
        1 / (1 + S**2 + eps)
    ]).T

def simulate_paths_discrete(S0, r, sigma, T, M, N, dividend_times, dividend_amounts):
    """Simulate stock paths with discrete dividends (no price drop, PV handled separately)"""
    dt = T / M
    increments = np.random.normal((r - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), (N, M))
    log_S = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_S)
    S = np.hstack((S0 * np.ones((N, 1)), S))
    return S

def calculate_pv_future_dividends(current_time, div_times, div_amounts, r):
    """Calculate present value of future dividends strictly after current_time"""
    pv = 0
    for t, amount in zip(div_times, div_amounts):
        if t > current_time:
            pv += amount * math.exp(-r * (t - current_time))
    return pv

def lsmc_exercise_boundary(S0=60.0, K=65.0, T=1.0, sigma=0.3, r=0.03,
                          paths=50000, steps=252, N=1.0, frequency='monthly', plot=False):
    """
    LSMC for Exercise Boundary only (q < c case)
    Dividends and Coupons scaled per frequency
    Boundaries computed and recorded only at dividend payment dates.
    If plot=True, generates individual plot; else returns data for external plotting.
    """
    print("="*80)
    print(f"LSMC - EXERCISE BOUNDARY CALCULATION ({frequency.upper()})")
    print("Case: q < c (Dividend < Coupon)")
    print("="*80)
    
    # Dividend configurations by frequency (annual total $1.20)
    annual_div = 1.20
    if frequency == 'monthly':
        monthly_dividend = annual_div / 12
        div_times = [round(i/12, 4) for i in range(1, 13) if i/12 <= T]
        div_amounts = [monthly_dividend] * len(div_times)
        print(f"Dividends: Monthly ${monthly_dividend:.2f}")
    elif frequency == 'quarterly':
        quarterly_dividend = annual_div / 4
        div_times = [round(i/4, 4) for i in range(1, 5) if i/4 <= T]
        div_amounts = [quarterly_dividend] * len(div_times)
        print(f"Dividends: Quarterly ${quarterly_dividend:.2f}")
    elif frequency == 'semi_annual':
        semi_dividend = annual_div / 2
        div_times = [0.5, 1.0]
        div_amounts = [semi_dividend] * len(div_times)
        print(f"Dividends: Semi-annual ${semi_dividend:.2f}")
    else:
        raise ValueError("Frequency must be 'monthly', 'quarterly', or 'semi_annual'")
    
    # Coupon configurations by frequency (annual total $2.40)
    annual_coup = 2.40
    if frequency == 'monthly':
        monthly_coupon = annual_coup / 12
        coup_times = div_times[:]
        coup_amounts = [monthly_coupon] * len(coup_times)
        print(f"Coupons: Monthly ${monthly_coupon:.2f}")
    elif frequency == 'quarterly':
        quarterly_coupon = annual_coup / 4
        coup_times = div_times[:]
        coup_amounts = [quarterly_coupon] * len(coup_times)
        print(f"Coupons: Quarterly ${quarterly_coupon:.2f}")
    elif frequency == 'semi_annual':
        semi_coupon = annual_coup / 2
        coup_times = div_times[:]
        coup_amounts = [semi_coupon] * len(coup_times)
        print(f"Coupons: Semi-annual ${semi_coupon:.2f}")
    
    dt = T / steps
    discount = np.exp(-r * dt)
    
    print(f"Dividend/Coupon times ({frequency}): {[f'{t:.3f}' for t in div_times]}")
    print(f"Paths: {paths:,}, Steps: {steps}")
    
    # Simulate paths (no dividend drop in price path)
    start_time = time.perf_counter()
    S = simulate_paths_discrete(S0, r, sigma, T, steps, paths, 
                                div_times, div_amounts)
    
    # Map steps
    div_steps_set = set(int(round(tt / dt)) for tt in div_times)
    coup_steps = {int(round(tt / dt)): amt for tt, amt in zip(coup_times, coup_amounts)}
    
    # Payoff matrix
    V = np.zeros_like(S)
    
    # Terminal condition
    final_coup = coup_steps.get(steps, 0)
    V[:, steps] = np.maximum(N * S[:, steps], K + final_coup)
    
    # Store boundaries (only at dividend dates)
    boundaries = []
    
    # Backward induction (full American every step, boundary only at div dates)
    for t in range(steps - 1, -1, -1):
        current_time = t * dt
        
        # Continuation value
        continuation_base = discount * V[:, t + 1]
        
        # Add coupon if NEXT step is a coupon payment step
        coup_amount = coup_steps.get(t + 1, 0)
        continuation = continuation_base + coup_amount
        
        # PV of future dividends (strictly after current_time)
        pv_future_div = calculate_pv_future_dividends(current_time, div_times, div_amounts, r)
        
        # Conversion value
        conversion = N * (S[:, t] + pv_future_div)
        
        # Default: continue
        V[:, t] = continuation
        
        # ITM paths for regression (lowered threshold for more early boundaries)
        itm = S[:, t] > (K / N) * 0.1  # Changed from 0.5 to 0.1
        exercised = False
        boundary_price = np.nan
        
        if np.sum(itm) > 5:  # Lowered from 10 to 5
            X = S[:, t][itm]
            Y = continuation[itm]
            A = rational_basis(X)
            ridge = Ridge(alpha=0.1, fit_intercept=False)
            ridge.fit(A, Y)
            cont_fitted = A @ ridge.coef_
            
            # Exercise decision
            exercise_now = conversion[itm] > cont_fitted
            if np.any(exercise_now):
                exercised = True
                V[:, t][itm] = np.where(exercise_now, conversion[itm], Y)
                min_ex_S = np.min(X[exercise_now])
            
            # Boundary detection ONLY at dividend dates using bisection
            if t in div_steps_set:
                def boundary_func(S_test):
                    basis = rational_basis(np.array([S_test]))
                    cont_val = basis @ ridge.coef_
                    return N * (S_test + pv_future_div) - cont_val
                
                S_low = (K / N) * 0.05  # Lowered low bound
                S_high = S0 * 5.0  # Increased high bound
                f_low = boundary_func(S_low)
                f_high = boundary_func(S_high)
                
                if f_low * f_high < 0:  # Sign change
                    # Bisection for precision
                    for _ in range(30):
                        S_mid = (S_low + S_high) / 2
                        f_mid = boundary_func(S_mid)
                        if f_low * f_mid <= 0:
                            S_high = S_mid
                            f_high = f_mid
                        else:
                            S_low = S_mid
                            f_low = f_mid
                    boundary_price = (S_low + S_high) / 2
                else:
                    # No crossing: determine based on signs
                    if f_low > 0:  # Exercise even at low S
                        boundary_price = S_low
                    elif f_high > 0:  # Exercise above high, but cap at high
                        boundary_price = S_high
                    else:  # Never exercise: set to very high value to indicate impractical
                        boundary_price = S_high * 2  # e.g., 600, beyond plot ylim
                    
                    # Override with min_ex_S if exercised
                    if exercised and (np.isnan(boundary_price) or boundary_price > min_ex_S):
                        boundary_price = min_ex_S
                
                if not np.isnan(boundary_price):
                    boundaries.append((current_time, boundary_price, pv_future_div))
        else:
            # Few ITM paths: direct exercise check (and attempt boundary)
            direct_exercise = conversion > continuation
            V[:, t] = np.maximum(conversion, continuation)
            if t in div_steps_set:
                if np.any(direct_exercise):
                    boundary_price = np.min(S[:, t][direct_exercise])
                else:
                    # Force a boundary estimate: use bisection with all paths for regression
                    X_all = S[:, t]
                    Y_all = continuation
                    A_all = rational_basis(X_all)
                    ridge_all = Ridge(alpha=0.1, fit_intercept=False)
                    ridge_all.fit(A_all, Y_all)
                    
                    def boundary_func_all(S_test):
                        basis = rational_basis(np.array([S_test]))
                        cont_val = basis @ ridge_all.coef_
                        return N * (S_test + pv_future_div) - cont_val
                    
                    S_low = (K / N) * 0.05
                    S_high = S0 * 5.0
                    f_low = boundary_func_all(S_low)
                    f_high = boundary_func_all(S_high)
                    
                    if f_low * f_high < 0:
                        for _ in range(30):
                            S_mid = (S_low + S_high) / 2
                            f_mid = boundary_func_all(S_mid)
                            if f_low * f_mid <= 0:
                                S_high = S_mid
                                f_high = f_mid
                            else:
                                S_low = S_mid
                                f_low = f_mid
                        boundary_price = (S_low + S_high) / 2
                    else:
                        if f_low > 0:
                            boundary_price = S_low
                        elif f_high > 0:
                            boundary_price = S_high
                        else:
                            boundary_price = S_high * 2  # Very high to indicate never optimal
                
                if not np.isnan(boundary_price):
                    boundaries.append((current_time, boundary_price, pv_future_div))
    
    # Add terminal boundary if applicable
    if steps in div_steps_set:
        current_time = T
        pv_future_div_term = 0.0
        final_coup = coup_steps.get(steps, 0)
        boundary_term = (K + final_coup) / N
        boundaries.append((current_time, boundary_term, pv_future_div_term))
        print(f"Added terminal boundary at t={T}: ${boundary_term:.2f}")
    
    # Price at t=0
    current_time = 0.0
    continuation_base = discount * V[:, 1]
    coup_amount = coup_steps.get(1, 0)
    continuation = continuation_base + coup_amount
    pv_future_div = calculate_pv_future_dividends(0.0, div_times, div_amounts, r)
    conversion = N * (S[:, 0] + pv_future_div)
    V[:, 0] = np.maximum(conversion, continuation)
    price = np.mean(V[:, 0])
    
    duration = time.perf_counter() - start_time
    print(f"\nBond Price: ${price:.4f}")
    print(f"Calculation time: {duration:.2f} seconds")
    
    # Straight bond value (PV coupons + par)
    pv_coupons = sum(c * math.exp(-r * tt) for c, tt in zip(coup_amounts, coup_times))
    straight_bond = pv_coupons + K * math.exp(-r * T)
    premium = price - straight_bond
    print(f"Straight Bond: ${straight_bond:.4f}")
    print(f"Conversion Premium: ${premium:.4f}")
    
    # ==================== EXERCISE BOUNDARY DATA PREP ====================
    
    # Filter valid boundaries (already only at div payments)
    valid_boundaries = [(t, p, pv) for t, p, pv in boundaries if not np.isnan(p)]
    
    if valid_boundaries:
        times, prices, pvs = zip(*valid_boundaries)
        
        print(f"\nExercise Boundary Points (at {frequency} payment dates):")
        print("-" * 70)
        print("Time (years)\tStock Price\tPV(Future Div)")
        print("-" * 70)
        for t, p, pv in sorted(valid_boundaries, key=lambda x: x[0]):
            note = " (Very High - Rarely Optimal)" if p > 200 else ""
            print(f"{t:.4f}\t\t${p:.2f}{note}\t\t${pv:.3f}")
        
        # Statistics
        print("\n" + "="*60)
        print("BOUNDARY STATISTICS:")
        print(f"Number of boundary points: {len(valid_boundaries)}")
        print(f"Minimum boundary: ${min(prices):.2f}")
        print(f"Maximum boundary: ${max(prices):.2f}")
        print(f"Average boundary: ${np.mean(prices):.2f}")
        
        # Data for plotting
        plot_data = {
            'frequency': frequency,
            'times': list(times),
            'prices': list(prices),
            'pvs': list(pvs),
            'div_times': div_times,
            'annual_div': annual_div,
            'annual_coup': annual_coup,
            'has_high_boundary': max(prices) > 200  # Flag for note in plot
        }
    else:
        print("No exercise boundaries found - never optimal to exercise early")
        plot_data = None
    
    if plot:  # Legacy individual plot
        plt.figure(figsize=(12, 8))
        
        if plot_data:
            times, prices, pvs = plot_data['times'], plot_data['prices'], plot_data['pvs']
            has_high = plot_data['has_high_boundary']
            
            # Plot exercise boundary as points only (no connecting line)
            plt.scatter(times, prices, s=100, c='red', marker='o', 
                       label='Exercise Boundary', zorder=5)
            
            # Add note as text box if high boundaries present
            if has_high:
                plt.text(0.98, 0.98, '(Very High - Rarely Optimal)\nfor Early Dates', 
                         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                         horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Reference lines
        plt.axhline(y=K / N, color='black', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Strike Price (K/N=${K/N:.1f})')
        plt.axhline(y=S0, color='green', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Initial Price (S0=${S0})')
        
        # Mark payment dates (div and coup same)
        for t in div_times:
            plt.axvline(x=t, color='orange', linestyle=':', linewidth=1, alpha=0.7)
            plt.text(t, 50 * 1.02, f't={t:.3f}', 
                    ha='center', va='bottom', fontsize=8, color='orange')
        
        # Plot formatting
        plt.xlabel('Time to Maturity (years)', fontsize=12)
        plt.ylabel('Stock Price at Exercise Boundary', fontsize=12)
        plt.title(f'Convertible Bond Exercise Boundary ({frequency.title()} Dividends & Coupons)\n' +
                  f'Case: q < c (Annual Div=${annual_div}, Annual Coup=${annual_coup})', 
                  fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Fixed y-axis limits (high boundaries clipped)
        plt.ylim(50, 200)
        
        plt.xlim(0, T)
        plt.tight_layout()
        plt.show()
    
    return plot_data

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Parameters from trinomial tree
    S0 = 60.0
    K = 65.0
    T = 1.0
    sigma = 0.3
    r = 0.03  # Risk-free rate
    N = 1.0  # Conversion ratio
    
    # Run for different frequencies (dividends and coupons match)
    frequencies = ['semi_annual', 'quarterly', 'monthly']
    plot_data_list = []
    for freq in frequencies:
        # Calculate exercise boundary (no individual plot)
        plot_data = lsmc_exercise_boundary(
            S0=S0,
            K=K,
            T=T,
            sigma=sigma,
            r=r,
            paths=50000,
            steps=252,
            N=N,
            frequency=freq,
            plot=False  # Disable individual plotting
        )
        if plot_data:
            plot_data_list.append(plot_data)
        
        # Interpretation
        if plot_data:
            print("\n" + "="*80)
            print("INTERPRETATION")
            print("="*80)
            print("The exercise boundary shows the minimum stock price at which")
            print("it's optimal to convert the bond to stock at each payment date.")
            print(f"\nFor {freq.replace('_', ' ').title()} dividends and coupons (q < c):")
            print("- Coupons provide higher income than dividends")
            print("- Therefore, need higher stock price to justify conversion")
            print("- Very high boundaries indicate early exercise is rarely optimal due to time value")
            print("- Boundary typically increases as time to maturity decreases")
        print("\n" + "="*80 + "\n")
    
    # ==================== COMBINED PLOT ====================
    if plot_data_list:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        freq_names = ['Semi-Annual', 'Quarterly', 'Monthly']
        
        for i, (plot_data, freq_name) in enumerate(zip(plot_data_list, freq_names)):
            ax = axes[i]
            times = plot_data['times']
            prices = plot_data['prices']
            pvs = plot_data['pvs']
            div_times = plot_data['div_times']
            annual_div = plot_data['annual_div']
            annual_coup = plot_data['annual_coup']
            has_high = plot_data['has_high_boundary']
            
            # Plot exercise boundary as points only (no connecting line)
            ax.scatter(times, prices, s=100, c='red', marker='o', 
                       label='Exercise Boundary', zorder=5)
            
            # Add note as text box if high boundaries present
            
            # Reference lines
            ax.axhline(y=K / N, color='black', linestyle='--', linewidth=2, 
                       alpha=0.7, label=f'Strike Price (K/N=${K/N:.1f})')
            ax.axhline(y=S0, color='green', linestyle='--', linewidth=2,
                       alpha=0.7, label=f'Initial Price (S0=${S0})')
            
            # Mark payment dates (div and coup same)
            for t in div_times:
                ax.axvline(x=t, color='orange', linestyle=':', linewidth=1, alpha=0.7)
                ax.text(t, 50 * 1.02, f't={t:.3f}', 
                        ha='center', va='bottom', fontsize=8, color='orange')
            
            # Plot formatting
            ax.set_xlabel('Time to Maturity (years)', fontsize=12)
            ax.set_ylabel('Stock Price at Exercise Boundary', fontsize=12)
            ax.set_title(f'{freq_name} Dividends & Coupons\n' +
                         f'Case: q < c (Annual Div=${annual_div}, Annual Coup=${annual_coup})', 
                         fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Fixed y-axis limits (high boundaries clipped)
            ax.set_ylim(50, 200)
            ax.set_xlim(0, T)
        
        plt.tight_layout()
        plt.savefig('exercise_boundaries_combined.png', dpi=300, bbox_inches='tight')
        print("\nCombined plot saved as 'exercise_boundaries_combined.png'")
        plt.show()  # Optional: show the combined plot