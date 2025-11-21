import numpy as np
import math
import time
import matplotlib.pyplot as plt

def trinomial_convertible_discrete_payments(
    S0=60.0, K=65.0, T=1.0, sigma=0.3, r=0.03, n=16384,
    lambda_=math.sqrt(3), save_path='convertible_bond_monthly_linked.png'
):
    """
    Trinomial tree for Convertible Bond with MONTHLY coupons & dividends
    Shows linked exercise boundaries right after each payment
    """
    print("Convertible Bond - Monthly Coupons & Dividends with Linked Boundaries")
    start_total = time.perf_counter()

    dt = T / n
    h = lambda_ * sigma * math.sqrt(dt)
    discount = math.exp(-r * dt)
    offset = n

    # Precompute full terminal grid
    js_full = np.arange(-n, n + 1)
    S_full = S0 * np.exp(js_full * h)

    # MONTHLY payment times (in steps) - 12 payments per year
    monthly_payments = []
    payment_times = []
    for month in range(1, 13):  # 1 to 12 months
        payment_time = month / 12.0  # Monthly intervals
        if payment_time <= T:  # Only include payments within bond maturity
            monthly_payments.append(int(payment_time / dt))
            payment_times.append(payment_time)
    
    pay_steps = monthly_payments
    print(f"Monthly payment times: {[f'{t:.3f}' for t in payment_times]}")
    print(f"Number of payments: {len(pay_steps)}")
    print(f"dt={dt:.2e}")

    cases = [
        (1.0, 1.0, 'Div=1, Coup=1 (q≈c)',    'blue',   '-'),
        (2.0, 2.0, 'Div=2, Coup=2 (q≈c high)', 'orange', '--'),
        (1.0, 2.0, 'Div=1, Coup=2 (q<c)',     'green',  '-.'),
        (2.0, 1.0, 'Div=2, Coup=1 (q>c)',     'red',    ':')
    ]

    plt.figure(figsize=(16, 10))
    results = {}

    for div_pay, coup_pay, label, color, ls in cases:
        print(f"\n--- {label} ---")
        start = time.perf_counter()

        # Drift: NO continuous q/c, only discrete → use r - 0.5σ² (pure risk-neutral)
        mu_drift = (r - 0.5 * sigma**2) * math.sqrt(dt) / (2 * lambda_ * sigma)
        qu = 0.5 / (lambda_**2) + mu_drift
        qm = 1 - 1 / (lambda_**2)
        qd = 0.5 / (lambda_**2) - mu_drift

        V = np.zeros((n + 1, 2 * n + 1))
        V[n, :] = np.maximum(S_full, K)  # At maturity: max(S, K)

        boundaries = []
        boundaries_after_payments = []  # Store boundaries right after payments
        payment_boundary_points = []    # Store exact boundary points at payment times

        for i in range(n - 1, -1, -1):
            start_idx = offset - i
            end_idx = offset + i + 1
            j_curr = np.arange(-i, i + 1)
            S_curr = S0 * np.exp(j_curr * h)

            # Expected future value
            exp_future = (
                qu * V[i + 1, start_idx + 1 : end_idx + 1] +
                qm * V[i + 1, start_idx : end_idx] +
                qd * V[i + 1, start_idx - 1 : end_idx - 1]
            )

            continuation = discount * exp_future

            # Track if this is right after a payment
            is_after_payment = i in pay_steps
            
            # Add discrete coupon at MONTHLY payment times
            if i + 1 in pay_steps:  # next step is payment
                continuation += coup_pay

            # Add discrete dividend: reduces continuation value (holder loses div if converts)
            if i + 1 in pay_steps:
                continuation -= div_pay  # dividend goes to equity holder

            # American conversion
            V[i, start_idx:end_idx] = np.maximum(S_curr, continuation)

            # Find early conversion boundary
            convert = S_curr > continuation
            idxs = np.where(convert)[0]
            current_time = i * dt
            
            if len(idxs) > 0:
                idx = idxs[0]
                boundary_S = S_curr[idx]
                # Interpolation for more accuracy
                if idx > 0:
                    S_l, S_h = S_curr[idx-1], S_curr[idx]
                    c_l, c_h = continuation[idx-1], continuation[idx]
                    if c_l > S_l:
                        frac = (c_l - S_l) / ((c_h - S_h) - (c_l - S_l) + 1e-12)
                        boundary_S = S_l + frac * (S_h - S_l)
                
                boundaries.append((current_time, boundary_S))
                
                # Store boundary points specifically right after payments
                if is_after_payment:
                    boundaries_after_payments.append((current_time, boundary_S))
                    payment_boundary_points.append((current_time, boundary_S))
                
                # Also store boundaries at exact payment times for linking
                if abs(current_time - 0.0) < dt/2 or any(abs(current_time - t) < dt/2 for t in payment_times):
                    payment_boundary_points.append((current_time, boundary_S))

        price = V[0, offset]
        duration = time.perf_counter() - start

        # Straight bond value (MONTHLY coupons + par)
        pv_coups = 0
        for month in range(1, 13):
            payment_time = month / 12.0
            if payment_time <= T:
                pv_coups += coup_pay * math.exp(-r * payment_time)
        straight_bond = pv_coups + K * math.exp(-r * T)
        premium = price - straight_bond

        print(f"Price: {price:.6f} | Straight: {straight_bond:.2f} | Premium: {premium:.2f} | Time: {duration:.2f}s")
        
        # Report boundaries at payment times
        print("Exercise Boundaries RIGHT AFTER payments:")
        for payment_time in payment_times + [0.0]:
            # Find closest boundary point to each payment time
            closest_points = []
            for t, s in payment_boundary_points:
                if abs(t - payment_time) < dt * 1.5:  # Allow some tolerance
                    closest_points.append((t, s))
            
            if closest_points:
                # Take the one with time closest to payment_time
                closest_point = min(closest_points, key=lambda x: abs(x[0] - payment_time))
                print(f"  t={payment_time:.3f}: S ≥ {closest_point[1]:.2f}")
            else:
                print(f"  t={payment_time:.3f}: No early conversion")
        
        if boundaries:
            b_min, b_max = min(b[1] for b in boundaries), max(b[1] for b in boundaries)
            print(f"Overall Boundary Range: {b_min:.1f} → {b_max:.1f}")
        else:
            print("No early conversion")

        results[(div_pay, coup_pay)] = {
            'price': price, 
            'premium': premium, 
            'boundaries': boundaries,
            'boundaries_after_payments': boundaries_after_payments,
            'payment_boundary_points': payment_boundary_points
        }

        # Plot 1: Full boundary curve
        if boundaries:
            t_b, S_b = zip(*boundaries)
            plt.plot(t_b, S_b, color=color, linestyle=ls, linewidth=1, alpha=0.3, label='_nolegend_')
        
        # Plot 2: Linked boundaries at payment points (thicker, more prominent)
        if payment_boundary_points:
            t_pay, S_pay = zip(*payment_boundary_points)
            # Sort by time for proper linking
            sorted_points = sorted(payment_boundary_points, key=lambda x: x[0])
            t_sorted, S_sorted = zip(*sorted_points)
            
            # Plot linked line (thicker and more prominent)
            plt.plot(t_sorted, S_sorted, color=color, linestyle=ls, linewidth=3, 
                    marker='o', markersize=6, label=label + ' (After Payments)')
            
            # Highlight the points
            plt.scatter(t_sorted, S_sorted, color=color, s=50, zorder=5)

    # Final plot with enhanced markers
    plt.axhline(y=S0, color='black', linestyle='-', linewidth=1.5, label=f'S0 = {S0}')
    plt.axhline(y=K, color='gray', linestyle='--', linewidth=1.5, label=f'Par = {K}')
    
    # Mark monthly payment dates with vertical lines
    for payment_time in payment_times:
        plt.axvline(x=payment_time, color='purple', linestyle=':', linewidth=1, alpha=0.7)
    
    # Add text labels for payment months
    for i, payment_time in enumerate(payment_times):
        plt.text(payment_time, plt.ylim()[1] * 0.95, f'M{i+1}', 
                ha='center', va='top', fontsize=8, color='purple', alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))

    plt.ylim(50, 400)
    plt.xlim(0, T)
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price - Early Conversion Boundary')
    plt.title('Convertible Bond: Linked Exercise Boundaries RIGHT AFTER Monthly Payments\n'
              'Thick lines show boundaries immediately after coupon/dividend payments\n'
              'Purple dotted: Monthly payment dates | M1-M12: Month numbers')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    total_time = time.perf_counter() - start_total
    print(f"Total time: {total_time:.1f}s")

    return results

# =============================================================================
# RUN ALL FOUR MONTHLY PAYMENT CASES WITH LINKED BOUNDARIES
# =============================================================================
results = trinomial_convertible_discrete_payments(
    S0=60.0, K=65.0, T=1.0, sigma=0.3, r=0.03, n=32768
)
