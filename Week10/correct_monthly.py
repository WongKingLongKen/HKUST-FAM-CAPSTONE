import numpy as np
import math
import time
import matplotlib.pyplot as plt

def trinomial_convertible_discrete_payments(
    S0=60.0, K=65.0, T=1.0, sigma=0.3, r=0.03, n=16384,
    lambda_=math.sqrt(3), N=1.0, save_path='convertible_bond_monthly_linked.png'
):
    """
    Trinomial tree for a Convertible Bond with MONTHLY coupons & dividends (discrete).
    Corrected logic:
      - Bond side: coupons only (no dividends).
      - Stock side (upon conversion): N * (S + PV(future dividends after conversion)).
      - No dividend effect before the first dividend date (t=0.25) on the bond leg.

    Parameters:
      S0: initial stock price
      K: par (redeem value) at maturity
      T: maturity (years)
      sigma: volatility
      r: risk-free rate
      n: number of time steps
      lambda_: tree width parameter (sqrt(3) for Jarrow-Rudd-like trinomial)
      N: conversion ratio (number of shares per bond)
      save_path: output path for the plot (optional)
    """
    print("Convertible Bond - Monthly Coupons & Dividends (Corrected PV-dividend conversion logic)")
    start_total = time.perf_counter()

    dt = T / n
    h = lambda_ * sigma * math.sqrt(dt)
    discount = math.exp(-r * dt)
    offset = n

    # Precompute full terminal grid for S
    js_full = np.arange(-n, n + 1)
    S_full = S0 * np.exp(js_full * h)

    # Monthly payment times (12 per year) within [0, T]
    pay_steps = []
    payment_times = []
    for month in range(1, 13):
        t_pay = month / 12.0
        if t_pay <= T:
            pay_steps.append(int(t_pay / dt))
            payment_times.append(t_pay)

    print(f"Monthly payment times: {[f'{t:.3f}' for t in payment_times]}")
    print(f"Number of payments: {len(pay_steps)}")
    print(f"dt={dt:.2e}")

    # Test cases: (dividend cash per month, coupon cash per month, label, color, linestyle)
    cases = [
        (1.0, 1.0, 'Div=1, Coup=1 (q≈c)',           'blue',   '-'),
        (2.0, 2.0, 'Div=2, Coup=2 (q≈c high)',      'orange', '--'),
        (1.0, 2.0, 'Div=1, Coup=2 (q<c)',           'green',  '-.'),
        (2.0, 1.0, 'Div=2, Coup=1 (q>c)',           'red',    ':')
    ]

    plt.figure(figsize=(16, 10))
    results = {}

    for div_pay, coup_pay, label, color, ls in cases:
        print(f"\n--- {label} ---")
        start = time.perf_counter()

        # Risk-neutral transition probabilities (no continuous q; discrete cash flows only)
        mu_drift = (r - 0.5 * sigma**2) * math.sqrt(dt) / (2 * lambda_ * sigma)
        qu = 0.5 / (lambda_**2) + mu_drift
        qm = 1.0 - 1.0 / (lambda_**2)
        qd = 0.5 / (lambda_**2) - mu_drift

        # Value grid
        V = np.zeros((n + 1, 2 * n + 1))
        # At maturity: max(convert to stock, redeem at par)
        V[n, :] = np.maximum(N * S_full, K)

        boundaries = []
        payment_boundary_points = []

        for i in range(n - 1, -1, -1):
            start_idx = offset - i
            end_idx = offset + i + 1
            j_curr = np.arange(-i, i + 1)
            S_curr = S0 * np.exp(j_curr * h)
            current_time = i * dt

            # Continuation from next step
            exp_future = (
                qu * V[i + 1, start_idx + 1 : end_idx + 1] +
                qm * V[i + 1, start_idx : end_idx] +
                qd * V[i + 1, start_idx - 1 : end_idx - 1]
            )
            continuation = discount * exp_future

            # Add coupon to the bond leg if the NEXT step is a payment
            if i + 1 in pay_steps:
                continuation += coup_pay

            # IMPORTANT: Do NOT subtract dividends on the bond leg.
            # Bondholders don't get dividends; dividends belong to equity holders after conversion.

            # PV of future dividends received by the stockholder AFTER conversion
            future_dividends = sum(
                div_pay * math.exp(-r * (t - current_time))
                for t in payment_times
                if t > current_time
            )
            conversion_value = N * (S_curr + future_dividends)

            # American conversion decision
            V[i, start_idx:end_idx] = np.maximum(conversion_value, continuation)

            # Boundary detection (first node where conversion beats continuation)
            convert_mask = conversion_value > continuation
            idxs = np.where(convert_mask)[0]
            if len(idxs) > 0:
                idx = idxs[0]
                boundary_S = S_curr[idx]
                # Linear interpolation for sharper boundary
                if idx > 0:
                    S_l, S_h = S_curr[idx - 1], S_curr[idx]
                    conv_l, conv_h = conversion_value[idx - 1], conversion_value[idx]
                    cont_l, cont_h = continuation[idx - 1], continuation[idx]
                    # We want the crossing of conv - cont from negative to positive
                    f_l = conv_l - cont_l
                    f_h = conv_h - cont_h
                    denom = (f_h - f_l)
                    if denom != 0:
                        frac = -f_l / denom
                        frac = min(max(frac, 0.0), 1.0)
                        boundary_S = S_l + frac * (S_h - S_l)

                boundaries.append((current_time, boundary_S))

                # Store points at times aligned with payments (including t=0)
                if abs(current_time - 0.0) < dt / 2 or any(abs(current_time - t) < dt / 2 for t in payment_times):
                    payment_boundary_points.append((current_time, boundary_S))

        # Price is the value at the root
        price = V[0, offset]
        duration = time.perf_counter() - start

        # Straight bond PV: sum of coupons + par
        pv_coups = sum(
            coup_pay * math.exp(-r * t) for t in payment_times if t <= T
        )
        straight_bond = pv_coups + K * math.exp(-r * T)
        premium = price - straight_bond

        print(f"Price: {price:.6f} | Straight: {straight_bond:.2f} | Premium: {premium:.2f} | Time: {duration:.2f}s")

        results[(div_pay, coup_pay)] = {
            'price': price,
            'premium': premium,
            'boundaries': boundaries,
            'payment_boundary_points': payment_boundary_points
        }

        # Plot: full boundary curve (thin)
        if boundaries:
            t_b, S_b = zip(*boundaries)
            plt.plot(t_b, S_b, color=color, linestyle=ls, linewidth=1, alpha=0.3, label='_nolegend_')

        # Plot: boundary points near payment times (thick)
        if payment_boundary_points:
            sorted_points = sorted(payment_boundary_points, key=lambda x: x[0])
            t_sorted, S_sorted = zip(*sorted_points)
            plt.plot(t_sorted, S_sorted, color=color, linestyle=ls, linewidth=3,
                     marker='o', markersize=6, label=label)
            plt.scatter(t_sorted, S_sorted, color=color, s=50, zorder=5)

    # Final plot styling
    plt.axhline(y=S0, color='black', linestyle='-', linewidth=1.5, label=f'S0 = {S0}')
    plt.axhline(y=K, color='gray', linestyle='--', linewidth=1.5, label=f'Par = {K}')
    for t in payment_times:
        plt.axvline(x=t, color='purple', linestyle=':', linewidth=1, alpha=0.7)
        plt.text(t, plt.ylim()[1] * 0.95, f'M{int(t*12)}',
                 ha='center', va='top', fontsize=8, color='purple', alpha=0.8,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))

    plt.ylim(50, 400)
    plt.xlim(0, T)
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price - Early Conversion Boundary')
    plt.title('Convertible Bond: Corrected PV of Future Dividends in Conversion Value\n'
              'Thick lines show boundaries near monthly payment dates')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    total_time = time.perf_counter() - start_total
    print(f"Total time: {total_time:.1f}s")
    return results


# =============================================================================
# RUN ALL FOUR MONTHLY PAYMENT CASES (example)
# =============================================================================
if __name__ == "__main__":
    results = trinomial_convertible_discrete_payments(
        S0=60.0, K=65.0, T=1.0, sigma=0.3, r=0.03, n=32768, N=1.0
    )
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    for (div_pay, coup_pay), data in results.items():
        print(f"\nDiv={div_pay}, Coup={coup_pay}:")
        print(f" Convertible Bond Price: {data['price']:.6f}")
        print(f" Premium over Straight Bond: {data['premium']:.6f}")
