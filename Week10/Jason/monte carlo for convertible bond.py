import matplotlib.pyplot as plt
import numpy as np

# Parameters
face_value = 1000
conversion_price = 50
conversion_ratio = face_value / conversion_price
r = 0.05
q = 0.02  # Continuous dividend yield
T = 1.0
M = 1000
dt = T / M
coupon_rate = 0.04
N = 32768
sigma = 0.2
S0 = 50

def quadratic_basis(S):
    return np.vstack([np.ones(len(S)), S, S**2]).T

# Simulate GBM paths (adjusted for dividend yield q)
np.random.seed(0)
Z = np.random.standard_normal((N, M))
S = np.zeros((N, M + 1))
S[:, 0] = S0
for t in range(1, M + 1):
    S[:, t] = S[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])

# Initialize bond values at maturity
bond_values = np.maximum(face_value, conversion_ratio * S[:, -1])

exercise_boundary = np.zeros(M + 1)

for t in range(M - 1, -1, -1):
    accrued_coupons = coupon_rate * dt * face_value
    discount_factor = np.exp(-r * dt)

    # Discount bond values from next step
    discounted_bond_values = discount_factor * bond_values + accrued_coupons
    Y = discounted_bond_values

    S_t = S[:, t]

    # Quadratic regression
    X = quadratic_basis(S_t)
    coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    cont_val = X @ coeffs

    conversion_values = conversion_ratio * S_t

    # Exercise decision
    exercise = conversion_values >= cont_val
    bond_values = np.where(exercise, conversion_values, discounted_bond_values)

    # Exercise boundary estimation: max S_t in hold region
    hold_indices = np.where(~exercise)[0]
    exercise_boundary[t] = np.max(S_t[hold_indices]) if len(hold_indices) > 0 else 0

exercise_boundary[-1] = conversion_price

# Plot exercise boundary
times = np.linspace(0, T, M + 1)
plt.plot(times, exercise_boundary, label='Exercise Boundary')
plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.title('Exercise Boundary for Convertible Bond (Quadratic Basis, q=0.02)')
plt.grid(True)
plt.legend()
plt.show()

# Print selected boundary values
print("Selected Exercise Boundary Values:")
print("| Time (years) | Exercise Boundary (Stock Price) |")
print("|--------------|---------------------------------|")
for i in range(0, M+1, M//10):
    print(f"| {times[i]:.2f}        | {exercise_boundary[i]:.4f}                        |")