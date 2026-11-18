import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# Data
# =============================================================================

N = np.array([
    50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
    750, 1250, 1500, 1750, 2000,
], dtype=float)

error = np.array([
    43.73718697, 45.8619237, 39.2436604, 51.13526775, 36.81341006,
    45.36160429, 45.73468768, 34.44849056, 41.3634181, 42.43903752,
    28.25122875, 24.75207851, 24.90991354, 22.4, 22.9,
], dtype=float)
# =============================================================================
# Model definitions
# =============================================================================

# 1. Power law with offset
# E(N) = a * N^b + c
def power_law_offset(N, a, b, c):
    return a * N**b + c


# 2. Inverse power law
# E(N) = a / N^p + c
def inverse_power_offset(N, a, p, c):
    return a / (N**p) + c


# 3. Square-root decay
# E(N) = a / sqrt(N) + c
def sqrt_decay_offset(N, a, c):
    return a / np.sqrt(N) + c


# 4. Inverse-linear decay
# E(N) = a / N + c
def inverse_linear_offset(N, a, c):
    return a / N + c


# 5. Exponential decay
# E(N) = a * exp(-kN) + c
def exponential_offset(N, a, k, c):
    return a * np.exp(-k * N) + c


# 6. Stretched exponential
# E(N) = a * exp(-k * N^p) + c
def stretched_exponential_offset(N, a, k, p, c):
    return a * np.exp(-k * N**p) + c


# 7. Logarithmic inverse decay
# E(N) = a / log(N + d) + c
def log_inverse_offset(N, a, d, c):
    return a / np.log(N + d) + c


# 8. Rational decay
# E(N) = a / (N + b) + c
def rational_offset(N, a, b, c):
    return a / (N + b) + c


# 9. Hill decay / saturation-style decay
# E(N) = c + a / (1 + (N / K)^p)
def hill_decay_offset(N, a, K, p, c):
    return c + a / (1 + (N / K)**p)


# 10. Two-term power law
# E(N) = c + a * N^(-p) + d * N^(-q)
def two_term_power_offset(N, a, p, d, q, c):
    return c + a * N**(-p) + d * N**(-q)


# =============================================================================
# Fit statistics
# =============================================================================

def fit_stats(y_true, y_pred, num_params):
    residuals = y_true - y_pred
    sse = np.sum(residuals**2)
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)

    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - sse / ss_tot

    n = len(y_true)

    # AIC and BIC for least-squares Gaussian residual assumption
    # Add tiny value to avoid log(0)
    eps = 1e-12
    aic = n * np.log(sse / n + eps) + 2 * num_params
    bic = n * np.log(sse / n + eps) + num_params * np.log(n)

    return {
        "R2": r2,
        "RMSE": rmse,
        "SSE": sse,
        "AIC": aic,
        "BIC": bic,
    }


# =============================================================================
# Models to fit
# =============================================================================
# Important:
# c is unbounded in every model below.
# Some non-c parameters still have practical bounds to avoid invalid math,
# for example N + d must be positive for log(N + d), and K must be positive.

models = [
    {
        "name": "Power law offset",
        "func": power_law_offset,
        "p0": [50000, -0.7, 0],
        "bounds": (
            [-np.inf, -np.inf, -np.inf],
            [ np.inf,  np.inf,  np.inf],
        ),
        "equation": "E(N) = a N^b + c",
    },
    {
        "name": "Inverse power offset",
        "func": inverse_power_offset,
        "p0": [50000, 0.7, 0],
        "bounds": (
            [-np.inf, -np.inf, -np.inf],
            [ np.inf,  np.inf,  np.inf],
        ),
        "equation": "E(N) = a / N^p + c",
    },
    {
        "name": "Square-root decay offset",
        "func": sqrt_decay_offset,
        "p0": [20000, 0],
        "bounds": (
            [-np.inf, -np.inf],
            [ np.inf,  np.inf],
        ),
        "equation": "E(N) = a / sqrt(N) + c",
    },
    {
        "name": "Inverse-linear offset",
        "func": inverse_linear_offset,
        "p0": [200000, 0],
        "bounds": (
            [-np.inf, -np.inf],
            [ np.inf,  np.inf],
        ),
        "equation": "E(N) = a / N + c",
    },
    {
        "name": "Exponential offset",
        "func": exponential_offset,
        "p0": [1800, 0.001, 100],
        "bounds": (
            [-np.inf, -np.inf, -np.inf],
            [ np.inf,  np.inf,  np.inf],
        ),
        "equation": "E(N) = a exp(-kN) + c",
    },
    {
        "name": "Stretched exponential offset",
        "func": stretched_exponential_offset,
        "p0": [20, 0.01, 0.5, 22],       # a>0, small k, p~0.5, c near plateau
        "bounds": (
            [0,     0,    0,   0],         # all positive: a, k, p, c
            [np.inf, np.inf, 2, 40],       # p<2 keeps shape physical, c<40
        ),
        "equation": "E(N) = a exp(-k N^p) + c",
    },
    {
        "name": "Log inverse offset",
        "func": log_inverse_offset,
        "p0": [10000, 1, 0],
        "bounds": (
            [-np.inf, -99.999, -np.inf],  # Need N + d > 0; min N is 100
            [ np.inf,  np.inf,  np.inf],
        ),
        "equation": "E(N) = a / log(N + d) + c",
    },
    {
        "name": "Rational offset",
        "func": rational_offset,
        "p0": [200000, 100, 100],
        "bounds": (
            [-np.inf, -99.999, -np.inf],  # Avoid N + b = 0 near min N
            [ np.inf,  np.inf,  np.inf],
        ),
        "equation": "E(N) = a / (N + b) + c",
    },
    {
        "name": "Hill decay offset",
        "func": hill_decay_offset,
        "p0": [1800, 700, 2, 100],
        "bounds": (
            [-np.inf, 1e-12, -np.inf, -np.inf],  # K must be nonzero positive
            [ np.inf, np.inf, np.inf,  np.inf],
        ),
        "equation": "E(N) = c + a / (1 + (N/K)^p)",
    },
    {
        "name": "Two-term power offset",
        "func": two_term_power_offset,
        "p0": [50000, 0.7, 10000, 0.2, 100],
        "bounds": (
            [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
            [ np.inf,  np.inf,  np.inf,  np.inf,  np.inf],
        ),
        "equation": "E(N) = c + a N^(-p) + d N^(-q)",
    },
]

# =============================================================================
# Fit all models
# =============================================================================

results = []

x_fit = np.linspace(N.min(), N.max(), 1000)

for model in models:
    name = model["name"]
    func = model["func"]
    p0 = model["p0"]
    bounds = model["bounds"]

    print("\n" + "=" * 80)
    print(f"Fitting: {name}")
    print(model["equation"])

    try:
        popt, pcov = curve_fit(
            func,
            N,
            error,
            p0=p0,
            bounds=bounds,
            maxfev=500000
        )

        y_pred = func(N, *popt)
        y_smooth = func(x_fit, *popt)

        stats = fit_stats(error, y_pred, len(popt))

        result = {
            "name": name,
            "equation": model["equation"],
            "params": popt,
            "R2": stats["R2"],
            "RMSE": stats["RMSE"],
            "SSE": stats["SSE"],
            "AIC": stats["AIC"],
            "BIC": stats["BIC"],
            "y_smooth": y_smooth,
            "success": True,
        }

        results.append(result)

        print("Success")
        print("Parameters:")
        for i, val in enumerate(popt):
            print(f"  p{i+1} = {val:.10g}")

        print(f"R²   = {stats['R2']:.6f}")
        print(f"RMSE = {stats['RMSE']:.3f} pcm")
        print(f"AIC  = {stats['AIC']:.3f}")
        print(f"BIC  = {stats['BIC']:.3f}")

    except Exception as e:
        print(f"Failed: {e}")

# =============================================================================
# Results table
# =============================================================================

summary_rows = []

for r in results:
    summary_rows.append({
        "model": r["name"],
        "equation": r["equation"],
        "R2": r["R2"],
        "RMSE": r["RMSE"],
        "SSE": r["SSE"],
        "AIC": r["AIC"],
        "BIC": r["BIC"],
        "params": r["params"],
    })

summary_df = pd.DataFrame(summary_rows)

# Sort by RMSE
summary_df = summary_df.sort_values(by="RMSE", ascending=True)

print("\n" + "=" * 100)
print("MODEL COMPARISON, SORTED BY RMSE")
print("=" * 100)
print(summary_df[["model", "R2", "RMSE", "AIC", "BIC"]].to_string(index=False))

# Save comparison table
summary_df.to_csv("convergence_model_fit_comparison.csv", index=False)
print("\nSaved model comparison table:")
print("convergence_model_fit_comparison.csv")

# =============================================================================
# Print best model equation
# =============================================================================

best = results[np.argmin([r["RMSE"] for r in results])]

print("\n" + "=" * 100)
print("BEST MODEL BY RMSE")
print("=" * 100)
print(f"Model: {best['name']}")
print(f"Equation: {best['equation']}")
print(f"R²   = {best['R2']:.6f}")
print(f"RMSE = {best['RMSE']:.3f} pcm")
print(f"AIC  = {best['AIC']:.3f}")
print(f"BIC  = {best['BIC']:.3f}")
print("Parameters:")
for i, val in enumerate(best["params"]):
    print(f"  p{i+1} = {val:.10g}")

# =============================================================================
# Plot all successful fits
# =============================================================================

plt.figure(figsize=(12, 7))

plt.scatter(
    N,
    error,
    s=40,
    color="black",
    alpha=0.85,
    label="Data",
    zorder=5
)

colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

for r, color in zip(results, colors):
    label = f"{r['name']} | RMSE={r['RMSE']:.1f}"
    plt.plot(
        x_fit,
        r["y_smooth"],
        linewidth=2,
        alpha=0.8,
        color=color,
        label=label
    )

plt.title("Convergence Curve Fits — Multiple Model Forms")
plt.xlabel("Dataset Size, Number of FMs")
plt.ylabel("Mean |PCM| Error")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("convergence_all_model_fits.png", dpi=300)
plt.show()

print("\nSaved plot:")
print("convergence_all_model_fits.png")

# =============================================================================
# Plot top 4 models only
# =============================================================================

top_n = 4
top_models = summary_df.head(top_n)["model"].tolist()

plt.figure(figsize=(12, 7))

plt.scatter(
    N,
    error,
    s=45,
    color="black",
    alpha=0.9,
    label="Data",
    zorder=5
)

for r in results:
    if r["name"] in top_models:
        plt.plot(
            x_fit,
            r["y_smooth"],
            linewidth=2.5,
            label=f"{r['name']} | R²={r['R2']:.4f}, RMSE={r['RMSE']:.1f}"
        )

plt.title(f"Top {top_n} Convergence Curve Fits")
plt.xlabel("Dataset Size, Number of FMs")
plt.ylabel("Mean |PCM| Error")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("convergence_top_model_fits.png", dpi=300)
plt.show()

print("\nSaved plot:")
print("convergence_top_model_fits.png")
