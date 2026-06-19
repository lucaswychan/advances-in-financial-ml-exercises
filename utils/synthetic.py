import numpy as np
from itertools import product
from random import gauss
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm


def fit_ou_process(price_series: pd.Series) -> dict:
    """
    Fit an Ornstein-Uhlenbeck process to a price series using OLS on the
    discretised AR(1) representation:
        ΔP_t = α + β·P_{t-1} + ε_t
    
    O-U parameters recovered as:
        κ  = -β / Δt          (mean-reversion speed)
        θ  = -α / β           (long-run mean)
        σ  = std(ε) / √Δt     (diffusion coefficient)
        t½ = ln(2) / κ        (half-life in bar units)
    """
    import statsmodels.api as sm

    delta_p = price_series.diff().dropna()
    p_lagged = price_series.shift(1).dropna()

    # Align
    y = delta_p
    X = sm.add_constant(p_lagged)

    result = sm.OLS(y, X).fit()
    alpha, beta = result.params["const"], result.params.iloc[1]
    residuals = result.resid

    # dt = 1 bar
    dt = 1.0
    kappa = -beta / dt
    theta = -alpha / beta
    sigma = residuals.std() / np.sqrt(dt)
    half_life = np.log(2) / kappa if kappa > 0 else np.nan

    return {
        "alpha": alpha,
        "beta": beta,
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "half_life_bars": half_life,
        "r_squared": result.rsquared,
        "p_value_beta": result.pvalues.iloc[1],
        "ols_result": result,
    }

def explain_and_plot_ou_params(ou_params, dollar_bars):
    print("=== O-U Process Parameters ===")
    print(f"  Long-run mean    θ  : {ou_params['theta']:.4f}")
    print(f"  Mean-rev. speed  κ  : {ou_params['kappa']:.6f}  (per bar)")
    print(f"  Diffusion coeff  σ  : {ou_params['sigma']:.4f}")
    print(f"  Half-life            : {ou_params['half_life_bars']:.1f}  bars")
    print(f"  β (AR coeff)         : {ou_params['beta']:.6f}  (negative ⟹ mean-reverting)")
    print(f"  R²                   : {ou_params['r_squared']:.4f}")
    print(f"  p-value (β)          : {ou_params['p_value_beta']:.4e}")

    # ------------------------------------------------------------------
    # Diagnostic plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: price series with long-run mean
    axes[0].plot(dollar_bars["price"].values, lw=0.8, label="Close price")
    axes[0].axhline(ou_params["theta"], color="red", linestyle="--", lw=1.5,
                    label=f"θ = {ou_params['theta']:.2f}")
    axes[0].set_title("Dollar Bars – Close Price vs O-U Long-Run Mean")
    axes[0].set_xlabel("Bar index")
    axes[0].set_ylabel("Price")
    axes[0].legend()

    # Right: residual distribution
    axes[1].hist(ou_params["ols_result"].resid, bins=60, density=True, alpha=0.7)
    x_range = np.linspace(ou_params["ols_result"].resid.min(),
                        ou_params["ols_result"].resid.max(), 200)
    axes[1].plot(x_range,
                norm.pdf(x_range, 0, ou_params["ols_result"].resid.std()),
                "r-", lw=2, label="Normal fit")
    axes[1].set_title("OLS Residuals (ε_t)")
    axes[1].set_xlabel("Residual")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def simulate_ou_process(coeffs, iterations=1e5, max_hp=100, r_pt=np.linspace(0.5, 10, 20), r_slm=np.linspace(0.5, 10, 20), seed=0):
    phi = coeffs['phi'] if "phi" in coeffs else 2 ** (-1.0 / coeffs['hl'])
    outputs = []
    # Describe what tqdm progress bar is showing: "Sweep over (profit-taking, stop-loss multiple) pairs"
    progress_desc = "Sweep (r_pt, r_slm) pairs"
    for comb_ in tqdm(product(r_pt, r_slm), desc=progress_desc, total=len(r_pt)*len(r_slm)):
        output_per_path = []
        # tqdm progress for each (pt, slm) pair: "Simulating paths"
        for _ in tqdm(range(int(iterations)), desc=f"Simulate {comb_}", leave=False, disable=True):
            p, hp = seed, 0
            for _ in range(max_hp):
                p = (1 - phi) * coeffs['forecast'] + phi * p + coeffs['sigma'] * gauss(0, 1)
                cP = p - seed
                hp += 1
                if cP > comb_[0] or cP < -comb_[1]:
                    output_per_path.append(cP)
                    break

        if len(output_per_path) > 1:
            mean, std = np.mean(output_per_path), np.std(output_per_path)
            # print(comb_[0], comb_[1], mean, std, mean / std)
            outputs.append((comb_[0], comb_[1], mean, std, mean / std))
        else:
            outputs.append((comb_[0], comb_[1], 0, 0, 0))
    return outputs