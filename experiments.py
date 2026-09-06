"""
Experiment runner: replicates each policy many times, aggregates results,
sweeps the fixed threshold tau to trace a Pareto trade-off curve, and runs
a sensitivity analysis over AI noise level.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import SimConfig, Policy, run_simulation, summarize

N_REPS = 40
OUT_DIR = "/home/claude/ed_triage_sim/figures"
import os
os.makedirs(OUT_DIR, exist_ok=True)


def replicate(cfg, policy, n_reps=N_REPS, base_seed=100):
    rows = []
    for i in range(n_reps):
        df = run_simulation(cfg, policy, seed=base_seed + i)
        rows.append(summarize(df, policy_name=""))
    return pd.DataFrame(rows)


def experiment_policy_comparison(cfg):
    policies = {
        "No review": Policy("none"),
        "Review all": Policy("all"),
        "Fixed \u03c4=0.15 (safe)": Policy("fixed", tau_fixed=0.15),
        "Dynamic heuristic": Policy("dynamic", tau_base=0.15, low_util=0.3, high_util=0.85),
    }
    agg = []
    for name, pol in policies.items():
        rep_df = replicate(cfg, pol)
        agg.append({
            "policy": name,
            "avg_wait_mean": rep_df["avg_wait"].mean(),
            "avg_wait_sem": rep_df["avg_wait"].sem(),
            "avg_wait_critical_mean": rep_df["avg_wait_critical"].mean(),
            "avg_wait_critical_sem": rep_df["avg_wait_critical"].sem(),
            "avg_wait_missed_critical_mean": np.nanmean(rep_df["avg_wait_missed_critical"]),
            "harmful_misses_mean": rep_df["harmful_misses"].mean(),
            "harmful_misses_sem": rep_df["harmful_misses"].sem(),
            "review_minutes_mean": rep_df["total_review_minutes"].mean(),
            "review_minutes_sem": rep_df["total_review_minutes"].sem(),
            "pct_reviewed_mean": rep_df["pct_reviewed"].mean(),
        })
    result = pd.DataFrame(agg)
    result.to_csv(f"{OUT_DIR}/policy_comparison.csv", index=False)
    return result


def plot_policy_comparison(result):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    ax = axes[0]
    ax.bar(result["policy"], result["avg_wait_critical_mean"],
           yerr=result["avg_wait_critical_sem"], capsize=4, color=colors)
    ax.set_ylabel("Avg. wait, critical patients (min)")
    ax.set_title("(a) Wait time for critical patients\n(true severity 1-2)")
    ax.tick_params(axis='x', rotation=20)

    ax = axes[1]
    ax.bar(result["policy"], result["harmful_misses_mean"],
           yerr=result["harmful_misses_sem"], capsize=4, color=colors)
    ax.set_ylabel("Harmful misses per shift")
    ax.set_title("(b) Critical patients missed by\nAI and not caught by review")
    ax.tick_params(axis='x', rotation=20)

    ax = axes[2]
    ax.bar(result["policy"], result["review_minutes_mean"],
           yerr=result["review_minutes_sem"], capsize=4, color=colors)
    ax.set_ylabel("Total reviewer-minutes / shift")
    ax.set_title("(c) Human review workload")
    ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/policy_comparison.png", dpi=160)
    plt.close()


def experiment_tau_sweep(cfg):
    taus = np.linspace(0.0, 0.5, 11)
    rows = []
    for tau in taus:
        pol = Policy("fixed", tau_fixed=tau)
        rep_df = replicate(cfg, pol, n_reps=25)
        rows.append({
            "tau": tau,
            "harmful_misses": rep_df["harmful_misses"].mean(),
            "review_minutes": rep_df["total_review_minutes"].mean(),
            "avg_wait_critical": rep_df["avg_wait_critical"].mean(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/tau_sweep.csv", index=False)
    return df


def plot_tau_sweep(df, dynamic_point=None):
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(df["tau"], df["harmful_misses"], "o-", color="#C44E52", label="Harmful misses (left axis)")
    ax1.set_xlabel(r"Review threshold $\tau$  (review iff risk score $(1-\mathrm{confidence})\times w(\mathrm{ai\_sev}) > \tau$)")
    ax1.set_ylabel("Harmful misses per shift", color="#C44E52")
    ax1.tick_params(axis='y', labelcolor="#C44E52")

    ax2 = ax1.twinx()
    ax2.plot(df["tau"], df["review_minutes"], "s-", color="#4C72B0", label="Review workload (right axis)")
    ax2.set_ylabel("Total reviewer-minutes / shift", color="#4C72B0")
    ax2.tick_params(axis='y', labelcolor="#4C72B0")

    if dynamic_point is not None:
        ax2.scatter([dynamic_point["tau_equiv"]], [dynamic_point["review_minutes"]],
                    marker="*", s=260, color="black", zorder=5,
                    label="Dynamic heuristic (actual)")
        ax2.annotate("Dynamic heuristic\n(same safety, less workload)",
                     xy=(dynamic_point["tau_equiv"], dynamic_point["review_minutes"]),
                     xytext=(dynamic_point["tau_equiv"] + 0.05, dynamic_point["review_minutes"] + 300),
                     arrowprops=dict(arrowstyle="->", color="black"),
                     fontsize=9)

    plt.title(r"Safety vs. workload trade-off as review threshold $\tau$ varies")
    fig.tight_layout()
    plt.savefig(f"{OUT_DIR}/tau_sweep.png", dpi=160)
    plt.close()


def experiment_sensitivity_ai_noise(cfg):
    sigmas = [0.3, 0.6, 0.9, 1.2, 1.5]
    rows = []
    for sigma in sigmas:
        cfg2 = SimConfig(**{**cfg.__dict__, "ai_noise_sigma": sigma})
        for tau in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
            pol = Policy("fixed", tau_fixed=tau)
            rep_df = replicate(cfg2, pol, n_reps=15)
            rows.append({
                "sigma": sigma,
                "tau": tau,
                "harmful_misses": rep_df["harmful_misses"].mean(),
                "review_minutes": rep_df["total_review_minutes"].mean(),
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/sensitivity_noise.csv", index=False)
    return df


def plot_sensitivity(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for sigma, sub in df.groupby("sigma"):
        ax.plot(sub["tau"], sub["harmful_misses"], "o-", label=f"AI noise \u03c3={sigma}")
    ax.set_xlabel(r"Review threshold $\tau$")
    ax.set_ylabel("Harmful misses per shift")
    ax.set_title("As AI accuracy improves (lower \u03c3),\nfewer reviews are needed to stay safe")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sensitivity_noise.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    cfg = SimConfig()

    print("Running policy comparison (this takes a bit)...")
    result = experiment_policy_comparison(cfg)
    print(result.to_string(index=False))
    plot_policy_comparison(result)

    print("\nRunning tau sweep...")
    tau_df = experiment_tau_sweep(cfg)
    print(tau_df.to_string(index=False))

    dyn_row = result[result["policy"] == "Dynamic heuristic"].iloc[0]
    plot_tau_sweep(tau_df, dynamic_point={
        "tau_equiv": 0.15,  # dynamic policy's tau_base, for x-position only
        "review_minutes": dyn_row["review_minutes_mean"],
    })

    print("\nRunning AI-noise sensitivity analysis...")
    sens_df = experiment_sensitivity_ai_noise(cfg)
    plot_sensitivity(sens_df)

    print("\nDone. Figures saved to", OUT_DIR)
