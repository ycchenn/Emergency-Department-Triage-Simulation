"""
Interactive dashboard for the AI-Assisted ED Triage simulation.

Run locally:
    streamlit run app.py

Deploy free at https://share.streamlit.io (Streamlit Community Cloud) by
pointing it at this file in a public GitHub repo.
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulation import SimConfig, Policy, run_simulation, summarize

st.set_page_config(page_title="AI Triage Trade-off Simulator", page_icon="🏥", layout="wide")

# ----------------------------------------------------------------------
# Cached simulation helpers
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def replicate(cfg_dict, policy_kind, policy_kwargs, n_reps, base_seed=100):
    cfg = SimConfig(**cfg_dict)
    pol = Policy(policy_kind, **policy_kwargs)
    rows = []
    for i in range(n_reps):
        df = run_simulation(cfg, pol, seed=base_seed + i)
        rows.append(summarize(df, policy_name=""))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def tau_sweep(cfg_dict, taus, n_reps, base_seed=300):
    cfg = SimConfig(**cfg_dict)
    rows = []
    for tau in taus:
        pol = Policy("fixed", tau_fixed=tau)
        rep = []
        for i in range(n_reps):
            df = run_simulation(cfg, pol, seed=base_seed + i)
            rep.append(summarize(df, ""))
        rep_df = pd.DataFrame(rep)
        rows.append({
            "tau": tau,
            "harmful_misses": rep_df["harmful_misses"].mean(),
            "review_minutes": rep_df["total_review_minutes"].mean(),
        })
    return pd.DataFrame(rows)


def agg_row(rep_df, label):
    return {
        "Policy": label,
        "Avg wait, all (min)": rep_df["avg_wait"].mean(),
        "Avg wait, critical (min)": rep_df["avg_wait_critical"].mean(),
        "Harmful misses / shift": rep_df["harmful_misses"].mean(),
        "Reviewer minutes / shift": rep_df["total_review_minutes"].mean(),
        "% patients reviewed": rep_df["pct_reviewed"].mean(),
    }


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.title("🎛️ Simulation controls")

sigma = st.sidebar.slider(
    "AI noise (σ)", 0.3, 1.8, 0.9, 0.1,
    help="How inaccurate the AI's severity score is. Higher = the AI makes bigger errors more often."
)

n_staff = st.sidebar.slider(
    "Clinical staff (N)", 8, 18, 12, 1,
    help="Shared pool of doctors/nurses who both treat patients and perform manual reviews."
)

tau_user = st.sidebar.slider(
    "Fixed policy threshold (τ)", 0.0, 0.5, 0.15, 0.01,
    help="Review a patient iff their risk score exceeds this. Lower = review more (safer, slower)."
)

n_reps = st.sidebar.slider(
    "Replications per policy", 5, 40, 15, 5,
    help="More replications = more statistically reliable averages, but slower to compute."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "All data shown is synthetically generated — no real patient data is used. "
    "See the **Methodology** section below for how the simulation works."
)

cfg_dict = dict(ai_noise_sigma=sigma, n_staff=n_staff)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🏥 AI-Assisted ED Triage: Safety vs. Workload Trade-off")
st.markdown(
    "An AI system scores every arriving patient's urgency. Should a human "
    "double-check it? This simulator compares four review policies under a "
    "discrete-event model of a busy emergency department, where the same "
    "clinical staff both **treat patients and perform manual reviews** — so "
    "every review has a real opportunity cost."
)

# ----------------------------------------------------------------------
# Run simulations
# ----------------------------------------------------------------------
with st.spinner("Running simulations..."):
    t0 = time.time()
    rep_none = replicate(cfg_dict, "none", {}, n_reps)
    rep_all = replicate(cfg_dict, "all", {}, n_reps)
    rep_fixed = replicate(cfg_dict, "fixed", {"tau_fixed": tau_user}, n_reps)
    rep_dyn = replicate(cfg_dict, "dynamic",
                         {"tau_base": 0.15, "low_util": 0.3, "high_util": 0.85}, n_reps)
    elapsed = time.time() - t0

summary = pd.DataFrame([
    agg_row(rep_none, "No review"),
    agg_row(rep_all, "Review all"),
    agg_row(rep_fixed, f"Fixed τ={tau_user:.2f}"),
    agg_row(rep_dyn, "Dynamic heuristic"),
])

# ----------------------------------------------------------------------
# KPI cards for the two policies people care about most
# ----------------------------------------------------------------------
st.subheader("Fixed threshold vs. dynamic heuristic, head to head")
c1, c2, c3 = st.columns(3)
fixed_row = summary[summary["Policy"].str.startswith("Fixed")].iloc[0]
dyn_row = summary[summary["Policy"] == "Dynamic heuristic"].iloc[0]

def pct_change(new, old):
    if old == 0:
        return None
    return 100 * (new - old) / old

c1.metric(
    "Harmful misses / shift",
    f"{dyn_row['Harmful misses / shift']:.2f}",
    delta=f"{dyn_row['Harmful misses / shift'] - fixed_row['Harmful misses / shift']:+.2f} vs fixed",
    delta_color="inverse",
)
c2.metric(
    "Reviewer minutes / shift",
    f"{dyn_row['Reviewer minutes / shift']:.0f}",
    delta=f"{pct_change(dyn_row['Reviewer minutes / shift'], fixed_row['Reviewer minutes / shift']):+.0f}% vs fixed",
    delta_color="inverse",
)
c3.metric(
    "Avg. wait, all patients (min)",
    f"{dyn_row['Avg wait, all (min)']:.1f}",
    delta=f"{pct_change(dyn_row['Avg wait, all (min)'], fixed_row['Avg wait, all (min)']):+.0f}% vs fixed",
    delta_color="inverse",
)
st.caption(f"({n_reps} replications per policy, computed in {elapsed:.1f}s)")

# ----------------------------------------------------------------------
# Full comparison table
# ----------------------------------------------------------------------
st.subheader("All four policies")
st.dataframe(
    summary.style.format({
        "Avg wait, all (min)": "{:.1f}",
        "Avg wait, critical (min)": "{:.1f}",
        "Harmful misses / shift": "{:.2f}",
        "Reviewer minutes / shift": "{:.0f}",
        "% patients reviewed": "{:.0f}%",
    }),
    width="stretch",
    hide_index=True,
)

# ----------------------------------------------------------------------
# Bar charts
# ----------------------------------------------------------------------
st.subheader("Comparison charts")
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].bar(summary["Policy"], summary["Avg wait, critical (min)"], color=colors)
axes[0].set_title("Avg. wait — critical patients")
axes[0].set_ylabel("minutes")
axes[0].tick_params(axis="x", rotation=25)

axes[1].bar(summary["Policy"], summary["Harmful misses / shift"], color=colors)
axes[1].set_title("Harmful misses / shift")
axes[1].tick_params(axis="x", rotation=25)

axes[2].bar(summary["Policy"], summary["Reviewer minutes / shift"], color=colors)
axes[2].set_title("Reviewer workload")
axes[2].set_ylabel("minutes")
axes[2].tick_params(axis="x", rotation=25)

plt.tight_layout()
st.pyplot(fig)

# ----------------------------------------------------------------------
# Tau sweep
# ----------------------------------------------------------------------
st.subheader("Safety / workload frontier")
st.markdown(
    "As the fixed review threshold τ increases, fewer patients are reviewed: "
    "workload drops, but so does safety. The dynamic heuristic (star) aims to "
    "sit **below the curve** — same safety as a conservative fixed policy, less workload."
)

taus = tuple(round(t, 2) for t in np.linspace(0.0, 0.5, 11))
sweep_reps = max(5, n_reps // 2)
sweep_df = tau_sweep(cfg_dict, taus, sweep_reps)

fig2, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(sweep_df["tau"], sweep_df["harmful_misses"], "o-", color="#C44E52", label="Harmful misses")
ax1.set_xlabel("Review threshold τ")
ax1.set_ylabel("Harmful misses / shift", color="#C44E52")
ax1.tick_params(axis="y", labelcolor="#C44E52")
ax1.axvline(tau_user, color="gray", linestyle=":", linewidth=1)

ax2 = ax1.twinx()
ax2.plot(sweep_df["tau"], sweep_df["review_minutes"], "s-", color="#4C72B0", label="Review workload")
ax2.set_ylabel("Reviewer minutes / shift", color="#4C72B0")
ax2.tick_params(axis="y", labelcolor="#4C72B0")
ax2.scatter([0.15], [dyn_row["Reviewer minutes / shift"]], marker="*", s=300,
            color="black", zorder=5, label="Dynamic heuristic")

plt.title(f"Trade-off curve (σ={sigma}, N={n_staff})")
fig2.tight_layout()
st.pyplot(fig2)

# ----------------------------------------------------------------------
# Methodology
# ----------------------------------------------------------------------
with st.expander("📖 Methodology — how this simulation works"):
    st.markdown(
        """
Each patient arriving at the ED is scored by a simulated AI (severity 1–5, plus a
confidence value). A **risk score** `(1 − confidence) × w(AI severity)` targets
review capacity at the boundary cases most likely to be a hidden emergency
(AI scores near level 3 get the highest weight). Four policies decide whether
to send a patient for manual review:

- **No review** — always trust the AI
- **Review all** — always double-check
- **Fixed threshold** — review iff risk score > τ (constant all shift)
- **Dynamic heuristic** — same rule, but τ adapts to real-time staff congestion:
  more thorough when quiet, more selective when busy

Reviews and treatment draw on the **same shared pool of clinical staff**
(discrete-event simulation, built with [SimPy](https://simpy.readthedocs.io)),
so every minute spent reviewing is a minute not spent treating — the
trade-off is real, not assumed away by separate resource pools.

A **harmful miss** = a truly critical patient (severity 1–2) that the AI
scored non-urgent (≥3) and that was never corrected by review.

All patient data is synthetically generated (Poisson arrivals with a peak
window, severity mix, AI noise model, log-normal treatment times) — no real
hospital data is used. See `simulation.py` for full parameter documentation.
        """
    )

st.markdown("---")
st.caption(
    "Built with SimPy + Streamlit as part of an Operations Research capstone project. "
    "[View source on GitHub](https://github.com/YOUR-USERNAME/ed-triage-simulation)"
)
