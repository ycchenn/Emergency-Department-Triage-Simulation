"""
AI-Assisted Emergency Department Triage Simulation
=====================================================
Models an ED where an AI system gives each arriving patient an estimated
severity score (1 = most critical, 5 = least urgent) plus a confidence
level. The hospital must decide, for each patient, whether to send them
for manual (human) review before assigning bed/doctor resources.

Trade-off modeled:
    - Reviewing more patients -> lower risk of missing a true emergency,
      but consumes scarce reviewer time (and therefore doctor time too,
      since reviewers are drawn from clinical staff).
    - Reviewing fewer patients -> faster throughput, but AI errors on
      truly critical patients go uncorrected, causing dangerous delays.

Policies compared:
    P0  No review          - always trust the AI score (tau = 0)
    P1  Review all         - always send to human review (tau = 1)
    P2  Fixed threshold    - review iff AI confidence < tau_fixed
    P3  Dynamic heuristic  - tau adapts to current reviewer congestion
"""

import simpy
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class SimConfig:
    sim_minutes: int = 720          # 12-hour shift
    n_staff: int = 12               # clinical staff pool: SAME staff both
                                     # treat patients and perform manual
                                     # reviews, so review time is real
                                     # opportunity cost of treatment time
    review_priority: float = 2.5    # queue priority given to a review task
                                     # (between severity-2 and severity-3
                                     # treatment -> reviews cut ahead of
                                     # routine patients but never ahead of
                                     # a patient already flagged critical)
    review_time_mean: float = 6.0   # minutes per manual review
    base_arrival_rate: float = 0.30 # patients / minute (off-peak)
    peak_multiplier: float = 1.5    # arrival rate multiplier during peak
    peak_window: tuple = (240, 480) # peak = hour 4-8 of the shift
    severity_probs: tuple = (0.05, 0.15, 0.30, 0.30, 0.20)  # levels 1..5
    ai_noise_sigma: float = 0.9     # AI scoring error std dev (severity units)
    treatment_base: dict = field(default_factory=lambda: {
        # mean treatment minutes by TRUE severity (more severe -> longer/more resource-intensive)
        1: 55, 2: 45, 3: 30, 4: 20, 5: 12
    })
    seed: int = 0


# severity weight used to penalize waiting time for more critical patients
# (level 1 = most critical -> highest weight)
def severity_weight(level: int) -> float:
    return 6 - level  # level1->5, level2->4, ... level5->1


# ----------------------------------------------------------------------
# Patient generation
# ----------------------------------------------------------------------
def arrival_rate(t, cfg: SimConfig):
    lo, hi = cfg.peak_window
    if lo <= t <= hi:
        return cfg.base_arrival_rate * cfg.peak_multiplier
    return cfg.base_arrival_rate


def draw_patient(rng, cfg: SimConfig, arrival_time):
    true_sev = rng.choice([1, 2, 3, 4, 5], p=cfg.severity_probs)
    noise = rng.normal(0, cfg.ai_noise_sigma)
    ai_sev = int(np.clip(round(true_sev + noise), 1, 5))
    err_mag = abs(noise)
    # confidence: higher when |noise| small; squashed to (0,1), with jitter
    confidence = float(np.clip(np.exp(-err_mag) + rng.normal(0, 0.05), 0.02, 0.99))
    treat_mean = cfg.treatment_base[true_sev]
    treat_time = max(3.0, rng.lognormal(mean=np.log(treat_mean), sigma=0.35))
    return {
        "arrival_time": arrival_time,
        "true_sev": true_sev,
        "ai_sev": ai_sev,
        "confidence": confidence,
        "treat_time": treat_time,
    }


def generate_patients(cfg: SimConfig, rng):
    patients = []
    t = 0.0
    while t < cfg.sim_minutes:
        rate = arrival_rate(t, cfg)
        t += rng.exponential(1.0 / rate)
        if t >= cfg.sim_minutes:
            break
        patients.append(draw_patient(rng, cfg, t))
    return patients


# ----------------------------------------------------------------------
# Policy: decide whether to review, given current system state
# ----------------------------------------------------------------------
# Risk weight by AI-assigned severity: a "harmful miss" only happens when
# the AI UNDER-estimates a truly critical (true_sev<=2) patient, i.e. gives
# them ai_sev>=3 so they lose queue priority. AI level-3 scores are the most
# dangerous blind spot (closest to the critical boundary, most likely to be
# hiding a true level-2 patient given the noise model); level-1 patients are
# already fast-tracked by the AI itself, so mis-scoring them causes little
# harm even if uncorrected.
AI_SEV_RISK_WEIGHT = {1: 0.05, 2: 0.35, 3: 1.00, 4: 0.55, 5: 0.20}


class Policy:
    """
    Review decision is based on a risk score:
        risk_score = (1 - confidence) * AI_SEV_RISK_WEIGHT[ai_sev]
    A patient is sent to manual review iff risk_score > threshold.
    This targets scarce review capacity at patients who are BOTH
    AI-uncertain AND sitting in the danger zone of the severity scale,
    rather than spreading review effort uniformly by confidence alone.
    """
    def __init__(self, kind, tau_fixed=None, tau_base=0.18,
                 low_util=0.3, high_util=0.85):
        self.kind = kind
        self.tau_fixed = tau_fixed
        self.tau_base = tau_base
        self.low_util = low_util
        self.high_util = high_util

    @staticmethod
    def risk_score(confidence, ai_sev):
        return (1 - confidence) * AI_SEV_RISK_WEIGHT[ai_sev]

    def review_threshold(self, reviewer_resource: simpy.Resource):
        if self.kind == "none":
            return 1.01   # nothing ever exceeds this -> never review
        if self.kind == "all":
            return -0.01  # everything exceeds this -> always review
        if self.kind == "fixed":
            return self.tau_fixed
        if self.kind == "dynamic":
            util = reviewer_resource.count / reviewer_resource.capacity
            if util <= self.low_util:
                return max(0.02, self.tau_base - 0.05)   # slack -> review a bit more liberally
            elif util >= self.high_util:
                return min(0.95, self.tau_base + 0.10)   # congested -> be more selective
            else:
                frac = (util - self.low_util) / (self.high_util - self.low_util)
                return (self.tau_base - 0.05) + frac * 0.15
        raise ValueError(self.kind)


# ----------------------------------------------------------------------
# Simulation core
# ----------------------------------------------------------------------
def run_simulation(cfg: SimConfig, policy: Policy, seed=None):
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    patients_raw = generate_patients(cfg, rng)

    env = simpy.Environment()
    # A single shared clinical-staff pool: the same doctors/nurses both treat
    # patients and perform manual severity reviews. This makes the safety/
    # efficiency trade-off real -- every minute spent reviewing is a minute
    # not spent treating, and vice versa. Priority resource: lower priority
    # value = served first.
    staff = simpy.PriorityResource(env, capacity=cfg.n_staff)

    log = []

    def patient_process(p):
        arrival = p["arrival_time"]
        yield env.timeout(max(0, arrival - env.now))

        threshold = policy.review_threshold(staff)
        risk = Policy.risk_score(p["confidence"], p["ai_sev"])
        will_review = risk > threshold
        review_wait = 0.0
        review_busy = 0.0

        if will_review:
            req_time = env.now
            with staff.request(priority=cfg.review_priority) as req:
                yield req
                review_wait = env.now - req_time
                rtime = max(1.0, rng.normal(cfg.review_time_mean, 1.5))
                yield env.timeout(rtime)
                review_busy = rtime
            est_sev = p["true_sev"]   # human review corrects to ground truth
        else:
            est_sev = p["ai_sev"]

        bed_req_time = env.now
        with staff.request(priority=est_sev) as req:
            yield req
            bed_wait = env.now - bed_req_time
            yield env.timeout(p["treat_time"])

        start_treatment = bed_req_time + bed_wait
        true_total_wait = start_treatment - arrival

        harmful_miss = (not will_review) and (p["true_sev"] <= 2) and (p["ai_sev"] >= 3)

        log.append({
            "true_sev": p["true_sev"],
            "ai_sev": p["ai_sev"],
            "confidence": p["confidence"],
            "reviewed": will_review,
            "wait": true_total_wait,
            "weighted_wait": true_total_wait * severity_weight(p["true_sev"]),
            "review_busy_time": review_busy,
            "harmful_miss": harmful_miss,
        })

    for p in patients_raw:
        env.process(patient_process(p))

    env.run()  # run until every generated patient has been fully processed

    df = pd.DataFrame(log)
    assert len(df) == len(patients_raw), "some patients were not processed"
    return df


def summarize(df: pd.DataFrame, policy_name: str) -> dict:
    if len(df) == 0:
        return {"policy": policy_name}
    critical = df[df["true_sev"] <= 2]
    missed = df[df["harmful_miss"]]
    return {
        "policy": policy_name,
        "n_patients": len(df),
        "avg_wait": df["wait"].mean(),
        "avg_weighted_wait": df["weighted_wait"].mean(),
        "avg_wait_critical": critical["wait"].mean() if len(critical) else np.nan,
        "avg_wait_missed_critical": missed["wait"].mean() if len(missed) else np.nan,
        "pct_reviewed": 100 * df["reviewed"].mean(),
        "harmful_misses": int(df["harmful_miss"].sum()),
        "total_review_minutes": df["review_busy_time"].sum(),
    }


if __name__ == "__main__":
    cfg = SimConfig()
    policies = {
        "P0 No review": Policy("none"),
        "P1 Review all": Policy("all"),
        "P2 Fixed tau=0.18": Policy("fixed", tau_fixed=0.18),
        "P3 Dynamic": Policy("dynamic", tau_base=0.18),
    }
    rows = []
    for name, pol in policies.items():
        df = run_simulation(cfg, pol, seed=1)
        rows.append(summarize(df, name))
    print(pd.DataFrame(rows).to_string(index=False))
