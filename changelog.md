# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project loosely follows [Semantic Versioning](https://semver.org/)
(MAJOR.MINOR.PATCH), treating each dashboard/feature addition as a minor bump.

## [Unreleased]

- Polish README with CI badge, live-demo link, and a "Key Findings" summary.

## [0.6.0] - 2026-09-07

### Added
- CSV download buttons on the interactive dashboard, for both the policy
  comparison table and the safety/workload trade-off curve data, so results
  can be exported for offline analysis.

## [0.5.0] - 2026-09-07

### Added
- GitHub Actions CI workflow (`.github/workflows/tests.yml`) that runs the
  test suite automatically on every push and pull request against `main`,
  across Python 3.10–3.12.

## [0.4.0] - 2026-09-07

### Added
- `tests/test_simulation.py`: a 15-case unit test suite covering the risk
  scoring model, all four review policies' threshold logic, arrival-rate
  generation, and end-to-end simulation invariants (every generated patient
  is processed, `Review all` never produces a harmful miss, results are
  reproducible given a fixed seed).

## [0.3.0] - 2026-09-07

### Added
- MIT license.

## [0.2.0] - 2026-09-06

### Added
- `dashboard/app.py`: an interactive Streamlit dashboard letting users adjust
  AI noise, staff capacity, and the fixed review threshold in real time and
  see the safety/workload trade-off update live, including a head-to-head
  comparison against the dynamic heuristic.
- Formal project report content (Problem Description, Model Formulation,
  Method, Data Generation, Results, Conclusions) written up separately for
  the course deliverable.

## [0.1.0] - 2026-09-06

### Added
- Core discrete-event simulation (`simulation.py`) of an AI-assisted ED
  triage system, built with SimPy: time-varying Poisson arrivals, AI
  severity scoring with confidence, a shared clinical-staff resource pool
  for both treatment and manual review, and four review policies (No
  review, Review all, Fixed threshold, Dynamic heuristic).
- `experiments.py`: replicated policy comparison, review-threshold sweep,
  and AI-noise sensitivity analysis, producing the figures in `figures/`.
- Initial `README.md` and `requirements.txt`.