# Redrob Intelligent Candidate Discovery & Ranking

**Team:** ThreeTwoOne  
**Members:** Vaibhav Sharma (Lead) · Shreya Khantal  
**Challenge:** Redrob Hackathon 2026 — Track 1: Intelligent Candidate Ranking

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://india-runs.streamlit.app)

---

## 🚀 Live Demo

**Streamlit App:** [india-runs.streamlit.app](https://india-runs.streamlit.app)

Upload a sample JSONL/JSON file and run the full two-stage ranking pipeline in your browser. The demo uses the bundled `data/sample_candidates.json` for a quick test run (no upload required in the Quick Demo tab).

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Pre-compute (one-time, ~45 minutes for 100K candidates)
```bash
python precompute.py \
  --candidates ./candidates.jsonl \
  --jd ./job_description.docx
```
This downloads the embedding model locally and pre-computes all 100K candidate embeddings. After this step, **no network access is needed** during ranking.

### Step 3 — Rank candidates (~3 minutes, no network)
```bash
python rank.py \
  --candidates ./candidates.jsonl \
  --jd ./job_description.docx \
  --out ./submission.csv
```

### Step 4 — Validate submission
```bash
python validate_submission.py submission.csv
```

### Step 5 — Run Streamlit demo locally
```bash
streamlit run app.py
```

---

## Architecture

Two-stage CPU-only ranking pipeline:

```
candidates.jsonl (100K)
        │
        ▼
┌───────────────────────────────────────────────────┐
│  Stage 1: Signal Extraction + Behavioral Gate    │
│  • 7 signals per candidate (see below)           │
│  • Hard filter: open_to_work + ≤60d + >10% resp │
│  • Semantic similarity via pre-computed embeddings│
│  • Top 500 advance                               │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────┐
│  Stage 2: Signal Fusion + Reranking              │
│  • LogisticRegression + IsotonicRegression       │
│  • 7 signals + 4 interaction terms               │
│  • 90% fusion + 10% semantic similarity          │
│  • Honeypot penalty enforcement                  │
│  • Score calibration to [0,1] + monotonic        │
│  • Tie-break: equal scores → candidate_id asc   │
│  • Template-based factual reasoning              │
└──────────────────────┬────────────────────────────┘
                       │
                       ▼
        submission.csv (top 100 candidates)
        candidate_id | rank | score | reasoning
```

### The 7 Signals

| # | Signal | Description | Weight |
|---|--------|-------------|--------|
| 1 | `title_career` | Title match, production evidence, product vs services company | 25% |
| 2 | `skill_depth` | Trust-weighted skill score (endorsements × duration × proficiency × JD multiplier) | 20% |
| 3 | `experience` | Years-of-experience fit for the [5-9] target band | 15% |
| 4 | `education` | Institution tier (Tier-1/2/3) + field relevance + degree level | 10% |
| 5 | `location` | Preferred city match (Pune/Noida=1.0, NCR=0.85) + relocation willingness | 5% |
| 6 | `behavioral` | Availability gate (open_to_work + ≤60d active + >10% response) + engagement | 15% |
| 7 | `honeypot_penalty` | Timeline impossibility, skill inflation, company age mismatch, salary anomalies | -15% |

### Honeypot Detection

The dataset contains ~80 honeypot candidates with subtly impossible profiles.
Our system detects them through:

1. **Tenure > total experience** — impossible career timeline
2. **Skill inflation** — expert proficiency in 10+ skills with <12 months duration
3. **Company age mismatch** — claimed tenure longer than company has existed
4. **Salary/experience mismatch** — <3 years requesting >40 LPA
5. **Education anomalies** — graduation year in the future
6. **Keyword stuffing** — AI skills on services background with no depth

### Reasoning Quality

All reasoning strings are template-filled from verified candidate fields only — **no LLM inference, no hallucination**. Each string includes:
- Title / Company / Company type / Years of experience
- Specific tech evidence from career descriptions (FAISS, NDCG, etc.)
- Behavioral signal (last active, response rate, GitHub activity)
- Up to 3 specific concerns (notice period, services background, location)

---

## Constraints Compliance

| Constraint | Status |
|-----------|--------|
| CPU only | ✅ PyTorch CPU build, no CUDA |
| No network during ranking | ✅ Model loaded from `models/embedding_model/` |
| < 5 minutes for 100K | ✅ ~3 min (pre-computed embeddings) |
| < 16 GB RAM | ✅ ~4 GB peak (100K × 384-dim float32 embeddings) |
| Exactly 100 rows | ✅ Hard-enforced |
| Ranks 1–100, unique | ✅ Validated |
| Scores non-increasing | ✅ Monotonic enforcement + final sweep |
| Tie-break by candidate_id ascending | ✅ Final stable sort enforces this |
| CAND_XXXXXXX format | ✅ Validated |

---

## File Structure

```
Implementation/
├── rank.py                      # Main CLI entry point
├── precompute.py                # One-time pre-computation script
├── app.py                       # Streamlit demo app (4 tabs)
├── validate_submission.py       # Submission format validator
├── requirements.txt             # Full requirements (with CPU torch)
├── requirements_cloud.txt       # Streamlit Cloud requirements
├── pyproject.toml               # Python project config
├── submission_metadata.yaml     # Team and approach metadata
├── README.md                    # This file
├── .streamlit/
│   └── config.toml              # Streamlit theme (dark mode)
├── data/
│   ├── job_description.md       # JD text (used on Streamlit Cloud)
│   └── sample_candidates.json  # 50-candidate sample for demo
├── src/
│   └── ranker/
│       ├── __init__.py
│       ├── __main__.py          # Pipeline orchestrator (Ranker class)
│       ├── candidate_loader.py  # JSONL/JSON/GZ reader & Candidate dataclass
│       ├── signals.py           # 7-signal extractor + SignalScores
│       ├── embeddings.py        # Offline-first embedding manager
│       ├── jd_parser.py         # JD requirements parser
│       ├── fusion.py            # LR + IsotonicRegression fusion
│       └── reasoning.py         # Template-based reasoning generator
└── models/
    ├── embedding_model/         # Saved sentence-transformers model (offline)
    ├── embeddings/              # Pre-computed candidate embeddings (gitignored)
    └── fusion/                  # Trained fusion model (~2KB, committed)
        ├── fusion_model.pkl
        ├── scaler.pkl
        ├── calibrator.pkl
        └── weights.npy
```

---

## AI Tools Declaration

Used Claude (Anthropic) for architectural discussion and debugging, and GitHub Copilot for code autocomplete. All core logic (signal design, honeypot rules, scoring formulas, reasoning templates) was authored and validated by the team. No candidate data was sent to any external API. The ranking pipeline runs fully offline.