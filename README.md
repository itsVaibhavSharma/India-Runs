# Redrob Intelligent Candidate Discovery & Ranking

**Team:** ThreeTwoOne  
**Leader:** Vaibhav Sharma  
**Track:** Intelligent Candidate Discovery & Ranking Challenge

---

## Overview

A CPU-only, two-stage candidate ranking system that goes beyond keyword matching to understand context, evaluate behavioral signals, and detect honeypots. Designed for the Redrob Hackathon constraints: 5-min runtime, 16GB RAM, CPU-only, no network.

### Key Features

- **Hybrid Scoring**: 7 independent signals (title/career, skill depth, experience, education, location, behavioral, honeypot penalty)
- **Semantic Understanding**: Sentence-transformer embeddings for JD-candidate matching
- **Behavioral Gating**: Filters inactive/unresponsive candidates early (JD requirement: "active on platform")
- **Honeypot Detection**: Timeline inconsistency, skill inflation, company-age mismatch detection
- **Explainable Output**: Per-candidate reasoning with specific facts, JD connections, and honest concerns
- **CPU Optimized**: Runs in ~3 minutes on 100K candidates with pre-computed embeddings

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Pre-computation (One-time)

```bash
# Compute embeddings for all 100K candidates (run once)
python -c "
from src.ranker.embeddings import EmbeddingManager
from src.ranker.candidate_loader import CandidateLoader
loader = CandidateLoader('../Dataset/candidates.jsonl')
candidates = loader.load_all_candidates()
mgr = EmbeddingManager()
mgr.get_candidate_embeddings(candidates)
print('Embeddings cached!')
"

# Train fusion model (run once)
python -c "
from src.ranker.fusion import SignalFusion
from src.ranker.jd_parser import parse_jd
req = parse_jd('../Dataset/job_description.docx')
fusion = SignalFusion()
fusion.train(req)
print('Fusion model trained!')
"
```

### Ranking

```bash
# Full ranking (produces submission.csv)
python rank.py --candidates ../Dataset/candidates.jsonl --out submission.csv

# With custom paths
python rank.py --candidates path/to/candidates.jsonl --jd path/to/jd.docx --out submission.csv
```

### Validate Submission

```bash
python ../Dataset/validate_submission.py submission.csv
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        100,000 Candidates                        │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Fast Filtering (~2.5 min)                              │
│ • Load candidates, extract 7 signals per candidate              │
│ • Behavioral availability gate (open_to_work, recent activity,  │
│   response_rate > 10%)                                          │
│ • Honeypot detection (timeline, skill inflation, salary/exp)    │
│ • Semantic similarity (pre-computed embeddings)                 │
│ • Top 500 advance                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Calibrated Reranking (~30 sec)                         │
│ • Signal fusion: LogisticRegression + Isotonic calibration      │
│ • Trained on synthetic pairs from JD requirements               │
│ • Combined with similarity score (90/10)                        │
│ • Honeypot penalty enforcement                                  │
│ • Monotonic score enforcement                                   │
│ • Template-based reasoning generation                           │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: submission.csv (top 100)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Signal Details

| Signal | Weight | Description |
|--------|--------|-------------|
| **Title/Career** | ~25% | Current title match, production evidence, ranking/retrieval experience, product vs services company |
| **Skill Depth** | ~20% | Trust-weighted skills (endorsements × duration × proficiency), 3x multiplier for required tech |
| **Experience** | ~15% | Years in [5,9] optimal, decays outside range |
| **Education** | ~5% | Tier-1 institution + CS/ML field bonus |
| **Location** | ~5% | Pune/Noida/Hyderabad/Mumbai/Delhi NCR preferred, relocation friendly |
| **Behavioral** | ~15% | **Multiplicative gate**: must have open_to_work + recent activity + response_rate > 10% |
| **Honeypot Penalty** | ~15% | Timeline impossibility, expert skill inflation, company-age mismatch, salary/exp mismatch |

---

## Requirements

- Python 3.10+
- 16 GB RAM
- CPU only (no GPU required)
- ~5 GB disk for cached embeddings

Dependencies in `requirements.txt`:
- numpy, pandas, scikit-learn
- sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- torch (CPU)
- python-docx, PyYAML, tqdm

---

## Output Format

`submission.csv` with columns:
- `candidate_id`: CAND_XXXXXXX
- `rank`: 1-100 (unique)
- `score`: float, non-increasing with rank
- `reasoning`: 1-2 sentence justification with specific facts

Example:
```csv
candidate_id,rank,score,reasoning
CAND_0042871,1,0.987,"Senior AI Engineer at Swiggy (product) with 6.2 yrs; built embedding-based retrieval for food recommendations using FAISS + sentence-transformers; strong Python (GitHub: 42); recent active (last_active: 2026-06-15); concern: 90-day notice period"
```

---

## Deployment (Streamlit)

```bash
streamlit run app.py
```

Upload a candidates JSONL file (or sample) to see interactive ranking results.

---

## Project Structure

```
Implementation/
├── rank.py                    # Entry point
├── app.py                     # Streamlit demo
├── requirements.txt
├── submission_metadata.yaml   # Hackathon metadata
├── models/                    # Cached artifacts (git-lfs)
│   ├── embeddings/
│   │   ├── candidate_embeddings.npy
│   │   ├── candidate_ids.pkl
│   │   └── candidate_norms.npy
│   └── fusion/
│       ├── fusion_model.pkl
│       ├── scaler.pkl
│       ├── calibrator.pkl
│       └── weights.npy
├── src/
│   └── ranker/
│       ├── __init__.py
│       ├── __main__.py        # Main pipeline
│       ├── jd_parser.py       # JD requirement extraction
│       ├── candidate_loader.py # JSONL streaming + parsing
│       ├── signals.py         # 7 signal extractors
│       ├── embeddings.py      # Embedding cache + similarity
│       ├── fusion.py          # LogisticRegression + calibration
│       └── reasoning.py       # Factual reasoning generator
└── tests/
    └── test_rank.py
```

---

## Hackathon Compliance

✅ **Format**: CSV with candidate_id, rank, score, reasoning (100 rows)  
✅ **Compute**: ≤5 min, ≤16 GB RAM, CPU-only, no network during ranking  
✅ **No API calls**: All models local (sentence-transformers, sklearn)  
✅ **Honeypot rate**: Conservative penalties, behavioral gate filters inactives  
✅ **Reasoning**: Specific facts, JD-connected, honest concerns, varied templates  
✅ **Reproducible**: Single command `python rank.py --candidates ... --out ...`  
✅ **Sandbox**: Streamlit app at `app.py` for small-sample verification  

---

## License

MIT License - Built for Redrob Hackathon 2026