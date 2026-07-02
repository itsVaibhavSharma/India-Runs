"""Streamlit App — Redrob Intelligent Candidate Ranker
Team: ThreeTwoOne | Vaibhav Sharma & Shreya Khantal

Run locally:  streamlit run app.py
Deploy:       Streamlit Cloud (connects to GitHub repo)
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

# Suppress TF/transformers noise in the UI
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redrob Ranker · ThreeTwoOne",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global fonts & palette ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero header */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

/* Metric pills */
.metric-pill {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: #c4b5fd;
    margin: 0.2rem;
}

/* Score bar */
.score-bar-outer {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.score-bar-inner {
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
}

/* Top candidate card */
.top-cand {
    background: linear-gradient(135deg, rgba(167,139,250,0.12), rgba(96,165,250,0.08));
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    transition: transform 0.15s;
}
.top-cand:hover { transform: translateY(-2px); }
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    color: white;
    font-weight: 700;
    font-size: 0.8rem;
    padding: 0.1rem 0.6rem;
    border-radius: 999px;
    margin-right: 0.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.8);
    border-right: 1px solid rgba(255,255,255,0.07);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Success/info colors */
.stSuccess { background: rgba(52,211,153,0.1) !important; border-radius: 8px !important; }
.stInfo    { background: rgba(96,165,250,0.1) !important; border-radius: 8px !important; }
.stWarning { background: rgba(251,191,36,0.1) !important; border-radius: 8px !important; }

/* DataFrame */
.dataframe { font-size: 0.85rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ── JD path resolution ───────────────────────────────────────────────────────
# Prefer the bundled markdown JD (works on Streamlit Cloud without extra deps)
_DEFAULT_JD = _HERE / "data" / "job_description.md"
if not _DEFAULT_JD.exists():
    # Fallback: look for docx in parent Dataset folder (local dev)
    _DEFAULT_JD = _HERE.parent / "Dataset" / "job_description.docx"

# ── Hero section ────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("<div style='padding-top:0.5rem;font-size:3rem;'>🎯</div>", unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="hero-title">Redrob Intelligent Candidate Ranker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Two-stage hybrid ranking • 7 explainable signals • '
        'Honeypot detection • CPU-only offline inference</div>',
        unsafe_allow_html=True,
    )

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    jd_path_input = st.text_input(
        "Job Description Path",
        value=str(_DEFAULT_JD),
        help="Path to the .docx or .md JD file. On Streamlit Cloud, uses the bundled data/job_description.md",
    )
    cache_dir = st.text_input(
        "Cache Directory",
        value=str(_HERE / "models"),
        help="Directory with cached embeddings and models",
    )
    stage1_k = st.slider(
        "Stage-1 Candidates (semantic filter)",
        min_value=50, max_value=2000, value=500, step=50,
    )
    force_recompute = st.checkbox(
        "Force Recompute Embeddings",
        value=False,
        help="Ignore cached embeddings and recompute (slow for large files)",
    )

    st.divider()
    st.markdown("**Team ThreeTwoOne**")
    st.markdown("Vaibhav Sharma · Shreya Khantal")
    st.markdown(
        "[GitHub](https://github.com/itsVaibhavSharma/India-Runs) | "
        "Redrob Hackathon 2026"
    )

# ── Main area ────────────────────────────────────────────────────────────────
tab_run, tab_demo, tab_about, tab_signals = st.tabs(
    ["🚀 Run Ranker", "🎬 Quick Demo", "📖 About the Pipeline", "📊 Signal Details"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Run Ranker
# ══════════════════════════════════════════════════════════════════════════════
with tab_run:
    st.markdown("### 📁 Upload Candidates File")
    st.markdown(
        '<div class="card">Upload a <code>.jsonl</code>, <code>.json</code>, or '
        '<code>.jsonl.gz</code> file. For the Streamlit demo, a sample of ≤1000 '
        'candidates works best. The full 100K run is designed for local execution.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop candidates file here",
        type=["jsonl", "json", "gz"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.info(
            "👆 Upload a candidates file to start ranking. "
            "You can use **sample_candidates.json** from the Dataset folder for a quick demo, "
            "or switch to the **🎬 Quick Demo** tab to run with pre-loaded sample data."
        )

    else:
        # ── Save upload to temp file ────────────────────────────────────────
        name = uploaded.name
        # Determine suffix (handle .jsonl.gz double extension)
        if name.endswith(".jsonl.gz"):
            suffix = ".jsonl.gz"
        else:
            suffix = "." + name.split(".")[-1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        st.success(f"✅ Uploaded **{uploaded.name}** ({uploaded.size:,} bytes)")

        # ── Preview ─────────────────────────────────────────────────────────
        with st.expander("🔍 Preview first 5 candidates"):
            try:
                if tmp_path.endswith(".gz"):
                    with gzip.open(tmp_path, "rt", encoding="utf-8") as f:
                        lines = []
                        for i, line in enumerate(f):
                            if i >= 5:
                                break
                            lines.append(json.loads(line))
                        raw = lines
                else:
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        first = f.read(1)
                        f.seek(0)
                        if first == "[":
                            raw = json.load(f)[:5]
                        else:
                            raw = []
                            for i, line in enumerate(f):
                                if i >= 5:
                                    break
                                if line.strip():
                                    raw.append(json.loads(line))

                for item in raw:
                    p = item.get("profile", {})
                    sig = item.get("redrob_signals", {})
                    st.json({
                        "candidate_id": item.get("candidate_id"),
                        "title": p.get("current_title"),
                        "company": p.get("current_company"),
                        "experience_yrs": p.get("years_of_experience"),
                        "location": p.get("location"),
                        "skills_count": len(item.get("skills", [])),
                        "open_to_work": sig.get("open_to_work_flag"),
                        "last_active": sig.get("last_active_date"),
                        "response_rate": sig.get("recruiter_response_rate"),
                        "github_activity": sig.get("github_activity_score"),
                    })
            except Exception as exc:
                st.warning(f"Preview failed: {exc}")

        # ── Run Ranking ──────────────────────────────────────────────────────
        run_btn = st.button("🚀 Run Ranking Pipeline", type="primary", use_container_width=True)

        if run_btn:
            _run_ranking_ui(tmp_path, jd_path_input, cache_dir, stage1_k, force_recompute)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _run_ranking_ui(candidates_path: str, jd_path: str, cache_dir: str,
                    stage1_k: int, force_recompute: bool) -> None:
    """Run the full ranking pipeline and display results in Streamlit."""
    out_path = str(_HERE / "submission_output.csv")
    progress = st.progress(0, text="Initialising pipeline…")
    status = st.empty()

    try:
        from ranker.__main__ import Ranker

        start = time.time()

        if not Path(jd_path).exists():
            st.error(f"❌ JD file not found: {jd_path}")
            st.info("Tip: Check the 'Job Description Path' in the sidebar. On Streamlit Cloud, the bundled `data/job_description.md` is used automatically.")
            return

        status.info("⚙️ Parsing job description…")
        progress.progress(5, text="Parsing JD…")

        ranker = Ranker(jd_path=jd_path, candidates_path=candidates_path, cache_dir=cache_dir)

        status.info("📐 Extracting candidate signals & embeddings…")
        progress.progress(20, text="Loading candidates and extracting signals…")

        ranker.run(
            output_path=out_path,
            force_recompute=force_recompute,
            final_top_k=100,
            stage1_top_k=stage1_k,
        )

        progress.progress(100, text="Done!")
        elapsed = time.time() - start
        status.success(f"✅ Ranking complete in **{elapsed:.1f}s** ({elapsed/60:.1f} min)")

    except Exception as exc:
        import traceback
        status.error(f"❌ Ranking failed: {exc}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return

    # ── Display results ──────────────────────────────────────────────────────
    if Path(out_path).exists():
        df = pd.read_csv(out_path)
        _display_results(df)


def _display_results(df: pd.DataFrame) -> None:
    """Render the ranking results to the Streamlit UI."""
    st.markdown(f"### 🏆 Top {len(df)} Ranked Candidates")

    # KPI metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top Score", f"{df['score'].max():.4f}")
    m2.metric("Bottom Score", f"{df['score'].min():.4f}")
    m3.metric("Score Range", f"{df['score'].max() - df['score'].min():.4f}")
    m4.metric("Candidates Ranked", len(df))

    # Top-10 cards
    st.markdown("#### 🥇 Top 10 Candidates")
    for _, row in df.head(10).iterrows():
        bar_pct = int(row["score"] * 100)
        st.markdown(
            f'<div class="top-cand">'
            f'<span class="rank-badge">#{int(row["rank"])}</span> '
            f'<strong>{row["candidate_id"]}</strong> — '
            f'Score: <strong>{row["score"]:.4f}</strong>'
            f'<div class="score-bar-outer"><div class="score-bar-inner" style="width:{bar_pct}%"></div></div>'
            f'<p style="color:#cbd5e1;font-size:0.82rem;margin:0.4rem 0 0 0">{row["reasoning"]}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # Full table
    st.markdown("#### 📋 Full Rankings Table")
    display_df = df.copy()
    display_df["score"] = display_df["score"].apply(lambda x: f"{x:.6f}")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "score": st.column_config.TextColumn("Score", width="small"),
            "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
        },
    )

    # Score distribution chart
    st.markdown("#### 📈 Score Distribution")
    score_df = df[["rank", "score"]].copy()
    st.line_chart(score_df.set_index("rank")["score"])

    # Download
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Quick Demo (pre-loaded sample data)
# ══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("### 🎬 Quick Demo — Pre-loaded Sample Candidates")
    st.markdown(
        '<div class="card">'
        'This demo runs the full ranking pipeline on the <strong>50 sample candidates</strong> '
        'bundled with the repository. No file upload required — just click Run!'
        '</div>',
        unsafe_allow_html=True,
    )

    sample_path = _HERE / "data" / "sample_candidates.json"
    if not sample_path.exists():
        st.warning(
            "Sample data not found at `data/sample_candidates.json`. "
            "Please upload a file in the **🚀 Run Ranker** tab instead."
        )
    else:
        st.info(f"📊 Sample file: `{sample_path.name}` — 50 candidates")

        demo_btn = st.button("▶️ Run Demo Pipeline", type="primary", use_container_width=True)
        if demo_btn:
            _run_ranking_ui(
                str(sample_path),
                jd_path_input,
                cache_dir,
                stage1_k,
                force_recompute,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: About the Pipeline
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("## 🧠 Pipeline Architecture")
    st.markdown(
        """
<div class="card">

**Two-stage CPU-only ranking system** targeting 100K candidates in &lt;5 minutes.

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: candidates.jsonl (100K)  +  job_description.docx    │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Signal Extraction + Behavioral Gate (~2.5 min)    │
│  • 7 independent signal scores per candidate                │
│  • Hard filter: open_to_work + active≤60d + response>10%   │
│  • Semantic similarity (pre-computed 384-dim embeddings)    │
│  • Top 500 advance                                          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Fusion + Calibration + Reasoning (~30 sec)        │
│  • LogisticRegression + IsotonicRegression                  │
│  • Trained on synthetic JD-derived preference pairs        │
│  • Combined 90% fusion + 10% semantic similarity            │
│  • Honeypot penalty enforcement                             │
│  • Monotonic score enforcement + [0,1] calibration         │
│  • Template-based reasoning (fact-grounded, no hallucin.)   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: submission.csv (100 rows)                          │
│  candidate_id | rank | score | reasoning                    │
└─────────────────────────────────────────────────────────────┘
```
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("## 🎯 JD Requirements Mapped to Signals")
    st.markdown(
        """
| Requirement | Signal | Weight |
|-------------|--------|--------|
| Production embeddings/retrieval | Title/Career | 25% |
| Vector DB / hybrid search | Skill Depth | 20% |
| Strong Python + code quality | Skill Depth + GitHub | 20% |
| 5-9 years experience | Experience | 15% |
| Active on Redrob platform | Behavioral Gate | 15% |
| Ranking evaluation (NDCG/MRR) | Title/Career | 10% |
| Preferred locations (Pune/Noida) | Location | 5% |
| Product company background | Title/Career | 5% |
| Education (Tier-1 institution) | Education | 5% |
""",
    )

    st.markdown("## 🚫 Disqualifiers Implemented")
    st.markdown(
        """
- **Pure research** (no production deployment evidence) → Title/Career penalty  
- **Services background** (TCS/Infosys/Wipro etc.) → 0.20 penalty on title score  
- **Inactive** (not open to work, >60d, <10% response) → Behavioral gate (score=0)  
- **Job hopping** (>4 companies, <18mo avg tenure) → Honeypot penalty  
- **Keyword stuffing** (expert in 10+ skills, all <12mo duration) → Honeypot penalty  
- **Timeline impossibilities** (tenure > company age, etc.) → Honeypot penalty  
"""
    )

    st.markdown("## ⏱️ Runtime Profile")
    st.markdown(
        """
| Stage | Time | Memory |
|-------|------|--------|
| Load 100K JSONL | ~30s | ~500 MB |
| Signal extraction | ~90s | ~800 MB |
| Embedding similarity | ~30s | ~200 MB |
| Fusion + calibration | ~10s | ~50 MB |
| Reasoning generation | ~20s | ~100 MB |
| **TOTAL** | **~3 min** | **< 2 GB** |
"""
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Signal Details
# ══════════════════════════════════════════════════════════════════════════════
with tab_signals:
    st.markdown("## 📊 Signal Breakdown Explorer")
    st.markdown("Paste a candidate JSON to see its full signal breakdown:")

    sample_json = json.dumps(
        {
            "candidate_id": "CAND_0000001",
            "profile": {
                "current_title": "Senior ML Engineer",
                "current_company": "Swiggy",
                "current_industry": "Technology",
                "years_of_experience": 6.5,
                "location": "Bengaluru",
                "country": "India",
                "summary": "ML engineer with 6.5 years building production retrieval systems.",
                "headline": "Senior ML Engineer | Embeddings | Ranking | Vector Search",
            },
            "career_history": [
                {
                    "company": "Swiggy",
                    "title": "Senior ML Engineer",
                    "start_date": "2022-01-01",
                    "end_date": None,
                    "duration_months": 30,
                    "is_current": True,
                    "industry": "Technology",
                    "company_size": "5001-10000",
                    "description": "Built embedding-based retrieval system using FAISS and sentence-transformers for food search. Deployed to production serving millions of users. Implemented learning-to-rank with XGBoost. Evaluated with NDCG@10 and MRR. A/B tested ranking improvements.",
                }
            ],
            "education": [
                {
                    "institution": "IIT Delhi",
                    "degree": "B.Tech",
                    "field_of_study": "Computer Science",
                    "start_year": 2015,
                    "end_year": 2019,
                    "grade": "8.9 CGPA",
                    "tier": "tier_1",
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 45, "duration_months": 72},
                {"name": "FAISS", "proficiency": "advanced", "endorsements": 12, "duration_months": 30},
                {"name": "sentence-transformers", "proficiency": "advanced", "endorsements": 8, "duration_months": 30},
                {"name": "XGBoost", "proficiency": "advanced", "endorsements": 15, "duration_months": 36},
            ],
            "certifications": [],
            "languages": [{"language": "English", "proficiency": "professional"}],
            "redrob_signals": {
                "profile_completeness_score": 92.0,
                "signup_date": "2025-01-01",
                "last_active_date": "2026-06-25",
                "open_to_work_flag": True,
                "profile_views_received_30d": 35,
                "applications_submitted_30d": 3,
                "recruiter_response_rate": 0.72,
                "avg_response_time_hours": 4.5,
                "skill_assessment_scores": {"Python": 88.0, "Machine Learning": 79.0},
                "connection_count": 420,
                "endorsements_received": 67,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 35, "max": 55},
                "preferred_work_mode": "hybrid",
                "willing_to_relocate": True,
                "github_activity_score": 62.0,
                "search_appearance_30d": 180,
                "saved_by_recruiters_30d": 8,
                "interview_completion_rate": 0.90,
                "offer_acceptance_rate": 0.75,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            },
        },
        indent=2,
    )

    cand_json_str = st.text_area("Candidate JSON", value=sample_json, height=300)

    if st.button("🔬 Analyse Signals", key="analyse_btn"):
        try:
            from ranker.candidate_loader import CandidateLoader
            from ranker.signals import SignalExtractor
            from ranker.fusion import SignalFusion
            from ranker.jd_parser import parse_jd
            from ranker.reasoning import generate_reasoning

            loader = CandidateLoader.__new__(CandidateLoader)
            loader.candidates_path = Path("dummy")
            cand_data = json.loads(cand_json_str)
            candidate = loader._parse_candidate(cand_data)

            req = None
            jd_p = Path(jd_path_input)
            if jd_p.exists():
                req = parse_jd(str(jd_p))

            extractor = SignalExtractor(req)
            scores = extractor.extract_all_signals(candidate)
            sig_dict = scores.to_dict()

            # Display signal bars
            st.markdown("### Signal Scores")
            signal_labels = {
                "title_career": "🏷️ Title / Career",
                "skill_depth": "🔧 Skill Depth",
                "experience": "📅 Experience",
                "education": "🎓 Education",
                "location": "📍 Location",
                "behavioral": "💚 Behavioral",
                "honeypot_penalty": "⚠️ Honeypot Penalty",
            }
            for key, label in signal_labels.items():
                val = sig_dict.get(key, 0.0)
                colour = "#ef4444" if key == "honeypot_penalty" and val > 0 else "#a78bfa"
                st.markdown(
                    f"**{label}**: `{val:.3f}`"
                    f'<div class="score-bar-outer"><div class="score-bar-inner" '
                    f'style="width:{val*100:.0f}%;background:{colour}"></div></div>',
                    unsafe_allow_html=True,
                )

            # Fusion score
            fusion = SignalFusion(model_dir=str(_HERE / "models" / "fusion"))
            if not fusion.load():
                fusion.train(req)
            final_score = fusion.predict_proba(sig_dict)
            st.metric("🎯 Fusion Score", f"{final_score:.4f}")

            # Reasoning
            reasoning = generate_reasoning(candidate, sig_dict, 1, final_score, req)
            st.markdown("### 📝 Generated Reasoning")
            st.info(reasoning)

        except Exception as exc:
            import traceback
            st.error(f"Analysis failed: {exc}")
            st.code(traceback.format_exc())

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="color:#475569;text-align:center;font-size:0.82rem;">'
    "Team ThreeTwoOne · Vaibhav Sharma &amp; Shreya Khantal · "
    "Redrob Hackathon 2026 · Intelligent Candidate Discovery &amp; Ranking</p>",
    unsafe_allow_html=True,
)