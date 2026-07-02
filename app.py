"""Streamlit App — Redrob Intelligent Candidate Ranker
Team: ThreeTwoOne | Vaibhav Sharma & Shreya Khantal

Run locally:  streamlit run app.py
Deploy:       Streamlit Cloud (connects to GitHub repo)
"""

from __future__ import annotations

import gzip
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

# ── JD path resolution ───────────────────────────────────────────────────────
_DEFAULT_JD = _HERE / "data" / "job_description.md"
if not _DEFAULT_JD.exists():
    _DEFAULT_JD = _HERE.parent / "Dataset" / "job_description.docx"

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}
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
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}
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
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.8);
    border-right: 1px solid rgba(255,255,255,0.07);
}
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
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    jd_path_input = st.text_input(
        "Job Description Path",
        value=str(_DEFAULT_JD),
        help="Path to the .docx or .md JD file",
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
        help="Ignore cached embeddings and recompute",
    )
    st.divider()
    st.markdown("**Team ThreeTwoOne**")
    st.markdown("Vaibhav Sharma · Shreya Khantal")
    st.markdown("[GitHub](https://github.com/itsVaibhavSharma/India-Runs) | Redrob Hackathon 2026")


# ═══════════════════════════════════════════════════════════════════════════
# ── Helper functions — MUST be defined before any call sites below ─────────
# ═══════════════════════════════════════════════════════════════════════════

def _display_results(df: pd.DataFrame) -> None:
    """Render ranking results to the Streamlit UI."""
    st.markdown(f"### 🏆 Top {len(df)} Ranked Candidates")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top Score",    f"{df['score'].max():.4f}")
    m2.metric("Bottom Score", f"{df['score'].min():.6f}")
    m3.metric("Score Range",  f"{df['score'].max() - df['score'].min():.4f}")
    m4.metric("Ranked",       len(df))

    st.markdown("#### 🥇 Top 10 Candidates")
    top_score = df["score"].max() or 1.0
    for _, row in df.head(10).iterrows():
        bar_pct = int((row["score"] / top_score) * 100)
        st.markdown(
            f'<div class="top-cand">'
            f'<span class="rank-badge">#{int(row["rank"])}</span> '
            f'<strong>{row["candidate_id"]}</strong> — '
            f'Score: <strong>{row["score"]:.6f}</strong>'
            f'<div class="score-bar-outer"><div class="score-bar-inner" style="width:{bar_pct}%"></div></div>'
            f'<p style="color:#cbd5e1;font-size:0.82rem;margin:0.4rem 0 0 0">{row["reasoning"]}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### 📋 Full Rankings Table")
    display_df = df.copy()
    display_df["score"] = display_df["score"].apply(lambda x: f"{x:.6f}")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
            "rank":         st.column_config.NumberColumn("Rank", width="small"),
            "score":        st.column_config.TextColumn("Score", width="small"),
            "reasoning":    st.column_config.TextColumn("Reasoning", width="large"),
        },
    )

    st.markdown("#### 📈 Score Distribution (top 50)")
    chart_df = df.head(50)[["rank", "score"]].set_index("rank")
    st.line_chart(chart_df["score"])

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _run_ranking_ui(
    candidates_path: str,
    jd_path: str,
    cache_dir: str,
    stage1_k: int,
    force_recompute: bool,
) -> None:
    """Run the full ranking pipeline and display results."""
    out_path = str(_HERE / "submission_output.csv")
    progress = st.progress(0, text="Initialising pipeline…")
    status   = st.empty()

    try:
        from ranker.__main__ import Ranker

        if not Path(jd_path).exists():
            st.error(f"❌ JD file not found: {jd_path}")
            st.info(
                "Check the **Job Description Path** in the sidebar. "
                "On Streamlit Cloud the bundled `data/job_description.md` is used automatically."
            )
            return

        start = time.time()
        status.info("⚙️ Parsing job description…")
        progress.progress(5, text="Parsing JD…")

        ranker = Ranker(
            jd_path=jd_path,
            candidates_path=candidates_path,
            cache_dir=cache_dir,
        )

        status.info("📐 Extracting candidate signals & embeddings…")
        progress.progress(20, text="Extracting signals…")

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

    if Path(out_path).exists():
        df = pd.read_csv(out_path)
        _display_results(df)


# ═══════════════════════════════════════════════════════════════════════════
# ── Hero section ────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("<div style='padding-top:0.5rem;font-size:3rem;'>🎯</div>", unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="hero-title">Redrob Intelligent Candidate Ranker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Two-stage hybrid ranking · 7 explainable signals · '
        'Honeypot detection · CPU-only offline inference</div>',
        unsafe_allow_html=True,
    )

tab_run, tab_demo, tab_about, tab_signals = st.tabs(
    ["🚀 Run Ranker", "🎬 Quick Demo", "📖 About the Pipeline", "📊 Signal Details"]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Run Ranker
# ═══════════════════════════════════════════════════════════════════════════
with tab_run:
    st.markdown("### 📁 Upload Candidates File")
    st.markdown(
        '<div class="card">Upload a <code>.jsonl</code>, <code>.json</code>, or '
        '<code>.jsonl.gz</code> file (up to 500 MB). For small demos the '
        '<strong>🎬 Quick Demo</strong> tab runs instantly with no upload.</div>',
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
            "You can also use the **🎬 Quick Demo** tab to run the pre-loaded 50-candidate sample."
        )
    else:
        name = uploaded.name
        suffix = ".jsonl.gz" if name.endswith(".jsonl.gz") else ("." + name.rsplit(".", 1)[-1])
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded.read())
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        st.success(f"✅ Uploaded **{uploaded.name}** ({uploaded.size / 1_048_576:.1f} MB)")

        # ── Preview ─────────────────────────────────────────────────────────
        with st.expander("🔍 Preview first 5 candidates"):
            try:
                if tmp_path.endswith(".gz"):
                    with gzip.open(tmp_path, "rt", encoding="utf-8") as f:
                        raw = [json.loads(l) for i, l in enumerate(f) if i < 5]
                else:
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        first = f.read(1); f.seek(0)
                        raw = json.load(f)[:5] if first == "[" else \
                              [json.loads(l) for i, l in enumerate(f) if i < 5 and l.strip()]

                for item in raw:
                    p   = item.get("profile", {})
                    sig = item.get("redrob_signals", {})
                    st.json({
                        "candidate_id":    item.get("candidate_id"),
                        "title":           p.get("current_title"),
                        "company":         p.get("current_company"),
                        "experience_yrs":  p.get("years_of_experience"),
                        "location":        p.get("location"),
                        "skills_count":    len(item.get("skills", [])),
                        "open_to_work":    sig.get("open_to_work_flag"),
                        "last_active":     sig.get("last_active_date"),
                        "response_rate":   sig.get("recruiter_response_rate"),
                        "github_activity": sig.get("github_activity_score"),
                    })
            except Exception as exc:
                st.warning(f"Preview failed: {exc}")

        run_btn = st.button("🚀 Run Ranking Pipeline", type="primary", use_container_width=True)
        if run_btn:
            _run_ranking_ui(tmp_path, jd_path_input, cache_dir, stage1_k, force_recompute)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Quick Demo
# ═══════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("### 🎬 Quick Demo — Pre-loaded Sample Candidates")
    st.markdown(
        '<div class="card">'
        'Runs the full ranking pipeline on the <strong>50 sample candidates</strong> '
        'bundled with the repository. No file upload needed.'
        '</div>',
        unsafe_allow_html=True,
    )

    sample_path = _HERE / "data" / "sample_candidates.json"
    if not sample_path.exists():
        st.warning("Sample data not found at `data/sample_candidates.json`. Use the **🚀 Run Ranker** tab instead.")
    else:
        st.info(f"📊 Sample file: `{sample_path.name}` — 50 candidates")
        demo_btn = st.button("▶️ Run Demo Pipeline", type="primary", use_container_width=True)
        if demo_btn:
            _run_ranking_ui(str(sample_path), jd_path_input, cache_dir, stage1_k, force_recompute)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — About the Pipeline
# ═══════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("## 🧠 Pipeline Architecture")
    st.markdown(
        """
<div class="card">

**Two-stage CPU-only ranking system** — designed for 100K candidates in &lt;5 minutes.

```
INPUT: candidates.jsonl (100K)  +  job_description.docx
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 1 · Signal Extraction + Behavioral Gate          │
│  • 7 independent signal scores per candidate            │
│  • Hard filter: open_to_work + active ≤60d + resp >10% │
│  • Pre-computed 384-dim sentence-transformer embeddings │
│  • Top 500 advance to Stage 2                          │
└─────────────────────────┬────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│  STAGE 2 · Fusion + Calibration + Reasoning             │
│  • LogisticRegression + IsotonicRegression              │
│  • 7 signals + 4 interaction terms (11 features total)  │
│  • 90% signal fusion + 10% semantic similarity          │
│  • Honeypot penalty multiplier                          │
│  • Monotonic score enforcement + [0,1] calibration      │
│  • Tie-break: equal scores → candidate_id ascending     │
│  • Template-based factual reasoning (no LLM needed)     │
└─────────────────────────┬────────────────────────────────┘
                          ▼
OUTPUT: submission.csv — exactly 100 rows
        candidate_id | rank | score | reasoning
```
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### 📊 The 7 Signals")
    st.table(
        pd.DataFrame([
            ["1", "title_career",      "Title match, production evidence, product vs services",    "25%"],
            ["2", "skill_depth",       "Trust-weighted (endorsements × duration × proficiency × JD weight)", "20%"],
            ["3", "experience",        "Years-of-experience fit for the [5-9yr] target band",      "15%"],
            ["4", "education",         "Institution tier (Tier-1/2/3) + field + degree level",     "10%"],
            ["5", "location",          "Preferred city match (Pune/Noida=1.0, NCR=0.85) + relocation", "5%"],
            ["6", "behavioral",        "Gate: open_to_work + ≤60d active + >10% response rate",   "15%"],
            ["7", "honeypot_penalty",  "Timeline impossibility, skill inflation, salary anomalies", "−15%"],
        ], columns=["#", "Signal", "Description", "Weight"])
    )

    st.markdown("### 🚫 Honeypot Detection Rules")
    st.markdown(
        """
1. **Tenure > total experience** — impossible career timeline  
2. **Skill inflation** — expert in 10+ skills, all with <12 months duration  
3. **Company age mismatch** — claimed tenure longer than company has existed  
4. **Salary/experience mismatch** — <3 years requesting >40 LPA  
5. **Education anomalies** — graduation year in the future  
6. **Keyword stuffing** — AI skills on services background with no depth  
"""
    )

    st.markdown("### ⏱️ Runtime Profile (100K candidates)")
    st.table(
        pd.DataFrame([
            ["Load + parse 100K JSONL",   "~30s",  "~500 MB"],
            ["Signal extraction (7×100K)","~90s",  "~800 MB"],
            ["Embedding similarity",      "~30s",  "~200 MB"],
            ["Fusion + calibration",      "~10s",  "~50 MB"],
            ["Reasoning generation",      "~20s",  "~100 MB"],
            ["**TOTAL**",                 "**~3 min**", "**<2 GB**"],
        ], columns=["Stage", "Time", "Memory"])
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Signal Details
# ═══════════════════════════════════════════════════════════════════════════
with tab_signals:
    st.markdown("## 📊 Live Signal Analyser")
    st.markdown("Paste any candidate JSON below to see the full signal breakdown and score.")

    _SAMPLE_CAND = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "current_title": "Senior ML Engineer",
            "current_company": "Swiggy",
            "current_industry": "Technology",
            "years_of_experience": 6.5,
            "location": "Bengaluru",
            "country": "India",
            "summary": "ML engineer building production retrieval and ranking systems.",
            "headline": "Senior ML Engineer | Embeddings | Ranking | Vector Search",
            "anonymized_name": "Candidate A",
            "current_company_size": "5001-10000",
        },
        "career_history": [{
            "company": "Swiggy",
            "title": "Senior ML Engineer",
            "start_date": "2021-01-01",
            "end_date": None,
            "duration_months": 42,
            "is_current": True,
            "industry": "Technology",
            "company_size": "5001-10000",
            "description": (
                "Built FAISS-based dense retrieval system with sentence-transformers for food search. "
                "Deployed to production serving 10M+ daily users. Learning-to-rank with XGBoost ranking. "
                "Evaluated with NDCG@10 and MRR. A/B tested ranking improvements. "
                "Hybrid search combining BM25 and dense embeddings."
            ),
        }],
        "education": [{
            "institution": "IIT Delhi",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "start_year": 2015,
            "end_year": 2019,
            "grade": "8.9 CGPA",
            "tier": "tier_1",
        }],
        "skills": [
            {"name": "Python",               "proficiency": "expert",    "endorsements": 45, "duration_months": 72},
            {"name": "FAISS",                "proficiency": "advanced",  "endorsements": 12, "duration_months": 42},
            {"name": "sentence-transformers","proficiency": "advanced",  "endorsements": 8,  "duration_months": 42},
            {"name": "XGBoost",              "proficiency": "advanced",  "endorsements": 15, "duration_months": 36},
            {"name": "Elasticsearch",        "proficiency": "advanced",  "endorsements": 10, "duration_months": 30},
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
    }

    cand_json_str = st.text_area(
        "Candidate JSON",
        value=json.dumps(_SAMPLE_CAND, indent=2),
        height=280,
    )

    if st.button("🔬 Analyse Signals", key="analyse_btn"):
        try:
            from ranker.candidate_loader import CandidateLoader
            from ranker.signals import SignalExtractor
            from ranker.fusion import SignalFusion
            from ranker.jd_parser import parse_jd
            from ranker.reasoning import generate_reasoning

            # Parse the candidate JSON
            loader = CandidateLoader.__new__(CandidateLoader)
            loader.candidates_path = Path("dummy")
            candidate = loader._parse_candidate(json.loads(cand_json_str))

            # Parse JD if available
            req = None
            jd_p = Path(jd_path_input)
            if jd_p.exists():
                req = parse_jd(str(jd_p))

            # Extract signals
            extractor = SignalExtractor(req)
            scores    = extractor.extract_all_signals(candidate)
            sig_dict  = scores.to_dict()

            st.markdown("### Signal Scores")
            _signal_labels = {
                "title_career":     "🏷️ Title / Career",
                "skill_depth":      "🔧 Skill Depth",
                "experience":       "📅 Experience",
                "education":        "🎓 Education",
                "location":         "📍 Location",
                "behavioral":       "💚 Behavioral Gate",
                "honeypot_penalty": "⚠️ Honeypot Penalty",
            }
            cols = st.columns(2)
            for i, (key, label) in enumerate(_signal_labels.items()):
                val = sig_dict.get(key, 0.0)
                colour = "#ef4444" if key == "honeypot_penalty" and val > 0 else "#a78bfa"
                with cols[i % 2]:
                    st.markdown(
                        f"**{label}**: `{val:.3f}`"
                        f'<div class="score-bar-outer"><div class="score-bar-inner" '
                        f'style="width:{val*100:.0f}%;background:{colour}"></div></div>',
                        unsafe_allow_html=True,
                    )
                    st.write("")

            # Fusion score
            fusion = SignalFusion(model_dir=str(_HERE / "models" / "fusion"))
            if not fusion.load():
                with st.spinner("Training fusion model…"):
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