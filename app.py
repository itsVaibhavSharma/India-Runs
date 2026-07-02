"""
Streamlit App for Redrob Candidate Ranker - Deployment Demo
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ranker.__main__ import Ranker

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Redrob Intelligent Candidate Ranker")
st.markdown("""
Rank candidates for **Senior AI Engineer - Founding Team** at Redrob AI.
Upload a candidate JSONL file and get top 100 ranked results with explanations.
""")

# Sidebar
st.sidebar.header("Configuration")
jd_path = st.sidebar.text_input(
    "Job Description Path",
    value="../Dataset/job_description.docx",
    help="Path to the JD file"
)

cache_dir = st.sidebar.text_input(
    "Cache Directory",
    value="models",
    help="Directory for cached embeddings and models"
)

force_recompute = st.sidebar.checkbox("Force Recompute Embeddings", value=False)

# File upload
st.header("📁 Upload Candidates")
uploaded_file = st.file_uploader(
    "Upload candidates.jsonl or candidates.jsonl.gz",
    type=['jsonl', 'gz'],
    help="Maximum 100K candidates. For demo, upload a small sample."
)

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size:,} bytes)")

    # Preview
    if st.loader
    if st.button("🔍 Preview Data (first 5)"):
        try:
            if tmp_path.endswith('.gz'):
                import gzip
                with gzip.open(tmp_path, 'rt', encoding='utf-8') as f:
                    lines = [next(f) for _ in range(5)]
            else:
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    lines = [next(f) for _ in range(5)]

            for i, line in enumerate(lines):
                data = json.loads(line)
                st.json({
                    "candidate_id": data.get("candidate_id"),
                    "name": data.get("profile", {}).get("anonymized_name"),
                    "title": data.get("profile", {}).get("current_title"),
                    "company": data.get("profile", {}).get("current_company"),
                    "experience": data.get("profile", {}).get("years_of_experience"),
                    "location": data.get("profile", {}).get("location"),
                    "skills_count": len(data.get("skills", [])),
                    "last_active": data.get("redrob_signals", {}).get("last_active_date"),
                    "response_rate": data.get("redrob_signals", {}).get("recruiter_response_rate"),
                })
        except Exception as e:
            st.error(f"Preview error: {e}")

    # Run ranking
    if st.button("🚀 Rank Candidates", type="primary"):
        output_path = "submission.csv"

        with st.spinner("Ranking candidates... This may take a few minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                ranker = Ranker(jd_path, tmp_path, cache_dir)

                # Patch to show progress
                original_run = ranker.run

                def run_with_progress(output_path, force_recompute=False):
                    # We can't easily hook into the internal progress, so just run
                    return original_run(output_path, force_recompute)

                run_with_progress(output_path, force_recompute)
                progress_bar.progress(1.0)
                status_text.success("Ranking complete!")

            except Exception as e:
                st.error(f"Ranking failed: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                # Cleanup temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        # Display results
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            st.header(f"📊 Top {len(df)} Ranked Candidates")

            # Format display
            display_df = df.copy()
            display_df['score'] = display_df['score'].apply(lambda x: f"{x:.4f}")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "candidate_id": "Candidate ID",
                    "rank": "Rank",
                    "score": "Score",
                    "reasoning": "Reasoning"
                }
            )

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download submission.csv",
                data=csv,
                file_name="submission.csv",
                mime="text/csv"
            )

            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Top Score", f"{df['score'].max():.4f}")
            with col2:
                st.metric("Score Range", f"{df['score'].max() - df['score'].min():.4f}")
            with col3:
                st.metric("Candidates", len(df))

else:
    st.info("👆 Upload a candidates JSONL file to start ranking")

# Footer
st.markdown("---")
st.markdown("""
**Team:** ThreeTwoOne | **Leader:** Vaibhav Sharma  
**Track:** Intelligent Candidate Discovery & Ranking Challenge
""")