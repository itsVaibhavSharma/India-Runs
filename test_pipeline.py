#!/usr/bin/env python3
"""
Quick test script to verify the ranking pipeline works on sample data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ranker import (
    CandidateLoader, parse_jd, SignalExtractor, EmbeddingManager,
    SignalFusion, generate_reasoning
)


def test_jd_parser():
    print("Testing JD Parser...")
    req = parse_jd('../Dataset/job_description.docx')
    print(f"  Must-have skills: {len(req.must_have_skills)}")
    print(f"  Nice-to-have: {len(req.nice_to_have_skills)}")
    print(f"  Exp range: {req.required_experience_range}")
    print(f"  Preferred locations: {req.preferred_locations}")
    print("  ✓ JD Parser works")


def test_candidate_loader():
    print("\nTesting Candidate Loader...")
    loader = CandidateLoader('../Dataset/sample_candidates.json')
    candidates = loader.load_all_candidates()
    print(f"  Loaded {len(candidates)} candidates")
    print(f"  First candidate: {candidates[0].candidate_id}")
    print(f"  Title: {candidates[0].current_title}")
    print(f"  Skills: {len(candidates[0].skills)}")
    print("  ✓ Candidate Loader works")


def test_signal_extractor():
    print("\nTesting Signal Extractor...")
    loader = CandidateLoader('../Dataset/sample_candidates.json')
    candidates = loader.load_all_candidates()

    req = parse_jd('../Dataset/job_description.docx')
    extractor = SignalExtractor(req)

    # Test on first 5 candidates
    for i, cand in enumerate(candidates[:5]):
        signals = extractor.extract_all_signals(cand)
        print(f"  {cand.candidate_id}:")
        print(f"    Title/Career: {signals.title_career:.3f}")
        print(f"    Skill Depth: {signals.skill_depth:.3f}")
        print(f"    Experience: {signals.experience:.3f}")
        print(f"    Education: {signals.education:.3f}")
        print(f"    Location: {signals.location:.3f}")
        print(f"    Behavioral: {signals.behavioral:.3f}")
        print(f"    Honeypot: {signals.honeypot_penalty:.3f}")
    print("  ✓ Signal Extractor works")


def test_reasoning():
    print("\nTesting Reasoning Generator...")
    loader = CandidateLoader('../Dataset/sample_candidates.json')
    candidates = loader.load_all_candidates()

    req = parse_jd('../Dataset/job_description.docx')
    extractor = SignalExtractor(req)

    for cand in candidates[:3]:
        signals = extractor.extract_all_signals(cand)
        signal_dict = {
            'title_career': signals.title_career,
            'skill_depth': signals.skill_depth,
            'experience': signals.experience,
            'education': signals.education,
            'location': signals.location,
            'behavioral': signals.behavioral,
            'honeypot_penalty': signals.honeypot_penalty
        }
        reasoning = generate_reasoning(cand, signal_dict, 1, 0.95, req)
        print(f"  {cand.candidate_id}: {reasoning[:100]}...")
    print("  ✓ Reasoning Generator works")


def test_embeddings():
    print("\nTesting Embedding Manager...")
    mgr = EmbeddingManager('models/embeddings')

    # Test JD embedding
    jd_emb = mgr.get_jd_embedding("Senior AI Engineer with production embeddings experience")
    print(f"  JD embedding shape: {jd_emb.shape}")

    # Test candidate embeddings (just verify cache exists)
    cache_dir = Path('models/embeddings')
    if (cache_dir / 'candidate_embeddings.npy').exists():
        print("  Candidate embeddings cached ✓")
    else:
        print("  Candidate embeddings not yet cached (run precompute.py)")
    print("  ✓ Embedding Manager works")


def test_fusion():
    print("\nTesting Fusion Model...")
    fusion = SignalFusion('models/fusion')

    # Test with sample signals
    sample_signals = {
        'title_career': 0.8,
        'skill_depth': 0.75,
        'experience': 0.9,
        'education': 0.7,
        'location': 0.8,
        'behavioral': 0.85,
        'honeypot_penalty': 0.0
    }

    # Try loading cached model
    if fusion.load():
        score = fusion.predict_proba(sample_signals)
        print(f"  Sample score: {score:.4f}")
        print("  ✓ Fusion model loaded from cache")
    else:
        # Train on synthetic data
        req = parse_jd('../Dataset/job_description.docx')
        fusion.train(req)
        score = fusion.predict_proba(sample_signals)
        print(f"  Sample score: {score:.4f}")
        print("  ✓ Fusion model trained and works")


def main():
    print("=" * 60)
    print("Running Component Tests")
    print("=" * 60)

    test_jd_parser()
    test_candidate_loader()
    test_signal_extractor()
    test_reasoning()
    test_embeddings()
    test_fusion()

    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == '__main__':
    main()