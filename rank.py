#!/usr/bin/env python3
"""
Redrob Candidate Ranker - Entry Point
Run: python rank.py --candidates candidates.jsonl --out submission.csv
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ranker.__main__ import main

if __name__ == '__main__':
    main()