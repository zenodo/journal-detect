#!/usr/bin/env python3
"""
Threshold Analysis for Journal Runner Detection
"""

import pandas as pd
import numpy as np

# Configuration
CONFIG = {
    'min_records_threshold': 20,
    'external_doi_consistency_threshold': 0.7,
    'min_upload_regularity': 0.1,
    'max_spam_ratio': 0.1,
    'repetitive_author_threshold': 0.3,
}

def analyze_thresholds():
    """Analyze each threshold step by step."""
    
    print("=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)
    
    # Load data
    try:
        users_df = pd.read_parquet('data/users.parquet')
        print(f"Loaded {len(users_df)} users")
    except FileNotFoundError:
        print("❌ Data files not found. Please run the main analysis first.")
        return
    
    total_users = len(users_df)
    current_candidates = users_df.copy()
    
    print(f"\n📊 Starting with {total_users} total users")
    
    # Test each threshold
    tests = [
        ("Min Records (≥20)", lambda df: df['n_records'] >= CONFIG['min_records_threshold']),
        ("External DOI Consistency (≥0.7)", lambda df: df['external_doi_consistency'] >= CONFIG['external_doi_consistency_threshold']),
        ("Upload Regularity (≥0.1)", lambda df: df['upload_regularity'] >= CONFIG['min_upload_regularity']),
        ("Spam Ratio (≤0.1)", lambda df: df['spam_record_ratio'] <= CONFIG['max_spam_ratio']),
        ("No Safe Communities", lambda df: ~df['has_safe_community']),
        ("Repetitive Author (≥0.3)", lambda df: df['repetitive_author_score'] >= CONFIG['repetitive_author_threshold'])
    ]
    
    print("\n🔍 Individual threshold impact:")
    for i, (test_name, test_func) in enumerate(tests):
        passed = len(users_df[test_func(users_df)])
        percentage = (passed / total_users) * 100
        print(f"   {i+1}. {test_name}: {passed:,} users ({percentage:.1f}%)")
    
    print("\n🔍 Sequential threshold application:")
    for i, (test_name, test_func) in enumerate(tests):
        before_count = len(current_candidates)
        current_candidates = current_candidates[test_func(current_candidates)]
        after_count = len(current_candidates)
        
        print(f"   {i+1}. {test_name}")
        print(f"      Before: {before_count:,} users")
        print(f"      After:  {after_count:,} users")
        print(f"      Removed: {before_count - after_count:,} users")
        
        if after_count == 0:
            print(f"      ❌ PROBLEM: This threshold eliminates ALL users!")
            break
    
    print(f"\n🎯 Final candidates: {len(current_candidates):,} users")

if __name__ == "__main__":
    analyze_thresholds() 