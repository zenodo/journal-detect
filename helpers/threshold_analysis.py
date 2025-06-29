#!/usr/bin/env python3
"""
Threshold Analysis for Journal Runner Detection
"""

import pandas as pd
import numpy as np
from typing import Callable, List, Tuple
from dataclasses import dataclass

@dataclass
class ThresholdTest:
    """Represents a threshold test with name and function."""
    name: str
    test_func: Callable[[pd.DataFrame], pd.Series]
    description: str = ""

# Configuration
CONFIG = {
    'min_records_threshold': 20,
    'external_doi_consistency_threshold': 0.7,
    'min_upload_regularity': 0.1,
    'max_spam_ratio': 0.1,
    'repetitive_author_threshold': 0.3,
}

def create_threshold_tests() -> List[ThresholdTest]:
    """Create list of threshold tests to apply."""
    return [
        ThresholdTest(
            "Min Records (≥20)", 
            lambda df: df['n_records'] >= CONFIG['min_records_threshold'],
            "Minimum number of records required"
        ),
        ThresholdTest(
            "External DOI Consistency (≥0.7)", 
            lambda df: df['external_doi_consistency'] >= CONFIG['external_doi_consistency_threshold'],
            "Consistency in external DOI usage"
        ),
        ThresholdTest(
            "Upload Regularity (≥0.1)", 
            lambda df: df['upload_regularity'] >= CONFIG['min_upload_regularity'],
            "Regularity in upload patterns"
        ),
        ThresholdTest(
            "Spam Ratio (≤0.1)", 
            lambda df: df['spam_record_ratio'] <= CONFIG['max_spam_ratio'],
            "Maximum allowed spam record ratio"
        ),
        ThresholdTest(
            "No Safe Communities", 
            lambda df: ~df['has_safe_community'],
            "Exclude users with safe community associations"
        ),
        ThresholdTest(
            "Repetitive Author (≥0.3)", 
            lambda df: df['no_repetitive_author_score'] >= CONFIG['repetitive_author_threshold'],
            "Minimum repetitive author score"
        )
    ]

def analyze_thresholds() -> None:
    """Analyze each threshold step by step."""
    
    print("=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)
    
    # Load data
    try:
        users_df = pd.read_parquet('data/users.parquet')
        print(f"✅ Loaded {len(users_df):,} users")
    except FileNotFoundError:
        print("❌ Data files not found. Please run the main analysis first.")
        return
    
    total_users = len(users_df)
    current_candidates = users_df.copy()
    threshold_tests = create_threshold_tests()
    
    print(f"\n📊 Starting with {total_users:,} total users")
    
    # Test each threshold individually
    print("\n🔍 Individual threshold impact:")
    for i, test in enumerate(threshold_tests, 1):
        passed = len(users_df[test.test_func(users_df)])
        percentage = (passed / total_users) * 100
        print(f"   {i}. {test.name}: {passed:,} users ({percentage:.1f}%)")
    
    # Test thresholds sequentially
    print("\n🔍 Sequential threshold application:")
    for i, test in enumerate(threshold_tests, 1):
        before_count = len(current_candidates)
        current_candidates = current_candidates[test.test_func(current_candidates)]
        after_count = len(current_candidates)
        
        print(f"   {i}. {test.name}")
        print(f"      Before: {before_count:,} users")
        print(f"      After:  {after_count:,} users")
        print(f"      Removed: {before_count - after_count:,} users")
        
        if after_count == 0:
            print(f"      ❌ PROBLEM: This threshold eliminates ALL users!")
            break
    
    print(f"\n🎯 Final candidates: {len(current_candidates):,} users")
    
    # Show final candidate characteristics if any exist
    if len(current_candidates) > 0:
        print(f"\n📈 Final candidate characteristics:")
        print(f"   • Average records: {current_candidates['n_records'].mean():.1f}")
        print(f"   • Average DOI consistency: {current_candidates['external_doi_consistency'].mean():.3f}")
        print(f"   • Average repetitive author score: {current_candidates['no_repetitive_author_score'].mean():.3f}")

if __name__ == "__main__":
    analyze_thresholds() 