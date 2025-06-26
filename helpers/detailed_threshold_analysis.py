#!/usr/bin/env python3
"""
Detailed Threshold Analysis - Focus on Primary Author Ratio
"""

import pandas as pd
import numpy as np

def analyze_primary_author_ratio():
    """Analyze why the primary author ratio threshold is so restrictive."""
    
    print("=" * 80)
    print("DETAILED ANALYSIS: PRIMARY AUTHOR RATIO THRESHOLD")
    print("=" * 80)
    
    # Load data
    try:
        users_df = pd.read_parquet('data/users.parquet')
        records_df = pd.read_parquet('data/records.parquet')
        print(f"Loaded {len(users_df)} users and {len(records_df)} records")
    except FileNotFoundError:
        print("❌ Data files not found. Please run the main analysis first.")
        return
    
    # Filter to users with >= 20 records and external DOI consistency >= 0.7
    filtered_users = users_df[
        (users_df['n_records'] >= 20) &
        (users_df['external_doi_consistency'] >= 0.7)
    ]
    
    print(f"\n📊 Users after first two filters: {len(filtered_users)}")
    
    # Analyze primary author ratio distribution
    print("\n🔍 Primary Author Ratio Analysis:")
    print("=" * 50)
    
    # Show distribution
    ratios = filtered_users['primary_author_ratio']
    print(f"   Min: {ratios.min():.3f}")
    print(f"   Max: {ratios.max():.3f}")
    print(f"   Mean: {ratios.mean():.3f}")
    print(f"   Median: {ratios.median():.3f}")
    print(f"   Std: {ratios.std():.3f}")
    
    # Show distribution in bins
    print(f"\n   Distribution:")
    print(f"     0.0: {len(ratios[ratios == 0.0]):,} users")
    print(f"     0.0-0.1: {len(ratios[(ratios > 0.0) & (ratios <= 0.1)]):,} users")
    print(f"     0.1-0.2: {len(ratios[(ratios > 0.1) & (ratios <= 0.2)]):,} users")
    print(f"     0.2-0.3: {len(ratios[(ratios > 0.2) & (ratios <= 0.3)]):,} users")
    print(f"     0.3-0.4: {len(ratios[(ratios > 0.3) & (ratios <= 0.4)]):,} users")
    print(f"     0.4-0.5: {len(ratios[(ratios > 0.4) & (ratios <= 0.5)]):,} users")
    print(f"     0.5+: {len(ratios[ratios > 0.5]):,} users")
    
    # Show users that would pass different thresholds
    print(f"\n   Users passing different thresholds:")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        passing = len(ratios[ratios <= threshold])
        print(f"     ≤{threshold}: {passing:,} users ({passing/len(ratios)*100:.1f}%)")
    
    # Show top users by primary author ratio
    print(f"\n📊 Top 10 users by primary author ratio (highest):")
    top_users = filtered_users.nlargest(10, 'primary_author_ratio')[['user_id', 'n_records', 'primary_author_ratio', 'external_doi_consistency']]
    print(top_users.to_string(index=False))
    
    # Show users with primary author ratio = 0
    zero_ratio_users = filtered_users[filtered_users['primary_author_ratio'] == 0.0]
    print(f"\n📊 Users with primary author ratio = 0.0: {len(zero_ratio_users)}")
    if len(zero_ratio_users) > 0:
        print("Sample users:")
        sample_cols = ['user_id', 'n_records', 'primary_author_ratio', 'external_doi_consistency']
        print(zero_ratio_users[sample_cols].head().to_string(index=False))
    
    # Analyze why primary author ratio might be high
    print(f"\n🔍 Investigating high primary author ratios:")
    print("=" * 50)
    
    # Look at a few high-ratio users in detail
    high_ratio_users = filtered_users[filtered_users['primary_author_ratio'] > 0.5].head(3)
    
    for _, user in high_ratio_users.iterrows():
        user_id = user['user_id']
        user_records = records_df[records_df['user_id'] == user_id]
        
        print(f"\n   User {user_id}:")
        print(f"     Records: {user['n_records']}")
        print(f"     Primary author ratio: {user['primary_author_ratio']:.3f}")
        print(f"     External DOI consistency: {user['external_doi_consistency']:.3f}")
        
        # Check if this user is actually the primary author in their records
        primary_author_count = len(user_records[user_records['is_primary_author'] == True])
        total_records = len(user_records)
        actual_ratio = primary_author_count / total_records if total_records > 0 else 0
        
        print(f"     Actual primary author records: {primary_author_count}/{total_records} ({actual_ratio:.3f})")
        
        # Show a few sample records
        print(f"     Sample records:")
        sample_records = user_records.head(3)
        for _, record in sample_records.iterrows():
            print(f"       Record {record['record_id']}: is_primary_author={record['is_primary_author']}, n_authors={record['n_authors']}")

def analyze_all_thresholds():
    """Analyze all thresholds in detail."""
    
    print("=" * 80)
    print("COMPREHENSIVE THRESHOLD ANALYSIS")
    print("=" * 80)
    
    # Load data
    try:
        users_df = pd.read_parquet('data/users.parquet')
        print(f"Loaded {len(users_df)} users")
    except FileNotFoundError:
        print("❌ Data files not found. Please run the main analysis first.")
        return
    
    total_users = len(users_df)
    
    # Analyze each threshold in detail
    thresholds = [
        ('n_records', '>=', 20, 'Min Records'),
        ('external_doi_consistency', '>=', 0.7, 'External DOI Consistency'),
        ('primary_author_ratio', '<=', 0.3, 'Primary Author Ratio'),
        ('upload_regularity', '>=', 0.1, 'Upload Regularity'),
        ('spam_record_ratio', '<=', 0.1, 'Spam Ratio'),
        ('repetitive_author_score', '>=', 0.3, 'Repetitive Author Score')
    ]
    
    for field, operator, value, name in thresholds:
        print(f"\n🔍 {name} Analysis:")
        print("-" * 40)
        
        data = users_df[field]
        
        if operator == '>=':
            passing = len(data[data >= value])
            condition = data >= value
        elif operator == '<=':
            passing = len(data[data <= value])
            condition = data <= value
        elif operator == '>':
            passing = len(data[data > value])
            condition = data > value
        elif operator == '<':
            passing = len(data[data < value])
            condition = data < value
        
        print(f"   Threshold: {operator} {value}")
        print(f"   Users passing: {passing:,} ({passing/total_users*100:.1f}%)")
        print(f"   Users failing: {total_users - passing:,} ({(total_users - passing)/total_users*100:.1f}%)")
        
        # Show distribution
        print(f"   Distribution:")
        if field in ['n_records']:
            # For counts, show ranges
            ranges = [(0, 10), (10, 20), (20, 50), (50, 100), (100, 500), (500, 1000), (1000, float('inf'))]
            for start, end in ranges:
                if end == float('inf'):
                    count = len(data[data >= start])
                    print(f"     {start}+: {count:,} users")
                else:
                    count = len(data[(data >= start) & (data < end)])
                    print(f"     {start}-{end-1}: {count:,} users")
        else:
            # For ratios, show percentiles
            percentiles = [0, 10, 25, 50, 75, 90, 100]
            for i in range(len(percentiles)-1):
                p1, p2 = percentiles[i], percentiles[i+1]
                val1 = data.quantile(p1/100)
                val2 = data.quantile(p2/100)
                count = len(data[(data >= val1) & (data <= val2)])
                print(f"     {p1}%-{p2}% ({val1:.3f}-{val2:.3f}): {count:,} users")
        
        # Show some examples of failing users
        failing_users = users_df[~condition]
        if len(failing_users) > 0:
            print(f"   Sample failing users:")
            sample = failing_users.head(3)[['user_id', field, 'n_records']]
            print(sample.to_string(index=False))

if __name__ == "__main__":
    analyze_primary_author_ratio()
    print("\n" + "="*80)
    analyze_all_thresholds() 