#!/usr/bin/env python3
"""
Debug script to examine the data and understand the new features
"""

import pandas as pd
import numpy as np

# Load the data
print("Loading data...")
users_df = pd.read_parquet('data/users.parquet')
records_df = pd.read_parquet('data/records.parquet')

print(f"Loaded {len(users_df)} users and {len(records_df)} records")

# Check the new features
print("\n=== NEW FEATURES ANALYSIS ===")

# Check repetitive author scores
print(f"Users with repetitive_author_score > 0.3: {len(users_df[users_df['repetitive_author_score'] > 0.3])}")
print(f"Users with repetitive_author_score > 0.5: {len(users_df[users_df['repetitive_author_score'] > 0.5])}")
print(f"Users with repetitive_author_score > 0.7: {len(users_df[users_df['repetitive_author_score'] > 0.7])}")

# Check safe communities
print(f"Users with safe communities: {users_df['has_safe_community'].sum()}")

# Check high-volume users
high_volume = users_df[users_df['n_records'] >= 50]
print(f"Users with >= 50 records: {len(high_volume)}")

# Check external DOI consistency
print(f"Users with external_doi_consistency >= 0.7: {len(users_df[users_df['external_doi_consistency'] >= 0.7])}")

# Check primary author ratio
print(f"Users with primary_author_ratio <= 0.3: {len(users_df[users_df['primary_author_ratio'] <= 0.3])}")

# Check upload regularity
print(f"Users with upload_regularity >= 0.1: {len(users_df[users_df['upload_regularity'] >= 0.1])}")

# Apply criteria step by step
print("\n=== STEP-BY-STEP CRITERIA ANALYSIS ===")

step1 = users_df[users_df['n_records'] >= 50]
print(f"Step 1 (>=50 records): {len(step1)} users")

step2 = step1[step1['external_doi_consistency'] >= 0.7]
print(f"Step 2 (+external DOI consistency >=0.7): {len(step2)} users")

step3 = step2[step2['primary_author_ratio'] <= 0.3]
print(f"Step 3 (+primary author ratio <=0.3): {len(step3)} users")

step4 = step3[step3['upload_regularity'] >= 0.1]
print(f"Step 4 (+upload regularity >=0.1): {len(step4)} users")

step5 = step4[step4['spam_record_ratio'] <= 0.1]
print(f"Step 5 (+spam ratio <=0.1): {len(step5)} users")

step6 = step5[~step5['has_safe_community']]
print(f"Step 6 (+no safe communities): {len(step6)} users")

step7 = step6[step6['repetitive_author_score'] >= 0.3]
print(f"Step 7 (+repetitive author score >=0.3): {len(step7)} users")

# Show top users by record count with their features
print("\n=== TOP 10 USERS BY RECORD COUNT ===")
top_users = users_df.nlargest(10, 'n_records')
for _, user in top_users.iterrows():
    print(f"User {user['user_id']}: {user['n_records']} records, "
          f"ext_doi_cons: {user['external_doi_consistency']:.3f}, "
          f"rep_author: {user['repetitive_author_score']:.3f}, "
          f"safe_comm: {user['has_safe_community']}, "
          f"primary_ratio: {user['primary_author_ratio']:.3f}")

# Check some specific users
print("\n=== EXAMINING SPECIFIC USERS ===")
user_1161 = users_df[users_df['user_id'] == 1161]
if len(user_1161) > 0:
    user = user_1161.iloc[0]
    print(f"User 1161: {user['n_records']} records")
    print(f"  - External DOI consistency: {user['external_doi_consistency']:.3f}")
    print(f"  - Repetitive author score: {user['repetitive_author_score']:.3f}")
    print(f"  - Has safe community: {user['has_safe_community']}")
    print(f"  - Primary author ratio: {user['primary_author_ratio']:.3f}")
    print(f"  - Upload regularity: {user['upload_regularity']:.3f}")

# Check what communities user 1161 has
user_1161_records = records_df[records_df['user_id'] == 1161]
if len(user_1161_records) > 0:
    all_communities = []
    for communities in user_1161_records['community_slugs'].dropna():
        if isinstance(communities, list):
            all_communities.extend(communities)
    print(f"  - Communities: {set(all_communities)}") 