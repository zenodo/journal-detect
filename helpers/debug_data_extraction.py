#!/usr/bin/env python3
"""
Debug script to investigate data extraction issues
"""

import json
import glob
import pandas as pd
from collections import Counter

def debug_record_extraction():
    """Debug the record extraction process"""
    
    # Load a few sample records
    json_files = glob.glob('records-json-2025/*.json')[:10]
    
    print("=== DEBUGGING RECORD EXTRACTION ===\n")
    
    journal_titles = []
    community_counts = []
    user_ids = []
    
    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            record = json.load(f)
            
        record_id = record.get('id')
        user_id = record.get('parent', {}).get('access', {}).get('owned_by', {}).get('user')
        user_ids.append(user_id)
        
        # Check journal extraction
        journal_title = None
        if 'custom_fields' in record.get('metadata', {}) and \
           'journal:journal' in record['metadata']['custom_fields']:
            journal_title = record['metadata']['custom_fields']['journal:journal'].get('title')
        
        if journal_title:
            journal_titles.append(journal_title)
            print(f"Record {record_id}: Found journal '{journal_title}'")
        
        # Check community extraction
        communities = record.get('communities', {})
        print(f"Record {record_id}: Communities structure: {type(communities)} = {communities}")
        
        if isinstance(communities, dict) and 'ids' in communities:
            community_list = communities['ids']
            if community_list:
                community_counts.extend(community_list)
                print(f"Record {record_id}: Found communities {community_list}")
        elif isinstance(communities, list):
            community_counts.extend(communities)
            print(f"Record {record_id}: Found communities {communities}")
        else:
            print(f"Record {record_id}: No communities found")
    
    print(f"\n=== SUMMARY ===")
    print(f"Records checked: {len(json_files)}")
    print(f"Records with journal titles: {len(journal_titles)}")
    print(f"Unique journal titles: {len(set(journal_titles))}")
    print(f"Records with communities: {len(community_counts)}")
    print(f"Unique communities: {len(set(community_counts))}")
    print(f"Unique user IDs: {len(set(user_ids))}")
    
    if journal_titles:
        print(f"\nSample journal titles: {journal_titles[:5]}")
    if community_counts:
        print(f"Sample communities: {list(set(community_counts))[:5]}")

def debug_spam_matching():
    """Debug spam record matching"""
    
    print("\n=== DEBUGGING SPAM MATCHING ===\n")
    
    # Load spam records
    spam_df = pd.read_csv('records-deleted-2025.csv')
    spam_record_ids = set(spam_df['record_id'].astype(str))
    
    print(f"Total spam records: {len(spam_df)}")
    print(f"Unique spam record IDs: {len(spam_record_ids)}")
    print(f"Sample spam record IDs: {list(spam_record_ids)[:5]}")
    
    # Check if any spam records exist in our JSON files
    json_files = glob.glob('records-json-2025/*.json')
    found_spam = 0
    
    for filepath in json_files[:100]:  # Check first 100 files
        record_id = filepath.split('/')[-1].replace('.json', '')
        if record_id in spam_record_ids:
            found_spam += 1
            print(f"Found spam record: {record_id}")
    
    print(f"Spam records found in JSON files (first 100): {found_spam}")

def debug_user_volume():
    """Debug user volume patterns"""
    
    print("\n=== DEBUGGING USER VOLUME ===\n")
    
    # Load a sample of records to check user distribution
    json_files = glob.glob('records-json-2025/*.json')
    user_counts = Counter()
    
    for filepath in json_files[:1000]:  # Check first 1000 files
        with open(filepath, 'r', encoding='utf-8') as f:
            record = json.load(f)
        
        user_id = record.get('parent', {}).get('access', {}).get('owned_by', {}).get('user')
        if user_id:
            user_counts[user_id] += 1
    
    print(f"Users in first 1000 records: {len(user_counts)}")
    print(f"Most active users:")
    for user_id, count in user_counts.most_common(10):
        print(f"  User {user_id}: {count} records")

if __name__ == "__main__":
    debug_record_extraction()
    debug_spam_matching()
    debug_user_volume() 