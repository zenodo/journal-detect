#!/usr/bin/env python3
"""
Test script to verify the fixed data extraction
"""

import json
import glob
import pandas as pd
from collections import Counter

def extract_record_features(record):
    """Extract relevant features from a single JSON record."""
    try:
        # Basic record info
        record_id = record.get('id')
        created = record.get('created')
        
        # User info
        user_id = record.get('parent', {}).get('access', {}).get('owned_by', {}).get('user')
        
        # DOI info
        doi = record.get('pids', {}).get('doi', {}).get('identifier', '')
        is_zenodo_doi = doi.startswith('10.5281/zenodo.') if doi else False
        
        # Journal info
        journal_title = None
        if 'custom_fields' in record.get('metadata', {}) and \
           'journal:journal' in record['metadata']['custom_fields']:
            journal_title = record['metadata']['custom_fields']['journal:journal'].get('title')
        
        # Community info - check both main record and parent
        communities = []
        if 'communities' in record and 'ids' in record['communities']:
            communities = record['communities']['ids']
        elif 'parent' in record and 'communities' in record['parent'] and 'ids' in record['parent']['communities']:
            communities = record['parent']['communities']['ids']
        
        # File info
        file_count = record.get('files', {}).get('count', 0)
        
        return {
            'record_id': record_id,
            'user_id': user_id,
            'created': created,
            'doi': doi,
            'is_zenodo_doi': is_zenodo_doi,
            'journal_title': journal_title,
            'communities': communities,
            'file_count': file_count
        }
    except Exception as e:
        print(f"Error processing record {record.get('id', 'unknown')}: {e}")
        return None

def test_extraction():
    """Test the fixed extraction on a sample of records"""
    
    # Test on the files we know have journal info
    test_files = [
        'records-json-2025/14966851.json',
        'records-json-2025/14852295.json', 
        'records-json-2025/15495132.json',
        'records-json-2025/15020817.json',
        'records-json-2025/15602496.json'
    ]
    
    print("=== TESTING FIXED EXTRACTION ===\n")
    
    journal_titles = []
    community_counts = []
    user_ids = []
    
    for filepath in test_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                record = json.load(f)
            
            features = extract_record_features(record)
            if features:
                record_id = features['record_id']
                user_id = features['user_id']
                journal_title = features['journal_title']
                communities = features['communities']
                
                user_ids.append(user_id)
                
                if journal_title:
                    journal_titles.append(journal_title)
                    print(f"Record {record_id}: Found journal '{journal_title}'")
                
                if communities:
                    community_counts.extend(communities)
                    print(f"Record {record_id}: Found communities {communities}")
                else:
                    print(f"Record {record_id}: No communities found")
                    
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    print(f"\n=== RESULTS ===")
    print(f"Records checked: {len(test_files)}")
    print(f"Records with journal titles: {len(journal_titles)}")
    print(f"Unique journal titles: {len(set(journal_titles))}")
    print(f"Records with communities: {len(community_counts)}")
    print(f"Unique communities: {len(set(community_counts))}")
    print(f"Unique user IDs: {len(set(user_ids))}")
    
    if journal_titles:
        print(f"\nSample journal titles: {journal_titles}")
    if community_counts:
        print(f"Sample communities: {list(set(community_counts))}")

if __name__ == "__main__":
    test_extraction() 