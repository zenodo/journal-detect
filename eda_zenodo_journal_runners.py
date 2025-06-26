# # Zenodo Journal Runner Detection - Exploratory Data Analysis
#
# This notebook performs an explainable exploratory analysis of Zenodo records to identify users who appear to be running journals on the platform.
#
# ## Overview
# - **Data Source**: 500,000+ JSON records from Zenodo (2025)
# - **Goal**: Identify users with journal-running behavior patterns
# - **Approach**: Feature engineering based on explainable heuristics
#
# ## Key Features Analyzed
# 1. **Volume metrics**: Number of records per user
# 2. **DOI patterns**: Zenodo DOI usage ratio
# 3. **Journal indicators**: Title consistency, community dedication
# 4. **Temporal patterns**: Upload burstiness
# 5. **Quality signals**: Spam record associations

# ## 1. Configuration and Setup

# +
# Configuration
import os
import json
import glob
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Data processing
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Statistics
from scipy import stats

# Set up plotting style
plt.style.use('seaborn-v0_8')
rcParams['figure.figsize'] = (12, 8)
rcParams['font.size'] = 10

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Configuration parameters
CONFIG = {
    'json_dir': 'records-json-2025',
    'spam_file': 'records-deleted-2025.csv',
    'sample_size': None,  # Set to number for sampling, None for full dataset
    'chunk_size': 1000,   # Process files in chunks
    'min_records_threshold': 20,
    'zenodo_ratio_threshold': 0.8,
    'community_ratio_threshold': 0,
    'max_external_prefixes': 1
}

print(f"Configuration loaded. Processing directory: {CONFIG['json_dir']}")
print(f"Spam records file: {CONFIG['spam_file']}")


# -

# ## 2. Data Loading Functions

# +
def load_spam_records(filepath):
    """Load spam records CSV file."""
    print(f"Loading spam records from {filepath}...")
    spam_df = pd.read_csv(filepath)
    print(f"Loaded {len(spam_df)} spam records")
    return spam_df

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
        if 'custom_fields' in record and \
           'journal:journal' in record['custom_fields']:
            journal_title = record['custom_fields']['journal:journal'].get('title')
        
        # Community info
        communities = record.get('communities', {}).get('ids', [])
        
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

def load_records_generator(json_dir, sample_size=None):
    """Generator function to load records in chunks for memory efficiency."""
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    
    if sample_size:
        json_files = json_files[:sample_size]
    
    print(f"Processing {len(json_files)} JSON files...")
    
    for filepath in tqdm(json_files, desc="Loading records"):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                record = json.load(f)
                features = extract_record_features(record)
                if features:
                    yield features
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

def load_records_to_dataframe(json_dir, sample_size=None):
    """Load all records into a pandas DataFrame."""
    records = list(load_records_generator(json_dir, sample_size))
    df = pd.DataFrame(records)
    
    # Convert created to datetime
    try:
        df['created'] = pd.to_datetime(df['created'], format='mixed', errors='coerce')
    except Exception as e:
        print(f"Warning: Some datetime parsing failed: {e}")
        # Fallback to more permissive parsing
        df['created'] = pd.to_datetime(df['created'], errors='coerce')
    
    # Remove rows where datetime parsing failed
    original_count = len(df)
    df = df.dropna(subset=['created'])
    if len(df) < original_count:
        print(f"Warning: Dropped {original_count - len(df)} records with invalid datetime")
    
    print(f"Loaded {len(df)} records from {df['user_id'].nunique()} unique users")
    return df


# -

# ## 3. Load and Prepare Data

# +
# Load spam records
spam_df = load_spam_records(CONFIG['spam_file'])
spam_record_ids = set(spam_df['record_id'].astype(str))

# Load main records
records_df = load_records_to_dataframe(CONFIG['json_dir'], CONFIG['sample_size'])

# Add spam flag
records_df['is_spam_record'] = records_df['record_id'].astype(str).isin(spam_record_ids)

print(f"\nData Summary:")
print(f"Total records: {len(records_df):,}")
print(f"Unique users: {records_df['user_id'].nunique():,}")
print(f"Spam records: {records_df['is_spam_record'].sum():,}")
print(f"Date range: {records_df['created'].min()} to {records_df['created'].max()}")

# Save records to parquet for future use
records_df.to_parquet('data/records.parquet', index=False)
print("\nRecords saved to data/records.parquet")


# -

# ## 4. Feature Engineering Functions

# +
def calculate_burstiness(upload_times):
    """
    Calculate burstiness using coefficient of variation of inter-upload times.
    
    Burstiness measures how irregular or clustered the upload timing patterns are.
    A high burstiness value indicates uploads happen in concentrated bursts with 
    long gaps between them (typical of journal runners who upload many papers at once).
    A low burstiness value indicates regular, evenly-spaced uploads (typical of 
    normal academic users who upload papers as they complete them).
    
    The coefficient of variation (std/mean) of time intervals between uploads 
    captures this pattern - higher values mean more bursty behavior.
    """
    if len(upload_times) < 2:
        return 0.0
    
    # Sort times and calculate intervals
    sorted_times = sorted(upload_times)
    intervals = np.diff([t.timestamp() for t in sorted_times])
    
    if len(intervals) == 0 or np.std(intervals) == 0:
        return 0.0
    
    # Coefficient of variation
    cv = np.std(intervals) / np.mean(intervals)
    return cv

def extract_user_features(user_records):
    """Extract features for a single user from their records."""
    user_id = user_records['user_id'].iloc[0]
    
    # Basic counts
    n_records = len(user_records)
    
    # DOI analysis
    zenodo_ratio = user_records['is_zenodo_doi'].mean()
    
    # DOI prefixes (external journals)
    external_dois = user_records[~user_records['is_zenodo_doi']]['doi'].dropna()
    if len(external_dois) > 0:
        prefixes = [doi.split('/')[0] for doi in external_dois if '/' in doi]
        n_external_prefixes = len(set(prefixes))
    else:
        n_external_prefixes = 0
    
    # Journal title consistency
    journal_titles = user_records['journal_title'].dropna()
    distinct_journal_title_cnt = journal_titles.nunique()
    
    # Community analysis
    all_communities = []
    for communities in user_records['communities'].dropna():
        if isinstance(communities, list):
            all_communities.extend(communities)
    
    if all_communities:
        community_counts = pd.Series(all_communities).value_counts()
        most_common_community = community_counts.index[0]
        same_comm_ratio = community_counts.iloc[0] / len(all_communities)
    else:
        same_comm_ratio = 0.0
    
    # Temporal burstiness
    upload_times = user_records['created'].dropna()
    burstiness = calculate_burstiness(upload_times)
    
    # Spam association
    spam_record_cnt = user_records['is_spam_record'].sum()
    
    return {
        'user_id': user_id,
        'n_records': n_records,
        'zenodo_ratio': zenodo_ratio,
        'n_external_prefixes': n_external_prefixes,
        'distinct_journal_title_cnt': distinct_journal_title_cnt,
        'same_comm_ratio': same_comm_ratio,
        'burstiness': burstiness,
        'spam_record_cnt': spam_record_cnt,
        'first_upload': upload_times.min() if len(upload_times) > 0 else None,
        'last_upload': upload_times.max() if len(upload_times) > 0 else None
    }

def create_user_features_df(records_df):
    """Create user-level features DataFrame."""
    print("Extracting user-level features...")
    
    user_features = []
    for user_id, user_records in tqdm(records_df.groupby('user_id'), desc="Processing users"):
        features = extract_user_features(user_records)
        user_features.append(features)
    
    users_df = pd.DataFrame(user_features)
    
    print(f"Created features for {len(users_df)} users")
    return users_df


# -

# ## 5. Generate User Features

# +
# Create user features
users_df = create_user_features_df(records_df)

# Display feature summary
print("\nUser Features Summary:")
print(users_df.describe())

# Save user features
users_df.to_parquet('data/users.parquet', index=False)
print("\nUser features saved to data/users.parquet")

# Show top users by record count
print("\nTop 10 users by record count:")
print(users_df.nlargest(10, 'n_records')[['user_id', 'n_records', 'zenodo_ratio', 'spam_record_cnt']])
# -

# ## 6. Exploratory Data Analysis - Feature Distributions

# +
# Set up plotting
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

# 1. Number of records distribution
axes[0].hist(users_df['n_records'], bins=50, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Number of Records')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Records per User')
axes[0].set_yscale('log')

# 2. Zenodo DOI ratio
axes[1].hist(users_df['zenodo_ratio'], bins=30, alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Zenodo DOI Ratio')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Zenodo DOI Usage')

# 3. External prefixes
axes[2].hist(users_df['n_external_prefixes'], bins=range(0, 10), alpha=0.7, edgecolor='black')
axes[2].set_xlabel('Number of External DOI Prefixes')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution of External DOI Prefixes')

# 4. Journal title count
axes[3].hist(users_df['distinct_journal_title_cnt'], bins=range(0, 20), alpha=0.7, edgecolor='black')
axes[3].set_xlabel('Distinct Journal Titles')
axes[3].set_ylabel('Frequency')
axes[3].set_title('Distribution of Distinct Journal Titles')

# 5. Community ratio
axes[4].hist(users_df['same_comm_ratio'], bins=30, alpha=0.7, edgecolor='black')
axes[4].set_xlabel('Most Common Community Ratio')
axes[4].set_ylabel('Frequency')
axes[4].set_title('Distribution of Community Dedication')

# 6. Burstiness
axes[5].hist(users_df['burstiness'], bins=50, alpha=0.7, edgecolor='black')
axes[5].set_xlabel('Burstiness (CV of Inter-upload Times)')
axes[5].set_ylabel('Frequency')
axes[5].set_title('Distribution of Upload Burstiness')

plt.tight_layout()
plt.savefig('figures/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("Feature distributions saved to figures/feature_distributions.png")
# -

# ## 7. Correlation Analysis

# +
# Select numeric features for correlation
numeric_features = ['n_records', 'zenodo_ratio', 'n_external_prefixes', 
                   'distinct_journal_title_cnt', 'same_comm_ratio', 'burstiness', 'spam_record_cnt']

# Create correlation matrix
correlation_matrix = users_df[numeric_features].corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("Correlation heatmap saved to figures/correlation_heatmap.png")

# Show top correlations
print("\nTop feature correlations:")
correlations = []
for i in range(len(numeric_features)):
    for j in range(i+1, len(numeric_features)):
        feat1, feat2 = numeric_features[i], numeric_features[j]
        corr = correlation_matrix.loc[feat1, feat2]
        correlations.append((feat1, feat2, corr))

correlations.sort(key=lambda x: abs(x[2]), reverse=True)
for feat1, feat2, corr in correlations[:10]:
    print(f"{feat1} vs {feat2}: {corr:.3f}")
# -

# ## 8. Scatter Plot Analysis

# +
# Create scatter plot: n_records vs zenodo_ratio
plt.figure(figsize=(12, 8))

# Color by spam record count
scatter = plt.scatter(users_df['n_records'], users_df['zenodo_ratio'], 
                     c=users_df['spam_record_cnt'], cmap='viridis', 
                     alpha=0.6, s=50)

plt.xlabel('Number of Records')
plt.ylabel('Zenodo DOI Ratio')
plt.title('Records vs Zenodo DOI Usage (colored by spam count)')
plt.xscale('log')
plt.colorbar(scatter, label='Spam Record Count')

# Add threshold lines
plt.axhline(y=CONFIG['zenodo_ratio_threshold'], color='red', linestyle='--', 
           alpha=0.7, label=f'Zenodo ratio threshold ({CONFIG["zenodo_ratio_threshold"]})')
plt.axvline(x=CONFIG['min_records_threshold'], color='red', linestyle='--', 
           alpha=0.7, label=f'Min records threshold ({CONFIG["min_records_threshold"]})')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/records_vs_zenodo_ratio.png', dpi=300, bbox_inches='tight')
plt.show()

print("Scatter plot saved to figures/records_vs_zenodo_ratio.png")
# -

# ## 9. Pair Plot for Key Features

# +
# Create pair plot for key features (sample for performance)
key_features = ['n_records', 'zenodo_ratio', 'same_comm_ratio', 'spam_record_cnt']
sample_size = min(1000, len(users_df))
sample_df = users_df.sample(n=sample_size, random_state=42)

print(f"Creating pair plot with {sample_size} sampled users...")

pair_plot = sns.pairplot(sample_df[key_features], diag_kind='hist', 
                        plot_kws={'alpha': 0.6, 's': 20})
pair_plot.fig.suptitle('Pair Plot of Key Features (Sampled)', y=1.02)
pair_plot.fig.set_size_inches(12, 10)
plt.tight_layout()
plt.savefig('figures/pair_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Pair plot saved to figures/pair_plot.png")


# -

# ## 10. Timeline Analysis for Top Candidates

# +
def plot_user_timeline(user_id, records_df):
    """Plot upload timeline for a specific user."""
    user_records = records_df[records_df['user_id'] == user_id].copy()
    user_records = user_records.sort_values('created')
    
    plt.figure(figsize=(12, 6))
    
    # Plot uploads over time
    plt.scatter(user_records['created'], range(len(user_records)), 
               alpha=0.7, s=30, c='blue', label='Uploads')
    
    # Highlight spam records
    spam_records = user_records[user_records['is_spam_record']]
    if len(spam_records) > 0:
        plt.scatter(spam_records['created'], 
                   [list(user_records['created']).index(t) for t in spam_records['created']], 
                   alpha=0.8, s=50, c='red', label='Spam Records', marker='x')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Upload Count')
    plt.title(f'Upload Timeline for User {user_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()

# Get top 5 users by record count
top_users = users_df.nlargest(5, 'n_records')['user_id'].tolist()

print("Creating timeline plots for top 5 users by record count...")

for i, user_id in enumerate(top_users):
    fig = plot_user_timeline(user_id, records_df)
    plt.savefig(f'figures/timeline_user_{user_id}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Timeline for user {user_id} saved to figures/timeline_user_{user_id}.png")
    
    # Show user stats
    user_stats = users_df[users_df['user_id'] == user_id].iloc[0]
    print(f"  Records: {user_stats['n_records']}, Zenodo ratio: {user_stats['zenodo_ratio']:.2f}, "
          f"Spam records: {user_stats['spam_record_cnt']}")
# -

# ## 11. Journal Runner Candidate Detection

# +
# Apply heuristic filter for journal runner candidates
print("Applying journal runner detection heuristics...")

candidates = users_df[
    (users_df['n_records'] >= CONFIG['min_records_threshold']) &
    ((users_df['zenodo_ratio'] >= CONFIG['zenodo_ratio_threshold']) | 
     (users_df['n_external_prefixes'] <= CONFIG['max_external_prefixes'])) &
    (users_df['same_comm_ratio'] >= CONFIG['community_ratio_threshold'])
].sort_values('zenodo_ratio', ascending=False)

print(f"\nFound {len(candidates)} users meeting journal runner criteria")
print(f"Out of {len(users_df)} total users with >= {CONFIG['min_records_threshold']} records")

# Display top candidates
print("\nTop 20 Journal Runner Candidates:")
display_columns = ['user_id', 'n_records', 'zenodo_ratio', 'n_external_prefixes', 
                  'distinct_journal_title_cnt', 'same_comm_ratio', 'burstiness', 'spam_record_cnt']

styled_candidates = candidates[display_columns].head(20).style\
    .background_gradient(subset=['n_records'], cmap='YlOrRd')\
    .background_gradient(subset=['zenodo_ratio'], cmap='Blues')\
    .background_gradient(subset=['same_comm_ratio'], cmap='Greens')\
    .format({
        'zenodo_ratio': '{:.3f}',
        'same_comm_ratio': '{:.3f}',
        'burstiness': '{:.3f}'
    })\
    .set_caption('Top 20 Journal Runner Candidates (Highlighted by Feature Strength)')

display(styled_candidates)

# Save candidates to CSV
candidates.to_csv('data/journal_runner_candidates.csv', index=False)
print("\nCandidates saved to data/journal_runner_candidates.csv")
# -

# ## 12. Summary Statistics and Insights

# +
# Generate summary statistics
print("=" * 60)
print("ZENODO JOURNAL RUNNER DETECTION - SUMMARY REPORT")
print("=" * 60)

total_users = len(users_df)
users_with_50_plus = len(users_df[users_df['n_records'] >= 50])
candidate_count = len(candidates)

print(f"\n📊 DATASET OVERVIEW:")
print(f"   • Total users analyzed: {total_users:,}")
print(f"   • Users with ≥50 records: {users_with_50_plus:,} ({users_with_50_plus/total_users*100:.1f}%)")
print(f"   • Journal runner candidates: {candidate_count:,} ({candidate_count/users_with_50_plus*100:.1f}% of high-volume users)")

print(f"\n🎯 DETECTION CRITERIA APPLIED:")
print(f"   • Minimum records: {CONFIG['min_records_threshold']}")
print(f"   • Zenodo DOI ratio threshold: {CONFIG['zenodo_ratio_threshold']}")
print(f"   • Community dedication threshold: {CONFIG['community_ratio_threshold']}")
print(f"   • Max external prefixes: {CONFIG['max_external_prefixes']}")

print(f"\n📈 CANDIDATE CHARACTERISTICS:")
if len(candidates) > 0:
    print(f"   • Average records per candidate: {candidates['n_records'].mean():.1f}")
    print(f"   • Average Zenodo DOI ratio: {candidates['zenodo_ratio'].mean():.3f}")
    print(f"   • Average community dedication: {candidates['same_comm_ratio'].mean():.3f}")
    print(f"   • Users with spam records: {len(candidates[candidates['spam_record_cnt'] > 0])} "
          f"({len(candidates[candidates['spam_record_cnt'] > 0])/len(candidates)*100:.1f}%)")

print(f"\n🔍 KEY INSIGHTS:")
print(f"   • {len(users_df[users_df['zenodo_ratio'] == 1.0]):,} users use exclusively Zenodo DOIs")
print(f"   • {len(users_df[users_df['same_comm_ratio'] == 1.0]):,} users are dedicated to a single community")
print(f"   • {len(users_df[users_df['spam_record_cnt'] > 0]):,} users have spam records associated")

print(f"\n💾 OUTPUT FILES GENERATED:")
print(f"   • data/records.parquet - Processed record data")
print(f"   • data/users.parquet - User-level features")
print(f"   • data/journal_runner_candidates.csv - Candidate list")
print(f"   • figures/ - Visualization plots")

print(f"\n" + "=" * 60)
print(f"ANALYSIS COMPLETE: Found {candidate_count} potential journal runners")
print("=" * 60)
# -

# ## 13. Additional Analysis: Feature Importance

# +
# Analyze feature importance for candidate detection
print("Analyzing feature importance for journal runner detection...")

# Create binary target: is candidate or not
users_df['is_candidate'] = users_df['user_id'].isin(candidates['user_id'])

# Calculate feature importance using correlation with target
feature_importance = {}
for feature in numeric_features:
    correlation = users_df[feature].corr(users_df['is_candidate'])
    feature_importance[feature] = abs(correlation)

# Sort by importance
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
features, importance = zip(*sorted_importance)
bars = plt.bar(range(len(features)), importance, color='skyblue', alpha=0.7)
plt.xlabel('Features')
plt.ylabel('Absolute Correlation with Candidate Status')
plt.title('Feature Importance for Journal Runner Detection')
plt.xticks(range(len(features)), features, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, imp in zip(bars, importance):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{imp:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("Feature importance plot saved to figures/feature_importance.png")
print("\nFeature importance ranking:")
for feature, imp in sorted_importance:
    print(f"  {feature}: {imp:.3f}") 
