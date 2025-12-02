"""
Parallel Subreddit Network Analyzer with Bipartite Structure
Fixed data loading and added EDA phase
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from karateclub import BigClam
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
import json
import os
import pickle
import gc
from datetime import datetime
import traceback
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
import psutil
from IPython.display import display
import logging
from collections import defaultdict, Counter
import sys

# Configure logging to file only, not console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_analysis.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)

# Custom print function for important debug info in notebook
def notebook_print(*args, **kwargs):
    """Print to notebook without going through logger"""
    print(*args, **kwargs)

warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

notebook_print("=== PARALLEL REDDIT NETWORK ANALYSIS WITH BIPARTITE STRUCTURE ===")

class ParallelRedditAnalyzer:
    def __init__(self, checkpoint_dir="checkpoints", n_workers=None):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.checkpoint_dir = checkpoint_dir
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        notebook_print(f"Initialized analyzer with {self.n_workers} parallel workers")
        
    def save_checkpoint(self, data, filename, compress=True):
        """Save intermediate results with better error handling"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            notebook_print(f"✓ Checkpoint saved: {filename}")
            logger.info(f"Checkpoint saved: {filename} ({os.path.getsize(filepath)/1024/1024:.2f} MB)")
            return True
        except Exception as e:
            notebook_print(f"✗ Failed to save checkpoint {filename}: {e}")
            logger.error(f"Failed to save checkpoint {filename}: {e}")
            return False
    
    def load_checkpoint(self, filename, max_age_hours=24):
        """Load intermediate results with age validation"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            try:
                # Check file age
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                file_age = (datetime.now() - file_time).total_seconds() / 3600
                
                if file_age > max_age_hours:
                    notebook_print(f"Checkpoint {filename} is {file_age:.1f}h old, ignoring")
                    logger.warning(f"Checkpoint {filename} is too old ({file_age:.1f}h)")
                    return None
                
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                notebook_print(f"✓ Checkpoint loaded: {filename}")
                logger.info(f"Checkpoint loaded: {filename}")
                return data
            except Exception as e:
                notebook_print(f"✗ Failed to load checkpoint {filename}: {e}")
                logger.error(f"Failed to load checkpoint {filename}: {e}")
        return None

    def exploratory_data_analysis(self, df_posts, df_comments):
        """
        Perform comprehensive exploratory data analysis
        """
        notebook_print("\n" + "="*60)
        notebook_print("EXPLORATORY DATA ANALYSIS")
        notebook_print("="*60)
        
        # Create figures directory
        os.makedirs('eda_plots', exist_ok=True)
        
        # Basic statistics
        notebook_print("\n📊 BASIC STATISTICS:")
        notebook_print(f"  Posts: {len(df_posts):,}")
        notebook_print(f"  Comments: {len(df_comments):,}")
        notebook_print(f"  Unique users: {df_comments['user'].nunique():,}")
        notebook_print(f"  Unique posts with comments: {df_comments['root'].nunique():,}")
        
        # Data quality checks
        notebook_print("\n🔍 DATA QUALITY:")
        notebook_print(f"  Comments with '[deleted]' users: {(df_comments['user'] == '[deleted]').sum():,}")
        notebook_print(f"  Comments with missing text: {df_comments['text'].isna().sum():,}")
        notebook_print(f"  Comments with empty text: {(df_comments['text'] == '').sum():,}")
        
        # Filter out deleted users and empty text
        original_count = len(df_comments)
        df_comments = df_comments[df_comments['user'] != '[deleted]']
        df_comments = df_comments[df_comments['text'].notna()]
        df_comments = df_comments[df_comments['text'] != '']
        
        notebook_print(f"  Comments after cleaning: {len(df_comments):,} ({len(df_comments)/original_count*100:.1f}% retained)")
        
        # Temporal analysis
        notebook_print("\n📅 TEMPORAL ANALYSIS:")
        df_comments['datetime'] = pd.to_datetime(df_comments['timestamp'], unit='s', errors='coerce')
        df_posts['datetime'] = pd.to_datetime(df_posts['timestamp'], unit='s', errors='coerce')
        
        # Remove invalid dates
        df_comments = df_comments[df_comments['datetime'].notna()]
        df_posts = df_posts[df_posts['datetime'].notna()]
        
        time_range_comments = df_comments['datetime'].min(), df_comments['datetime'].max()
        time_range_posts = df_posts['datetime'].min(), df_posts['datetime'].max()
        
        notebook_print(f"  Comment date range: {time_range_comments[0].strftime('%Y-%m-%d')} to {time_range_comments[1].strftime('%Y-%m-%d')}")
        notebook_print(f"  Post date range: {time_range_posts[0].strftime('%Y-%m-%d')} to {time_range_posts[1].strftime('%Y-%m-%d')}")
        
        # Create temporal plots
        self._create_temporal_plots(df_posts, df_comments)
        
        # User activity analysis
        self._analyze_user_activity(df_comments)
        
        # Post engagement analysis
        self._analyze_post_engagement(df_posts, df_comments)
        
        # Sentiment analysis (if available)
        if 'sentiment' in df_comments.columns:
            self._analyze_sentiment(df_comments)
        
        # Network preparation metrics
        self._network_prep_analysis(df_comments)
        
        notebook_print("\n✅ EDA complete! Check 'eda_plots/' directory for visualizations")
        return df_posts, df_comments

    def _create_temporal_plots(self, df_posts, df_comments):
        """Create temporal distribution plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Monthly activity
        df_comments['year_month'] = df_comments['datetime'].dt.to_period('M')
        monthly_comments = df_comments.groupby('year_month').size()
        monthly_comments.plot(ax=axes[0,0], title='Comments per Month', linewidth=2)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Daily activity pattern
        df_comments['hour'] = df_comments['datetime'].dt.hour
        hourly_comments = df_comments.groupby('hour').size()
        hourly_comments.plot(ax=axes[0,1], kind='bar', title='Comments by Hour of Day', color='skyblue')
        axes[0,1].tick_params(axis='x', rotation=0)
        
        # Posts over time
        df_posts['year_month'] = df_posts['datetime'].dt.to_period('M')
        monthly_posts = df_posts.groupby('year_month').size()
        monthly_posts.plot(ax=axes[1,0], title='Posts per Month', color='orange', linewidth=2)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Comments per post distribution
        comments_per_post = df_comments.groupby('root').size()
        comments_per_post.hist(ax=axes[1,1], bins=50, alpha=0.7, color='green')
        axes[1,1].set_title('Distribution of Comments per Post')
        axes[1,1].set_xlabel('Comments per Post')
        axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('eda_plots/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _analyze_user_activity(self, df_comments):
        """Analyze user activity patterns"""
        user_activity = df_comments['user'].value_counts()
        
        notebook_print("\n👥 USER ACTIVITY ANALYSIS:")
        notebook_print(f"  Most active user: {user_activity.index[0]} ({user_activity.iloc[0]:,} comments)")
        notebook_print(f"  Average comments per user: {user_activity.mean():.1f}")
        notebook_print(f"  Median comments per user: {user_activity.median():.1f}")
        notebook_print(f"  Users with only 1 comment: {(user_activity == 1).sum():,} ({((user_activity == 1).sum()/len(user_activity))*100:.1f}%)")
        
        # User activity distribution plot
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        # Top 20 users
        user_activity.head(20).plot(kind='bar', color='coral')
        plt.title('Top 20 Most Active Users')
        plt.xlabel('User')
        plt.ylabel('Number of Comments')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        # Distribution (log scale)
        user_activity.hist(bins=50, log=True, alpha=0.7, color='lightblue')
        plt.title('User Activity Distribution (Log Scale)')
        plt.xlabel('Comments per User')
        plt.ylabel('Frequency (log)')
        
        plt.tight_layout()
        plt.savefig('eda_plots/user_activity.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _analyze_post_engagement(self, df_posts, df_comments):
        """Analyze post engagement patterns"""
        comments_per_post = df_comments.groupby('root').size()
        post_engagement = df_posts.set_index('id')['num_comments']
        
        notebook_print("\n📈 POST ENGAGEMENT ANALYSIS:")
        notebook_print(f"  Average comments per post: {comments_per_post.mean():.1f}")
        notebook_print(f"  Median comments per post: {comments_per_post.median():.1f}")
        notebook_print(f"  Posts with no comments: {len(df_posts) - len(comments_per_post):,}")
        notebook_print(f"  Most commented post: {comments_per_post.idxmax()} ({comments_per_post.max():,} comments)")
        
        # Engagement distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        comments_per_post.hist(bins=50, alpha=0.7, color='lightgreen', log=True)
        plt.title('Comments per Post Distribution (Log Scale)')
        plt.xlabel('Comments per Post')
        plt.ylabel('Frequency (log)')
        
        plt.subplot(1, 2, 2)
        # Top 20 most engaged posts
        comments_per_post.sort_values(ascending=False).head(20).plot(kind='bar', color='orange')
        plt.title('Top 20 Most Commented Posts')
        plt.xlabel('Post ID')
        plt.ylabel('Number of Comments')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('eda_plots/post_engagement.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _analyze_sentiment(self, df_comments):
        """Analyze sentiment distribution"""
        notebook_print("\n😊 SENTIMENT ANALYSIS:")
        notebook_print(f"  Average sentiment: {df_comments['sentiment'].mean():.3f}")
        notebook_print(f"  Sentiment std: {df_comments['sentiment'].std():.3f}")
        notebook_print(f"  Positive comments (>0.5): {(df_comments['sentiment'] > 0.5).sum():,}")
        notebook_print(f"  Negative comments (<-0.5): {(df_comments['sentiment'] < -0.5).sum():,}")
        notebook_print(f"  Neutral comments (-0.5 to 0.5): {((df_comments['sentiment'] >= -0.5) & (df_comments['sentiment'] <= 0.5)).sum():,}")
        
        # Sentiment distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(df_comments['sentiment'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Distribution of Comment Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('eda_plots/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _network_prep_analysis(self, df_comments):
        """Analyze data for network construction readiness"""
        notebook_print("\n🔗 NETWORK PREPARATION ANALYSIS:")
        
        # User-post interactions
        user_post_counts = df_comments.groupby('user')['root'].nunique()
        post_user_counts = df_comments.groupby('root')['user'].nunique()
        
        notebook_print(f"  Average posts per user: {user_post_counts.mean():.2f}")
        notebook_print(f"  Average users per post: {post_user_counts.mean():.2f}")
        notebook_print(f"  Users commenting on multiple posts: {(user_post_counts > 1).sum():,}")
        notebook_print(f"  Posts with multiple users: {(post_user_counts > 1).sum():,}")
        
        # Potential network size estimation
        active_users = user_post_counts[user_post_counts >= 2].index
        active_posts = post_user_counts[post_user_counts >= 2].index
        
        notebook_print(f"  Estimated active users (≥2 posts): {len(active_users):,}")
        notebook_print(f"  Estimated active posts (≥2 users): {len(active_posts):,}")
        notebook_print(f"  Estimated bipartite edges: {len(df_comments[df_comments['user'].isin(active_users) & df_comments['root'].isin(active_posts)]):,}")

    def build_bipartite_network(self, df_comments):
        """
        Build bipartite network of user-post interactions with memory optimization
        """
        notebook_print("Building bipartite user-post network...")
        logger.info("Building bipartite user-post network...")
        
        try:
            # Input validation and filtering
            required_cols = ['user', 'root', 'text']
            missing_cols = [col for col in required_cols if col not in df_comments.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Filter out null users and posts, and deleted users
            df_filtered = df_comments.dropna(subset=['user', 'root']).copy()
            df_filtered = df_filtered[df_filtered['user'] != '[deleted]']
            
            notebook_print(f"Filtered data: {len(df_filtered):,} comments")
            logger.info(f"Filtered data: {len(df_filtered)} comments (from {len(df_comments)} original)")
            
            if len(df_filtered) == 0:
                raise ValueError("No valid user-post interactions after filtering")
            
            # Create bipartite graph
            B = nx.Graph()
            
            # Add nodes with attributes - more memory efficient
            users = df_filtered['user'].unique()
            posts = df_filtered['root'].unique()
            
            notebook_print(f"Adding {len(users):,} users and {len(posts):,} posts...")
            logger.info(f"Adding {len(users)} users and {len(posts)} posts to bipartite network...")
            
            # Add user nodes in batches
            batch_size = 5000
            for i in range(0, len(users), batch_size):
                batch = users[i:i+batch_size]
                for user in batch:
                    B.add_node(user, bipartite=0, type='user')
            
            # Add post nodes and count comments per post
            post_comment_counts = df_filtered.groupby('root').size()
            for i in range(0, len(posts), batch_size):
                batch = posts[i:i+batch_size]
                for post in batch:
                    comment_count = post_comment_counts.get(post, 0)
                    B.add_node(post, bipartite=1, type='post', comment_count=comment_count)
            
            # Add edges efficiently
            edges_added = 0
            for _, row in df_filtered.iterrows():
                B.add_edge(row['user'], row['root'])
                edges_added += 1
                if edges_added % 10000 == 0:
                    logger.debug(f"Added {edges_added} edges...")
            
            notebook_print(f"Bipartite network: {B.number_of_nodes():,} nodes, {B.number_of_edges():,} edges")
            logger.info(f"Bipartite network built: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
            
            # Validate network structure
            user_nodes = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 0]
            post_nodes = [n for n, d in B.nodes(data=True) if d.get('bipartite') == 1]
            
            logger.info(f"Validation - User nodes: {len(user_nodes)}, Post nodes: {len(post_nodes)}")
            
            # Calculate basic network statistics
            if B.number_of_edges() > 0:
                density = nx.density(B)
                logger.info(f"Bipartite network density: {density:.6f}")
            else:
                logger.warning("No edges in bipartite network")
                
            return B
            
        except Exception as e:
            notebook_print(f"✗ Failed to build bipartite network: {e}")
            logger.error(f"Failed to build bipartite network: {e}")
            traceback.print_exc()
            return None

    def parallel_data_loading(self, data_path, use_checkpoints=True):
        """Parallel data loading with proper JSON structure handling"""
        notebook_print("Loading and preprocessing data...")
        logger.info("Loading and preprocessing Reddit data...")
        
        if use_checkpoints:
            checkpoint_data = self.load_checkpoint("preprocessed_data.pkl")
            if checkpoint_data is not None:
                df_posts, df_comments = checkpoint_data
                notebook_print(f"✓ Using cached data: {len(df_posts):,} posts, {len(df_comments):,} comments")
                logger.info("Using existing preprocessed data")
                return df_posts, df_comments
        
        try:
            # Load posts - handle the JSON structure properly
            notebook_print("Loading posts...")
            posts_file = f"{data_path}/conversations.json"
            if not os.path.exists(posts_file):
                raise FileNotFoundError(f"Posts file not found: {posts_file}")
            
            # Load the JSON with post IDs as keys
            with open(posts_file, 'r') as f:
                posts_data = json.load(f)
            
            # Convert to dataframe with post IDs as a column
            df_posts = pd.DataFrame.from_dict(posts_data, orient='index')
            df_posts['id'] = df_posts.index  # Add post ID as a column
            df_posts = df_posts.reset_index(drop=True)
            
            notebook_print(f"✓ Loaded {len(df_posts):,} posts")
            logger.info(f"Posts loaded: {len(df_posts)} rows")
            
            # Load comments
            notebook_print("Loading comments...")
            df_comments = self._load_comments_smart(data_path)
            
            if df_comments is None or len(df_comments) == 0:
                raise ValueError("Failed to load comments")

            notebook_print(f"✓ Loaded {len(df_comments):,} comments")
            logger.info(f"Loaded {len(df_comments)} comments")

            # Parallel preprocessing with progress
            notebook_print("Running sentiment analysis...")
            df_comments = self._parallel_preprocessing(df_comments)
            
            # Save checkpoint
            if use_checkpoints:
                self.save_checkpoint((df_posts, df_comments), "preprocessed_data.pkl")
            
            # Force garbage collection
            gc.collect()
            
            notebook_print("✓ Data loading complete")
            return df_posts, df_comments
            
        except Exception as e:
            notebook_print(f"✗ Data loading failed: {e}")
            logger.error(f"Data loading failed: {e}")
            traceback.print_exc()
            return None, None

    def _load_comments_smart(self, data_path):
        """Smart comment loading with memory optimization"""
        csv_path = f"{data_path}/utterances.csv"
        jsonl_path = f"{data_path}/utterances.jsonl"
        
        if os.path.exists(csv_path):
            notebook_print("Using CSV file...")
            try:
                # Read in chunks to manage memory
                chunks = []
                for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=100000):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
                notebook_print(f"✓ Loaded {len(df):,} comments from CSV")
                return df
            except Exception as e:
                notebook_print(f"CSV loading failed: {e}")
        
        if os.path.exists(jsonl_path):
            notebook_print("Reading JSONL in chunks...")
            try:
                df = self._load_jsonl_chunked(jsonl_path)
                notebook_print(f"✓ Loaded {len(df):,} comments from JSONL")
                return df
            except Exception as e:
                notebook_print(f"JSONL loading failed: {e}")
                return None
        
        raise FileNotFoundError(f"No comments file found in {data_path}")

    def _load_jsonl_chunked(self, file_path, chunk_size=50000):
        """Load JSONL file in chunks with progress"""
        chunks = []
        current_chunk = []
        line_count = 0
        
        notebook_print(f"Reading {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    current_chunk.append(data)
                    line_count += 1
                    
                    if len(current_chunk) >= chunk_size:
                        chunks.append(pd.DataFrame(current_chunk))
                        current_chunk = []
                        notebook_print(f"  Processed {line_count:,} lines...")
                        
                except json.JSONDecodeError:
                    continue
            
            if current_chunk:
                chunks.append(pd.DataFrame(current_chunk))
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            notebook_print(f"✓ Loaded {len(df):,} total comments")
            
            # Save as CSV for faster future loading
            csv_path = file_path.replace('.jsonl', '.csv')
            if not os.path.exists(csv_path):
                notebook_print("Creating CSV for faster future loading...")
                df.to_csv(csv_path, index=False)
            
            return df
        
        return None

    def _parallel_preprocessing(self, df_comments):
        """Parallel preprocessing with memory management"""
        # Split data for parallel processing
        n_chunks = min(self.n_workers, len(df_comments) // 10000 + 1)
        chunks = np.array_split(df_comments, n_chunks)
        
        notebook_print(f"Processing {len(chunks)} chunks in parallel...")
        logger.info(f"Processing {len(chunks)} chunks in parallel...")
        
        # Process sentiment in parallel
        with Pool(self.n_workers) as pool:
            processed_chunks = pool.map(process_sentiment_chunk, chunks)
        
        # Combine results
        df_comments = pd.concat(processed_chunks, ignore_index=True)
        
        # Add temporal features
        notebook_print("Adding temporal features...")
        df_comments['datetime'] = pd.to_datetime(df_comments['timestamp'], unit='s', errors='coerce')
        df_comments = df_comments[df_comments['datetime'].notna()]
        df_comments['week'] = df_comments['datetime'].dt.to_period('W').astype(str)
        df_comments['month'] = df_comments['datetime'].dt.strftime('%Y-%m')
        
        return df_comments

    # ... (rest of the network analysis methods remain the same as previous version)

# WORKER FUNCTIONS (must be at module level for multiprocessing)

def process_sentiment_chunk(chunk):
    """Process sentiment for a chunk (worker function)"""
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        try:
            return analyzer.polarity_scores(str(text))['compound']
        except:
            return 0.0
    
    chunk = chunk.copy()
    chunk['sentiment'] = chunk['text'].apply(get_sentiment)
    return chunk

# ... (rest of worker functions remain the same)

# MAIN EXECUTION FUNCTION
def run_analysis(data_path, checkpoint_dir="checkpoints", n_workers=None, use_checkpoints=True):
    """
    Main analysis function - call this from Jupyter notebook
    
    Args:
        data_path: Path to Reddit data directory
        checkpoint_dir: Directory for checkpoints
        n_workers: Number of parallel workers (None = auto)
        use_checkpoints: Whether to use checkpoints
    
    Returns:
        analyzer, community_results, stability_metrics
    """
    notebook_print("="*60)
    notebook_print("PARALLEL REDDIT NETWORK ANALYSIS WITH BIPARTITE STRUCTURE")
    notebook_print("="*60)
    
    # Initialize analyzer
    analyzer = ParallelRedditAnalyzer(checkpoint_dir=checkpoint_dir, n_workers=n_workers)
    
    try:
        # Phase 1: Load Data
        notebook_print("\n[PHASE 1] Loading data...")
        df_posts, df_comments = analyzer.parallel_data_loading(data_path, use_checkpoints)
        
        if df_comments is None:
            notebook_print("✗ Failed to load data")
            return analyzer, None, None

        # Phase 1.5: Exploratory Data Analysis
        notebook_print("\n[PHASE 1.5] Exploratory Data Analysis...")
        df_posts, df_comments = analyzer.exploratory_data_analysis(df_posts, df_comments)
        
        # Ask user if they want to continue after seeing EDA
        notebook_print("\n" + "="*60)
        response = input("Continue with network analysis? (y/n): ")
        if response.lower() != 'y':
            notebook_print("Analysis stopped by user.")
            return analyzer, None, None

        # Phase 2: Build Networks with Bipartite Structure
        notebook_print("\n[PHASE 2] Building networks with bipartite structure...")
        temporal_networks = analyzer.parallel_network_construction(
            df_comments, temporal_unit='month', max_periods=6, use_checkpoints=use_checkpoints
        )
        
        if not temporal_networks:
            notebook_print("✗ Failed to build networks")
            return analyzer, None, None
        
        # Phase 3: Community Detection
        notebook_print("\n[PHASE 3] Detecting communities...")
        community_results = analyzer.parallel_community_detection(temporal_networks, use_checkpoints)
        
        # Phase 4: Hub Analysis
        notebook_print("\n[PHASE 4] Analyzing hubs...")
        hub_evolution = analyzer.parallel_hub_analysis(community_results, use_checkpoints)
        
        # Phase 5: Stability Metrics
        notebook_print("\n[PHASE 5] Calculating stability...")
        stability_metrics = analyzer.calculate_stability_metrics(community_results, hub_evolution)
        
        # Phase 6: Export Results
        notebook_print("\n[PHASE 6] Exporting results...")
        analyzer.export_results(community_results, hub_evolution, stability_metrics)
        
        # Generate Report
        analyzer.generate_report(stability_metrics)
        
        notebook_print("\n" + "="*60)
        notebook_print("ANALYSIS COMPLETED SUCCESSFULLY!")
        notebook_print("="*60)
        
        return analyzer, community_results, stability_metrics
        
    except Exception as e:
        notebook_print(f"\n✗ Analysis failed: {e}")
        traceback.print_exc()
        return analyzer, None, None