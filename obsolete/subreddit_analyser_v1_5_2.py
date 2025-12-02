"""
Reddit Reply Network Analyzer
Builds networks based on direct reply interactions between users
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

class RedditReplyAnalyzer:
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
        Perform comprehensive exploratory data analysis with reply analysis
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
        
        # Reply analysis
        notebook_print("\n🔄 REPLY ANALYSIS:")
        df_comments_with_replies = df_comments[df_comments['reply_to'].notna()]
        notebook_print(f"  Comments that are replies: {len(df_comments_with_replies):,} ({len(df_comments_with_replies)/len(df_comments)*100:.1f}%)")
        
        # Analyze reply chains
        if len(df_comments_with_replies) > 0:
            # Create mapping from comment ID to user
            comment_to_user = df_comments.set_index('id')['user']
            
            # Map reply_to to target users
            df_comments_with_replies = df_comments_with_replies.copy()
            df_comments_with_replies['reply_to_user'] = df_comments_with_replies['reply_to'].map(comment_to_user)
            
            valid_replies = df_comments_with_replies[df_comments_with_replies['reply_to_user'].notna()]
            valid_replies = valid_replies[valid_replies['reply_to_user'] != '[deleted]']
            valid_replies = valid_replies[valid_replies['user'] != valid_replies['reply_to_user']]  # Remove self-replies
            
            notebook_print(f"  Valid user-to-user replies: {len(valid_replies):,}")
            notebook_print(f"  Unique reply pairs: {valid_replies[['user', 'reply_to_user']].drop_duplicates().shape[0]:,}")
            
            # Analyze reply patterns
            reply_counts = valid_replies.groupby(['user', 'reply_to_user']).size().reset_index(name='count')
            notebook_print(f"  Average replies per pair: {reply_counts['count'].mean():.2f}")
            notebook_print(f"  Max replies between a pair: {reply_counts['count'].max():,}")
        
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
        
        # Reply network analysis
        self._analyze_reply_network(df_comments)
        
        # Sentiment analysis (if available)
        if 'sentiment' in df_comments.columns:
            self._analyze_sentiment(df_comments)
        
        # Network preparation metrics
        self._network_prep_analysis(df_comments)
        
        notebook_print("\n✅ EDA complete! Check 'eda_plots/' directory for visualizations")
        return df_posts, df_comments

    def _analyze_reply_network(self, df_comments):
        """Analyze potential reply network structure"""
        notebook_print("\n🔗 REPLY NETWORK ANALYSIS:")
        
        # Create mapping from comment ID to user
        comment_to_user = df_comments.set_index('id')['user']
        
        # Prepare reply data
        df_replies = df_comments[df_comments['reply_to'].notna()].copy()
        df_replies['reply_to_user'] = df_replies['reply_to'].map(comment_to_user)
        
        # Filter valid replies
        valid_replies = df_replies[
            (df_replies['reply_to_user'].notna()) & 
            (df_replies['reply_to_user'] != '[deleted]') &
            (df_replies['user'] != df_replies['reply_to_user'])  # Remove self-replies
        ]
        
        if len(valid_replies) == 0:
            notebook_print("  No valid reply data found")
            return
        
        # Analyze reply patterns
        reply_pairs = valid_replies.groupby(['user', 'reply_to_user']).size().reset_index(name='weight')
        
        notebook_print(f"  Total reply interactions: {len(valid_replies):,}")
        notebook_print(f"  Unique user pairs with replies: {len(reply_pairs):,}")
        notebook_print(f"  Average replies per pair: {reply_pairs['weight'].mean():.2f}")
        notebook_print(f"  Most active reply pair: {reply_pairs.loc[reply_pairs['weight'].idxmax()].to_dict()}")
        
        # Degree analysis
        out_degree = reply_pairs.groupby('user')['weight'].sum()
        in_degree = reply_pairs.groupby('reply_to_user')['weight'].sum()
        
        notebook_print(f"  Users who sent replies: {len(out_degree):,}")
        notebook_print(f"  Users who received replies: {len(in_degree):,}")
        notebook_print(f"  Average replies sent per user: {out_degree.mean():.2f}")
        notebook_print(f"  Average replies received per user: {in_degree.mean():.2f}")
        
        # Create reply network visualization
        self._create_reply_network_plots(reply_pairs, out_degree, in_degree)

    def _create_reply_network_plots(self, reply_pairs, out_degree, in_degree):
        """Create visualization for reply network analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Distribution of reply weights
        weights = reply_pairs['weight'].values
        axes[0,0].hist(weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black', log=True)
        axes[0,0].set_title('Distribution of Reply Interactions per Pair')
        axes[0,0].set_xlabel('Number of Replies between Pair')
        axes[0,0].set_ylabel('Frequency (log)')
        
        # Plot 2: Top users by replies sent
        out_degree_sorted = out_degree.sort_values(ascending=False).head(20)
        axes[0,1].bar(range(len(out_degree_sorted)), out_degree_sorted.values, color='coral')
        axes[0,1].set_title('Top 20 Users by Replies Sent')
        axes[0,1].set_xlabel('User Rank')
        axes[0,1].set_ylabel('Replies Sent')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Top users by replies received
        in_degree_sorted = in_degree.sort_values(ascending=False).head(20)
        axes[1,0].bar(range(len(in_degree_sorted)), in_degree_sorted.values, color='lightgreen')
        axes[1,0].set_title('Top 20 Users by Replies Received')
        axes[1,0].set_xlabel('User Rank')
        axes[1,0].set_ylabel('Replies Received')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Reciprocity analysis (if we have bidirectional data)
        if len(reply_pairs) > 0:
            # Create a set of bidirectional pairs
            pair_set = set(zip(reply_pairs['user'], reply_pairs['reply_to_user']))
            reverse_pair_set = set(zip(reply_pairs['reply_to_user'], reply_pairs['user']))
            bidirectional_pairs = pair_set & reverse_pair_set
            
            reciprocality = len(bidirectional_pairs) / len(pair_set) if len(pair_set) > 0 else 0
            
            labels = ['One-way', 'Bidirectional']
            sizes = [len(pair_set) - len(bidirectional_pairs), len(bidirectional_pairs)]
            colors = ['lightcoral', 'gold']
            
            axes[1,1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1,1].set_title(f'Reply Reciprocity\n({reciprocality*100:.1f}% bidirectional)')
        
        plt.tight_layout()
        plt.savefig('eda_plots/reply_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def build_reply_network(self, df_comments, min_interactions=1):
        """
        Build directed network based on reply interactions
        Edge from user A to user B means A replied to B's comment
        """
        notebook_print("Building reply network...")
        logger.info("Building reply network...")
        
        try:
            # Input validation
            required_cols = ['id', 'user', 'reply_to']
            missing_cols = [col for col in required_cols if col not in df_comments.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Filter out deleted users and null values
            df_filtered = df_comments[
                (df_comments['user'] != '[deleted]') & 
                (df_comments['user'].notna()) &
                (df_comments['reply_to'].notna())
            ].copy()
            
            notebook_print(f"Processing {len(df_filtered):,} comments with replies...")
            
            # Create mapping from comment ID to user
            comment_to_user = df_comments.set_index('id')['user']
            
            # Map reply_to to target users
            df_filtered['reply_to_user'] = df_filtered['reply_to'].map(comment_to_user)
            
            # Filter valid replies (exclude deleted users and self-replies)
            valid_replies = df_filtered[
                (df_filtered['reply_to_user'].notna()) & 
                (df_filtered['reply_to_user'] != '[deleted]') &
                (df_filtered['user'] != df_filtered['reply_to_user'])
            ]
            
            notebook_print(f"Valid user-to-user replies: {len(valid_replies):,}")
            
            if len(valid_replies) == 0:
                notebook_print("No valid reply interactions found")
                return nx.DiGraph()
            
            # Create directed graph
            G = nx.DiGraph()
            
            # Count interactions between users
            interaction_counts = defaultdict(int)
            for _, row in valid_replies.iterrows():
                pair = (row['user'], row['reply_to_user'])
                interaction_counts[pair] += 1
            
            # Add edges with weights
            edges_added = 0
            for (source, target), weight in interaction_counts.items():
                if weight >= min_interactions:
                    G.add_edge(source, target, weight=weight)
                    edges_added += 1
            
            # Add nodes that only sent or received replies
            all_users = set()
            for source, target in interaction_counts.keys():
                all_users.add(source)
                all_users.add(target)
            
            G.add_nodes_from(all_users)
            
            notebook_print(f"Reply network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            
            # Calculate network statistics
            if G.number_of_nodes() > 0:
                # Basic stats
                density = nx.density(G)
                avg_in_degree = sum(dict(G.in_degree()).values()) / G.number_of_nodes()
                avg_out_degree = sum(dict(G.out_degree()).values()) / G.number_of_nodes()
                
                notebook_print(f"Network density: {density:.6f}")
                notebook_print(f"Average in-degree: {avg_in_degree:.2f}")
                notebook_print(f"Average out-degree: {avg_out_degree:.2f}")
                
                # Reciprocity
                try:
                    reciprocity = nx.reciprocity(G)
                    notebook_print(f"Reciprocity: {reciprocity:.3f}")
                except:
                    reciprocity = 0
                    notebook_print("Could not calculate reciprocity")
                
                # Add network statistics as graph attributes
                G.graph['density'] = density
                G.graph['avg_in_degree'] = avg_in_degree
                G.graph['avg_out_degree'] = avg_out_degree
                G.graph['reciprocity'] = reciprocity
                G.graph['network_type'] = 'reply'
                G.graph['construction_time'] = datetime.now().isoformat()
            
            return G
            
        except Exception as e:
            notebook_print(f"✗ Failed to build reply network: {e}")
            logger.error(f"Failed to build reply network: {e}")
            traceback.print_exc()
            return nx.DiGraph()

    def build_undirected_reply_network(self, df_comments, min_interactions=1):
        """
        Build undirected version of reply network for community detection
        Combines bidirectional interactions
        """
        notebook_print("Building undirected reply network...")
        
        directed_net = self.build_reply_network(df_comments, min_interactions)
        
        if directed_net.number_of_nodes() == 0:
            return nx.Graph()
        
        # Convert to undirected graph, combining edge weights
        undirected_net = nx.Graph()
        
        # Add all nodes
        undirected_net.add_nodes_from(directed_net.nodes())
        
        # For each directed edge, add to undirected graph
        for u, v, data in directed_net.edges(data=True):
            weight = data.get('weight', 1)
            if undirected_net.has_edge(u, v):
                undirected_net[u][v]['weight'] += weight
            else:
                undirected_net.add_edge(u, v, weight=weight)
        
        # Also consider the reverse direction if it exists
        for u, v, data in directed_net.edges(data=True):
            if directed_net.has_edge(v, u):
                reverse_weight = directed_net[v][u].get('weight', 1)
                if undirected_net.has_edge(u, v):
                    undirected_net[u][v]['weight'] += reverse_weight
                else:
                    undirected_net.add_edge(u, v, weight=reverse_weight)
        
        notebook_print(f"Undirected reply network: {undirected_net.number_of_nodes():,} nodes, {undirected_net.number_of_edges():,} edges")
        
        # Calculate undirected network statistics
        if undirected_net.number_of_nodes() > 0:
            density = nx.density(undirected_net)
            avg_degree = sum(dict(undirected_net.degree()).values()) / undirected_net.number_of_nodes()
            
            if undirected_net.number_of_edges() > 0:
                try:
                    avg_clustering = nx.average_clustering(undirected_net)
                    notebook_print(f"Average clustering: {avg_clustering:.4f}")
                except:
                    avg_clustering = 0
            else:
                avg_clustering = 0
            
            notebook_print(f"Undirected density: {density:.6f}")
            notebook_print(f"Average degree: {avg_degree:.2f}")
            
            # Add statistics
            undirected_net.graph.update(directed_net.graph)
            undirected_net.graph['avg_degree'] = avg_degree
            undirected_net.graph['avg_clustering'] = avg_clustering
            undirected_net.graph['network_type'] = 'undirected_reply'
        
        return undirected_net

    def parallel_network_construction(self, df_comments, temporal_unit='month', min_replies=15000, use_checkpoints=True):
        """Smart period selection: randomly select 6 months per year with sufficient reply activity"""
        notebook_print(f"Building temporal reply networks ({temporal_unit})...")
        logger.info(f"Building temporal reply networks ({temporal_unit})...")
        
        # Extract year from temporal unit
        df_comments['year'] = df_comments['datetime'].dt.year
        
        # Count valid user-user replies per period
        notebook_print("Analyzing reply activity across periods...")
        
        # Create mapping from comment ID to user
        comment_to_user = df_comments.set_index('id')['user']
        
        # Count valid replies per period
        period_reply_counts = {}
        
        for period in df_comments[temporal_unit].unique():
            period_df = df_comments[df_comments[temporal_unit] == period]
            
            # Count valid user-user replies
            df_replies = period_df[period_df['reply_to'].notna()].copy()
            df_replies['reply_to_user'] = df_replies['reply_to'].map(comment_to_user)
            
            valid_replies = df_replies[
                (df_replies['reply_to_user'].notna()) & 
                (df_replies['reply_to_user'] != '[deleted]') &
                (df_replies['user'] != df_replies['reply_to_user'])
            ]
            
            period_reply_counts[period] = len(valid_replies)
        
        # Convert to DataFrame for easier manipulation
        reply_stats = pd.DataFrame({
            'period': list(period_reply_counts.keys()),
            'reply_count': list(period_reply_counts.values()),
            'year': [p.split('-')[0] if '-' in p else str(p) for p in period_reply_counts.keys()]
        })
        
        # Filter periods with sufficient replies
        sufficient_periods = reply_stats[reply_stats['reply_count'] >= min_replies]
        notebook_print(f"Periods with ≥{min_replies:,} replies: {len(sufficient_periods)} out of {len(reply_stats)} total periods")
        
        # Select 6 random months per year
        selected_periods = []
        for year in sufficient_periods['year'].unique():
            year_periods = sufficient_periods[sufficient_periods['year'] == year]
            if len(year_periods) > 6:
                # Randomly select 6 periods from this year
                selected = year_periods.sample(n=6, random_state=42)  # Fixed seed for reproducibility
                selected_periods.extend(selected['period'].tolist())
                notebook_print(f"  Year {year}: randomly selected 6 periods from {len(year_periods)} available")
            else:
                # Use all available periods for this year
                selected_periods.extend(year_periods['period'].tolist())
                notebook_print(f"  Year {year}: using all {len(year_periods)} periods (less than 6 available)")
        
        notebook_print(f"Selected {len(selected_periods)} periods for analysis")
        
        if len(selected_periods) == 0:
            notebook_print("✗ No periods with sufficient reply activity found")
            return {}
        
        # Individual period checkpointing
        temporal_networks = {}
        
        for period in selected_periods:
            period_filename = f"reply_network_{temporal_unit}_{period}.pkl"
            
            if use_checkpoints:
                period_data = self.load_checkpoint(period_filename)
                if period_data is not None:
                    temporal_networks[period] = period_data
                    notebook_print(f"✓ Loaded reply network for {period}")
                    continue
            
            # Build network for this period
            notebook_print(f"Building reply network for {period}...")
            period_df = df_comments[df_comments[temporal_unit] == period]
            
            if len(period_df) < 10:
                notebook_print(f"  Skipping {period} - insufficient data")
                continue
            
            networks = self._build_reply_networks_for_period(period, period_df)
            
            if networks and networks['user_network'].number_of_nodes() > 0:
                temporal_networks[period] = networks
                if use_checkpoints:
                    self.save_checkpoint(networks, period_filename)
                notebook_print(f"✓ Built reply network for {period}: {networks['user_network'].number_of_nodes():,} users, {networks['user_network'].number_of_edges():,} edges")
            else:
                notebook_print(f"✗ Failed to build reply network for {period}")
        
        notebook_print(f"✓ Built reply networks for {len(temporal_networks)} periods")
        return temporal_networks

    def _build_reply_networks_for_period(self, period, period_data):
        """Build reply networks for a single period"""
        try:
            # Build directed reply network
            directed_reply_net = self.build_reply_network(period_data, min_interactions=1)
            
            # Build undirected version for community detection
            undirected_reply_net = self.build_undirected_reply_network(period_data, min_interactions=1)
            
            if undirected_reply_net.number_of_nodes() == 0:
                return None
            
            # Build sentiment network
            sentiment_net = build_sentiment_network(period_data)
            
            networks = {
                'directed_reply_network': directed_reply_net,
                'user_network': undirected_reply_net,  # Use undirected for community detection
                'sentiment_network': sentiment_net,
                'user_sentiments': period_data.groupby('user')['sentiment'].mean().to_dict(),
                'activity_levels': period_data.groupby('user').size().to_dict(),
                'reply_stats': period_data[period_data['reply_to'].notna()].groupby('user').size().to_dict(),
                'network_stats': {
                    'n_users': undirected_reply_net.number_of_nodes(),
                    'n_edges': undirected_reply_net.number_of_edges(),
                    'density': undirected_reply_net.graph.get('density', 0),
                    'avg_degree': undirected_reply_net.graph.get('avg_degree', 0),
                    'avg_clustering': undirected_reply_net.graph.get('avg_clustering', 0),
                    'reciprocity': directed_reply_net.graph.get('reciprocity', 0)
                }
            }
            
            return networks
            
        except Exception as e:
            logger.error(f"Failed to build reply networks for period {period}: {e}")
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
            
            # Debug: print columns to see what we have
            notebook_print(f"  Post columns: {list(df_posts.columns)}")
            
            # Load comments
            notebook_print("Loading comments...")
            df_comments = self._load_comments_smart(data_path)
            
            if df_comments is None or len(df_comments) == 0:
                raise ValueError("Failed to load comments")

            notebook_print(f"✓ Loaded {len(df_comments):,} comments")
            logger.info(f"Loaded {len(df_comments)} comments")
            notebook_print(f"  Comment columns: {list(df_comments.columns)}")

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
            #TODO csv re-formating has issues, especially in the text field
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

    def parallel_community_detection(self, temporal_networks, use_checkpoints=True):
        """Parallel community detection with individual checkpointing"""
        notebook_print("Running community detection...")
        logger.info("Running community detection...")
        
        community_results = {}
        
        for period, networks in temporal_networks.items():
            comm_filename = f"communities_{period}.pkl"
            
            if use_checkpoints:
                period_result = self.load_checkpoint(comm_filename)
                if period_result is not None:
                    community_results[period] = period_result
                    notebook_print(f"✓ Loaded communities for {period}")
                    continue
            
            # Detect communities for this period
            notebook_print(f"Detecting communities for {period}...")
            user_network = networks['user_network']
            
            if user_network.number_of_nodes() < 10:
                notebook_print(f"  Skipping {period} - too few users")
                continue
            
            communities = detect_communities_worker((period, user_network))
            
            if communities[1]:  # communities found
                period_result = {
                    'communities': communities[1],
                    'network': user_network
                }
                community_results[period] = period_result
                if use_checkpoints:
                    self.save_checkpoint(period_result, comm_filename)
                notebook_print(f"✓ Found {len(communities[1]):,} communities for {period}")
            else:
                notebook_print(f"✗ No communities found for {period}")
        
        notebook_print(f"✓ Community detection complete for {len(community_results)} periods")
        return community_results

    def parallel_hub_analysis(self, community_results, use_checkpoints=True):
        """Parallel hub analysis with individual checkpointing"""
        notebook_print("Analyzing network hubs...")
        logger.info("Analyzing network hubs...")
        
        hub_evolution = {}
        
        for period, result in community_results.items():
            hub_filename = f"hubs_{period}.pkl"
            
            if use_checkpoints:
                period_hubs = self.load_checkpoint(hub_filename)
                if period_hubs is not None:
                    hub_evolution[period] = period_hubs
                    notebook_print(f"✓ Loaded hubs for {period}")
                    continue
            
            # Analyze hubs for this period
            notebook_print(f"Analyzing hubs for {period}...")
            network = result['network']
            
            hubs_result = analyze_hubs_worker((period, network))
            
            if hubs_result[1]:  # hubs found
                period_hubs = {
                    'hubs': hubs_result[1],
                    'degrees': hubs_result[2]
                }
                hub_evolution[period] = period_hubs
                if use_checkpoints:
                    self.save_checkpoint(period_hubs, hub_filename)
                notebook_print(f"✓ Found {len(hubs_result[1]):,} hubs for {period}")
            else:
                notebook_print(f"✗ No hubs found for {period}")
        
        notebook_print(f"✓ Hub analysis complete for {len(hub_evolution)} periods")
        return hub_evolution

    def calculate_stability_metrics(self, community_results, hub_evolution):
        """Calculate stability metrics across periods"""
        notebook_print("Calculating stability metrics...")
        logger.info("Calculating stability metrics...")
        
        try:
            periods = sorted(community_results.keys())
            
            if len(periods) < 2:
                notebook_print("Need at least 2 periods for stability analysis")
                return {}
            
            stability_metrics = {}
            
            for i in range(len(periods) - 1):
                period1, period2 = periods[i], periods[i+1]
                
                # Hub persistence
                hubs1 = set(hub_evolution.get(period1, {}).get('hubs', {}).keys())
                hubs2 = set(hub_evolution.get(period2, {}).get('hubs', {}).keys())
                
                hub_overlap = 0
                if hubs1 and hubs2:
                    hub_overlap = len(hubs1 & hubs2) / len(hubs1 | hubs2) if (hubs1 | hubs2) else 0
                
                # Community persistence
                comm1 = community_results.get(period1, {}).get('communities', {})
                comm2 = community_results.get(period2, {}).get('communities', {})
                
                community_similarity = self._calculate_community_similarity(comm1, comm2)
                
                metrics = {
                    'hub_overlap': hub_overlap,
                    'community_similarity': community_similarity,
                    'transition_period': f"{period1}→{period2}"
                }
                
                stability_metrics[f"{period1}_{period2}"] = metrics
                notebook_print(f"  {period1}→{period2}: Hub overlap={hub_overlap:.3f}, Community similarity={community_similarity:.3f}")
            
            return stability_metrics
            
        except Exception as e:
            notebook_print(f"✗ Stability calculation failed: {e}")
            logger.error(f"Stability calculation failed: {e}")
            return {}

    def _calculate_community_similarity(self, comm1, comm2):
        """Calculate community similarity efficiently"""
        if not comm1 or not comm2:
            return 0
        
        common_users = set(comm1.keys()) & set(comm2.keys())
        if not common_users:
            return 0
        
        similarities = []
        for user in common_users:
            aff1 = set(comm1[user])
            aff2 = set(comm2[user])
            if aff1 or aff2:
                jaccard = len(aff1 & aff2) / len(aff1 | aff2) if (aff1 | aff2) else 0
                similarities.append(jaccard)
        
        return np.mean(similarities) if similarities else 0

    def export_results(self, community_results, hub_evolution, stability_metrics):
        """Export results to CSV files"""
        notebook_print("Exporting results to CSV...")
        logger.info("Exporting results to CSV...")
        
        try:
            # Export community data
            comm_data = []
            for period, result in community_results.items():
                for user, communities in result['communities'].items():
                    comm_data.append({
                        'period': period,
                        'user': user,
                        'communities': list(communities),
                        'num_communities': len(communities)
                    })
            
            if comm_data:
                pd.DataFrame(comm_data).to_csv('community_memberships.csv', index=False)
                notebook_print("✓ Exported community_memberships.csv")
            
            # Export hub data
            hub_data = []
            for period, result in hub_evolution.items():
                for user, degree in result['degrees'].items():
                    is_hub = user in result['hubs']
                    hub_data.append({
                        'period': period,
                        'user': user,
                        'degree': degree,
                        'is_hub': is_hub
                    })
            
            if hub_data:
                pd.DataFrame(hub_data).to_csv('hub_analysis.csv', index=False)
                notebook_print("✓ Exported hub_analysis.csv")
            
            # Export stability metrics
            if stability_metrics:
                stability_df = pd.DataFrame.from_dict(stability_metrics, orient='index')
                stability_df.to_csv('stability_metrics.csv', index=True)
                notebook_print("✓ Exported stability_metrics.csv")
            
            return True
            
        except Exception as e:
            notebook_print(f"✗ Export failed: {e}")
            logger.error(f"Export failed: {e}")
            return False

    def generate_report(self, stability_metrics):
        """Generate summary report"""
        notebook_print("\n" + "="*60)
        notebook_print("ANALYSIS SUMMARY")
        notebook_print("="*60)
        
        if stability_metrics:
            hub_stabilities = [m['hub_overlap'] for m in stability_metrics.values()]
            comm_stabilities = [m['community_similarity'] for m in stability_metrics.values()]
            
            notebook_print(f"\nSTABILITY METRICS:")
            notebook_print(f"  Average Hub Stability: {np.mean(hub_stabilities):.3f}")
            notebook_print(f"  Average Community Stability: {np.mean(comm_stabilities):.3f}")
            notebook_print(f"  Period Transitions Analyzed: {len(stability_metrics)}")
        
        # Memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 ** 3
        notebook_print(f"\nRESOURCE USAGE:")
        notebook_print(f"  Memory Usage: {memory_usage:.2f} GB")
        notebook_print(f"  CPU Cores Used: {self.n_workers}")
        
        notebook_print("\n" + "="*60)

    def _create_temporal_plots(self, df_posts, df_comments):
        """Create temporal distribution plots with log scaling"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Monthly activity with log scale
        df_comments['year_month'] = df_comments['datetime'].dt.to_period('M')
        monthly_comments = df_comments.groupby('year_month').size()
        monthly_comments.plot(ax=axes[0,0], title='Comments per Month (Log Scale)', linewidth=2)
        axes[0,0].set_yscale('log')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Monthly replies with log scale
        monthly_replies = df_comments[df_comments['reply_to'].notna()].groupby('year_month').size()
        monthly_replies.plot(ax=axes[0,0], linewidth=2, color='red', alpha=0.7, label='Replies')
        axes[0,0].legend()
        
        # Daily activity pattern with log scale
        df_comments['hour'] = df_comments['datetime'].dt.hour
        hourly_comments = df_comments.groupby('hour').size()
        hourly_comments.plot(ax=axes[0,1], kind='bar', title='Comments by Hour of Day (Log Scale)', color='skyblue')
        axes[0,1].set_yscale('log')
        axes[0,1].tick_params(axis='x', rotation=0)
        
        # Posts over time with log scale
        df_posts['year_month'] = df_posts['datetime'].dt.to_period('M')
        monthly_posts = df_posts.groupby('year_month').size()
        monthly_posts.plot(ax=axes[1,0], title='Posts per Month (Log Scale)', color='orange', linewidth=2)
        axes[1,0].set_yscale('log')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Reply ratio over time
        monthly_total = df_comments.groupby('year_month').size()
        monthly_reply_ratio = (monthly_replies / monthly_total).fillna(0)
        ax2 = axes[1,0].twinx()
        monthly_reply_ratio.plot(ax=ax2, color='green', linewidth=1, linestyle='--', alpha=0.7, label='Reply Ratio')
        ax2.set_ylabel('Reply Ratio')
        axes[1,0].legend(['Posts'], loc='upper left')
        ax2.legend(loc='upper right')
        
        # Comments per post distribution with log scale
        comments_per_post = df_comments.groupby('root').size()
        axes[1,1].hist(comments_per_post, bins=50, alpha=0.7, color='green', log=True)
        axes[1,1].set_title('Distribution of Comments per Post (Log Scale)')
        axes[1,1].set_xlabel('Comments per Post')
        axes[1,1].set_ylabel('Frequency (log10)')
        
        plt.tight_layout()
        plt.savefig('eda_plots/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _analyze_user_activity(self, df_comments):
        """Analyze user activity patterns with log scaling"""
        user_activity = df_comments['user'].value_counts()
        user_replies = df_comments[df_comments['reply_to'].notna()]['user'].value_counts()
        
        notebook_print("\n👥 USER ACTIVITY ANALYSIS:")
        notebook_print(f"  Most active user: {user_activity.index[0]} ({user_activity.iloc[0]:,} comments)")
        notebook_print(f"  Most replying user: {user_replies.index[0] if len(user_replies) > 0 else 'N/A'} ({user_replies.iloc[0] if len(user_replies) > 0 else 0:,} replies)")
        notebook_print(f"  Average comments per user: {user_activity.mean():.1f}")
        notebook_print(f"  Users who sent replies: {len(user_replies):,} ({len(user_replies)/len(user_activity)*100:.1f}% of active users)")
        
        # User activity distribution plot with log scaling
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        # Top 20 users with log scale
        user_activity.head(20).plot(kind='bar', color='coral')
        plt.title('Top 20 Most Active Users (Log Scale)')
        plt.xlabel('User')
        plt.ylabel('Number of Comments (log10)')
        plt.yscale('log')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        # Distribution with log scale
        user_activity.hist(bins=50, log=True, alpha=0.7, color='lightblue')
        plt.title('User Activity Distribution (Log Scale)')
        plt.xlabel('Comments per User')
        plt.ylabel('Frequency (log10)')
        
        plt.tight_layout()
        plt.savefig('eda_plots/user_activity.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _analyze_post_engagement(self, df_posts, df_comments):
        """Analyze post engagement patterns with log scaling"""
        comments_per_post = df_comments.groupby('root').size()
        
        notebook_print("\n📈 POST ENGAGEMENT ANALYSIS:")
        notebook_print(f"  Average comments per post: {comments_per_post.mean():.1f}")
        notebook_print(f"  Median comments per post: {comments_per_post.median():.1f}")
        notebook_print(f"  Posts with no comments: {len(df_posts) - len(comments_per_post):,}")
        notebook_print(f"  Most commented post: {comments_per_post.idxmax()} ({comments_per_post.max():,} comments)")
        
        # Engagement distribution with log scaling
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        comments_per_post.hist(bins=50, alpha=0.7, color='lightgreen', log=True)
        plt.title('Comments per Post Distribution (Log Scale)')
        plt.xlabel('Comments per Post')
        plt.ylabel('Frequency (log10)')
        
        plt.subplot(1, 2, 2)
        # Top 20 most engaged posts with log scale
        comments_per_post.sort_values(ascending=False).head(20).plot(kind='bar', color='orange')
        plt.title('Top 20 Most Commented Posts (Log Scale)')
        plt.xlabel('Post ID')
        plt.ylabel('Number of Comments (log10)')
        plt.yscale('log')
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

def build_sentiment_network(period_data):
    """Build sentiment similarity network"""
    try:
        G = nx.Graph()
        
        # Only consider users with multiple comments for sentiment analysis
        user_stats = period_data.groupby('user')['sentiment'].agg(['mean', 'count'])
        user_stats = user_stats[user_stats['count'] >= 2]
        
        # Limit to top users for performance
        users = user_stats.index.tolist()[:300]
        user_means = user_stats['mean'].values
        
        # Create edges between users with similar sentiment
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                similarity = 1 - abs(user_means[i] - user_means[j])
                if similarity > 0.5:  # Only connect if sentiment similarity is high
                    G.add_edge(users[i], users[j], weight=similarity)
        
        return G
    except Exception as e:
        logger.error(f"Failed to build sentiment network: {e}")
        return nx.Graph()

def detect_communities_worker(network_tuple):
    """Community detection worker with multiple fallback methods"""
    period, network = network_tuple
    
    try:
        if network.number_of_nodes() < 20:
            logger.info(f"Network too small for community detection: {network.number_of_nodes()} nodes")
            return (period, {}, network)
        
        # Try multiple community detection methods
        
        # Method 1: Try BigCLAM first
        try:
            dimensions = min(16, max(8, network.number_of_nodes() // 50))
            bigclam = BigClam(
                dimensions=dimensions,
                iterations=500,
                random_state=42
            )
            
            # Convert to adjacency matrix
            adj_matrix = nx.to_scipy_sparse_array(network)
            bigclam.fit(adj_matrix)
            
            memberships = bigclam.get_memberships()
            
            communities = {}
            node_list = list(network.nodes())
            for node_idx, comm_affiliations in enumerate(memberships):
                if comm_affiliations:
                    node_name = node_list[node_idx]
                    communities[node_name] = comm_affiliations
            
            if communities:
                logger.info(f"BigCLAM successful for {period}: {len(communities)} users in communities")
                return (period, communities, network)
            else:
                logger.warning(f"BigCLAM found no communities for {period}")
                
        except Exception as e:
            logger.warning(f"BigCLAM failed for {period}: {e}")
        
        # Method 2: Fallback to Louvain method
        try:
            import community as community_louvain
            
            # Create a copy with weights for Louvain
            weighted_net = network.copy()
            for u, v, d in weighted_net.edges(data=True):
                if 'weight' not in d:
                    d['weight'] = 1.0
            
            partition = community_louvain.best_partition(weighted_net, weight='weight')
            
            # Convert to BigCLAM format
            communities = {}
            for node, comm_id in partition.items():
                communities[node] = [comm_id]
            
            if communities:
                logger.info(f"Louvain successful for {period}: {len(set(partition.values()))} communities")
                return (period, communities, network)
                
        except Exception as e:
            logger.warning(f"Louvain failed for {period}: {e}")
        
        # Method 3: Fallback to connected components as communities
        try:
            communities = {}
            for i, component in enumerate(nx.connected_components(network)):
                for node in component:
                    communities[node] = [i]
            
            if communities and len(communities) > 1:
                logger.info(f"Connected components as communities for {period}: {len(set([list(v)[0] for v in communities.values()]))} components")
                return (period, communities, network)
                
        except Exception as e:
            logger.warning(f"Connected components failed for {period}: {e}")
        
        logger.error(f"All community detection methods failed for {period}")
        return (period, {}, network)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Community detection crashed for {period}: {e}")
        return (period, {}, network)

def analyze_hubs_worker(network_tuple):
    """Hub analysis worker"""
    period, network = network_tuple
    
    try:
        degrees = dict(network.degree())
        
        if not degrees:
            return (period, {}, {})
        
        degree_threshold = np.percentile(list(degrees.values()), 95)
        hubs = {node: deg for node, deg in degrees.items() if deg >= degree_threshold}
        
        return (period, hubs, degrees)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Hub analysis for {period}: {e}")
        return (period, {}, {})


# MAIN EXECUTION FUNCTION
def run_analysis(data_path, checkpoint_dir="checkpoints", n_workers=None, use_checkpoints=True, run_network_analysis=True):
    """
    Main analysis function - call this from Jupyter notebook
    
    Args:
        data_path: Path to Reddit data directory
        checkpoint_dir: Directory for checkpoints
        n_workers: Number of parallel workers (None = auto)
        use_checkpoints: Whether to use checkpoints
        run_network_analysis: Whether to run the computationally intensive network analysis
    
    Returns:
        analyzer, community_results, stability_metrics
    """
    notebook_print("="*60)
    notebook_print("REDDIT REPLY NETWORK ANALYSIS")
    notebook_print("="*60)
    
    # Initialize analyzer
    analyzer = RedditReplyAnalyzer(checkpoint_dir=checkpoint_dir, n_workers=n_workers)
    
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
        
        # Check if we should run network analysis
        if not run_network_analysis:
            notebook_print("\n" + "="*60)
            notebook_print("Network analysis skipped as requested")
            notebook_print("="*60)
            return analyzer, None, None

        notebook_print("\n" + "="*60)
        notebook_print("Continuing with reply network analysis...")
        notebook_print("="*60)

        # Phase 2: Build Reply Networks
        notebook_print("\n[PHASE 2] Building reply networks...")
        temporal_networks = analyzer.parallel_network_construction(
            df_comments, 
            temporal_unit='month', 
            min_replies=15000,  # Minimum 15,000 user-user replies
            use_checkpoints=use_checkpoints
        )
        
        if not temporal_networks:
            notebook_print("✗ Failed to build reply networks")
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
        notebook_print("REPLY NETWORK ANALYSIS COMPLETED SUCCESSFULLY!")
        notebook_print("="*60)
        
        return analyzer, community_results, stability_metrics
        
    except Exception as e:
        notebook_print(f"\n✗ Analysis failed: {e}")
        traceback.print_exc()
        return analyzer, None, None