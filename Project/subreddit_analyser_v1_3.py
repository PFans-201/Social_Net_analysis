"""
Parallel Subreddit Network Analyzer with Bipartite Structure
Enhanced logging and checkpoint management
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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
        # No StreamHandler - this prevents terminal output
    ]
)
logger = logging.getLogger(__name__)

# Custom print function for important debug info in notebook
def notebook_print(*args, **kwargs):
    """Print to notebook without going through logger"""
    print(*args, **kwargs)

warnings.filterwarnings('ignore')

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

    def build_bipartite_network(self, df_comments):
        """
        Build bipartite network of user-post interactions with memory optimization
        """
        notebook_print("Building bipartite user-post network...")
        logger.info("Building bipartite user-post network...")
        
        try:
            # Input validation
            required_cols = ['user', 'root', 'text']
            missing_cols = [col for col in required_cols if col not in df_comments.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Filter out null users and posts 
            # TODO filter out [deleted] users
            df_filtered = df_comments.dropna(subset=['user', 'root']).copy()
            notebook_print(f"Filtered data: {len(df_filtered)} comments")
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

    def project_to_user_network(self, bipartite_graph, min_common_posts=1):
        """
        Project bipartite network to user-user network with memory optimization
        """
        notebook_print("Projecting to user-user network...")
        logger.info("Projecting bipartite network to user-user network...")
        
        try:
            if bipartite_graph is None or bipartite_graph.number_of_nodes() == 0:
                raise ValueError("Invalid bipartite graph for projection")
            
            # Get user nodes
            user_nodes = [n for n, d in bipartite_graph.nodes(data=True) 
                         if d.get('bipartite') == 0 and d.get('type') == 'user']
            
            notebook_print(f"Projecting network for {len(user_nodes):,} users...")
            logger.info(f"Projecting network for {len(user_nodes)} users...")
            
            # Create user-user projection
            user_network = nx.Graph()
            
            # Get post nodes
            post_nodes = [n for n, d in bipartite_graph.nodes(data=True) 
                         if d.get('bipartite') == 1 and d.get('type') == 'post']
            
            notebook_print(f"Processing {len(post_nodes):,} posts...")
            logger.info(f"Processing {len(post_nodes)} posts for projection...")
            
            # Build user-post mapping for efficient lookup
            user_post_map = defaultdict(set)
            for post in post_nodes:
                post_users = list(bipartite_graph.neighbors(post))
                for user in post_users:
                    if user in user_nodes:
                        user_post_map[user].add(post)
            
            # Convert to list for iteration
            user_list = list(user_post_map.keys())
            user_network.add_nodes_from(user_list)
            
            # Create edges based on common posts
            edges_created = 0
            for i, user1 in enumerate(user_list):
                posts1 = user_post_map[user1]
                for j in range(i + 1, len(user_list)):
                    user2 = user_list[j]
                    posts2 = user_post_map[user2]
                    common_posts = posts1 & posts2
                    
                    if len(common_posts) >= min_common_posts:
                        user_network.add_edge(user1, user2, weight=len(common_posts))
                        edges_created += 1
                
                if (i + 1) % 1000 == 0:
                    notebook_print(f"  Processed {i+1:,} users, created {edges_created:,} edges...")
            
            notebook_print(f"User network: {user_network.number_of_nodes():,} nodes, {user_network.number_of_edges():,} edges")
            logger.info(f"User network projection complete: {user_network.number_of_nodes()} nodes, {user_network.number_of_edges()} edges")
            
            # Calculate network statistics
            if user_network.number_of_nodes() > 0:
                density = nx.density(user_network)
                avg_degree = sum(dict(user_network.degree()).values()) / user_network.number_of_nodes()
                
                # Clustering coefficient
                if user_network.number_of_edges() > 0:
                    avg_clustering = nx.average_clustering(user_network)
                    logger.info(f"Average clustering coefficient: {avg_clustering:.4f}")
                else:
                    avg_clustering = 0
                    logger.warning("No edges for clustering calculation")
                
                notebook_print(f"Network density: {density:.6f}, avg degree: {avg_degree:.2f}")
                logger.info(f"User network density: {density:.6f}")
                logger.info(f"Average degree: {avg_degree:.2f}")
                
                # Add network statistics as graph attributes
                user_network.graph['density'] = density
                user_network.graph['avg_degree'] = avg_degree
                user_network.graph['avg_clustering'] = avg_clustering
                user_network.graph['projection_time'] = datetime.now().isoformat()
            
            return user_network
            
        except Exception as e:
            notebook_print(f"✗ Failed to project user network: {e}")
            logger.error(f"Failed to project user network: {e}")
            traceback.print_exc()
            return None

    def parallel_data_loading(self, data_path, use_checkpoints=True):
        """Parallel data loading with better memory management"""
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
            # Load posts
            notebook_print("Loading posts...")
            posts_file = f"{data_path}/conversations.json"
            if not os.path.exists(posts_file):
                raise FileNotFoundError(f"Posts file not found: {posts_file}")
            
            df_posts = pd.read_json(posts_file).T.reset_index(drop=False)
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

    def parallel_network_construction(self, df_comments, temporal_unit='month', max_periods=6, use_checkpoints=True):
        """Parallel network construction with better checkpointing"""
        notebook_print(f"Building temporal networks ({temporal_unit})...")
        logger.info(f"Building temporal networks ({temporal_unit})...")
        
        # Individual period checkpointing
        temporal_networks = {}
        periods = df_comments[temporal_unit].value_counts().head(max_periods).index.tolist()
        
        notebook_print(f"Processing {len(periods)} periods...")
        
        for period in periods:
            period_filename = f"network_{temporal_unit}_{period}.pkl"
            
            if use_checkpoints:
                period_data = self.load_checkpoint(period_filename)
                if period_data is not None:
                    temporal_networks[period] = period_data
                    notebook_print(f"✓ Loaded network for {period}")
                    continue
            
            # Build network for this period
            notebook_print(f"Building network for {period}...")
            period_df = df_comments[df_comments[temporal_unit] == period]
            
            if len(period_df) < 10:
                notebook_print(f"  Skipping {period} - insufficient data")
                continue
            
            networks = self._build_networks_for_period(period, period_df)
            
            if networks and networks['user_network'].number_of_nodes() > 0:
                temporal_networks[period] = networks
                if use_checkpoints:
                    self.save_checkpoint(networks, period_filename)
                notebook_print(f"✓ Built network for {period}: {networks['user_network'].number_of_nodes():,} users")
            else:
                notebook_print(f"✗ Failed to build network for {period}")
        
        notebook_print(f"✓ Built networks for {len(temporal_networks)} periods")
        return temporal_networks

    def _build_networks_for_period(self, period, period_data):
        """Build networks for a single period"""
        try:
            # Build bipartite network
            bipartite_net = self.build_bipartite_network(period_data)
            if not bipartite_net:
                return None
            
            # Project to user network
            user_net = self.project_to_user_network(bipartite_net, min_common_posts=1)
            
            if not user_net or user_net.number_of_nodes() == 0:
                return None
            
            # Build additional networks
            sentiment_net = build_sentiment_network(period_data)
            temporal_net = build_temporal_network(period_data)
            
            networks = {
                'bipartite_network': bipartite_net,
                'user_network': user_net,
                'sentiment_network': sentiment_net,
                'temporal_network': temporal_net,
                'user_sentiments': period_data.groupby('user')['sentiment'].mean().to_dict(),
                'activity_levels': period_data.groupby('user').size().to_dict(),
                'network_stats': {
                    'n_users': user_net.number_of_nodes(),
                    'n_edges': user_net.number_of_edges(),
                    'density': user_net.graph.get('density', 0),
                    'avg_clustering': user_net.graph.get('avg_clustering', 0)
                }
            }
            
            return networks
            
        except Exception as e:
            logger.error(f"Failed to build networks for period {period}: {e}")
            return None

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
        user_stats = period_data.groupby('user')['sentiment'].agg(['mean', 'count'])
        user_stats = user_stats[user_stats['count'] >= 2]
        
        users = user_stats.index.tolist()[:300]  # Limit for performance
        user_means = user_stats['mean'].values
        
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                similarity = 1 - abs(user_means[i] - user_means[j])
                if similarity > 0.5:
                    G.add_edge(users[i], users[j], weight=similarity)
        
        return G
    except Exception as e:
        return nx.Graph()

def build_temporal_network(period_data):
    """Build temporal co-occurrence network"""
    try:
        G = nx.Graph()
        user_activity = period_data['user'].value_counts()
        active_users = user_activity[user_activity >= 2].index.tolist()[:150]  # Limit for performance
        
        if len(active_users) < 2:
            return G
        
        user_posts = period_data.groupby('user')['root'].agg(set)
        
        for i, user1 in enumerate(active_users):
            posts1 = user_posts.get(user1, set())
            for user2 in active_users[i+1:]:
                posts2 = user_posts.get(user2, set())
                common = len(posts1 & posts2)
                if common > 0:
                    G.add_edge(user1, user2, weight=common)
        
        return G
    except Exception as e:
        return nx.Graph()

def detect_communities_worker(network_tuple):
    """Community detection worker"""
    period, network = network_tuple
    
    try:
        if network.number_of_nodes() < 10:
            return (period, {}, network)
            
        dimensions = min(16, max(8, network.number_of_nodes() // 100))
        bigclam = BigClam(dimensions=dimensions)
        
        adj_matrix = nx.to_scipy_sparse_array(network)
        bigclam.fit(adj_matrix)
        
        memberships = bigclam.get_memberships()
        
        communities = {}
        node_list = list(network.nodes())
        for node_idx, comm_affiliations in enumerate(memberships):
            if comm_affiliations:
                node_name = node_list[node_idx]
                communities[node_name] = comm_affiliations
        
        return (period, communities, network)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"BigCLAM for {period}: {e}")
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


# For script execution
if __name__ == "__main__":
    # Set multiprocessing start method
    import sys
    if sys.platform.startswith('win') or 'ipykernel' in sys.modules:
        # Windows or Jupyter
        try:
            mp.set_start_method('spawn', force=True)
        except:
            pass
    
    # Run analysis
    analyzer, community_results, stability_metrics = run_analysis(
        data_path="../Documentaries.corpus",
        checkpoint_dir="parallel_checkpoints",
        use_checkpoints=True
    )