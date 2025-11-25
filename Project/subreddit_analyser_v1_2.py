"""
Parallel Subreddit Network Analyzer with Bipartite Structure
Supports both notebook and script execution with multiprocessing
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

print("=== PARALLEL REDDIT NETWORK ANALYSIS WITH BIPARTITE STRUCTURE ===")

class ParallelRedditAnalyzer:
    def __init__(self, checkpoint_dir="checkpoints", n_workers=None):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.checkpoint_dir = checkpoint_dir
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"Initialized analyzer with {self.n_workers} parallel workers")
        
    def save_checkpoint(self, data, filename):
        """Save intermediate results with compression"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Checkpoint saved: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filename}: {e}")
            return False
    
    def load_checkpoint(self, filename):
        """Load intermediate results"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f"Checkpoint loaded: {filename}")
                return data
            except Exception as e:
                logger.error(f"Failed to load checkpoint {filename}: {e}")
        return None

    def build_bipartite_network(self, df_comments):
        """
        Build bipartite network of user-post interactions
        Nodes: users and posts, Edges: user commented on post
        """
        logger.info("Building bipartite user-post network...")
        
        try:
            # Input validation
            required_cols = ['user', 'root', 'text']
            missing_cols = [col for col in required_cols if col not in df_comments.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Filter out null users and posts
            df_filtered = df_comments.dropna(subset=['user', 'root']).copy()
            logger.info(f"Filtered data: {len(df_filtered)} comments (from {len(df_comments)} original)")
            
            if len(df_filtered) == 0:
                raise ValueError("No valid user-post interactions after filtering")
            
            # Create bipartite graph
            B = nx.Graph()
            
            # Add nodes with attributes
            users = df_filtered['user'].unique()
            posts = df_filtered['root'].unique()
            
            logger.info(f"Adding {len(users)} users and {len(posts)} posts to bipartite network...")
            
            # Add user nodes
            for user in users:
                B.add_node(user, bipartite=0, type='user')
            
            # Add post nodes and count comments per post
            post_comment_counts = df_filtered.groupby('root').size()
            for post in posts:
                comment_count = post_comment_counts.get(post, 0)
                B.add_node(post, bipartite=1, type='post', comment_count=comment_count)
            
            # Add edges
            edges_added = 0
            for _, row in df_filtered.iterrows():
                B.add_edge(row['user'], row['root'])
                edges_added += 1
            
            logger.info(f"Bipartite network built: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
            logger.info(f"Users: {len(users)}, Posts: {len(posts)}, Edges added: {edges_added}")
            
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
            logger.error(f"Failed to build bipartite network: {e}")
            traceback.print_exc()
            return None

    def project_to_user_network(self, bipartite_graph, min_common_posts=1):
        """
        Project bipartite network to user-user network
        Users are connected if they commented on the same post
        Edge weight = number of common posts they commented on
        """
        logger.info("Projecting bipartite network to user-user network...")
        
        try:
            if bipartite_graph is None or bipartite_graph.number_of_nodes() == 0:
                raise ValueError("Invalid bipartite graph for projection")
            
            # Get user nodes
            user_nodes = [n for n, d in bipartite_graph.nodes(data=True) 
                         if d.get('bipartite') == 0 and d.get('type') == 'user']
            
            logger.info(f"Projecting network for {len(user_nodes)} users...")
            
            # Create user-user projection
            user_network = nx.Graph()
            user_network.add_nodes_from(user_nodes)
            
            # Get post nodes
            post_nodes = [n for n, d in bipartite_graph.nodes(data=True) 
                         if d.get('bipartite') == 1 and d.get('type') == 'post']
            
            logger.info(f"Processing {len(post_nodes)} posts for projection...")
            
            # For each post, connect all users who commented on it
            edges_created = 0
            for post in post_nodes:
                # Get all users who commented on this post
                post_users = [n for n in bipartite_graph.neighbors(post) 
                             if n in user_nodes]
                
                # Connect every pair of users who commented on this post
                for i in range(len(post_users)):
                    for j in range(i + 1, len(post_users)):
                        user1, user2 = post_users[i], post_users[j]
                        
                        if user_network.has_edge(user1, user2):
                            user_network[user1][user2]['weight'] += 1
                            user_network[user1][user2]['common_posts'].add(post)
                        else:
                            user_network.add_edge(user1, user2, weight=1, 
                                                common_posts={post})
                        edges_created += 1
            
            logger.info(f"User network projection complete: {user_network.number_of_nodes()} nodes, {user_network.number_of_edges()} edges")
            
            # Filter edges by minimum common posts
            if min_common_posts > 1:
                edges_to_remove = [(u, v) for u, v, d in user_network.edges(data=True) 
                                 if d.get('weight', 0) < min_common_posts]
                user_network.remove_edges_from(edges_to_remove)
                logger.info(f"After filtering (min_common_posts={min_common_posts}): {user_network.number_of_edges()} edges")
            
            # Calculate network statistics
            if user_network.number_of_nodes() > 0:
                # Basic stats
                density = nx.density(user_network)
                avg_degree = sum(dict(user_network.degree()).values()) / user_network.number_of_nodes()
                
                # Clustering coefficient
                if user_network.number_of_edges() > 0:
                    avg_clustering = nx.average_clustering(user_network)
                    logger.info(f"Average clustering coefficient: {avg_clustering:.4f}")
                else:
                    avg_clustering = 0
                    logger.warning("No edges for clustering calculation")
                
                logger.info(f"User network density: {density:.6f}")
                logger.info(f"Average degree: {avg_degree:.2f}")
                
                # Add network statistics as graph attributes
                user_network.graph['density'] = density
                user_network.graph['avg_degree'] = avg_degree
                user_network.graph['avg_clustering'] = avg_clustering
                user_network.graph['projection_time'] = datetime.now().isoformat()
            
            return user_network
            
        except Exception as e:
            logger.error(f"Failed to project user network: {e}")
            traceback.print_exc()
            return None

    def parallel_data_loading(self, data_path, use_checkpoints=True):
        """Parallel data loading with fallback options"""
        logger.info("Loading and preprocessing Reddit data...")
        
        if use_checkpoints:
            checkpoint_data = self.load_checkpoint("preprocessed_data.pkl")
            if checkpoint_data:
                logger.info("Using existing preprocessed data")
                df_posts, df_comments = checkpoint_data
            
                logger.info(f"Loaded {len(df_posts)} posts")
                logger.info(f"Posts memory usage: {df_posts.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

                logger.info(f"Loaded {len(df_comments)} comments")
                logger.info(f"Comments memory usage: {df_comments.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                return checkpoint_data
        
        try:
            # Load posts
            logger.info("Loading posts...")
            posts_file = f"{data_path}/conversations.json"
            if not os.path.exists(posts_file):
                raise FileNotFoundError(f"Posts file not found: {posts_file}")
            
            df_posts = pd.read_json(posts_file).T.reset_index(drop=False)
            logger.info(f"Posts loaded: {len(df_posts)} rows")
            logger.info(f"Posts memory usage: {df_posts.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Load comments - try multiple methods
            logger.info("Loading comments...")
            df_comments = self._load_comments_smart(data_path)
            
            if df_comments is None or len(df_comments) == 0:
                raise ValueError("Failed to load comments")

            # Validation
            logger.info(f"Loaded {len(df_comments)} comments")
            logger.info(f"Comments memory usage: {df_comments.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            # Parallel preprocessing
            logger.info("Parallel sentiment analysis...")
            df_comments = self._parallel_preprocessing(df_comments)
            
            # Save checkpoint
            if use_checkpoints:
                self.save_checkpoint((df_posts, df_comments), "preprocessed_data.pkl")
            
            logger.info(f"Successfully loaded {len(df_comments)} comments")
            return df_posts, df_comments
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            traceback.print_exc()
            return None, None

    def _load_comments_smart(self, data_path):
        """Smart comment loading with multiple fallback methods"""
        # Try CSV first (fastest if it exists)
        csv_path = f"{data_path}/utterances.csv"
        jsonl_path = f"{data_path}/utterances.jsonl"
        
        if os.path.exists(csv_path):
            logger.info("Using existing utterances.csv file")
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                logger.info(f"Loaded {len(df)} comments from CSV")
                return df
            except Exception as e:
                logger.warning(f"CSV loading failed: {e}")
        
        # Try JSONL with chunked reading
        if os.path.exists(jsonl_path):
            logger.info("Reading JSONL in chunks...")
            try:
                df = self._load_jsonl_chunked(jsonl_path)
                logger.info(f"Loaded {len(df)} comments from JSONL")
                return df
            except Exception as e:
                logger.error(f"JSONL loading failed: {e}")
                return None
        
        raise FileNotFoundError(f"No comments file found in {data_path}")

    def _load_jsonl_chunked(self, file_path, chunk_size=100000):
        """Load JSONL file in chunks to avoid memory issues"""
        chunks = []
        chunk = []
        
        logger.info("Reading JSONL file...")
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    chunk.append(data)
                    
                    if len(chunk) >= chunk_size:
                        chunks.append(pd.DataFrame(chunk))
                        chunk = []
                        if (i + 1) % 500000 == 0:
                            logger.info(f"Processed {i+1} lines...")
                        
                except json.JSONDecodeError:
                    continue
            
            # Add remaining data
            if chunk:
                chunks.append(pd.DataFrame(chunk))
        
        # Combine all chunks
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Loaded {len(df)} comments from utterances.jsonl")
            
            # Optionally save as CSV for faster future loading
            csv_path = file_path.replace('.jsonl', '.csv')
            if not os.path.exists(csv_path):
                logger.info("Creating utterances.csv for faster future loading")
                df.to_csv(csv_path, index=False)
            
            return df
        
        return None

    def _parallel_preprocessing(self, df_comments):
        """Parallel preprocessing with sentiment analysis"""
        # Split data for parallel processing
        n_chunks = min(self.n_workers, len(df_comments) // 10000 + 1)
        chunks = np.array_split(df_comments, n_chunks)
        
        logger.info(f"Processing {len(chunks)} chunks in parallel...")
        
        # Process sentiment in parallel
        with Pool(self.n_workers) as pool:
            processed_chunks = pool.map(process_sentiment_chunk, chunks)
        
        # Combine results
        df_comments = pd.concat(processed_chunks, ignore_index=True)
        
        # Add temporal features
        logger.info("Adding temporal features...")
        df_comments['datetime'] = pd.to_datetime(df_comments['timestamp'], unit='s', errors='coerce')
        df_comments = df_comments[df_comments['datetime'].notna()]
        df_comments['week'] = df_comments['datetime'].dt.to_period('W').astype(str)
        df_comments['month'] = df_comments['datetime'].dt.strftime('%Y-%m')
        
        return df_comments

    def parallel_network_construction(self, df_comments, temporal_unit='month', max_periods=6, use_checkpoints=True):
        """Parallel network construction using proper bipartite structure"""
        logger.info(f"Building temporal networks ({temporal_unit})...")
        
        if use_checkpoints:
            checkpoint_data = self.load_checkpoint(f"temporal_networks_{temporal_unit}.pkl")
            if checkpoint_data:
                return checkpoint_data
        
        try:
            # Get unique periods and limit
            periods = df_comments[temporal_unit].value_counts().head(max_periods).index.tolist()
            logger.info(f"Processing {len(periods)} periods...")
            
            # Prepare arguments for parallel processing
            period_data_list = [
                (period, df_comments[df_comments[temporal_unit] == period])
                for period in periods
            ]
            
            # Build networks in parallel using proper bipartite structure
            logger.info("Building networks in parallel with bipartite structure...")
            with Pool(self.n_workers) as pool:
                results = pool.map(self._build_bipartite_networks_worker, period_data_list)
            
            # Combine results
            temporal_networks = {}
            for period, networks in results:
                if networks and networks['user_network'].number_of_nodes() > 0:
                    temporal_networks[period] = networks
            
            # Save checkpoint
            if use_checkpoints:
                self.save_checkpoint(temporal_networks, f"temporal_networks_{temporal_unit}.pkl")
            
            logger.info(f"Successfully built networks for {len(temporal_networks)} periods")
            return temporal_networks
            
        except Exception as e:
            logger.error(f"Network construction failed: {e}")
            traceback.print_exc()
            return {}

    def _build_bipartite_networks_worker(self, period_data_tuple):
        """Worker function for building bipartite-based networks"""
        period, period_data = period_data_tuple
        
        try:
            logger.debug(f"Worker processing period {period} with {len(period_data)} comments")
            
            if len(period_data) < 10:
                return (period, None)
            
            # Build bipartite network
            bipartite_net = self.build_bipartite_network(period_data)
            if not bipartite_net:
                return (period, None)
            
            # Project to user network
            user_net = self.project_to_user_network(bipartite_net, min_common_posts=1)
            
            if not user_net or user_net.number_of_nodes() == 0:
                return (period, None)
            
            # Build additional networks if meaningful
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
            
            logger.debug(f"Period {period}: {user_net.number_of_nodes()} users, {user_net.number_of_edges()} edges")
            return (period, networks)
            
        except Exception as e:
            logger.error(f"Worker failed for period {period}: {e}")
            return (period, None)

    def parallel_community_detection(self, temporal_networks, use_checkpoints=True):
        """Parallel community detection with BigCLAM"""
        logger.info("Running community detection...")
        
        if use_checkpoints:
            checkpoint_data = self.load_checkpoint("community_results.pkl")
            if checkpoint_data:
                return checkpoint_data
        
        try:
            # Prepare networks for parallel processing
            network_list = [
                (period, networks['user_network'])  # Use the proper user network from bipartite projection
                for period, networks in temporal_networks.items()
                if networks['user_network'].number_of_nodes() >= 10
                and networks['user_network'].number_of_nodes() <= 3000
            ]
            
            logger.info(f"Running BigCLAM on {len(network_list)} networks...")
            
            # Run BigCLAM in parallel
            with Pool(self.n_workers) as pool:
                results = pool.map(detect_communities_worker, network_list)
            
            # Combine results
            community_results = {}
            for period, communities, network in results:
                if communities:
                    community_results[period] = {
                        'communities': communities,
                        'network': network
                    }
            
            # Save checkpoint
            if use_checkpoints:
                self.save_checkpoint(community_results, "community_results.pkl")
            
            logger.info(f"Found communities in {len(community_results)} periods")
            return community_results
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            traceback.print_exc()
            return {}

    def parallel_hub_analysis(self, community_results, use_checkpoints=True):
        """Parallel hub identification and analysis"""
        logger.info("Analyzing network hubs...")
        
        if use_checkpoints:
            checkpoint_data = self.load_checkpoint("hub_evolution.pkl")
            if checkpoint_data:
                return checkpoint_data
        
        try:
            # Prepare networks
            network_list = [
                (period, result['network'])
                for period, result in community_results.items()
            ]
            
            # Analyze hubs in parallel
            with Pool(self.n_workers) as pool:
                results = pool.map(analyze_hubs_worker, network_list)
            
            # Combine results
            hub_evolution = {}
            for period, hubs, degrees in results:
                hub_evolution[period] = {
                    'hubs': hubs,
                    'degrees': degrees
                }
            
            # Save checkpoint
            if use_checkpoints:
                self.save_checkpoint(hub_evolution, "hub_evolution.pkl")
            
            logger.info(f"Analyzed hubs for {len(hub_evolution)} periods")
            return hub_evolution
            
        except Exception as e:
            logger.error(f"Hub analysis failed: {e}")
            traceback.print_exc()
            return {}

    def calculate_stability_metrics(self, community_results, hub_evolution):
        """Calculate stability metrics across periods"""
        logger.info("Calculating stability metrics...")
        
        try:
            periods = sorted(community_results.keys())
            
            if len(periods) < 2:
                logger.warning("Need at least 2 periods for stability analysis")
                return {}
            
            stability_metrics = {}
            
            for i in range(len(periods) - 1):
                period1, period2 = periods[i], periods[i+1]
                
                # Hub persistence
                hubs1 = set(hub_evolution.get(period1, {}).get('hubs', {}).keys())
                hubs2 = set(hub_evolution.get(period2, {}).get('hubs', {}).keys())
                
                hub_overlap = 0
                if hubs1 and hubs2:
                    hub_overlap = len(hubs1 & hubs2) / len(hubs1 | hubs2)
                
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
                logger.info(f"  {period1}→{period2}: Hub={hub_overlap:.3f}, Comm={community_similarity:.3f}")
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Stability calculation failed: {e}")
            traceback.print_exc()
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
                jaccard = len(aff1 & aff2) / len(aff1 | aff2)
                similarities.append(jaccard)
        
        return np.mean(similarities) if similarities else 0

    def export_results(self, community_results, hub_evolution, stability_metrics):
        """Export results to CSV files"""
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
                logger.info("Exported community_memberships.csv")
            
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
                logger.info("Exported hub_analysis.csv")
            
            # Export stability metrics
            if stability_metrics:
                stability_df = pd.DataFrame.from_dict(stability_metrics, orient='index')
                stability_df.to_csv('stability_metrics.csv', index=True)
                logger.info("Exported stability_metrics.csv")
            
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def generate_report(self, stability_metrics):
        """Generate summary report"""
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        
        if stability_metrics:
            hub_stabilities = [m['hub_overlap'] for m in stability_metrics.values()]
            comm_stabilities = [m['community_similarity'] for m in stability_metrics.values()]
            
            logger.info(f"\nSTABILITY METRICS:")
            logger.info(f"  Average Hub Stability: {np.mean(hub_stabilities):.3f}")
            logger.info(f"  Average Community Stability: {np.mean(comm_stabilities):.3f}")
            logger.info(f"  Period Transitions Analyzed: {len(stability_metrics)}")
        
        # Memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 ** 3
        logger.info(f"\nRESOURCE USAGE:")
        logger.info(f"  Memory Usage: {memory_usage:.2f} GB")
        logger.info(f"  CPU Cores Used: {self.n_workers}")
        
        logger.info("\n" + "="*60)


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
        
        users = user_stats.index.tolist()[:300]
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
        active_users = user_activity[user_activity >= 2].index.tolist()[:150]
        
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
        
        logger.info(f"  Period {period}: {len(communities)} users in communities")
        return (period, communities, network)
        
    except Exception as e:
        logger.error(f"BigCLAM for {period}: {e}")
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
        
        logger.info(f"  Period {period}: {len(hubs)} hubs identified")
        return (period, hubs, degrees)
        
    except Exception as e:
        logger.error(f"Hub analysis for {period}: {e}")
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
    logger.info("="*60)
    logger.info("PARALLEL REDDIT NETWORK ANALYSIS WITH BIPARTITE STRUCTURE")
    logger.info("="*60)
    
    # Initialize analyzer
    analyzer = ParallelRedditAnalyzer(checkpoint_dir=checkpoint_dir, n_workers=n_workers)
    
    try:
        # Phase 1: Load Data
        logger.info("\n[PHASE 1] Loading data...")
        df_posts, df_comments = analyzer.parallel_data_loading(data_path, use_checkpoints)
        
        if df_comments is None:
            logger.error("Failed to load data")
            return None, None, None

        # Phase 2: Build Networks with Bipartite Structure
        logger.info("\n[PHASE 2] Building networks with bipartite structure...")
        temporal_networks = analyzer.parallel_network_construction(
            df_comments, temporal_unit='month', max_periods=6, use_checkpoints=use_checkpoints
        )
        
        if not temporal_networks:
            logger.error("Failed to build networks")
            return analyzer, None, None
        
        # Phase 3: Community Detection
        logger.info("\n[PHASE 3] Detecting communities...")
        community_results = analyzer.parallel_community_detection(temporal_networks, use_checkpoints)
        
        # Phase 4: Hub Analysis
        logger.info("\n[PHASE 4] Analyzing hubs...")
        hub_evolution = analyzer.parallel_hub_analysis(community_results, use_checkpoints)
        
        # Phase 5: Stability Metrics
        logger.info("\n[PHASE 5] Calculating stability...")
        stability_metrics = analyzer.calculate_stability_metrics(community_results, hub_evolution)
        
        # Phase 6: Export Results
        logger.info("\n[PHASE 6] Exporting results...")
        analyzer.export_results(community_results, hub_evolution, stability_metrics)
        
        # Generate Report
        analyzer.generate_report(stability_metrics)
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        return analyzer, community_results, stability_metrics
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
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