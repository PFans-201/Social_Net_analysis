"""
Parallel subReddit Network Analyzer:
Supports both notebook and script execution with proper multiprocessing
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

warnings.filterwarnings('ignore')

print("=== PARALLEL NETWORK STABILITY ANALYSIS (JUPYTER COMPATIBLE) ===")

class ParallelRedditAnalyzer:
    def __init__(self, checkpoint_dir="checkpoints", n_workers=None):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.checkpoint_dir = checkpoint_dir
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"[INIT] Using {self.n_workers} parallel workers")
        
    def save_checkpoint(self, data, filename):
        """Save intermediate results with compression"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[CHECKPOINT] Saved {filename}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save {filename}: {e}")
            return False
    
    def load_checkpoint(self, filename):
        """Load intermediate results"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"[CHECKPOINT] Loaded {filename}")
                return data
            except Exception as e:
                print(f"[ERROR] Failed to load {filename}: {e}")
        return None

    def parallel_data_loading(self, data_path, use_checkpoints=True):
        """Parallel data loading with fallback options"""
        print("Loading and preprocessing Reddit data...")
        
        if use_checkpoints:
            checkpoint_data = self.load_checkpoint("preprocessed_data.pkl")
            if checkpoint_data:
                print("[CHECKPOINT] Using existing preprocessed data")
                df_posts, df_comments = checkpoint_data
            
                print(f"[SUCCESS] Loaded {len(df_posts)} posts")
                print("[DEBUG] Sample posts data:")
                display(df_posts.head())
                print(f"[INFO] Posts memory usage: {df_posts.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

                print(f"[SUCCESS] Loaded {len(df_comments)} comments")
                print("[DEBUG] Sample comments data:")
                display(df_comments.head())
                print(f"[INFO] Comments memory usage: {df_comments.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                return checkpoint_data
        
        try:
            # Load posts
            print("[LOADING] Loading posts...")
            posts_file = f"{data_path}/conversations.json"
            if not os.path.exists(posts_file):
                raise FileNotFoundError(f"Posts file not found: {posts_file}")
            
            df_posts = pd.read_json(posts_file).T.reset_index(drop=False)
            print(f"[SUCCESS] Posts loaded: {len(df_posts)} rows")
            print(f"[INFO] Posts memory usage: {df_posts.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print("[DEBUG] Sample posts data:")
            display(df_posts.head())
            
            # Load comments - try multiple methods
            print("[LOADING] Loading comments...")
            df_comments = self._load_comments_smart(data_path)
            
            if df_comments is None or len(df_comments) == 0:
                raise ValueError("Failed to 'smart' load comments")

            # Validation
            print(f"[SUCCESS] Loaded {len(df_comments)} comments")
            print(f"[INFO] Comments memory usage: {df_comments.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print("[DEBUG] Sample comments data:")
            display(df_comments.head())

            # Parallel preprocessing
            print("[PROCESSING] Parallel sentiment analysis...")
            df_comments = self._parallel_preprocessing(df_comments)
            
            # Save checkpoint
            if use_checkpoints:
                self.save_checkpoint((df_posts, df_comments), "preprocessed_data.pkl")
            
            print(f"[SUCCESS] Loaded {len(df_comments)} comments")
            return df_posts, df_comments
            
        except Exception as e:
            print(f"[CRITICAL] Data loading failed: {e}")
            traceback.print_exc()
            return None, None

    def _load_comments_smart(self, data_path):
        """Smart comment loading with multiple fallback methods"""
        # Try CSV first (fastest if it exists)
        csv_path = f"{data_path}/utterances.csv"
        jsonl_path = f"{data_path}/utterances.jsonl"
        
        if os.path.exists(csv_path):
            print("[LOADING] Using existing utterances.csv file")
            try:
                # Read CSV without dask pyarrow issues
                df = pd.read_csv(csv_path, low_memory=False)
                print(f"[SUCCESS] Loaded {len(df)} comments from CSV")
                display(df.head())
                return df
            except Exception as e:
                print(f"[WARNING] CSV loading failed: {e}")
        
        # Try JSONL with chunked reading (sequential but memory efficient)
        if os.path.exists(jsonl_path):
            print("[LOADING] Reading JSONL in chunks...")
            try:
                df = self._load_jsonl_chunked(jsonl_path)
                print(f"[SUCCESS] Loaded {len(df)} comments from JSONL")
                display(df.head())
                return df
            except Exception as e:
                print(f"[ERROR] loading comments from utterances.jsonl failed: {e}")
                return None
        
        raise FileNotFoundError(f"No comments file found in {data_path}")

    def _load_jsonl_chunked(self, file_path, chunk_size=100000):
        """Load JSONL file in chunks to avoid memory issues"""
        chunks = []
        chunk = []
        
        print("[LOADING] Reading JSONL file...")
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    chunk.append(data)
                    
                    if len(chunk) >= chunk_size:
                        chunks.append(pd.DataFrame(chunk))
                        chunk = []
                        if (i + 1) % 500000 == 0:
                            print(f"  Processed {i+1} lines...")
                        
                except json.JSONDecodeError:
                    continue
            
            # Add remaining data
            if chunk:
                chunks.append(pd.DataFrame(chunk))
        
        # Combine all chunks
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            print(f"[SUCCESS] Loaded {len(df)} comments from utterances.jsonl")
            display(df.head())
            
            # Optionally save as CSV for faster future loading
            csv_path = file_path.replace('.jsonl', '.csv')
            if not os.path.exists(csv_path):
                print("[SAVING] Creating utterances.csv for faster future loading!")
                df.to_csv(csv_path, index=False)
            
            return df
        
        return None

    def _parallel_preprocessing(self, df_comments):
        """Parallel preprocessing with sentiment analysis"""
        # Split data for parallel processing
        n_chunks = min(self.n_workers, len(df_comments) // 10000 + 1)
        chunks = np.array_split(df_comments, n_chunks)
        
        print(f"[PROCESSING] Processing {len(chunks)} chunks in parallel...")
        
        # Process sentiment in parallel
        with Pool(self.n_workers) as pool:
            processed_chunks = pool.map(process_sentiment_chunk, chunks)
        
        # Combine results
        df_comments = pd.concat(processed_chunks, ignore_index=True)
        
        # Add temporal features
        print("[PROCESSING] Adding temporal features...")
        df_comments['datetime'] = pd.to_datetime(df_comments['timestamp'], unit='s', errors='coerce')
        df_comments = df_comments[df_comments['datetime'].notna()]
        df_comments['week'] = df_comments['datetime'].dt.to_period('W').astype(str)
        df_comments['month'] = df_comments['datetime'].dt.strftime('%Y-%m')
        
        return df_comments

    def parallel_network_construction(self, df_comments, temporal_unit='month', max_periods=6, use_checkpoints=True):
        """Parallel network construction across time periods"""
        print(f"[PARALLEL] Building temporal networks ({temporal_unit})...")
        
        if use_checkpoints:
            checkpoint_data = self.load_checkpoint(f"temporal_networks_{temporal_unit}.pkl")
            if checkpoint_data:
                return checkpoint_data
        
        try:
            # Get unique periods and limit
            periods = df_comments[temporal_unit].value_counts().head(max_periods).index.tolist()
            print(f"[PARALLEL] Processing {len(periods)} periods...")
            
            # Prepare arguments for parallel processing
            period_data_list = [
                (period, df_comments[df_comments[temporal_unit] == period])
                for period in periods
            ]
            
            # Build networks in parallel
            print("[PARALLEL] Building networks in parallel...")
            with Pool(self.n_workers) as pool:
                results = pool.map(build_networks_for_period, period_data_list)
            
            # Combine results
            temporal_networks = {}
            for period, networks in results:
                if networks:
                    temporal_networks[period] = networks
            
            # Save checkpoint
            if use_checkpoints:
                self.save_checkpoint(temporal_networks, f"temporal_networks_{temporal_unit}.pkl")
            
            print(f"[SUCCESS] Built networks for {len(temporal_networks)} periods")
            return temporal_networks
            
        except Exception as e:
            print(f"[CRITICAL] Network construction failed: {e}")
            traceback.print_exc()
            return {}

    def parallel_community_detection(self, temporal_networks, use_checkpoints=True):
        """Parallel community detection with BigCLAM"""
        print("[PARALLEL] Running community detection...")
        
        if use_checkpoints:
            checkpoint_data = self.load_checkpoint("community_results.pkl")
            if checkpoint_data:
                return checkpoint_data
        
        try:
            # Prepare networks for parallel processing
            network_list = [
                (period, networks['conversation_network'])
                for period, networks in temporal_networks.items()
                if networks['conversation_network'].number_of_nodes() >= 10
                and networks['conversation_network'].number_of_nodes() <= 3000
            ]
            
            print(f"[PARALLEL] Running BigCLAM on {len(network_list)} networks...")
            
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
            
            print(f"[SUCCESS] Found communities in {len(community_results)} periods")
            return community_results
            
        except Exception as e:
            print(f"[CRITICAL] Community detection failed: {e}")
            traceback.print_exc()
            return {}

    def parallel_hub_analysis(self, community_results, use_checkpoints=True):
        """Parallel hub identification and analysis"""
        print("[PARALLEL] Analyzing network hubs...")
        
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
            
            print(f"[SUCCESS] Analyzed hubs for {len(hub_evolution)} periods")
            return hub_evolution
            
        except Exception as e:
            print(f"[CRITICAL] Hub analysis failed: {e}")
            traceback.print_exc()
            return {}

    def calculate_stability_metrics(self, community_results, hub_evolution):
        """Calculate stability metrics across periods"""
        print("[ANALYZING] Calculating stability metrics...")
        
        try:
            periods = sorted(community_results.keys())
            
            if len(periods) < 2:
                print("[WARNING] Need at least 2 periods for stability analysis")
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
                print(f"  {period1}→{period2}: Hub={hub_overlap:.3f}, Comm={community_similarity:.3f}")
            
            return stability_metrics
            
        except Exception as e:
            print(f"[ERROR] Stability calculation failed: {e}")
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
        print("[EXPORT] Exporting results to CSV...")
        
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
                print("[SUCCESS] Exported community_memberships.csv")
            
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
                print("[SUCCESS] Exported hub_analysis.csv")
            
            # Export stability metrics
            if stability_metrics:
                stability_df = pd.DataFrame.from_dict(stability_metrics, orient='index')
                stability_df.to_csv('stability_metrics.csv', index=True)
                print("[SUCCESS] Exported stability_metrics.csv")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
            return False

    def generate_report(self, stability_metrics):
        """Generate summary report"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        if stability_metrics:
            hub_stabilities = [m['hub_overlap'] for m in stability_metrics.values()]
            comm_stabilities = [m['community_similarity'] for m in stability_metrics.values()]
            
            print(f"\nSTABILITY METRICS:")
            print(f"  Average Hub Stability: {np.mean(hub_stabilities):.3f}")
            print(f"  Average Community Stability: {np.mean(comm_stabilities):.3f}")
            print(f"  Period Transitions Analyzed: {len(stability_metrics)}")
        
        # Memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 ** 3
        print(f"\nRESOURCE USAGE:")
        print(f"  Memory Usage: {memory_usage:.2f} GB")
        print(f"  CPU Cores Used: {self.n_workers}")
        
        print("\n" + "="*60)


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


def build_networks_for_period(period_data_tuple):
    """Build networks for a single period (worker function)"""
    period, period_data = period_data_tuple
    
    try:
        if len(period_data) < 10:
            return (period, None)
        
        # Build conversation network
        G_conversation = build_conversation_network(period_data)
        
        # Build additional networks if meaningful
        if G_conversation.number_of_nodes() > 5:
            G_sentiment = build_sentiment_network(period_data)
            G_temporal = build_temporal_network(period_data)
        else:
            G_sentiment = nx.Graph()
            G_temporal = nx.Graph()
        
        networks = {
            'conversation_network': G_conversation,
            'sentiment_network': G_sentiment,
            'temporal_network': G_temporal,
            'user_sentiments': period_data.groupby('user')['sentiment'].mean().to_dict(),
            'activity_levels': period_data.groupby('user').size().to_dict()
        }
        
        print(f"  [WORKER] {period}: {G_conversation.number_of_nodes()} users, {G_conversation.number_of_edges()} edges")
        return (period, networks)
        
    except Exception as e:
        print(f"[WORKER ERROR] Network building for {period}: {e}")
        return (period, None)


def build_conversation_network(period_data, max_users_per_post=100):
    """Build conversation network"""
    G = nx.Graph()
    
    try:
        post_user_groups = period_data.groupby('root')['user'].agg(set)
        
        for post, users in post_user_groups.items():
            if len(users) > max_users_per_post:
                continue
            
            users_list = list(users)
            for i in range(len(users_list)):
                for j in range(i + 1, len(users_list)):
                    user_i, user_j = users_list[i], users_list[j]
                    
                    if G.has_edge(user_i, user_j):
                        G[user_i][user_j]['weight'] += 1
                    else:
                        G.add_edge(user_i, user_j, weight=1)
        
        return G
    except Exception as e:
        print(f"[ERROR] Conversation network: {e}")
        return nx.Graph()


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
        
        print(f"  [WORKER] {period}: {len(communities)} users in communities")
        return (period, communities, network)
        
    except Exception as e:
        print(f"[WORKER ERROR] BigCLAM for {period}: {e}")
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
        
        print(f"  [WORKER] {period}: {len(hubs)} hubs identified")
        return (period, hubs, degrees)
        
    except Exception as e:
        print(f"[WORKER ERROR] Hub analysis for {period}: {e}")
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
    print("="*60)
    print("PARALLEL REDDIT NETWORK ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = ParallelRedditAnalyzer(checkpoint_dir=checkpoint_dir, n_workers=n_workers)
    
    try:
        # Phase 1: Load Data
        print("\n[PHASE 1] Loading data...")
        df_posts, df_comments = analyzer.parallel_data_loading(data_path, use_checkpoints)
        
        if df_comments is None:
            print("[CRITICAL] Failed to load data")
            return None, None, None

        # Phase 2: Build Networks
        print("\n[PHASE 2] Building networks...")
        temporal_networks = analyzer.parallel_network_construction(
            df_comments, temporal_unit='month', max_periods=6, use_checkpoints=use_checkpoints
        )
        
        if not temporal_networks:
            print("[CRITICAL] Failed to build networks")
            return analyzer, None, None
        
        # Phase 3: Community Detection
        print("\n[PHASE 3] Detecting communities...")
        community_results = analyzer.parallel_community_detection(temporal_networks, use_checkpoints)
        
        # Phase 4: Hub Analysis
        print("\n[PHASE 4] Analyzing hubs...")
        hub_evolution = analyzer.parallel_hub_analysis(community_results, use_checkpoints)
        
        # Phase 5: Stability Metrics
        print("\n[PHASE 5] Calculating stability...")
        stability_metrics = analyzer.calculate_stability_metrics(community_results, hub_evolution)
        
        # Phase 6: Export Results
        print("\n[PHASE 6] Exporting results...")
        analyzer.export_results(community_results, hub_evolution, stability_metrics)
        
        # Generate Report
        analyzer.generate_report(stability_metrics)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return analyzer, community_results, stability_metrics
        
    except Exception as e:
        print(f"\n[CRITICAL] Analysis failed: {e}")
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