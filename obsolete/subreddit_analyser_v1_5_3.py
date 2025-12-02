"""
Reddit Reply Network Analyzer - FIXED VERSION
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
from collections import defaultdict, Counter

# Configure logging to file ONLY
import logging
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_analysis.log', mode='w'),
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Set up matplotlib
plt.style.use('default')
sns.set_palette("husl")

class RedditReplyAnalyzer:
    def __init__(self, checkpoint_dir="checkpoints", n_workers=None):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.checkpoint_dir = checkpoint_dir
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"🚀 Initialized analyzer with {self.n_workers} parallel workers")
        print(f"{'='*70}\n")
        
    def save_checkpoint(self, data, filename):
        """Save checkpoint silently"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"   ✓ Checkpoint saved: {filename}")
            return True
        except Exception as e:
            print(f"   ✗ Failed to save {filename}: {e}")
            return False
    
    def load_checkpoint(self, filename, max_age_hours=48):
        """Load checkpoint silently"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                file_age = (datetime.now() - file_time).total_seconds() / 3600
                
                if file_age > max_age_hours:
                    return None
                
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"   ✓ Loaded checkpoint: {filename}")
                return data
            except Exception as e:
                print(f"   ✗ Failed to load checkpoint {filename}: {e}")
                return None
        return None

    def parallel_data_loading(self, data_path, use_checkpoints=True):
        """Load and preprocess data"""
        print("\n" + "="*70)
        print("📚 PHASE 1: DATA LOADING")
        print("="*70)
        
        if use_checkpoints:
            checkpoint_data = self.load_checkpoint("preprocessed_data.pkl")
            if checkpoint_data is not None:
                df_posts, df_comments = checkpoint_data
                print("\n✓ Using cached data:")
                print(f"   • Posts: {len(df_posts):,}")
                print(f"   • Comments: {len(df_comments):,}")
                print(f"   • Unique users: {df_comments['user'].nunique():,}")
                return df_posts, df_comments
        
        try:
            print("\n📖 Loading posts...")
            posts_file = f"{data_path}/conversations.json"
            with open(posts_file, 'r') as f:
                posts_data = json.load(f)
            
            df_posts = pd.DataFrame.from_dict(posts_data, orient='index')
            df_posts['id'] = df_posts.index
            df_posts = df_posts.reset_index(drop=True)
            print(f"   ✓ Loaded {len(df_posts):,} posts")
            
            print("\n💬 Loading comments...")
            df_comments = self._load_comments_smart(data_path)
            
            if df_comments is None or len(df_comments) == 0:
                raise ValueError("Failed to load comments")

            print(f"   ✓ Loaded {len(df_comments):,} comments")

            print("\n🔍 Running sentiment analysis...")
            df_comments = self._parallel_preprocessing(df_comments)
            
            if use_checkpoints:
                self.save_checkpoint((df_posts, df_comments), "preprocessed_data.pkl")
            
            gc.collect()
            
            print("\n✓ Data loading complete\n")
            return df_posts, df_comments
            
        except Exception as e:
            print(f"\n✗ Data loading failed: {e}\n")
            traceback.print_exc()
            return None, None

    def _load_comments_smart(self, data_path):
        """Smart comment loading"""
        jsonl_path = f"{data_path}/utterances.jsonl"
        
        if os.path.exists(jsonl_path):
            print("   Reading JSONL in chunks...")
            try:
                df = self._load_jsonl_chunked(jsonl_path)
                return df
            except Exception as e:
                print(f"   ✗ JSONL loading failed: {e}")
                return None
        
        raise FileNotFoundError(f"No comments file found in {data_path}")

    def _load_jsonl_chunked(self, file_path, chunk_size=100000):
        """Load JSONL in chunks with progress"""
        chunks = []
        current_chunk = []
        line_count = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    current_chunk.append(data)
                    line_count += 1
                    
                    if len(current_chunk) >= chunk_size:
                        chunks.append(pd.DataFrame(current_chunk))
                        current_chunk = []
                        if line_count % 500000 == 0:
                            print(f"      Processed {line_count:,} lines...")
                        
                except json.JSONDecodeError:
                    continue
            
            if current_chunk:
                chunks.append(pd.DataFrame(current_chunk))
        
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            return df
        
        return None

    def _parallel_preprocessing(self, df_comments):
        """Parallel preprocessing with sentiment"""
        n_chunks = min(self.n_workers, len(df_comments) // 10000 + 1)
        chunks = np.array_split(df_comments, n_chunks)
        
        print(f"   Processing {len(chunks)} chunks in parallel...")
        
        with Pool(self.n_workers) as pool:
            processed_chunks = pool.map(process_sentiment_chunk, chunks)
        
        df_comments = pd.concat(processed_chunks, ignore_index=True)
        
        # Add temporal features
        df_comments['datetime'] = pd.to_datetime(df_comments['timestamp'], unit='s', errors='coerce')
        df_comments = df_comments[df_comments['datetime'].notna()]
        df_comments['week'] = df_comments['datetime'].dt.to_period('W').astype(str)
        df_comments['month'] = df_comments['datetime'].dt.strftime('%Y-%m')
        df_comments['year'] = df_comments['datetime'].dt.year
        
        return df_comments

    def build_reply_network(self, df_comments, min_interactions=1):
        """Build directed reply network"""
        try:
            required_cols = ['id', 'user', 'reply_to']
            missing_cols = [col for col in required_cols if col not in df_comments.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            df_filtered = df_comments[
                (df_comments['user'] != '[deleted]') & 
                (df_comments['user'].notna()) &
                (df_comments['reply_to'].notna())
            ].copy()
            
            comment_to_user = df_comments.set_index('id')['user']
            df_filtered['reply_to_user'] = df_filtered['reply_to'].map(comment_to_user)
            
            valid_replies = df_filtered[
                (df_filtered['reply_to_user'].notna()) & 
                (df_filtered['reply_to_user'] != '[deleted]') &
                (df_filtered['user'] != df_filtered['reply_to_user'])
            ]
            
            if len(valid_replies) == 0:
                return nx.DiGraph()
            
            G = nx.DiGraph()
            
            interaction_counts = defaultdict(int)
            for _, row in valid_replies.iterrows():
                pair = (row['user'], row['reply_to_user'])
                interaction_counts[pair] += 1
            
            for (source, target), weight in interaction_counts.items():
                if weight >= min_interactions:
                    G.add_edge(source, target, weight=weight)
            
            all_users = set()
            for source, target in interaction_counts.keys():
                all_users.add(source)
                all_users.add(target)
            
            G.add_nodes_from(all_users)
            
            if G.number_of_nodes() > 0:
                density = nx.density(G)
                avg_in_degree = sum(dict(G.in_degree()).values()) / G.number_of_nodes()
                avg_out_degree = sum(dict(G.out_degree()).values()) / G.number_of_nodes()
                reciprocity = nx.reciprocity(G) if G.number_of_edges() > 0 else 0
                
                G.graph['density'] = density
                G.graph['avg_in_degree'] = avg_in_degree
                G.graph['avg_out_degree'] = avg_out_degree
                G.graph['reciprocity'] = reciprocity
            
            return G
            
        except Exception as e:
            print(f"      ✗ Failed to build reply network: {e}")
            return nx.DiGraph()

    def build_undirected_reply_network(self, df_comments, min_interactions=1):
        """Build undirected reply network"""
        directed_net = self.build_reply_network(df_comments, min_interactions)
        
        if directed_net.number_of_nodes() == 0:
            return nx.Graph()
        
        undirected_net = nx.Graph()
        undirected_net.add_nodes_from(directed_net.nodes())
        
        for u, v, data in directed_net.edges(data=True):
            weight = data.get('weight', 1)
            if undirected_net.has_edge(u, v):
                undirected_net[u][v]['weight'] += weight
            else:
                undirected_net.add_edge(u, v, weight=weight)
        
        for u, v, data in directed_net.edges(data=True):
            if directed_net.has_edge(v, u):
                reverse_weight = directed_net[v][u].get('weight', 1)
                if undirected_net.has_edge(u, v):
                    undirected_net[u][v]['weight'] += reverse_weight
        
        if undirected_net.number_of_nodes() > 0:
            density = nx.density(undirected_net)
            avg_degree = sum(dict(undirected_net.degree()).values()) / undirected_net.number_of_nodes()
            avg_clustering = nx.average_clustering(undirected_net) if undirected_net.number_of_edges() > 0 else 0
            
            undirected_net.graph.update(directed_net.graph)
            undirected_net.graph['avg_degree'] = avg_degree
            undirected_net.graph['avg_clustering'] = avg_clustering
        
        return undirected_net

    def parallel_network_construction(self, df_comments, temporal_unit='month', 
                                     min_gcc_size=15000, periods_per_year=6, 
                                     use_checkpoints=True):
        """
        FIXED: Select periods based on giant component size, not reply count
        """
        print("\n" + "="*70)
        print("🕸️  PHASE 2: BUILDING REPLY NETWORKS")
        print("="*70)
        print(f"\nCriterion: Giant Component must have ≥ {min_gcc_size:,} nodes")
        print(f"Selection: {periods_per_year} random periods per year (if available)")
        
        # Analyze all periods first
        print("\n📊 Analyzing network sizes across all periods...")
        
        period_gcc_sizes = {}
        all_periods = sorted(df_comments[temporal_unit].unique())
        
        print(f"\nScanning {len(all_periods)} periods for giant component sizes...")
        print("(This may take a few minutes)\n")
        
        for i, period in enumerate(all_periods, 1):
            period_df = df_comments[df_comments[temporal_unit] == period]
            
            # Build quick undirected network
            undirected_net = self.build_undirected_reply_network(period_df, min_interactions=1)
            
            if undirected_net.number_of_nodes() == 0:
                gcc_size = 0
            else:
                # Get giant component size
                components = list(nx.connected_components(undirected_net))
                gcc_size = len(max(components, key=len)) if components else 0
            
            period_gcc_sizes[period] = gcc_size
            
            # Show progress every 10 periods
            if i % 10 == 0 or i == len(all_periods):
                print(f"   Progress: {i}/{len(all_periods)} periods scanned...")
        
        # Create summary DataFrame
        gcc_stats = pd.DataFrame({
            'period': list(period_gcc_sizes.keys()),
            'gcc_size': list(period_gcc_sizes.values()),
            'year': [p.split('-')[0] if '-' in p else str(p) for p in period_gcc_sizes.keys()]
        })
        
        # Filter periods meeting size criterion
        sufficient_periods = gcc_stats[gcc_stats['gcc_size'] >= min_gcc_size]
        
        print(f"\n✓ Analysis complete!")
        print(f"\n📈 NETWORK SIZE SUMMARY:")
        print(f"   • Periods analyzed: {len(gcc_stats)}")
        print(f"   • Periods with GCC ≥ {min_gcc_size:,} nodes: {len(sufficient_periods)}")
        print(f"   • Largest GCC: {gcc_stats['gcc_size'].max():,} nodes in {gcc_stats.loc[gcc_stats['gcc_size'].idxmax(), 'period']}")
        print(f"   • Average GCC size: {gcc_stats['gcc_size'].mean():,.0f} nodes")
        
        if len(sufficient_periods) == 0:
            print(f"\n✗ No periods meet the {min_gcc_size:,} node criterion!")
            print("\nTop 10 periods by GCC size:")
            top_periods = gcc_stats.nlargest(10, 'gcc_size')
            for _, row in top_periods.iterrows():
                print(f"   • {row['period']}: {row['gcc_size']:,} nodes")
            return {}
        
        # Select periods per year
        print(f"\n🎲 Selecting up to {periods_per_year} periods per year:")
        selected_periods = []
        
        for year in sorted(sufficient_periods['year'].unique()):
            year_periods = sufficient_periods[sufficient_periods['year'] == year].sort_values('gcc_size', ascending=False)
            
            if len(year_periods) > periods_per_year:
                # Take top periods by GCC size, then randomize
                top_candidates = year_periods.head(min(periods_per_year * 2, len(year_periods)))
                selected = top_candidates.sample(n=min(periods_per_year, len(top_candidates)), random_state=42)
            else:
                selected = year_periods
            
            selected_periods.extend(selected['period'].tolist())
            
            avg_gcc = selected['gcc_size'].mean()
            print(f"   • Year {year}: {len(selected)} periods selected (avg GCC: {avg_gcc:,.0f} nodes)")
        
        print(f"\n✓ Total: {len(selected_periods)} periods selected for analysis\n")
        
        # Build networks for selected periods
        print("="*70)
        print("🔨 BUILDING NETWORKS FOR SELECTED PERIODS")
        print("="*70 + "\n")
        
        temporal_networks = {}
        
        for idx, period in enumerate(selected_periods, 1):
            print(f"\n{'─'*70}")
            print(f"📅 Period {idx}/{len(selected_periods)}: {period}")
            print(f"{'─'*70}")
            
            period_filename = f"reply_network_{temporal_unit}_{period}.pkl"
            
            if use_checkpoints:
                period_data = self.load_checkpoint(period_filename)
                if period_data is not None:
                    temporal_networks[period] = period_data
                    gcc_size = period_data.get('network_stats', {}).get('gcc_size', 'unknown')
                    print(f"   ✓ Loaded from checkpoint (GCC: {gcc_size:,} nodes)")
                    continue
            
            period_df = df_comments[df_comments[temporal_unit] == period]
            networks = self._build_reply_networks_for_period(period, period_df)
            
            if networks and networks['user_network'].number_of_nodes() > 0:
                temporal_networks[period] = networks
                if use_checkpoints:
                    self.save_checkpoint(networks, period_filename)
            else:
                print(f"   ✗ Failed to build network")
        
        print(f"\n{'='*70}")
        print(f"✓ Built networks for {len(temporal_networks)} periods")
        print(f"{'='*70}\n")
        
        return temporal_networks

    def _build_reply_networks_for_period(self, period, period_data):
        """Build networks for a period with detailed output"""
        try:
            # Build directed network
            print("   🔨 Building directed reply network...")
            directed_reply_net = self.build_reply_network(period_data, min_interactions=1)
            
            if directed_reply_net.number_of_nodes() == 0:
                print("   ✗ No valid reply network")
                return None
            
            print(f"      • Nodes: {directed_reply_net.number_of_nodes():,}")
            print(f"      • Edges: {directed_reply_net.number_of_edges():,}")
            print(f"      • Density: {directed_reply_net.graph.get('density', 0):.6f}")
            print(f"      • Reciprocity: {directed_reply_net.graph.get('reciprocity', 0):.3f}")
            
            # Build undirected network
            print("\n   🔨 Building undirected reply network...")
            undirected_reply_net = self.build_undirected_reply_network(period_data, min_interactions=1)
            
            if undirected_reply_net.number_of_nodes() == 0:
                print("   ✗ No valid undirected network")
                return None
            
            # Get giant component
            components = list(nx.connected_components(undirected_reply_net))
            gcc = max(components, key=len)
            gcc_subgraph = undirected_reply_net.subgraph(gcc).copy()
            
            print(f"      • Nodes: {undirected_reply_net.number_of_nodes():,}")
            print(f"      • Edges: {undirected_reply_net.number_of_edges():,}")
            print(f"      • Components: {len(components)}")
            print(f"      • Giant Component: {len(gcc):,} nodes ({len(gcc)/undirected_reply_net.number_of_nodes()*100:.1f}%)")
            print(f"      • Avg Clustering: {undirected_reply_net.graph.get('avg_clustering', 0):.4f}")
            print(f"      • Avg Degree: {undirected_reply_net.graph.get('avg_degree', 0):.2f}")
            
            # Build sentiment network
            sentiment_net = build_sentiment_network(period_data)
            
            networks = {
                'directed_reply_network': directed_reply_net,
                'user_network': gcc_subgraph,  # Use GCC for community detection
                'full_network': undirected_reply_net,  # Keep full network
                'sentiment_network': sentiment_net,
                'user_sentiments': period_data.groupby('user')['sentiment'].mean().to_dict(),
                'activity_levels': period_data.groupby('user').size().to_dict(),
                'reply_stats': period_data[period_data['reply_to'].notna()].groupby('user').size().to_dict(),
                'network_stats': {
                    'n_users_full': undirected_reply_net.number_of_nodes(),
                    'n_edges_full': undirected_reply_net.number_of_edges(),
                    'gcc_size': len(gcc),
                    'n_components': len(components),
                    'density': undirected_reply_net.graph.get('density', 0),
                    'avg_degree': undirected_reply_net.graph.get('avg_degree', 0),
                    'avg_clustering': undirected_reply_net.graph.get('avg_clustering', 0),
                    'reciprocity': directed_reply_net.graph.get('reciprocity', 0)
                }
            }
            
            print(f"\n   ✓ Network construction complete")
            return networks
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return None

    def parallel_community_detection(self, temporal_networks, use_checkpoints=True):
        """Community detection with better output"""
        print("\n" + "="*70)
        print("🏘️  PHASE 3: COMMUNITY DETECTION")
        print("="*70 + "\n")
        
        community_results = {}
        
        for idx, (period, networks) in enumerate(temporal_networks.items(), 1):
            print(f"\n{'─'*70}")
            print(f"📅 Period {idx}/{len(temporal_networks)}: {period}")
            print(f"{'─'*70}")
            
            comm_filename = f"communities_{period}.pkl"
            
            if use_checkpoints:
                period_result = self.load_checkpoint(comm_filename)
                if period_result is not None:
                    community_results[period] = period_result
                    n_comms = len(set([c[0] for c in period_result['communities'].values() if c]))
                    print(f"   ✓ Loaded from checkpoint: {n_comms} communities")
                    continue
            
            user_network = networks['user_network']
            
            if user_network.number_of_nodes() < 10:
                print(f"   ✗ Network too small ({user_network.number_of_nodes()} nodes)")
                continue
            
            print(f"   🔍 Detecting communities (network: {user_network.number_of_nodes():,} nodes)...")
            communities = detect_communities_worker((period, user_network))
            
            if communities[1]:
                # Count unique communities
                all_comms = [c[0] for c in communities[1].values() if c]
                n_unique_comms = len(set(all_comms))
                
                period_result = {
                    'communities': communities[1],
                    'network': user_network
                }
                community_results[period] = period_result
                
                if use_checkpoints:
                    self.save_checkpoint(period_result, comm_filename)
                
                print(f"   ✓ Found {n_unique_comms} communities")
                print(f"      • Users assigned: {len(communities[1]):,}")
                print(f"      • Avg community size: {len(communities[1])/n_unique_comms:.1f}")
            else:
                print(f"   ✗ No communities detected")
        
        print(f"\n{'='*70}")
        print(f"✓ Community detection complete for {len(community_results)} periods")
        print(f"{'='*70}\n")
        
        return community_results

    def parallel_hub_analysis(self, community_results, use_checkpoints=True):
        """Hub analysis with better output"""
        print("\n" + "="*70)
        print("⭐ PHASE 4: HUB ANALYSIS")
        print("="*70 + "\n")
        
        hub_evolution = {}
        
        for idx, (period, result) in enumerate(community_results.items(), 1):
            print(f"📅 Period {idx}/{len(community_results)}: {period}")
            
            hub_filename = f"hubs_{period}.pkl"
            
            if use_checkpoints:
                period_hubs = self.load_checkpoint(hub_filename)
                if period_hubs is not None:
                    hub_evolution[period] = period_hubs
                    print(f"   ✓ Loaded: {len(period_hubs['hubs']):,} hubs\n")
                    continue
            
            network = result['network']
            hubs_result = analyze_hubs_worker((period, network))
            
            if hubs_result[1]:
                period_hubs = {
                    'hubs': hubs_result[1],
                    'degrees': hubs_result[2]
                }
                hub_evolution[period] = period_hubs
                
                if use_checkpoints:
                    self.save_checkpoint(period_hubs, hub_filename)
                
                print(f"   ✓ Found {len(hubs_result[1]):,} hubs (top 5% by degree)\n")
            else:
                print(f"   ✗ No hubs found\n")
        
        print(f"{'='*70}")
        print(f"✓ Hub analysis complete for {len(hub_evolution)} periods")
        print(f"{'='*70}\n")
        
        return hub_evolution

    def calculate_stability_metrics(self, community_results, hub_evolution):
        """Calculate stability metrics"""
        print("\n" + "="*70)
        print("📊 PHASE 5: STABILITY METRICS")
        print("="*70 + "\n")
        
        try:
            periods = sorted(community_results.keys())
            
            if len(periods) < 2:
                print("⚠️  Need at least 2 periods for stability analysis\n")
                return {}
            
            stability_metrics = {}
            
            print(f"Analyzing {len(periods)-1} period transitions:\n")
            
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
                print(f"   {period1} → {period2}:")
                print(f"      • Hub overlap: {hub_overlap:.3f}")
                print(f"      • Community similarity: {community_similarity:.3f}\n")
            
            return stability_metrics
            
        except Exception as e:
            print(f"\n✗ Stability calculation failed: {e}\n")
            return {}

    def _calculate_community_similarity(self, comm1, comm2):
        """Calculate Jaccard similarity of community assignments"""
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
        """Export results"""
        print("\n" + "="*70)
        print("💾 PHASE 6: EXPORTING RESULTS")
        print("="*70 + "\n")
        
        try:
            # Community data
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
                print("✓ Exported: community_memberships.csv")
            
            # Hub data
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
                print("✓ Exported: hub_analysis.csv")
            
            # Stability metrics
            if stability_metrics:
                stability_df = pd.DataFrame.from_dict(stability_metrics, orient='index')
                stability_df.to_csv('stability_metrics.csv', index=True)
                print("✓ Exported: stability_metrics.csv")
            
            print("\n" + "="*70 + "\n")
            return True
            
        except Exception as e:
            print(f"\n✗ Export failed: {e}\n")
            return False

    def generate_report(self, stability_metrics):
        """Generate summary report"""
        print("\n" + "="*70)
        print("📋 ANALYSIS SUMMARY")
        print("="*70)
        
        if stability_metrics:
            hub_stabilities = [m['hub_overlap'] for m in stability_metrics.values()]
            comm_stabilities = [m['community_similarity'] for m in stability_metrics.values()]
            
            print(f"\n📊 STABILITY METRICS:")
            print(f"   • Average hub stability: {np.mean(hub_stabilities):.3f}")
            print(f"   • Average community stability: {np.mean(comm_stabilities):.3f}")
            print(f"   • Transitions analyzed: {len(stability_metrics)}")
        
        # Memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 ** 3
        print(f"\n💻 RESOURCE USAGE:")
        print(f"   • Memory: {memory_usage:.2f} GB")
        print(f"   • CPU cores: {self.n_workers}")
        
        print("\n" + "="*70 + "\n")


# WORKER FUNCTIONS (must be at module level)

def process_sentiment_chunk(chunk):
    """Process sentiment for a chunk"""
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
    except:
        return nx.Graph()


def detect_communities_worker(network_tuple):
    """
    FIXED: Community detection that actually works
    Uses Louvain as primary method, BigCLAM as fallback
    """
    period, network = network_tuple
    
    try:
        if network.number_of_nodes() < 20:
            return (period, {}, network)
        
        # Method 1: Louvain (most reliable for large networks)
        try:
            import community as community_louvain
            
            # Ensure all edges have weights
            weighted_net = network.copy()
            for u, v, d in weighted_net.edges(data=True):
                if 'weight' not in d:
                    d['weight'] = 1.0
            
            # Run Louvain
            partition = community_louvain.best_partition(weighted_net, weight='weight', random_state=42)
            
            # Convert to expected format
            communities = {}
            for node, comm_id in partition.items():
                communities[node] = [comm_id]  # List format for consistency
            
            # Verify we have real communities
            n_communities = len(set(partition.values()))
            avg_size = len(partition) / n_communities if n_communities > 0 else 0
            
            if n_communities > 1 and n_communities < len(partition) * 0.9:
                # Good communities found
                return (period, communities, network)
            else:
                # Degenerate case - try next method
                pass
                
        except Exception as e:
            pass
        
        # Method 2: BigCLAM (for overlapping communities)
        try:
            # Calculate appropriate dimensions
            n_nodes = network.number_of_nodes()
            dimensions = max(2, min(20, int(np.sqrt(n_nodes) / 10)))
            
            bigclam = BigClam(
                dimensions=dimensions,
                iterations=500,
                random_state=42
            )
            
            adj_matrix = nx.to_scipy_sparse_array(network)
            bigclam.fit(adj_matrix)
            
            memberships = bigclam.get_memberships()
            
            communities = {}
            node_list = list(network.nodes())
            for node_idx, comm_affiliations in enumerate(memberships):
                if comm_affiliations and len(comm_affiliations) > 0:
                    node_name = node_list[node_idx]
                    communities[node_name] = list(comm_affiliations)
            
            # Verify communities
            if communities:
                all_comms = set()
                for affs in communities.values():
                    all_comms.update(affs)
                
                n_communities = len(all_comms)
                
                if n_communities > 1 and n_communities < len(communities) * 0.9:
                    return (period, communities, network)
                    
        except Exception as e:
            pass
        
        # Method 3: Label Propagation (fast and simple)
        try:
            import networkx.algorithms.community as nx_comm
            
            label_prop_comms = list(nx_comm.label_propagation_communities(network))
            
            communities = {}
            for comm_id, comm_nodes in enumerate(label_prop_comms):
                for node in comm_nodes:
                    communities[node] = [comm_id]
            
            n_communities = len(label_prop_comms)
            
            if n_communities > 1 and n_communities < len(communities) * 0.9:
                return (period, communities, network)
                
        except Exception as e:
            pass
        
        # Method 4: Greedy modularity (fallback)
        try:
            import networkx.algorithms.community as nx_comm
            
            greedy_comms = list(nx_comm.greedy_modularity_communities(network))
            
            communities = {}
            for comm_id, comm_nodes in enumerate(greedy_comms):
                for node in comm_nodes:
                    communities[node] = [comm_id]
            
            n_communities = len(greedy_comms)
            
            if n_communities > 1:
                return (period, communities, network)
                
        except Exception as e:
            pass
        
        # All methods failed
        return (period, {}, network)
        
    except Exception as e:
        return (period, {}, network)


def analyze_hubs_worker(network_tuple):
    """Hub analysis worker"""
    period, network = network_tuple
    
    try:
        degrees = dict(network.degree())
        
        if not degrees:
            return (period, {}, {})
        
        # Top 5% by degree
        degree_threshold = np.percentile(list(degrees.values()), 95)
        hubs = {node: deg for node, deg in degrees.items() if deg >= degree_threshold}
        
        return (period, hubs, degrees)
        
    except Exception as e:
        return (period, {}, {})


# MAIN EXECUTION FUNCTION
def run_analysis(data_path, checkpoint_dir="checkpoints", n_workers=None, 
                use_checkpoints=True, min_gcc_size=15000, periods_per_year=6):
    """
    Main analysis function
    
    Args:
        data_path: Path to Reddit data directory
        checkpoint_dir: Directory for checkpoints
        n_workers: Number of parallel workers (None = auto)
        use_checkpoints: Whether to use checkpoints
        min_gcc_size: Minimum giant component size (default: 15,000 nodes)
        periods_per_year: Number of periods to select per year (default: 6)
    
    Returns:
        analyzer, community_results, stability_metrics
    """
    print("\n" + "="*70)
    print("🎯 REDDIT REPLY NETWORK ANALYSIS")
    print("="*70)
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   • Minimum GCC size: {min_gcc_size:,} nodes")
    print(f"   • Periods per year: {periods_per_year}")
    print(f"   • Workers: {n_workers or 'auto'}")
    print(f"   • Checkpoints: {'enabled' if use_checkpoints else 'disabled'}")
    print("="*70)
    
    analyzer = RedditReplyAnalyzer(checkpoint_dir=checkpoint_dir, n_workers=n_workers)
    
    try:
        # Phase 1: Load Data
        df_posts, df_comments = analyzer.parallel_data_loading(data_path, use_checkpoints)
        
        if df_comments is None:
            print("\n✗ Failed to load data\n")
            return analyzer, None, None

        # Phase 2: Build Networks
        temporal_networks = analyzer.parallel_network_construction(
            df_comments, 
            temporal_unit='month', 
            min_gcc_size=min_gcc_size,
            periods_per_year=periods_per_year,
            use_checkpoints=use_checkpoints
        )
        
        if not temporal_networks:
            print("\n✗ Failed to build networks\n")
            return analyzer, None, None
        
        # Phase 3: Community Detection
        community_results = analyzer.parallel_community_detection(temporal_networks, use_checkpoints)
        
        if not community_results:
            print("\n⚠️  No communities detected\n")
            return analyzer, None, None
        
        # Phase 4: Hub Analysis
        hub_evolution = analyzer.parallel_hub_analysis(community_results, use_checkpoints)
        
        # Phase 5: Stability Metrics
        stability_metrics = analyzer.calculate_stability_metrics(community_results, hub_evolution)
        
        # Phase 6: Export Results
        analyzer.export_results(community_results, hub_evolution, stability_metrics)
        
        # Generate Report
        analyzer.generate_report(stability_metrics)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📁 Results available in:")
        print("   • community_memberships.csv")
        print("   • hub_analysis.csv")
        print("   • stability_metrics.csv")
        print("   • checkpoints/ (for recovery)")
        print("\n" + "="*70 + "\n")
        
        return analyzer, community_results, stability_metrics
        
    except Exception as e:
        print(f"\n\n{'='*70}")
        print(f"✗ ANALYSIS FAILED")
        print(f"{'='*70}")
        print(f"\nError: {e}\n")
        traceback.print_exc()
        print("\n" + "="*70 + "\n")
        return analyzer, None, None


# For script/notebook execution
if __name__ == "__main__":
    import sys
    
    # Set multiprocessing start method
    if sys.platform.startswith('win') or 'ipykernel' in sys.modules:
        try:
            mp.set_start_method('spawn', force=True)
        except:
            pass
    
    # Example usage
    analyzer, community_results, stability_metrics = run_analysis(
        data_path="../Documentaries.corpus",
        checkpoint_dir="checkpoints",
        min_gcc_size=15000,  # Require 15k nodes in giant component
        periods_per_year=6,  # Select 6 periods per year
        use_checkpoints=True
    )