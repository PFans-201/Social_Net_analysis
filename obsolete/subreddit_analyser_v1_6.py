"""
Reddit Reply Network Analyzer
Builds networks with overlapping communities and better visualization.

This module contains the RedditReplyAnalyzer class and helper worker functions
for loading Reddit conversation data, preprocessing, constructing reply networks,
detecting communities (including overlapping communities with BigClam),
analyzing hubs, computing stability metrics, and exporting results.

Usage: import and call run_analysis(...) or use the RedditReplyAnalyzer class
directly in a Jupyter notebook for interactive exploration.
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
import csv
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
import logging

# Configure logging: write everything to a file only. Do NOT add a StreamHandler,
# so logger.* calls do NOT appear on console/notebook output. Keep print(...) for user feedback.
LOG_FILE = 'subreddit_analyser.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Clear any inherited handlers (e.g., from other modules or basicConfig)
logger.handlers = []
file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.propagate = False

warnings.filterwarnings('ignore')

# Set up matplotlib
plt.style.use('default')
sns.set_palette("husl")


class RedditReplyAnalyzer:
    """
    Core class for loading Reddit conversation data, building reply networks,
    detecting communities, analyzing hubs, computing stability metrics, and
    exporting results.

    Attributes:
        sentiment_analyzer: VADER sentiment analyzer instance.
        checkpoint_dir: Directory path where checkpoints are saved.
        n_workers: Number of parallel workers to use for multiprocessing tasks.
    """

    def __init__(self, checkpoint_dir="checkpoints", n_workers=None):
        """
        Initialize the analyzer.

        Args:
            checkpoint_dir (str): Path to directory used for checkpoints.
            n_workers (int or None): Number of parallel workers to use. If None,
                                     automatically set to cpu_count() - 1.
            global_comment_map (pd.Series or None): Optional mapping from comment IDs to user IDs. If provided, replies
                                     to comments outside the current period will be correctly attributed.
        """
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.checkpoint_dir = checkpoint_dir
        # Default to all but one CPU to leave system responsive
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.global_comment_map = None  # Will be set in parallel_data_loading

        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Friendly console initialization message
        print(f"\n{'='*70}")
        print(f"🚀 Initialized analyzer with {self.n_workers} parallel workers")
        print(f"{'='*70}\n")

    def save_checkpoint(self, data, filename):
        """
        Save a Python object to a pickle checkpoint file.

        Args:
            data: Any picklable Python object to save.
            filename (str): Filename to use for the checkpoint (saved inside checkpoint_dir).

        Returns:
            bool: True if saved successfully, False otherwise.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"   ✓ Checkpoint saved: {filename}")
            return True
        except Exception as e:
            # Avoid raising here; callers may continue without checkpoint
            print(f"   ✗ Failed to save {filename}: {e}")
            return False

    def load_checkpoint(self, filename, max_age_hours=48):
        """
        Load a checkpoint file if it exists and is not older than max_age_hours.

        Args:
            filename (str): Filename (inside checkpoint_dir) to load.
            max_age_hours (int): Maximum allowed age of checkpoint in hours.

        Returns:
            object or None: The unpickled object or None if loading failed or file is old.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                file_age = (datetime.now() - file_time).total_seconds() / 3600

                # Respect checkpoint freshness to avoid stale results
                if file_age > max_age_hours:
                    return None

                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"   ✓ Loaded checkpoint: {filename}")
                return data
            except Exception as e:
                # Log but do not crash; caller can fallback to recomputation
                logger.warning(f"Failed to load checkpoint {filename}: {e}")
                return None
        return None

    def parallel_data_loading(self, data_path, use_checkpoints=True):
        """
        Load posts and comments, preprocess and run sentiment analysis in parallel.

        This method attempts to reuse existing checkpoints to speed up load times.

        Args:
            data_path (str): Directory containing 'conversations.json' and comment files.
            use_checkpoints (bool): Whether to attempt loading/saving preprocessed checkpoints.

        Returns:
            tuple(pd.DataFrame, pd.DataFrame) or (None, None): posts and comments DataFrames.
        """
        print("\n" + "="*70)
        print("📚 PHASE 1: DATA LOADING")
        print("="*70)

        # Try to reuse cached preprocessed data
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
            # Load posts - handle JSON structure with post IDs as keys
            print("Loading posts...")
            posts_file = f"{data_path}/conversations.json"
            if not os.path.exists(posts_file):
                raise FileNotFoundError(f"Posts file not found: {posts_file}")

            with open(posts_file, 'r', encoding='utf-8') as f:
                posts_data = json.load(f)

            # Convert posts JSON dict to DataFrame; ensure id column exists
            df_posts = pd.DataFrame.from_dict(posts_data, orient='index')
            df_posts['id'] = df_posts.index
            df_posts = df_posts.reset_index(drop=True)
            print(f"   ✓ Loaded {len(df_posts):,} posts")
            logger.info(f"Posts loaded: {len(df_posts)} rows")
            print(f"  Post columns: {list(df_posts.columns)}")

            # Load comments using a memory- and encoding-resilient helper
            print("\n💬 Loading comments...")
            df_comments = self._load_comments_smart(data_path)

            if df_comments is None or len(df_comments) == 0:
                raise ValueError("Failed to load comments")

            # Create global comment->user mapping for cross-period reply resolution
            try:
                self.global_comment_map = df_comments.set_index('id')['user']
                logger.debug(f"Created global comment->user map with {len(self.global_comment_map):,} entries")
            except Exception as e:
                self.global_comment_map = None
                logger.warning(f"Could not create global_comment_map: {e}")

            print(f"   ✓ Loaded {len(df_comments):,} comments")

            # Run sentiment preprocessing in parallel
            print("\n🔍 Running sentiment analysis (might take a few minutes)")
            df_comments = self._parallel_preprocessing(df_comments)

            # Save a combined checkpoint for faster future runs
            if use_checkpoints:
                self.save_checkpoint((df_posts, df_comments), "preprocessed_data.pkl")

            gc.collect()
            print("\n✓ Data loading complete\n")
            return df_posts, df_comments

        except Exception as e:
            # Provide friendly error messages and include full traceback for debugging
            print(f"\n✗ Data loading failed: {e}\n")
            traceback.print_exc()
            return None, None

    def _load_comments_smart(self, data_path):
        """
        Load comments from CSV or JSONL using chunked reading and robust text handling.

        Prefers CSV (faster) if present; otherwise reads JSONL and will save a CSV copy
        for future speedups.

        Args:
            data_path (str): Directory containing 'utterances.csv' or 'utterances.jsonl'.

        Returns:
            pd.DataFrame or None: Loaded comments dataframe or None on failure.
        """
        csv_path = f"{data_path}/utterances.csv"
        jsonl_path = f"{data_path}/utterances.jsonl"

        # If CSV exists, read in chunks to bound memory usage
        if os.path.exists(csv_path):
            print("Using CSV file...")
            try:
                chunks = []
                for chunk in pd.read_csv(csv_path,
                                         low_memory=False,
                                         chunksize=100000,
                                         encoding='utf-8',
                                         escapechar='\\',
                                         quoting=csv.QUOTE_ALL):
                    # Ensure text column is string and missing values are handled
                    if 'text' in chunk.columns:
                        chunk['text'] = chunk['text'].fillna('')
                        chunk['text'] = chunk['text'].astype(str)
                    chunks.append(chunk)

                df = pd.concat(chunks, ignore_index=True)

                # Log text coverage
                non_empty_texts = (df['text'] != '').sum()
                logger.info(f"Loaded {len(df):,} comments from CSV, {non_empty_texts:,} with non-empty text")
                print(f"✓ Loaded {len(df):,} comments from CSV")

                # Debug sample of texts
                sample_size = min(5, len(df))
                logger.debug("Text sample validation:")
                for _, row in df.sample(n=sample_size, random_state=42).iterrows():
                    logger.debug(f"Sample text ({len(row['text'])} chars): {row['text'][:100]}...")

                return df

            except Exception as e:
                logger.error(f"CSV loading failed: {e}")
                print(f"CSV loading failed: {e}")

        # Fallback to JSONL chunked loading
        if os.path.exists(jsonl_path):
            print("Reading JSONL in chunks...")
            try:
                df = self._load_jsonl_chunked(jsonl_path)
                if df is not None:
                    print(f"✓ Loaded {len(df):,} comments from JSONL")
                    return df
            except Exception as e:
                logger.error(f"JSONL loading failed: {e}")
                print(f"JSONL loading failed: {e}")
                return None

        # Neither file found
        raise FileNotFoundError(f"No comments file found in {data_path}")

    def _load_jsonl_chunked(self, file_path, chunk_size=100000):
        """
        Read JSONL line-by-line into pandas DataFrame in chunks with cleaning.

        This avoids memory spikes and handles malformed lines gracefully.

        Args:
            file_path (str): Path to the JSONL file.
            chunk_size (int): Number of JSON objects per chunk to accumulate.

        Returns:
            pd.DataFrame or None: Concatenated dataframe or None on failure.
        """
        logger.debug(f"Starting JSONL loading from {file_path}")
        chunks = []
        current_chunk = []
        line_count = 0

        print(f"Reading {file_path}...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())

                        # Clean and normalize 'text' field proactively
                        if 'text' in data:
                            # Replace deleted markers with empty text
                            if data['text'] == '[deleted]':
                                data['text'] = ''
                            else:
                                # Remove null bytes and normalize newlines
                                text = data['text'].replace('\x00', '')
                                text = text.replace('\r\n', '\n').replace('\r', '\n')
                                # Remove lines that are empty to avoid huge redundant gaps
                                text = '\n'.join(ln for ln in text.splitlines() if ln.strip())
                                data['text'] = text

                        current_chunk.append(data)
                        line_count += 1

                        # Flush chunk to DataFrame to keep memory bounded
                        if len(current_chunk) >= chunk_size:
                            df_chunk = pd.DataFrame(current_chunk)
                            if 'text' in df_chunk.columns:
                                df_chunk['text'] = df_chunk['text'].astype(str)
                            chunks.append(df_chunk)
                            current_chunk = []
                            if line_count % 500000 == 0:
                                logger.debug(f"Processed {line_count:,} lines")
                                print(f"      Processed {line_count:,} lines...")

                    except json.JSONDecodeError as e:
                        # Skip malformed JSON lines but continue processing
                        logger.warning(f"Skipping malformed JSON at line {line_count}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line {line_count}: {e}")
                        continue

                # Handle any remaining records
                if current_chunk:
                    df_chunk = pd.DataFrame(current_chunk)
                    if 'text' in df_chunk.columns:
                        df_chunk['text'] = df_chunk['text'].astype(str)
                    chunks.append(df_chunk)

            if chunks:
                df = pd.concat(chunks, ignore_index=True)

                # Final cleaning: ensure text column is str and remove problematic characters
                if 'text' in df.columns:
                    df['text'] = df['text'].fillna('')
                    df['text'] = df['text'].astype(str)
                    # Remove non-ascii characters to avoid plotting/serialization issues downstream
                    df['text'] = df['text'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii') if x else '')

                    # Log number of empty texts for diagnostics
                    empty_texts = (df['text'] == '').sum()
                    logger.info(f"Text validation: {empty_texts} empty texts out of {len(df)} total")

                # Save CSV copy to speed up future loads
                csv_path = file_path.replace('.jsonl', '.csv')
                if not os.path.exists(csv_path):
                    print("Creating CSV for faster future loading...")
                    df.to_csv(csv_path, index=False, encoding='utf-8', escapechar='\\', quoting=csv.QUOTE_ALL)
                    logger.info(f"Saved processed data to {csv_path}")

                return df

            return None

        except Exception as e:
            logger.error(f"Failed to load JSONL file: {e}")
            return None

    def _parallel_preprocessing(self, df_comments):
        """
        Run sentiment analysis in parallel over comment DataFrame partitions and
        add basic temporal features.

        Args:
            df_comments (pd.DataFrame): Raw comments DataFrame with a 'text' column.

        Returns:
            pd.DataFrame: DataFrame enriched with 'sentiment', 'datetime', 'week', 'month', 'year'.
        """
        # Determine number of chunks based on dataset size and available workers
        n_chunks = min(self.n_workers, len(df_comments) // 10000 + 1)
        chunks = np.array_split(df_comments, n_chunks)

        print(f"   Processing {len(chunks)} chunks in parallel...")

        # Use multiprocessing Pool to compute sentiment in parallel
        with Pool(self.n_workers) as pool:
            processed_chunks = pool.map(process_sentiment_chunk, chunks)

        # Filter out any None chunks due to failures
        processed_chunks = [c for c in processed_chunks if c is not None]

        # Concatenate processed partitions
        df_comments = pd.concat(processed_chunks, ignore_index=True)

        # Add temporal features for downstream analysis
        df_comments['datetime'] = pd.to_datetime(df_comments['timestamp'], unit='s', errors='coerce')
        df_comments = df_comments[df_comments['datetime'].notna()]  # drop rows with invalid timestamps
        df_comments['week'] = df_comments['datetime'].dt.to_period('W').astype(str)
        df_comments['month'] = df_comments['datetime'].dt.strftime('%Y-%m')
        df_comments['year'] = df_comments['datetime'].dt.year

        return df_comments

    def build_reply_network(self, df_comments, min_interactions=1, global_comment_map=None, verbose=True):
        """
        Build a directed reply network from comments.
        Each edge represents replies from one user to another, weighted by the number
        of replies and annotated with sentiment statistics.

        Args:
            df_comments (pd.DataFrame): Comment DataFrame for a single period.
            min_interactions (int): Minimum number of replies between user pairs to
                                    include an edge in the network.
            global_comment_map (pd.Series or None): Optional mapping from comment IDs
                                    to user IDs. If provided, replies comments outside the current
                                    period will be correctly attributed.
        Returns:
            nx.DiGraph: Directed graph representing the reply network.
        """
        logger.info("Building reply network...")

        try:
            # Validate that required columns are present
            required_cols = ['id', 'user', 'reply_to', 'text']
            missing_cols = [col for col in required_cols if col not in df_comments.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Filter out deleted or missing users and ensure reply_to exists
            df_filtered = df_comments[
                (df_comments['user'] != '[deleted]') &
                (df_comments['user'].notna()) &
                (df_comments['reply_to'].notna())
            ].copy()

            if verbose:
                print(f"Processing {len(df_filtered):,} comments with replies...")
                logger.debug(f"Processing {len(df_filtered):,} comments with replies...")

            # Map comment IDs to users to identify target users for replies.
            # Use provided global_comment_map (from full dataset) if available so
            # replies that target comments outside the current period are resolved.
            if global_comment_map is not None:
                comment_to_user = global_comment_map
            else:
                comment_to_user = df_comments.set_index('id')['user']
            df_filtered['reply_to_user'] = df_filtered['reply_to'].map(comment_to_user)

            # Compute sentiment per reply (use local analyzer to be multiprocessing-safe)
            logger.debug("Calculating sentiment scores...")
            df_filtered['sentiment'] = df_filtered['text'].apply(
                lambda x: analyzer.polarity_scores(str(x))['compound']
            )

            # Keep only valid user-to-user replies, remove self-replies and deleted targets
            valid_replies = df_filtered[
                (df_filtered['reply_to_user'].notna()) &
                (df_filtered['reply_to_user'] != '[deleted]') &
                (df_filtered['user'] != df_filtered['reply_to_user'])
            ]

            logger.info(f"Valid user-to-user replies: {len(valid_replies):,}")
            print(f"Valid user-to-user replies: {len(valid_replies):,}")

            if len(valid_replies) == 0:
                logger.warning("No valid reply interactions found")
                print("No valid reply interactions found")
                return nx.DiGraph()

            logger.debug("Building network structure...")
            G = nx.DiGraph()

            # Aggregate interactions per user pair and collect sentiment values
            interaction_data = defaultdict(lambda: {'count': 0, 'sentiments': []})

            for _, row in valid_replies.iterrows():
                pair = (row['user'], row['reply_to_user'])
                interaction_data[pair]['count'] += 1
                interaction_data[pair]['sentiments'].append(row['sentiment'])

            # Add edges for pairs meeting the minimum interaction threshold
            for (source, target), data in interaction_data.items():
                if data['count'] >= min_interactions:
                    G.add_edge(
                        source,
                        target,
                        weight=data['count'],
                        avg_sentiment=np.mean(data['sentiments']),
                        min_sentiment=min(data['sentiments']),
                        max_sentiment=max(data['sentiments']),
                        sentiment_std=np.std(data['sentiments']) if len(data['sentiments']) > 1 else 0
                    )

            # Ensure all users appearing in interactions are present as nodes
            all_users = set()
            for source, target in interaction_data.keys():
                all_users.add(source)
                all_users.add(target)
            G.add_nodes_from(all_users)

            if G.number_of_nodes() > 0:
                # Compute simple network-level metrics
                density = nx.density(G)
                avg_in_degree = sum(dict(G.in_degree()).values()) / G.number_of_nodes()
                avg_out_degree = sum(dict(G.out_degree()).values()) / G.number_of_nodes()

                if verbose:
                    print(f"Network density: {density:.6f}")
                    print(f"Average in-degree: {avg_in_degree:.2f}")
                    print(f"Average out-degree: {avg_out_degree:.2f}")

                # Reciprocity can be expensive; compute in a try/except
                try:
                    reciprocity = nx.reciprocity(G)
                    if verbose:
                        print(f"Reciprocity: {reciprocity:.3f}")
                except Exception as e:
                    reciprocity = 0
                    logger.warning(f"Could not calculate reciprocity: {e}")
                    if verbose:
                        print("Could not calculate reciprocity")

                # Store metadata on the graph for downstream export
                G.graph['density'] = density
                G.graph['avg_in_degree'] = avg_in_degree
                G.graph['avg_out_degree'] = avg_out_degree
                G.graph['reciprocity'] = reciprocity
                G.graph['network_type'] = 'reply'
                G.graph['construction_time'] = datetime.now().isoformat()

            logger.info(f"Reply network built successfully: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            return G

        except Exception as e:
            logger.error(f"Failed to build reply network: {str(e)}", exc_info=True)
            if verbose:
                print(f"      ✗ Failed to build reply network: {e}")
            return nx.DiGraph()

    def build_undirected_reply_network(self, df_comments, min_interactions=1, global_comment_map=None, verbose=True):
        """
        Convert the directed reply network into an undirected network by symmetrizing edges.

        This is useful for clustering/community detection algorithms that expect undirected graphs.

        Args:
            df_comments (pd.DataFrame): Comment DataFrame for a single period.
            min_interactions (int): Minimum interactions threshold used when building directed network.
            global_comment_map (pd.Series or None): Optional mapping from comment IDs to user IDs. If provided, replies
                                                    to comments outside the current period will be correctly attributed.        Returns:

        Returns:
            nx.Graph: Undirected network (may be empty).
        """
        print("\nBuilding undirected reply network...")
        directed_net = self.build_reply_network(df_comments, min_interactions, global_comment_map=global_comment_map)
        if directed_net.number_of_nodes() == 0:
            return nx.Graph()

        # Build undirected graph (keep existing logic)
        undirected_net = nx.Graph()
        undirected_net.add_nodes_from(directed_net.nodes())

        # Add directed edges as undirected edges (will accumulate weights)
        for u, v, data in directed_net.edges(data=True):
            weight = data.get('weight', 1)
            if undirected_net.has_edge(u, v):
                undirected_net[u][v]['weight'] += weight
            else:
                undirected_net.add_edge(u, v, weight=weight)

        # Consider reverse direction explicitly to ensure bidirectional weight aggregation
        for u, v, data in directed_net.edges(data=True):
            if directed_net.has_edge(v, u):
                reverse_weight = directed_net[v][u].get('weight', 1)
                if undirected_net.has_edge(u, v):
                    undirected_net[u][v]['weight'] += reverse_weight
                else:
                    undirected_net.add_edge(u, v, weight=reverse_weight)

        if verbose:
            print(f"\nUndirected reply network: {undirected_net.number_of_nodes():,} nodes, {undirected_net.number_of_edges():,} edges")

            # Compute summary statistics and attach to graph attributes
            if undirected_net.number_of_nodes() > 0:
                density = nx.density(undirected_net)
                avg_degree = sum(dict(undirected_net.degree()).values()) / undirected_net.number_of_nodes()

                if undirected_net.number_of_edges() > 0:
                    try:
                        avg_clustering = nx.average_clustering(undirected_net)
                        print(f"Average clustering: {avg_clustering:.4f}")
                    except Exception as e:
                        logger.warning(f"Could not calculate average clustering: {e}")
                        avg_clustering = 0
                else:
                    avg_clustering = 0

                print(f"Undirected density: {density:.6f}")
                print(f"Average degree: {avg_degree:.2f}")

                # Copy directed metadata and augment
                undirected_net.graph.update(directed_net.graph)
                undirected_net.graph['avg_degree'] = avg_degree
                undirected_net.graph['avg_clustering'] = avg_clustering
                undirected_net.graph['network_type'] = 'undirected_reply'
                undirected_net.graph['construction_time'] = datetime.now().isoformat()

        return undirected_net

    def exploratory_data_analysis(self, df_posts, df_comments):
        """
        Perform exploratory data analysis and produce diagnostic plots.

        This includes basic statistics, reply analysis, temporal plots, user activity,
        post engagement, sentiment analysis, and network preparation metrics.

        Args:
            df_posts (pd.DataFrame): Posts DataFrame.
            df_comments (pd.DataFrame): Comments DataFrame.

        Returns:
            tuple(pd.DataFrame, pd.DataFrame): Possibly cleaned/augmented posts and comments DataFrames.
        """
        print("\n" + "="*70)
        print("📊 PHASE 1.5: EXPLORATORY DATA ANALYSIS")
        print("="*70)

        # Create output directory for EDA plots
        os.makedirs('eda_plots', exist_ok=True)

        # Basic statistics summary
        print("\n📊 BASIC STATISTICS:")
        print(f"  Posts: {len(df_posts):,}")
        print(f"  Comments: {len(df_comments):,}")
        print(f"  Unique users: {df_comments['user'].nunique():,}")
        print(f"  Unique posts with comments: {df_comments['root'].nunique():,}")

        # Data quality diagnostics
        print("\n🔍 DATA QUALITY:")
        print(f"  Comments with '[deleted]' users: {(df_comments['user'] == '[deleted]').sum():,}")
        print(f"  Comments with missing text: {df_comments['text'].isna().sum():,}")
        print(f"  Comments with empty text: {(df_comments['text'] == '').sum():,}")

        # Filter out deleted users and empty texts for the EDA analysis
        original_count = len(df_comments)
        df_comments = df_comments[df_comments['user'] != '[deleted]']
        df_comments = df_comments[df_comments['text'].notna()]
        df_comments = df_comments[df_comments['text'] != '']

        print(f"  Comments after cleaning: {len(df_comments):,} ({len(df_comments)/original_count*100:.1f}% retained)")

        # Reply analysis to understand pairwise interactions
        print("\n🔄 REPLY ANALYSIS:")
        df_comments_with_replies = df_comments[df_comments['reply_to'].notna()]
        print(f"  Comments that are replies: {len(df_comments_with_replies):,} ({len(df_comments_with_replies)/len(df_comments)*100:.1f}%)")

        # Build mapping from comment id to user to analyze reply chains
        if len(df_comments_with_replies) > 0:
            comment_to_user = df_comments.set_index('id')['user']
            df_comments_with_replies = df_comments_with_replies.copy()
            df_comments_with_replies['reply_to_user'] = df_comments_with_replies['reply_to'].map(comment_to_user)

            valid_replies = df_comments_with_replies[df_comments_with_replies['reply_to_user'].notna()]
            valid_replies = valid_replies[valid_replies['reply_to_user'] != '[deleted]']
            valid_replies = valid_replies[valid_replies['user'] != valid_replies['reply_to_user']]  # Remove self-replies

            print(f"  Valid user-to-user replies: {len(valid_replies):,}")
            print(f"  Unique reply pairs: {valid_replies[['user', 'reply_to_user']].drop_duplicates().shape[0]:,}")

            # Reply pattern statistics
            reply_counts = valid_replies.groupby(['user', 'reply_to_user']).size().reset_index(name='count')
            print(f"  Average replies per pair: {reply_counts['count'].mean():.2f}")
            print(f"  Max replies between a pair: {reply_counts['count'].max():,}")

        # Temporal analysis and plots
        print("\n📅 TEMPORAL ANALYSIS:")
        df_comments['datetime'] = pd.to_datetime(df_comments['timestamp'], unit='s', errors='coerce')
        df_posts['datetime'] = pd.to_datetime(df_posts['timestamp'], unit='s', errors='coerce')

        # Discard invalid dates to avoid plotting errors
        df_comments = df_comments[df_comments['datetime'].notna()]
        df_posts = df_posts[df_posts['datetime'].notna()]

        time_range_comments = df_comments['datetime'].min(), df_comments['datetime'].max()
        time_range_posts = df_posts['datetime'].min(), df_posts['datetime'].max()

        print(f"  Comment date range: {time_range_comments[0].strftime('%Y-%m-%d')} to {time_range_comments[1].strftime('%Y-%m-%d')}")
        print(f"  Post date range: {time_range_posts[0].strftime('%Y-%m-%d')} to {time_range_posts[1].strftime('%Y-%m-%d')}")

        # Create and save temporal plots
        self._create_temporal_plots(df_posts, df_comments)

        # User and post activity diagnostics and visualizations
        self._analyze_user_activity(df_comments)
        self._analyze_post_engagement(df_posts, df_comments)

        # Explore reply network patterns and visualize associated diagnostics
        self._analyze_reply_network(df_comments)

        # Sentiment diagnostics (if sentiment column exists)
        if 'sentiment' in df_comments.columns:
            self._analyze_sentiment(df_comments)

        # Metrics to inspect suitability for network construction
        self._network_prep_analysis(df_comments)

        print("\n✅ EDA complete! Check 'eda_plots/' directory for visualizations")
        return df_posts, df_comments

    def _create_temporal_plots(self, df_posts, df_comments):
        """
        Create a set of temporal distribution plots with log-scaled frequency axes.

        Saves 'eda_plots/temporal_analysis.png' and shows the figure.

        Args:
            df_posts (pd.DataFrame): Posts with 'datetime'.
            df_comments (pd.DataFrame): Comments with 'datetime' and 'reply_to'.
        """
        def _set_log10(ax, axis='y'):
            # Compatibility wrapper for different matplotlib versions
            try:
                if axis == 'y':
                    ax.set_yscale('log', base=10)
                else:
                    ax.set_xscale('log', base=10)
            except TypeError:
                if axis == 'y':
                    ax.set_yscale('log', basey=10)
                else:
                    ax.set_xscale('log', basex=10)

        _, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Monthly comment counts (log10)
        df_comments['year_month'] = df_comments['datetime'].dt.to_period('M')
        monthly_comments = df_comments.groupby('year_month').size()
        monthly_comments.plot(ax=axes[0, 0], linewidth=2)
        _set_log10(axes[0, 0], 'y')
        axes[0, 0].set_title('Monthly Comment Volume')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Overlay replies
        monthly_replies = df_comments[df_comments['reply_to'].notna()].groupby('year_month').size()
        monthly_replies.plot(ax=axes[0, 0], linewidth=2, color='red', alpha=0.7, label='Replies')
        axes[0, 0].legend()
        axes[0, 0].set_ylabel('Number of Comments (log10)')

        # Hourly activity (log10)
        df_comments['hour'] = df_comments['datetime'].dt.hour
        hourly_comments = df_comments.groupby('hour').size()
        hourly_comments.plot(ax=axes[0, 1], kind='bar', color='skyblue')
        _set_log10(axes[0, 1], 'y')
        axes[0, 1].set_title('Comments by Hour of Day')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Frequency (log10)')
        axes[0, 1].tick_params(axis='x', rotation=0)

        # Monthly posts with reply ratio on secondary axis
        df_posts['year_month'] = df_posts['datetime'].dt.to_period('M')
        monthly_posts = df_posts.groupby('year_month').size()
        monthly_posts.plot(ax=axes[1, 0], color='orange', linewidth=2)
        _set_log10(axes[1, 0], 'y')
        axes[1, 0].set_title('Monthly Post Volume')
        axes[1, 0].set_ylabel('Number of Posts (log10)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        monthly_total = df_comments.groupby('year_month').size()
        monthly_reply_ratio = (monthly_replies / monthly_total).fillna(0)
        ax2 = axes[1, 0].twinx()
        monthly_reply_ratio.plot(ax=ax2, color='green', linewidth=1, linestyle='--', alpha=0.8, label='Reply Ratio')
        ax2.set_ylabel('Reply Ratio (monthly replies / total comments)')
        ax2.set_xlabel('Months')
        axes[1, 0].legend(['Posts'], loc='lower left')
        ax2.legend(loc='lower right')

        # Comments per post distribution
        comments_per_post = df_comments.groupby('root').size()
        axes[1, 1].hist(comments_per_post, bins=50, alpha=0.7, color='green', edgecolor='black')
        _set_log10(axes[1, 1], 'y')
        axes[1, 1].set_title('Distribution of Comments per Post')
        axes[1, 1].set_xlabel('Comments per Post')
        axes[1, 1].set_ylabel('Frequency (log10)')

        plt.tight_layout()
        plt.savefig('eda_plots/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _analyze_user_activity(self, df_comments):
        """
        Analyze and visualize user activity patterns.

        Produces 'eda_plots/user_activity.png'.

        Args:
            df_comments (pd.DataFrame): Comments DataFrame with 'user' and 'reply_to'.
        """
        user_activity = df_comments['user'].value_counts()
        user_replies = df_comments[df_comments['reply_to'].notna()]['user'].value_counts()

        print("\n👥 USER ACTIVITY ANALYSIS:")
        if len(user_activity) > 0:
            print(f"  Most active user: {user_activity.index[0]} ({user_activity.iloc[0]:,} comments)")
        else:
            print("  Most active user: N/A")
        if len(user_replies) > 0:
            print(f"  Most replying user: {user_replies.index[0]} ({user_replies.iloc[0]:,} replies)")
        else:
            print("  Most replying user: N/A (0 replies)")
        print(f"  Average comments per user: {user_activity.mean():.1f}")
        print(f"  Users who sent replies: {len(user_replies):,} ({(len(user_replies)/len(user_activity)*100) if len(user_activity) else 0:.1f}% of active users)")

        # Top users bar and distribution histogram with log10 y-scale
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        user_activity.head(10).plot(kind='bar', color='coral')
        ax = plt.gca()
        try:
            ax.set_yscale('log', base=10)
        except TypeError:
            ax.set_yscale('log', basey=10)
        plt.title('Top 10 Most Active Users')
        plt.xlabel('Users')
        plt.ylabel('Number of Comments (log10)')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        ax2 = plt.gca()
        ax2.hist(user_activity, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        plt.title('User Activity Distribution')
        plt.xlabel('Comments per User')
        plt.ylabel('Frequency (log10)')

        plt.tight_layout()
        plt.savefig('eda_plots/user_activity.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _analyze_post_engagement(self, df_posts, df_comments):
        """
        Analyze post engagement: comments per post statistics and visualizations.

        Args:
            df_posts (pd.DataFrame): Posts DataFrame.
            df_comments (pd.DataFrame): Comments DataFrame with 'root' for post id mapping.
        """
        comments_per_post = df_comments.groupby('root').size()

        print("\n📈 POST ENGAGEMENT ANALYSIS:")
        print(f"  Average comments per post: {comments_per_post.mean():.1f}")
        print(f"  Median comments per post: {comments_per_post.median():.1f}")
        print(f"  Posts with no comments: {len(df_posts) - len(comments_per_post):,}")
        if len(comments_per_post) > 0:
            print(f"  Most commented post: {comments_per_post.idxmax()} ({comments_per_post.max():,} comments)")
        else:
            print("  Most commented post: N/A")

        # Engagement distribution and top posts with log10 frequency axis
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        ax = plt.gca()
        ax.hist(comments_per_post, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        try:
            ax.set_yscale('log', base=10)
        except TypeError:
            ax.set_yscale('log', basey=10)
        plt.title('Comments per Post Distribution')
        plt.xlabel('Comments per Post')
        plt.ylabel('Frequency (log10)')

        plt.subplot(1, 2, 2)
        ax2 = plt.gca()
        comments_per_post.sort_values(ascending=False).head(20).plot(kind='bar', color='orange')
        try:
            ax2.set_yscale('log', base=10)
        except TypeError:
            ax2.set_yscale('log', basey=10)
        plt.title('Top 20 Most Commented Posts')
        plt.xlabel('Post ID')
        plt.ylabel('Number of Comments (log10)')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('eda_plots/post_engagement.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _analyze_sentiment(self, df_comments):
        """
        Produce summary statistics and a distribution plot for comment sentiment scores.

        Args:
            df_comments (pd.DataFrame): DataFrame containing a 'sentiment' column.
        """
        logger.info("Performing sentiment analysis...")

        mean_sentiment = df_comments['sentiment'].mean()
        std_sentiment = df_comments['sentiment'].std()
        pos_comments = (df_comments['sentiment'] > 0.5).sum()

        logger.debug(f"Mean sentiment: {mean_sentiment:.3f}")
        logger.debug(f"Sentiment std: {std_sentiment:.3f}")
        logger.info(f"Found {pos_comments:,} positive comments")

        print("\n😊 SENTIMENT ANALYSIS:")
        print(f"  Average sentiment: {mean_sentiment:.3f}")
        print(f"  Sentiment std: {std_sentiment:.3f}")
        print(f"  Positive comments (>0.5): {pos_comments:,}")
        print(f"  Negative comments (<-0.5): {(df_comments['sentiment'] < -0.5).sum():,}")
        print(f"  Neutral comments (-0.5 to 0.5): {((df_comments['sentiment'] >= -0.5) & (df_comments['sentiment'] <= 0.5)).sum():,}")

        # Sentiment histogram (frequency log 10 scale)
        plt.figure(figsize=(10, 6))
        plt.hist(df_comments['sentiment'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        try:
            plt.yscale('log', base=10)
        except TypeError:
            plt.yscale('log', basey=10)
        plt.title('Distribution of Comment Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency (log10)')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        plt.legend()
        plt.savefig('eda_plots/sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _network_prep_analysis(self, df_comments):
        """
        Provide metrics useful for estimating network sizes and potential bipartite edges.

        Args:
            df_comments (pd.DataFrame): Cleaned comments DataFrame with 'user' and 'root' columns.
        """
        print("\n🔗 NETWORK PREPARATION ANALYSIS:")

        # Compute user-post and post-user incidence statistics
        user_post_counts = df_comments.groupby('user')['root'].nunique()
        post_user_counts = df_comments.groupby('root')['user'].nunique()

        print(f"  Average posts per user: {user_post_counts.mean():.2f}")
        print(f"  Average users per post: {post_user_counts.mean():.2f}")
        print(f"  Users commenting on multiple posts: {(user_post_counts > 1).sum():,}")
        print(f"  Posts with multiple users: {(post_user_counts > 1).sum():,}")

        # Estimate active sets for bipartite edges
        active_users = user_post_counts[user_post_counts >= 2].index
        active_posts = post_user_counts[post_user_counts >= 2].index

        print(f"  Estimated active users (≥2 posts): {len(active_users):,}")
        print(f"  Estimated active posts (≥2 users): {len(active_posts):,}")
        print(f"  Estimated bipartite edges: {len(df_comments[df_comments['user'].isin(active_users) & df_comments['root'].isin(active_posts)]):,}")

    def _analyze_reply_network(self, df_comments):
        """
        Produce diagnostics about potential reply network structure and produce plots.

        Args:
            df_comments (pd.DataFrame): Comments DataFrame with 'id', 'user', 'reply_to'.
        """
        print("\n🔗 REPLY NETWORK ANALYSIS:")

        # Use global map if available, otherwise create period-specific map
        comment_to_user = getattr(self, 'global_comment_map', None) or df_comments.set_index('id')['user']

        # Filter replies and map to target users
        df_replies = df_comments[df_comments['reply_to'].notna()].copy()
        df_replies['reply_to_user'] = df_replies['reply_to'].map(comment_to_user)

        # Keep only valid user-to-user replies
        valid_replies = df_replies[
            (df_replies['reply_to_user'].notna()) &
            (df_replies['reply_to_user'] != '[deleted]') &
            (df_replies['user'] != df_replies['reply_to_user'])  # Remove self-replies
        ]

        if len(valid_replies) == 0:
            print("  No valid reply data found")
            return

        # Summarize reply pairs and degree info
        reply_pairs = valid_replies.groupby(['user', 'reply_to_user']).size().reset_index(name='weight')

        print(f"  Total reply interactions: {len(valid_replies):,}")
        print(f"  Unique user pairs with replies: {len(reply_pairs):,}")
        print(f"  Average replies per pair: {reply_pairs['weight'].mean():.2f}")
        print(f"  Most active reply pair: {reply_pairs.loc[reply_pairs['weight'].idxmax()].to_dict()}")

        # Out/in degree proxy using aggregated weights
        out_degree = reply_pairs.groupby('user')['weight'].sum()
        in_degree = reply_pairs.groupby('reply_to_user')['weight'].sum()

        print(f"  Users who sent replies: {len(out_degree):,}")
        print(f"  Users who received replies: {len(in_degree):,}")
        print(f"  Average replies sent per user: {out_degree.mean():.2f}")
        print(f"  Average replies received per user: {in_degree.mean():.2f}")

        # Visualize reply network diagnostics
        self._create_reply_network_plots(reply_pairs, out_degree, in_degree)

    def _create_reply_network_plots(self, reply_pairs, out_degree, in_degree):
        """
        Create a set of plots summarizing reply pair weights and degree distributions.

        Args:
            reply_pairs (pd.DataFrame): DataFrame with columns ['user', 'reply_to_user', 'weight'].
            out_degree (pd.Series): Aggregated replies sent per user.
            in_degree (pd.Series): Aggregated replies received per user.
        """
        _, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Distribution of pair weights (log frequency)
        weights = reply_pairs['weight'].values
        axes[0, 0].hist(weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black', log=True)
        axes[0, 0].set_title('Distribution of Reply Interactions per Pair')
        axes[0, 0].set_xlabel('Number of Replies between Pair')
        axes[0, 0].set_ylabel('Frequency (log)')

        # Top users by replies sent
        out_degree_sorted = out_degree.sort_values(ascending=False).head(20)
        axes[0, 1].bar(range(len(out_degree_sorted)), out_degree_sorted.values, color='coral')
        axes[0, 1].set_title('Top 20 Users by Replies Sent (k_out)')
        axes[0, 1].set_xlabel('User Rank')
        axes[0, 1].set_ylabel('Replies Sent')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Top users by replies received
        in_degree_sorted = in_degree.sort_values(ascending=False).head(20)
        axes[1, 0].bar(range(len(in_degree_sorted)), in_degree_sorted.values, color='lightgreen')
        axes[1, 0].set_title('Top 20 Users by Replies Received')
        axes[1, 0].set_xlabel('User Rank')
        axes[1, 0].set_ylabel('Replies Received')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Reciprocity pie chart: how many pairs are bidirectional
        if len(reply_pairs) > 0:
            pair_set = set(zip(reply_pairs['user'], reply_pairs['reply_to_user']))
            reverse_pair_set = {(b, a) for (a, b) in pair_set}

            bidirectional_pairs = pair_set & reverse_pair_set

            reciprocity = len(bidirectional_pairs) / len(pair_set) if len(pair_set) > 0 else 0

            labels = ['One-way', 'Bidirectional']
            sizes = [len(pair_set) - len(bidirectional_pairs), len(bidirectional_pairs)]
            colors = ['lightcoral', 'gold']

            axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title(f'Reply Reciprocity\n({reciprocity*100:.1f}% bidirectional)')

        plt.tight_layout()
        plt.savefig('eda_plots/reply_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def parallel_network_construction(self, df_comments, temporal_unit='month', 
                                      min_gcc_size=15000, periods_per_year=6,
                                      use_checkpoints=True):
        """
        Construct temporal reply networks for periods that meet a minimum giant component size.

        Steps:
          - Scan all temporal periods to estimate giant component (GCC) sizes for each.
          - Select top periods per year according to provided rules.
          - Build reply networks for selected periods in sequence and optionally checkpoint.

        Args:
            df_comments (pd.DataFrame): Comments DataFrame with a temporal_unit column (e.g., 'month').
            temporal_unit (str): Column name in df_comments to use for period partitioning.
            min_gcc_size (int): Minimum GCC size to consider a period sufficiently large.
            periods_per_year (int): Max periods to select per year.
            use_checkpoints (bool): Whether to reuse or save period build checkpoints.

        Returns:
            dict: Mapping period -> networks dict containing at least 'user_network' and 'network_stats'.
        """
        try:
            logger.info("Starting parallel network construction...")
            
            print("\n📊 Analyzing network sizes across all periods...")
            period_gcc_sizes = {}
            all_periods = sorted(df_comments[temporal_unit].unique())

            print(f"\nScanning {len(all_periods)} periods in parallel (quiet mode)...")

            # Prepare args for worker: (period, period_df, global_comment_map)
            global_map = getattr(self, 'global_comment_map', None)
            period_args = [(period, df_comments[df_comments[temporal_unit] == period], global_map) for period in all_periods]

            # Process periods in parallel using top-level worker to avoid pickling local functions
            with Pool(self.n_workers) as pool:
                total = len(period_args)
                for i, (period, gcc_size) in enumerate(pool.imap_unordered(estimate_gcc_size_worker, period_args), 1):
                    period_gcc_sizes[period] = gcc_size
                    # print lightweight progress updates (overwrite line)
                    if i % max(1, min(10, total//10)) == 0 or i == total:
                        print(f"   Progress: {i}/{total} periods analyzed...", end='\r')
                print()  # newline after progress
            
            print("\n✓ Size analysis complete!")

            # Summarize distribution of GCC sizes
            gcc_stats = pd.DataFrame({
                'period': list(period_gcc_sizes.keys()),
                'gcc_size': list(period_gcc_sizes.values()),
                'year': [p.split('-')[0] if '-' in p else str(p) for p in period_gcc_sizes.keys()]
            })

            # Filter to periods meeting the size criterion
            sufficient_periods = gcc_stats[gcc_stats['gcc_size'] >= min_gcc_size]

            print("\n✓ Analysis complete!")
            print("\n📈 NETWORK SIZE SUMMARY:")
            print(f"   • Periods analyzed: {len(gcc_stats)}")
            print(f"   • Periods with GCC ≥ {min_gcc_size:,} nodes: {len(sufficient_periods)}")
            if len(gcc_stats) > 0:
                print(f"   • Largest GCC: {gcc_stats['gcc_size'].max():,} nodes in {gcc_stats.loc[gcc_stats['gcc_size'].idxmax(), 'period']}")
                print(f"   • Average GCC size: {gcc_stats['gcc_size'].mean():,.0f} nodes")

            if len(sufficient_periods) == 0:
                print(f"\n✗ No periods meet the {min_gcc_size:,} node criterion!")
                print("\nTop 10 periods by GCC size:")
                top_periods = gcc_stats.nlargest(10, 'gcc_size')
                for _, row in top_periods.iterrows():
                    print(f"   • {row['period']}: {row['gcc_size']:,} nodes")
                return {}

            # Select up to periods_per_year per calendar year using a mix of top and random sampling
            print(f"\n🎲 Selecting up to {periods_per_year} periods per year:")
            selected_periods = []

            for year in sorted(sufficient_periods['year'].unique()):
                year_periods = sufficient_periods[sufficient_periods['year'] == year].sort_values('gcc_size', ascending=False)

                if len(year_periods) > periods_per_year:
                    top_candidates = year_periods.head(min(periods_per_year * 2, len(year_periods)))
                    selected = top_candidates.sample(n=min(periods_per_year, len(top_candidates)), random_state=42)
                else:
                    selected = year_periods

                selected_periods.extend(selected['period'].tolist())

                avg_gcc = selected['gcc_size'].mean()
                print(f"   • Year {year}: {len(selected)} periods selected (avg GCC: {avg_gcc:,.0f} nodes)")

            print(f"\n✓ Total: {len(selected_periods)} periods selected for analysis\n")

            # Build networks for each selected period and optionally checkpoint results
            print("="*70)
            print("🔨 BUILDING NETWORKS FOR SELECTED PERIODS")
            print("="*70 + "\n")

            temporal_networks = {}

            # Prepare args for pool workers
            global_comment_map = getattr(self, 'global_comment_map', None)
            period_build_args = []
            for period in selected_periods:
                # allow checkpoint loading to skip work
                period_filename = f"reply_network_{temporal_unit}_{period}.pkl"
                if use_checkpoints:
                    period_data = self.load_checkpoint(period_filename)
                    if period_data is not None:
                        temporal_networks[period] = period_data
                        continue

                period_df = df_comments[df_comments[temporal_unit] == period]
                period_build_args.append((period, period_df, global_comment_map, 1))

            # Run builds in parallel using the top-level worker
            if period_build_args:
                with Pool(self.n_workers) as pool:
                    total = len(period_build_args)
                    for i, (period, networks) in enumerate(pool.imap_unordered(build_period_network_worker, period_build_args), 1):
                        if networks and networks.get('user_network') and networks['user_network'].number_of_nodes() > 0:
                            temporal_networks[period] = networks
                            if use_checkpoints:
                                try:
                                    period_filename = f"reply_network_{temporal_unit}_{period}.pkl"
                                    self.save_checkpoint(networks, period_filename)
                                except Exception as e:
                                    logger.warning(f"   ⚠️ Failed to save checkpoint for {period}: {e}")
                        # lightweight progress
                        if i % max(1, total//10) == 0 or i == total:
                            print(f"   Built networks for {i}/{total} selected periods...", end='\r')
                print()  # newline after progress
            else:
                print("   No periods required building (all loaded from checkpoints)")

            # Compute simple network-level metrics
            network_metrics = []
            for period, networks in temporal_networks.items():
                stats = networks.get('network_stats', {})
                stats['period'] = period
                network_metrics.append(stats)

            if network_metrics:
                pd.DataFrame(network_metrics).to_csv('network_metrics.csv', index=False)
                print("✓ Exported network_metrics.csv")

            return temporal_networks

        except Exception as e:
            logger.error("Network construction failed", exc_info=True)
            print(f"✗ Network construction failed: {e}")
            return {}

    # Helper to build multiple networks for a period (kept as a method for clarity)
    def _build_reply_networks_for_period(self, period, period_df, global_comment_map=None):
        """
        Build directed reply network and undirected user network for a given period.
        Args:
            period (str): Temporal period identifier.
            period_df (pd.DataFrame): Comments DataFrame for the period.    
            global_comment_map (pd.Series): Optional mapping from comment id to user for cross-period

        Returns:
            dict: Contains:
                'directed_reply_network'- directed reply network (nx.DiGraph)
                'user_network'- undirected user network (nx.Graph)
                'network_stats'- statistics about the networks
        """
        try:
            # Build directed reply network using default min_interactions=1.
            # If a global_comment_map is passed, use it so replies targeting comments
            # outside the current period can be resolved to their authors.
            directed_net = self.build_reply_network(
                period_df, min_interactions=1, global_comment_map=global_comment_map
            )

            # Convert to undirected user network for community detection
            user_network = nx.Graph()
            if directed_net.number_of_edges() > 0:
                # Add undirected edges by summing directed weights
                for u, v, d in directed_net.edges(data=True):
                    w = d.get('weight', 1)
                    if user_network.has_edge(u, v):
                        user_network[u][v]['weight'] += w
                    else:
                        user_network.add_edge(u, v, weight=w)
            else:
                # If no directed edges, create an empty graph of nodes
                user_network.add_nodes_from(directed_net.nodes())

            # Compute GCC size for the undirected network
            if user_network.number_of_nodes() == 0:
                gcc_size = 0
            else:
                components = list(nx.connected_components(user_network))
                gcc_size = len(max(components, key=len)) if components else 0

            network_stats = {
                'gcc_size': gcc_size,
                'nodes': user_network.number_of_nodes(),
                'edges': user_network.number_of_edges()
            }

            return {
                'directed_reply_network': directed_net,
                'user_network': user_network,
                'network_stats': network_stats
            }

        except Exception as e:
            logger.error(f"Failed to build networks for period {period}: {e}", exc_info=True)
            return {}

    def detect_overlapping_communities(self, network, method='bigclam'):
        """
        Detect overlapping communities in a user network using BigClam or fallback.

        Args:
            network (nx.Graph or nx.DiGraph): Undirected or directed user-level network.
            method (str): 'bigclam' to prefer BigClam, otherwise fallback to Louvain.

        Returns:
            dict: Mapping user -> list of community ids (possibly multiple).
        """
        print(f"   🔍 Detecting overlapping communities with {method}...")

        # Skip very small networks
        if network.number_of_nodes() < 20:
            return {}

        try:
            if method == 'bigclam':
                # Choose a reasonable embedding dimensionality based on network size
                n_nodes = network.number_of_nodes()
                dimensions = max(2, min(20, int(np.sqrt(n_nodes) / 5)))

                bigclam = BigClam(
                    dimensions=dimensions,
                    iterations=1000,  # More iterations for better convergence
                    random_state=42
                )

                # Convert to sparse adjacency matrix (karateclub expects scipy sparse)
                adj_matrix = nx.to_scipy_sparse_array(network)
                bigclam.fit(adj_matrix)

                memberships = bigclam.get_memberships()

                communities = {}
                node_list = list(network.nodes())

                for node_idx, comm_affiliations in enumerate(memberships):
                    if comm_affiliations and len(comm_affiliations) > 0:
                        node_name = node_list[node_idx]
                        communities[node_name] = list(comm_affiliations)

                # Report simple overlap statistics
                overlapping_users = sum(1 for affs in communities.values() if len(affs) > 1)
                total_users = len(communities)

                print(f"      • Users in communities: {total_users:,}")
                print(f"      • Users in multiple communities: {overlapping_users:,}")
                print(f"      • Overlap ratio: {overlapping_users/total_users*100:.1f}%")

                return communities

            else:
                # Fallback to non-overlapping Louvain partitioning
                return self._detect_non_overlapping_communities(network)

        except Exception as e:
            # If BigClam fails for any reason, fallback gracefully
            print(f"      ✗ BigClam failed: {e}")
            return self._detect_non_overlapping_communities(network)

    def _detect_non_overlapping_communities(self, network):
        """
        Detect non-overlapping communities using Louvain (community-louvain package).

        Args:
            network (nx.Graph): Undirected network.

        Returns:
            dict: Mapping user -> [community_id]
        """
        try:
            import community as community_louvain

            weighted_net = network.copy()
            # Ensure each edge has a weight for Louvain
            for u, v, d in weighted_net.edges(data=True):
                if 'weight' not in d:
                    d['weight'] = 1.0

            partition = community_louvain.best_partition(weighted_net, weight='weight', random_state=42)

            communities = {}
            for node, comm_id in partition.items():
                communities[node] = [comm_id]  # single community per user

            print(f"      • Using Louvain (non-overlapping): {len(set(partition.values()))} communities")
            return communities

        except Exception as e:
            print(f"      ✗ Louvain also failed: {e}")
            return {}

    def parallel_community_detection(self, temporal_networks, use_checkpoints=True, overlapping=True):
        """
        Detect communities for each temporal network.

        Supports overlapping detection via BigClam or non-overlapping via Louvain.

        Args:
            temporal_networks (dict): Mapping period -> networks dict (from parallel_network_construction).
            use_checkpoints (bool): Whether to attempt to load/save per-period community checkpoints.
            overlapping (bool): If True, use overlapping detection (BigClam), otherwise Louvain.

        Returns:
            dict: Mapping period -> result dict {'communities', 'network', 'overlapping'}.
        """
        print("\n" + "="*70)
        print("🏘️  PHASE 3: COMMUNITY DETECTION")
        print("="*70)
        print(f"\nMode: {'OVERLAPPING communities (BigClam)' if overlapping else 'Non-overlapping communities (Louvain)'}")

        community_results = {}

        # Prepare inputs for parallel community detection, honoring checkpoints and small networks
        detect_args = []
        for period, networks in temporal_networks.items():
            comm_filename = f"communities_{period}.pkl"
            if use_checkpoints:
                period_result = self.load_checkpoint(comm_filename)
                if period_result is not None:
                    community_results[period] = period_result
                    continue
            user_network = networks.get('user_network', None)
            if user_network is None or user_network.number_of_nodes() < 10:
                # skip tiny networks
                continue
            detect_args.append((period, user_network))

        if detect_args:
            with Pool(self.n_workers) as pool:
                total = len(detect_args)
                for i, (period, communities, network) in enumerate(pool.imap_unordered(detect_communities_worker, detect_args), 1):
                    if communities:
                        period_result = {'communities': communities, 'network': network, 'overlapping': overlapping}
                        community_results[period] = period_result
                        if use_checkpoints:
                            self.save_checkpoint(period_result, f"communities_{period}.pkl")
                    if i % max(1, total//10) == 0 or i == total:
                        print(f"   Community detection progress: {i}/{total}...", end='\r')
            print()
        else:
            print("   No networks eligible for community detection (checkpoints or size filters applied)")

        print("\n{'='*70}")
        print(f"✓ Community detection complete for {len(community_results)} periods")
        print("{'='*70}\n")

        return community_results

    def visualize_network_period(self, period, community_results, hub_evolution,
                                 max_nodes=500, layout='spring', show_labels=True):
        """
        Visualize a network for a specific period with community coloring and hub highlighting.

        Args:
            period (str): Period identifier present in community_results dict.
            community_results (dict): Mapping period -> result produced by parallel_community_detection.
            hub_evolution (dict): Mapping period -> hub diagnostics produced by parallel_hub_analysis.
            max_nodes (int): Maximum nodes to display (sampling applied if exceeded).
            layout (str): One of 'spring', 'kamada_kawai', 'circular' for layout algorithm.
            show_labels (bool): Whether to draw node labels for hubs.

        Returns:
            tuple: (network, communities, hubs) used for the visualization.
        """
        if period not in community_results:
            print(f"Period {period} not found in results")
            return

        result = community_results[period]
        network = result['network']
        communities = result['communities']
        hubs = hub_evolution.get(period, {}).get('hubs', {})

        if network.number_of_nodes() == 0:
            print("Network is empty")
            return

        print(f"\n📊 Visualizing network for {period}:")
        print(f"   • Nodes: {network.number_of_nodes():,}")
        print(f"   • Edges: {network.number_of_edges():,}")
        print(f"   • Communities: {len(set([c for comm_list in communities.values() for c in comm_list])):,}")
        print(f"   • Hubs: {len(hubs):,}")

        # Sampling logic if graph too large for plotting
        if network.number_of_nodes() > max_nodes:
            print(f"   • Sampling {max_nodes} nodes for visualization")
            # Keep hubs and sample others uniformly at random
            hub_nodes = list(hubs.keys())[:max_nodes//2]
            other_nodes = [n for n in network.nodes() if n not in hub_nodes]
            if len(other_nodes) > max_nodes - len(hub_nodes):
                rng = np.random.default_rng(seed=42)
                other_nodes = list(rng.choice(other_nodes, size=max_nodes - len(hub_nodes), replace=False))
            sample_nodes = hub_nodes + list(other_nodes)
            network = network.subgraph(sample_nodes)
            # Restrict communities and hubs to sampled nodes
            communities = {node: communities[node] for node in sample_nodes if node in communities}
            hubs = {node: hubs[node] for node in sample_nodes if node in hubs}

        # Prepare figure and layout
        plt.figure(figsize=(15, 10))

        if layout == 'spring':
            pos = nx.spring_layout(network, k=1/np.sqrt(network.number_of_nodes()), iterations=50)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(network)
        else:
            pos = nx.circular_layout(network)

        # Color nodes by their primary community (first affiliation)
        if communities:
            all_community_ids = set()
            for comm_list in communities.values():
                all_community_ids.update(comm_list)

            community_colors = {comm_id: i for i, comm_id in enumerate(all_community_ids)}
            cmap = plt.cm.get_cmap('tab20', len(all_community_ids))

            node_colors = []
            for node in network.nodes():
                if node in communities and communities[node]:
                    primary_comm = communities[node][0]
                    # Map community id to color index between 0 and 1 for colormap
                    node_colors.append(cmap(community_colors[primary_comm] / max(1, len(all_community_ids))))
                else:
                    node_colors.append('lightgray')
        else:
            node_colors = ['lightblue'] * network.number_of_nodes()

        # Size nodes by degree and emphasize hubs
        node_sizes = []
        node_degrees = dict(network.degree())
        for node in network.nodes():
            if node in hubs:
                node_sizes.append(300)  # clearly larger for hubs
            else:
                node_sizes.append(50 + node_degrees.get(node, 1) * 5)

        # Draw nodes and edges
        nx.draw_networkx_nodes(network, pos,
                               node_color=node_colors,
                               node_size=node_sizes,
                               alpha=0.8)

        nx.draw_networkx_edges(network, pos, alpha=0.2, edge_color='gray')

        # Draw labels for hubs only (to avoid clutter)
        if show_labels and hubs:
            hub_labels = {node: node for node in network.nodes() if node in hubs}
            nx.draw_networkx_labels(network, pos, labels=hub_labels,
                                    font_size=8, font_weight='bold')

        plt.title(f"Network Visualization: {period}\n"
                  f"Nodes: {network.number_of_nodes()}, Edges: {network.number_of_edges()}\n"
                  f"Communities: {len(set([c for comm_list in communities.values() for c in comm_list]))}, Hubs: {len(hubs)}",
                  size=14)
        plt.axis('off')

        # Add a legend for communities if number of communities is small
        if communities and len(all_community_ids) <= 20:
            legend_elements = []
            for i, comm_id in enumerate(list(all_community_ids)[:10]):  # show up to first 10
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=cmap(i/max(1, len(all_community_ids))),
                                                  markersize=10, label=f'Community {comm_id}'))
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1))

        plt.tight_layout()
        plt.savefig(f'network_visualization_{period}.png', dpi=300, bbox_inches='tight')
        plt.show()

        return network, communities, hubs

    def parallel_hub_analysis(self, community_results, use_checkpoints=True):
        """
        Analyze hubs (high-degree nodes) per period and optionally reuse checkpoints.

        Args:
            community_results (dict): Output of parallel_community_detection.
            use_checkpoints (bool): Whether to load/save hub analysis checkpoints.

        Returns:
            dict: Mapping period -> {'hubs': {user:deg}, 'degrees': {user:deg}}
        """
        print("\n" + "="*70)
        print("⭐ PHASE 4: HUB ANALYSIS")
        print("="*70 + "\n")

        hub_evolution = {}

        # Prepare hub worker args (skip if checkpoint exists)
        hub_args = []
        for period, result in community_results.items():
            hub_filename = f"hubs_{period}.pkl"
            if use_checkpoints:
                period_hubs = self.load_checkpoint(hub_filename)
                if period_hubs is not None:
                    hub_evolution[period] = period_hubs
                    continue
            network = result.get('network', None)
            if network is None or network.number_of_nodes() == 0:
                continue
            hub_args.append((period, network))

        if hub_args:
            with Pool(self.n_workers) as pool:
                total = len(hub_args)
                for i, (period, hubs, degrees) in enumerate(pool.imap_unordered(analyze_hubs_worker, hub_args), 1):
                    if hubs:
                        period_hubs = {'hubs': hubs, 'degrees': degrees}
                        hub_evolution[period] = period_hubs
                        if use_checkpoints:
                            self.save_checkpoint(period_hubs, f"hubs_{period}.pkl")
                    if i % max(1, total//10) == 0 or i == total:
                        print(f"   Hub analysis progress: {i}/{total}...", end='\r')
            print()
        else:
            print("   No hub analyses required (checkpoints or empty networks)")

        print("{'='*70}")
        print(f"✓ Hub analysis complete for {len(hub_evolution)} periods")
        print("{'='*70}\n")

        return hub_evolution

    def calculate_stability_metrics(self, community_results, hub_evolution):
        """
        Calculate stability metrics between consecutive periods.

        Computes:
            - hub overlap Jaccard between hub sets
            - average Jaccard similarity of individual users' community assignments

        Args:
            community_results (dict): Period -> communities
            hub_evolution (dict): Period -> hub dicts

        Returns:
            dict: Mapping 'period1_period2' -> metrics dict
        """
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

                # Compute hub persistence (Jaccard)
                hubs1 = set(hub_evolution.get(period1, {}).get('hubs', {}).keys())
                hubs2 = set(hub_evolution.get(period2, {}).get('hubs', {}).keys())

                hub_overlap = 0
                if hubs1 and hubs2:
                    hub_overlap = len(hubs1 & hubs2) / len(hubs1 | hubs2)

                # Community persistence measured by average per-user Jaccard of affiliation sets
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
        """
        Compute average Jaccard similarity of community affiliations for users present in both periods.

        Args:
            comm1 (dict): user -> [community_ids] for period1.
            comm2 (dict): user -> [community_ids] for period2.

        Returns:
            float: Mean Jaccard similarity across common users (0..1).
        """
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
        """
        Export community, hub, and stability results into CSV files.

        Files produced:
            - community_memberships.csv
            - hub_analysis.csv
            - stability_metrics.csv

        Args:
            community_results (dict): Per-period community information.
            hub_evolution (dict): Per-period hub information.
            stability_metrics (dict): Calculated stability metrics.

        Returns:
            bool: True if export completed successfully, False on failure.
        """
        print("\n" + "="*70)
        print("💾 PHASE 6: EXPORTING RESULTS")
        print("="*70 + "\n")

        try:
            # Export community memberships (flat table: period,user,communities,num_communities)
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

            # Export hub analysis per period
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
        """
        Print a concise textual report of key summary statistics and resource usage.

        Args:
            stability_metrics (dict): Stability metrics computed earlier.
        """
        print("\n" + "="*70)
        print("📋 ANALYSIS SUMMARY")
        print("="*70)

        if stability_metrics:
            hub_stabilities = [m['hub_overlap'] for m in stability_metrics.values()]
            comm_stabilities = [m['community_similarity'] for m in stability_metrics.values()]

            print("\n📊 STABILITY METRICS:")
            print(f"   • Average hub stability: {np.mean(hub_stabilities):.3f}")
            print(f"   • Average community stability: {np.mean(comm_stabilities):.3f}")
            print(f"   • Transitions analyzed: {len(stability_metrics)}")

        # Resource usage snapshot
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 ** 3
        print("\n💻 RESOURCE USAGE:")
        print(f"   • Memory: {memory_usage:.2f} GB")
        print(f"   • CPU cores: {self.n_workers}")

        print("\n" + "="*70 + "\n")

    def plot_stability_metrics(self, stability_metrics, output_file='stability_analysis.png'):
        """
        Plot hub overlap and community similarity over time and perform a permutation test
        to assess significance of correlation between the two metrics.

        Args:
            stability_metrics (dict): Mapping transition_key -> metrics dict.
            output_file (str): File name to save the generated figure.
        """
        if not stability_metrics:
            print("No stability metrics to plot")
            return

        # Convert to DataFrame for plotting
        stability_df = pd.DataFrame.from_dict(stability_metrics, orient='index')
        stability_df['period_pair'] = stability_df.index
        stability_df[['period1', 'period2']] = stability_df['period_pair'].str.split('_', expand=True)

        # Bar chart for both stability metrics
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        x_pos = np.arange(len(stability_df))
        width = 0.35

        ax1.bar(x_pos - width/2, stability_df['hub_overlap'], width,
                label='Hub Overlap', alpha=0.7, color='skyblue')
        ax1.bar(x_pos + width/2, stability_df['community_similarity'], width,
                label='Community Similarity', alpha=0.7, color='lightcoral')

        ax1.set_xlabel('Period Transition')
        ax1.set_ylabel('Stability Score')
        ax1.set_title('Network Stability Metrics Over Time')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stability_df['transition_period'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add average lines
        avg_hub = stability_df['hub_overlap'].mean()
        avg_comm = stability_df['community_similarity'].mean()
        ax1.axhline(y=avg_hub, color='blue', linestyle='--', alpha=0.8, label=f'Avg Hub: {avg_hub:.3f}')
        ax1.axhline(y=avg_comm, color='red', linestyle='--', alpha=0.8, label=f'Avg Comm: {avg_comm:.3f}')
        ax1.legend()

        # Scatter plot and linear fit between the two metrics
        ax2.scatter(stability_df['hub_overlap'], stability_df['community_similarity'],
                    alpha=0.7, s=100, color='purple')

        for i, row in stability_df.iterrows():
            ax2.annotate(row['transition_period'],
                         (row['hub_overlap'], row['community_similarity']),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Compute Pearson correlation coefficient (may be NaN with small samples)
        correlation = np.corrcoef(stability_df['hub_overlap'], stability_df['community_similarity'])[0, 1]

        # Add regression line
        z = np.polyfit(stability_df['hub_overlap'], stability_df['community_similarity'], 1)
        p = np.poly1d(z)
        ax2.plot(stability_df['hub_overlap'], p(stability_df['hub_overlap']), "r--", alpha=0.8)

        ax2.set_xlabel('Hub Overlap Stability')
        ax2.set_ylabel('Community Similarity Stability')
        ax2.set_title(f'Stability Correlation (r = {correlation:.3f})')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

        # Print statistical summary
        print("\n📊 STABILITY METRICS STATISTICAL SUMMARY:")
        print("   Hub Stability:")
        print(f"      • Mean: {stability_df['hub_overlap'].mean():.3f} ± {stability_df['hub_overlap'].std():.3f}")
        print(f"      • Range: [{stability_df['hub_overlap'].min():.3f}, {stability_df['hub_overlap'].max():.3f}]")

        print("   Community Stability:")
        print(f"      • Mean: {stability_df['community_similarity'].mean():.3f} ± {stability_df['community_similarity'].std():.3f}")
        print(f"      • Range: [{stability_df['community_similarity'].min():.3f}, {stability_df['community_similarity'].max():.3f}]")

        print(f"   Correlation between hub and community stability: {correlation:.3f}")

        # Permutation test for correlation significance
        n_permutations = 1000
        random_hub_corrs = []
        rng = np.random.default_rng(seed=42)

        hub_vals = stability_df['hub_overlap'].values
        comm_vals = stability_df['community_similarity'].values

        for _ in range(n_permutations):
            shuffled_hub = rng.permutation(hub_vals)
            corr = np.corrcoef(shuffled_hub, comm_vals)[0, 1]
            random_hub_corrs.append(corr)

        random_hub_corrs = np.array(random_hub_corrs)
        valid_corrs = random_hub_corrs[~np.isnan(random_hub_corrs)]
        if valid_corrs.size == 0 or np.isnan(correlation):
            p_value = 1.0
        else:
            p_value = np.mean(np.abs(valid_corrs) >= np.abs(correlation))
        print(f"   Permutation test p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("   ✅ Significant correlation detected (p < 0.05)")
        else:
            print("   ❌ No significant correlation detected")

    def export_network_metrics(self, temporal_networks, output_file='network_metrics.csv'):
        """
        Export detailed network metrics for each period into a CSV.

        Args:
            temporal_networks (dict): Mapping period -> network dict created earlier.
            output_file (str): Filename for the CSV export.

        Returns:
            pd.DataFrame or None: DataFrame with exported metrics if any, otherwise None.
        """
        network_metrics = []

        for period, networks in temporal_networks.items():
            stats = networks.get('network_stats', {})
            stats['period'] = period

            # Include directed network stats when available
            directed_net = networks.get('directed_reply_network')
            if directed_net and directed_net.number_of_nodes() > 0:
                stats['directed_nodes'] = directed_net.number_of_nodes()
                stats['directed_edges'] = directed_net.number_of_edges()
                stats['reciprocity'] = directed_net.graph.get('reciprocity', 0)

            network_metrics.append(stats)

        if network_metrics:
            df_metrics = pd.DataFrame(network_metrics)
            cols = ['period'] + [col for col in df_metrics.columns if col != 'period']
            df_metrics = df_metrics[cols]
            df_metrics.to_csv(output_file, index=False)
            print(f"✓ Exported detailed network metrics to {output_file}")
            return df_metrics
        else:
            print("✗ No network metrics to export")
            return None

    def compare_period_transition(self, period1, period2, community_results, hub_evolution):
        """
        Compare two periods: user overlap, hub changes, and provide sample lists.

        Args:
            period1 (str): First period identifier.
            period2 (str): Second period identifier.
            community_results (dict): Period -> communities mapping.
            hub_evolution (dict): Period -> hub mapping.

        Returns:
            dict: Containing sets for common_users, users_only_in_1, users_only_in_2,
                  common_hubs, new_hubs, lost_hubs
        """
        if period1 not in community_results or period2 not in community_results:
            logger.warning("One or both periods not found in community_results: %s, %s", period1, period2)
            print("One or both periods not found in results")
            return None

        print(f"\n{'='*70}")
        print(f"🔄 COMPARING PERIODS: {period1} vs {period2}")
        print(f"{'='*70}")

        comm1 = community_results.get(period1, {}).get('communities', {})
        comm2 = community_results.get(period2, {}).get('communities', {})
        hubs1 = hub_evolution.get(period1, {}).get('hubs', {})
        hubs2 = hub_evolution.get(period2, {}).get('hubs', {})

        # User overlap and exclusive sets
        common_users = set(comm1.keys()) & set(comm2.keys())
        users_only_in_1 = set(comm1.keys()) - set(comm2.keys())
        users_only_in_2 = set(comm2.keys()) - set(comm1.keys())

        print("\n👥 USER OVERLAP ANALYSIS:")
        print(f"   • Users in {period1}: {len(comm1):,}")
        print(f"   • Users in {period2}: {len(comm2):,}")
        print(f"   • Common users: {len(common_users):,}")
        print(f"   • Only in {period1}: {len(users_only_in_1):,}")
        print(f"   • Only in {period2}: {len(users_only_in_2):,}")

        # Hub transitions
        common_hubs = set(hubs1.keys()) & set(hubs2.keys())
        new_hubs = set(hubs2.keys()) - set(hubs1.keys())
        lost_hubs = set(hubs1.keys()) - set(hubs2.keys())

        print("\n⭐ HUBS:")
        print(f"   • Hubs in {period1}: {len(hubs1):,}")
        print(f"   • Hubs in {period2}: {len(hubs2):,}")
        print(f"   • Common hubs: {len(common_hubs):,}")
        print(f"   • New hubs in {period2}: {len(new_hubs):,}")
        print(f"   • Lost hubs from {period1}: {len(lost_hubs):,}")

        # Show sample lists to assist quick inspection
        sample_new = list(new_hubs)[:10]
        sample_lost = list(lost_hubs)[:10]
        if sample_new:
            print(f"   • Sample new hubs: {sample_new}")
        if sample_lost:
            print(f"   • Sample lost hubs: {sample_lost}")

        logger.info("Compared periods %s vs %s: users %d/%d common, hubs new=%d lost=%d",
                    period1, period2, len(common_users), len(set(comm1.keys()) | set(comm2.keys())),
                    len(new_hubs), len(lost_hubs))

        return {
            'common_users': common_users,
            'users_only_in_1': users_only_in_1,
            'users_only_in_2': users_only_in_2,
            'common_hubs': common_hubs,
            'new_hubs': new_hubs,
            'lost_hubs': lost_hubs
        }


# Worker functions:
def process_sentiment_chunk(chunk):
    """
    Worker function to compute VADER sentiment for a chunk of comments.

    Designed to be called via multiprocessing.Pool.map. Returns the processed DataFrame
    or None in case of failure, allowing the caller to drop failed partitions.

    Args:
        chunk (pd.DataFrame): Partition of the comments DataFrame.

    Returns:
        pd.DataFrame or None: Processed partition with 'sentiment' column or None on error.
    """
    logger.debug(f"Processing sentiment chunk of size {len(chunk)}")
    try:
        analyzer = SentimentIntensityAnalyzer()
        chunk = chunk.copy()
        chunk['sentiment'] = chunk['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
        logger.debug("Successfully processed chunk")
        return chunk
    except Exception as e:
        logger.error(f"Chunk processing failed: {e}", exc_info=True)
        # Returning None signals the caller to drop or handle this chunk
        return None


def build_sentiment_network(period_data):
    """
    Build a simple sentiment similarity network between users based on average sentiment.

    This is a lightweight network for exploratory analysis and visualization.

    Args:
        period_data (pd.DataFrame): Comments for a single period with 'user' and 'sentiment'.

    Returns:
        nx.Graph: Undirected graph connecting users with similar average sentiment.
    """
    try:
        G = nx.Graph()

        # Compute mean sentiment and count per user, require at least 2 comments per user
        user_stats = period_data.groupby('user')['sentiment'].agg(['mean', 'count'])
        user_stats = user_stats[user_stats['count'] >= 2]

        users = user_stats.index.tolist()[:300]  # limit to 300 users to keep computations tractable
        user_means = user_stats['mean'].values

        # Connect users with high sentiment similarity
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                similarity = 1 - abs(user_means[i] - user_means[j])
                if similarity > 0.5:
                    G.add_edge(users[i], users[j], weight=similarity)

        return G
    except Exception as e:
        logger.warning(f"build_sentiment_network failed: {e}")
        return nx.Graph()


def detect_communities_worker(network_tuple):
    """
    Worker function for community detection used in possible parallel pipelines.

    Tries BigClam for overlapping communities, otherwise falls back to Louvain.

    Args:
        network_tuple (tuple): (period, network) pair.

    Returns:
        tuple: (period, communities_dict, network) where communities_dict maps user -> [community_ids].
    """
    period, network = network_tuple

    try:
        if network.number_of_nodes() < 20:
            return (period, {}, network)

        # Attempt BigClam first (overlapping communities)
        try:
            n_nodes = network.number_of_nodes()
            dimensions = max(2, min(20, int(np.sqrt(n_nodes) / 5)))

            bigclam = BigClam(
                dimensions=dimensions,
                iterations=1000,
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

            # Validate result for reasonableness
            if communities:
                all_comm_ids = set()
                for affs in communities.values():
                    all_comm_ids.update(affs)

                n_communities = len(all_comm_ids)
                overlapping_users = sum(1 for affs in communities.values() if len(affs) > 1)

                if n_communities > 1 and n_communities < len(communities) * 0.8:
                    logger.info(f"BigCLAM successful for {period}: {n_communities} communities, {overlapping_users} overlapping users")
                    return (period, communities, network)

        except Exception as e:
            logger.warning(f"BigCLAM failed for {period}: {e}")

        # Fallback to Louvain if BigClam fails or returns poor results
        try:
            import community as community_louvain

            weighted_net = network.copy()
            for u, v, d in weighted_net.edges(data=True):
                if 'weight' not in d:
                    d['weight'] = 1.0

            partition = community_louvain.best_partition(weighted_net, weight='weight', random_state=42)

            communities = {}
            for node, comm_id in partition.items():
                communities[node] = [comm_id]

            n_communities = len(set(partition.values()))
            logger.info(f"Louvain fallback for {period}: {n_communities} communities")
            return (period, communities, network)

        except Exception as e:
            logger.error(f"All community detection failed for {period}: {e}")
            return (period, {}, network)

    except Exception as e:
        logger.error(f"Community detection crashed for {period}: {e}")
        return (period, {}, network)


def analyze_hubs_worker(network_tuple):
    """
    Worker function to identify hubs (top percentile by degree) for a period.

    Args:
        network_tuple (tuple): (period, network)

    Returns:
        tuple: (period, hubs_dict, degrees_dict)
               hubs_dict maps user -> degree for users above the 95th percentile.
               degrees_dict maps user -> degree for all users.
    """
    period, network = network_tuple

    try:
        degrees = dict(network.degree())

        if not degrees:
            return (period, {}, {})

        # Define hubs as nodes in the top 5% by degree
        degree_threshold = np.percentile(list(degrees.values()), 95)
        hubs = {node: deg for node, deg in degrees.items() if deg >= degree_threshold}

        return (period, hubs, degrees)

    except Exception:
        return (period, {}, {})

def estimate_gcc_size_worker(args):
    """
    Top-level worker for multiprocessing: estimate GCC size for a period.
    Args:
        args (tuple): (period, period_df, global_comment_map)
    Returns:
        tuple: (period, gcc_size)
    """
    period, period_df, global_comment_map = args
    try:
        # Minimal, quiet reconstruction of undirected reply network for GCC estimation
        if period_df is None or len(period_df) == 0:
            return period, 0

        # Map comment ids to users (prefer global map if provided)
        if global_comment_map is not None:
            comment_to_user = global_comment_map
        else:
            try:
                comment_to_user = period_df.set_index('id')['user']
            except Exception:
                comment_to_user = pd.Series(dtype=object)

        df = period_df
        df_filtered = df[
            (df['user'] != '[deleted]') &
            (df['user'].notna()) &
            (df['reply_to'].notna())
        ].copy()
        if df_filtered.empty:
            return period, 0

        df_filtered['reply_to_user'] = df_filtered['reply_to'].map(comment_to_user)
        valid_replies = df_filtered[
            (df_filtered['reply_to_user'].notna()) &
            (df_filtered['reply_to_user'] != '[deleted]') &
            (df_filtered['user'] != df_filtered['reply_to_user'])
        ]
        if valid_replies.empty:
            return period, 0

        # Build undirected edges (aggregate irrespective of direction)
        edge_weights = {}
        for _, r in valid_replies.iterrows():
            a, b = r['user'], r['reply_to_user']
            if pd.isna(a) or pd.isna(b):
                continue
            u, v = (a, b) if a <= b else (b, a)
            edge_weights[(u, v)] = edge_weights.get((u, v), 0) + 1

        G = nx.Graph()
        for (u, v), w in edge_weights.items():
            G.add_edge(u, v, weight=w)

        if G.number_of_nodes() == 0:
            return period, 0

        gcc_size = len(max(nx.connected_components(G), key=len))
        return period, int(gcc_size)
    except Exception as e:
        logger.error(f"estimate_gcc_size_worker failed for {period}: {e}", exc_info=True)
        return period, 0

def build_period_network_worker(args):
    """
    Worker that builds directed reply network and undirected user network for a single period.
    Args:
        args (tuple): (period, period_df, global_comment_map, min_interactions)
    Returns:
        tuple: (period, networks_dict)
    """
    period, period_df, global_comment_map, min_interactions = args
    try:
        if period_df is None or len(period_df) == 0:
            return period, {}

        analyzer = SentimentIntensityAnalyzer()

        # Map comment ids to users (prefer global map if provided)
        if global_comment_map is not None:
            comment_to_user = global_comment_map
        else:
            try:
                comment_to_user = period_df.set_index('id')['user']
            except Exception:
                comment_to_user = pd.Series(dtype=object)

        df = period_df
        df_filtered = df[
            (df['user'] != '[deleted]') &
            (df['user'].notna()) &
            (df['reply_to'].notna())
        ].copy()
        if df_filtered.empty:
            # create empty graphs
            return period, {
                'directed_reply_network': nx.DiGraph(),
                'user_network': nx.Graph(),
                'network_stats': {'gcc_size': 0, 'nodes': 0, 'edges': 0}
            }

        df_filtered['reply_to_user'] = df_filtered['reply_to'].map(comment_to_user)
        valid_replies = df_filtered[
            (df_filtered['reply_to_user'].notna()) &
            (df_filtered['reply_to_user'] != '[deleted]') &
            (df_filtered['user'] != df_filtered['reply_to_user'])
        ].copy()
        if valid_replies.empty:
            return period, {
                'directed_reply_network': nx.DiGraph(),
                'user_network': nx.Graph(),
                'network_stats': {'gcc_size': 0, 'nodes': 0, 'edges': 0}
            }

        # Compute sentiment per reply
        valid_replies['sentiment'] = valid_replies['text'].apply(
            lambda x: analyzer.polarity_scores(str(x))['compound']
        )

        # Build directed graph with aggregated stats
        G = nx.DiGraph()
        interaction_data = defaultdict(lambda: {'count': 0, 'sentiments': []})
        for _, row in valid_replies.iterrows():
            pair = (row['user'], row['reply_to_user'])
            interaction_data[pair]['count'] += 1
            interaction_data[pair]['sentiments'].append(row['sentiment'])

        for (source, target), data in interaction_data.items():
            if data['count'] >= min_interactions:
                G.add_edge(
                    source,
                    target,
                    weight=data['count'],
                    avg_sentiment=np.mean(data['sentiments']),
                    min_sentiment=min(data['sentiments']),
                    max_sentiment=max(data['sentiments']),
                    sentiment_std=np.std(data['sentiments']) if len(data['sentiments']) > 1 else 0
                )

        all_users = set()
        for s, t in interaction_data.keys():
            all_users.add(s)
            all_users.add(t)
        G.add_nodes_from(all_users)

        # Build undirected user network by aggregating directed weights
        U = nx.Graph()
        U.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            w = d.get('weight', 1)
            if U.has_edge(u, v):
                U[u][v]['weight'] += w
            else:
                U.add_edge(u, v, weight=w)

        # aggregate reverse direction if exists
        for u, v, d in G.edges(data=True):
            if G.has_edge(v, u):
                rev_w = G[v][u].get('weight', 1)
                if U.has_edge(u, v):
                    U[u][v]['weight'] += rev_w
                else:
                    U.add_edge(u, v, weight=rev_w)

        # Compute gcc size
        if U.number_of_nodes() == 0:
            gcc_size = 0
        else:
            try:
                gcc_size = len(max(nx.connected_components(U), key=len))
            except Exception:
                gcc_size = 0

        network_stats = {
            'gcc_size': int(gcc_size),
            'nodes': U.number_of_nodes(),
            'edges': U.number_of_edges()
        }

        return period, {
            'directed_reply_network': G,
            'user_network': U,
            'network_stats': network_stats
        }

    except Exception as e:
        logger.error(f"build_period_network_worker failed for {period}: {e}", exc_info=True)
        return period, {}

# MAIN EXECUTION FUNCTION
def run_analysis(data_path, checkpoint_dir="checkpoints", n_workers=None,
                 use_checkpoints=True, min_gcc_size=15000, periods_per_year=6,
                 overlapping_communities=True):
    """
    Run the full Reddit reply network analysis pipeline.

    High-level phases:
      1) Load and preprocess data (with optional checkpointing)
      2) Exploratory data analysis
      3) Construct temporal networks for selected periods
      4) Detect communities (overlapping or not)
      5) Hub analysis
      6) Stability metrics, plots, and exports

    Args:
        data_path (str): Path to Reddit data directory (should contain posts/comments).
        checkpoint_dir (str): Directory for checkpoints.
        n_workers (int or None): Number of workers for parallel processing.
        use_checkpoints (bool): Whether to use or produce checkpoints.
        min_gcc_size (int): Minimum GCC size to consider a period.
        periods_per_year (int): Number of periods to select per year.
        overlapping_communities (bool): If True use overlapping detection (BigClam).

    Returns:
        tuple: (analyzer, community_results, stability_metrics, network_metrics_df)
    """
    print("\n" + "="*70)
    print("🎯 REDDIT REPLY NETWORK ANALYSIS")
    print("="*70)
    print("\n⚙️  CONFIGURATION:")
    print(f"   • Minimum GCC size: {min_gcc_size:,} nodes")
    print(f"   • Periods per year: {periods_per_year}")
    print(f"   • Overlapping communities: {'Yes (BigClam)' if overlapping_communities else 'No (Louvain)'}")
    print(f"   • Workers: {n_workers or 'auto'}")
    print(f"   • Checkpoints: {'enabled' if use_checkpoints else 'disabled'}")
    print("="*70)

    analyzer = RedditReplyAnalyzer(checkpoint_dir=checkpoint_dir, n_workers=n_workers)

    try:
        # Phase 1: Load Data
        df_posts, df_comments = analyzer.parallel_data_loading(data_path, use_checkpoints)

        if df_comments is None:
            print("\n✗ Failed to load data\n")
            return analyzer, None, None, None

        # Phase 1.5: Exploratory Data Analysis
        df_posts, df_comments = analyzer.exploratory_data_analysis(df_posts, df_comments)

        # Phase 2: Build Reply Networks for selected periods
        temporal_networks = analyzer.parallel_network_construction(
            df_comments,
            temporal_unit='month',
            min_gcc_size=min_gcc_size,
            periods_per_year=periods_per_year,
            use_checkpoints=use_checkpoints
        )

        if not temporal_networks:
            print("\n✗ Failed to build networks\n")
            return analyzer, None, None, None

        # Export network metrics as CSV
        network_metrics_df = analyzer.export_network_metrics(temporal_networks)

        # Phase 3: Community Detection (optionally overlapping)
        community_results = analyzer.parallel_community_detection(
            temporal_networks,
            use_checkpoints=use_checkpoints,
            overlapping=overlapping_communities
        )

        if not community_results:
            print("\n⚠️  No communities detected\n")
            return analyzer, None, None, network_metrics_df

        # Phase 4: Hub Analysis
        hub_evolution = analyzer.parallel_hub_analysis(community_results, use_checkpoints)

        # Phase 5: Stability Metrics
        stability_metrics = analyzer.calculate_stability_metrics(community_results, hub_evolution)

        # Phase 6: Visualization and Analysis (stability plots)
        if stability_metrics:
            analyzer.plot_stability_metrics(stability_metrics)

        # Export final results and generate report
        analyzer.export_results(community_results, hub_evolution, stability_metrics)
        analyzer.generate_report(stability_metrics)

        print("\n" + "="*70)
        print("✅ ENHANCED ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\n📁 Results available in:")
        print("   • community_memberships.csv")
        print("   • hub_analysis.csv")
        print("   • stability_metrics.csv")
        print("   • network_metrics.csv")
        print("   • stability_analysis.png")
        print("   • checkpoints/ (for recovery)")
        print("\n🎨 Available visualization methods:")
        print("   • analyzer.visualize_network_period(period, community_results, hub_evolution)")
        print("   • analyzer.compare_period_transition(period1, period2, community_results, hub_evolution)")
        print("   • analyzer.plot_stability_metrics(stability_metrics)")
        print("\n" + "="*70 + "\n")

        return analyzer, community_results, stability_metrics, network_metrics_df

    except Exception as e:
        print(f"\n\n{'='*70}")
        print("✗ ANALYSIS FAILED")
        print(f"{'='*70}")
        print(f"\nError: {e}\n")
        traceback.print_exc()
        print("\n" + "="*70 + "\n")
        return analyzer, None, None, None


# Usage example in Jupyter notebook:
def demo_visualizations(analyzer, community_results, hub_evolution, stability_metrics):
    """
    Demo helper showing how to call visualization functions interactively.

    Args:
        analyzer (RedditReplyAnalyzer): An initialized analyzer instance.
        community_results (dict): Output of community detection per period.
        hub_evolution (dict): Hub diagnostics per period.
        stability_metrics (dict): Stability metrics to plot and inspect.
    """
    # 1. Visualize a specific period (choose middle period by default)
    print("1. 📊 Visualizing a specific network period:")
    periods = list(community_results.keys())
    if periods:
        sample_period = periods[len(periods)//2]  # Middle period
        analyzer.visualize_network_period(sample_period, community_results, hub_evolution)

    # 2. Compare two periods
    print("\n2. 🔄 Comparing period transitions:")
    if len(periods) >= 2:
        analyzer.compare_period_transition(periods[0], periods[1], community_results, hub_evolution)

    # 3. Plot stability metrics
    print("\n3. 📈 Plotting stability metrics:")
    if stability_metrics:
        analyzer.plot_stability_metrics(stability_metrics)

    # 4. Show network metrics (example)
    print("\n4. 📋 Network metrics summary:")
    network_metrics_df = analyzer.export_network_metrics(
        {p: community_results[p]['network'] for p in community_results.keys()}
    )
    if network_metrics_df is not None:
        print(network_metrics_df.head())


if __name__ == "__main__":
    import sys

    # Use 'spawn' on Windows and in some notebook environments to avoid forking issues
    if sys.platform.startswith('win') or 'ipykernel' in sys.modules:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError as e:
            # mp.set_start_method raises RuntimeError if the start method was already set
            logger.info(f"Multiprocessing start method already set: {e}")

    # Example usage (update data_path as needed)
    analyzer, community_results, stability_metrics, network_metrics_df = run_analysis(
        data_path="../Documentaries.corpus",
        checkpoint_dir="checkpoints",
        min_gcc_size=15000,
        periods_per_year=6,
        overlapping_communities=True,  # Set to False for non-overlapping
        use_checkpoints=True
    )

# RUN IN JUPYTER NOTEBOOK:
# # Run the analysis
# analyzer, communities, stability, metrics_df = run_analysis(
#     data_path="your_data_path",
#     overlapping_communities=True
# )

# # MANUAL INSPECTION AND VISUALIZATION
# analyzer.visualize_network_period('2018-03', communities, hubs)
# analyzer.compare_period_transition('2018-03', '2018-04', communities, hubs)

# Each edge in the networks can be accessed via:
# temporal_networks['2018-03']['reply_network'].edges(data=True)
# Example usage:
# for u, v, data in G.edges(data=True):
#     print(f"Edge {u} -> {v}:")
#     print(f"  Interactions: {data['weight']}")
#     print(f"  Average sentiment: {data['avg_sentiment']:.3f}")
#     print(f"  Sentiment range: [{data['min_sentiment']:.3f}, {data['max_sentiment']:.3f}]")
#     print(f"  Sentiment std: {data['sentiment_std']:.3f}")

# # After loading data:
# df_comments = analyzer._load_comments_smart(data_path)

# # Check some sample texts
# print("\nSample texts from loaded data:")
# for _, row in df_comments.sample(n=3, random_state=42).iterrows():
#     print(f"\nUser: {row['user']}")
#     print(f"Text length: {len(row['text'])}")
#     print(f"Text preview: {row['text'][:200]}...")

# # Check overlapping communities
# period = '2018-03'
# user_communities = communities[period]['communities']
# overlapping_users = [user for user, comms in user_communities.items() if len(comms) > 1]
# print(f"Users in multiple communities: {len(overlapping_users)}")

# # Load metrics for analysis
# metrics_df = pd.read_csv('network_metrics.csv')
# print(metrics_df.head())