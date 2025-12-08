"""
RedditNetworkAnalyzer class

This module contains the RedditNetworkAnalyzer class and helper methods to:
- load the raw subreddit conversation dataset;
- preprocess the raw dataset to create snapshot networks of user-user interactions;
- detect communities (overlapping or otherwise);
- generate reports with hub analysis for each network (description and stability metrics).

Usage:
- Import and call the `run_analysis` function, or use the `RedditNetworkAnalyzer` class directly for interactive exploration.
"""

import csv
import itertools
import json
import logging
import os
import pickle
import traceback
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import community as community_louvain
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from karateclub import BigClam

from src.helpers import metrics
from src.helpers import sentiment_analysis
from src.helpers import threading


# Supress warnings
warnings.filterwarnings("ignore")

# Configure logging: write everything to a file
# Do NOT add a StreamHandler so logger calls do NOT appear on console/notebook output
# Keep print(...) for user feedback.
LOG_FILE = "analyzer.log"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Clear any inherited handlers (e.g., from other modules or basicConfig)
logger.handlers = []
file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.propagate = False

# Configure matplotlib and seaborn palletes
plt.style.use("default")
sns.set_palette("husl")


class RedditNetworkAnalyzer:
    """
    Core class for loading Reddit conversation data, building reply networks,
    detecting communities, analyzing hubs, computing stability metrics, and
    exporting results.
    """

    def __init__(
            self,
            data_path: str,
            snapshots_per_year: int = 4,
            min_giant_component_size: int = 10_000,
            overlapping_communities: bool = True,
            use_checkpoints: bool = True,
            checkpoint_dir: str = "./checkpoints",
            reports_dir: str = "./reports",
            n_workers: int = 1,
            max_threads_per_worker: int = 1,
            bigclam_node_threshold: int = 20_000,
            bigclam_edge_threshold: int = 200_000,
            random_seed: int = 28,
        ):
        """
        Initialize the analyzer.

        Args:
            data_path (str): Path to the raw Reddit data directory
                (should contain JSON files with posts and comment data).
            snapshots_per_year (int, optional): Number of network snapshots
                per year to be generated for analysis (default: 4).
            min_giant_component_size (int, optional): Minimum number of nodes
                required to be found in the largest connected component to
                consider a snapshot for analysis (default: 10_000 nodes).
            overlapping_communities (bool, optional): If True, use BigClam
                algorithm for overlapping community detection; otherwise,
                use Louvain algorithm (default: True).
            use_checkpoints (bool, optional): Whether to save checkpoints
                of the preprocessed data (default: True).
            checkpoint_dir (str, optional): Directory in which to save the
                checkpoints (default: "./checkpoints").
            reports_dir (str, optional): Directory in which to save the
                reports (default: "./reports").
            n_workers (int, optional): Number of workers to use for parallel
                processing (default: 1).
            max_threads_per_worker (int, optional): Maximum number of threads
                per worker process for BLAS/OpenMP (default: 1).
            bigclam_node_threshold (int, optional): Node count threshold
                to enable BigClam community detection (default: 20_000).
            bigclam_edge_threshold (int, optional): Edge count threshold
                to enable BigClam community detection (default: 200_000).
        """

        self.data_path = data_path
        self.snapshots_per_year = snapshots_per_year
        self.min_giant_component_size = min_giant_component_size
        self.overlapping_communities = overlapping_communities
        self.use_checkpoints = use_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.reports_dir = reports_dir
        self.random_seed = random_seed

        self.checkpoint_loading_step = "full_data_raw.pkl"
        self.checkpoint_preprocessing_step = "full_data_preprocessed.pkl"
        self.community_memberships_filename = "community_memberships.pkl"
        self.metrics_filename = "metrics.pkl"

        # To be set in `load_data`
        self.global_comment_map = None
        self.df_posts = None
        self.df_comments = None

        # To be set in `create_periodical_snapshots`
        self.temporal_unit = None
        self.snapshots = None

        # To be set in `detect_communities`
        self.detected_communities = None
        self.detected_communities_synthetic_data = None

        # To be set in `run_network_analysis`
        self.numeric_metrics = {"network": [], "metric_name": [], "metric_value": []}
        self.distribution_metrics = {}

        # Default to all but one CPU to leave system responsive
        self.n_workers = n_workers or max(1, cpu_count() - 1)

        # Stability guardrails
        self.max_threads_per_worker = int(max(1, max_threads_per_worker))
        self.bigclam_node_threshold = int(bigclam_node_threshold)
        self.bigclam_edge_threshold = int(bigclam_edge_threshold)

        # Ensure reports and checkpoint directories exists
        os.makedirs(reports_dir, exist_ok=True)
        if use_checkpoints:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Friendly console initialization message
        print(f"Initialized Reddit Network Analyzer with {self.n_workers} parallel workers")

    def save_checkpoint(self, data: object, filename: str) -> bool:
        """
        Save a Python object to a pickle checkpoint file.

        Args:
            data (object): Any picklable Python object to save.
            filename (str): Filename to use for the checkpoint (saved inside checkpoint_dir).

        Returns:
            bool: True if checkpoint is saved successfully, False otherwise.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Checkpoint saved: {filename}")
            return True
        except Exception as e:
            # Avoid raising here; callers may continue without checkpoint
            print(f"Failed to save checkpoint {filename}: {e}")
            return False

    def load_checkpoint(self, filename: str) -> object:
        """
        Load a checkpoint file. Fails gracefully if the checkpoint does not exist.

        Args:
            filename (str): Name of the file to load (located in `self.checkpoint_dir`).

        Returns:
            object or None: The unpickled object or None if loading failed.
        """
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                print(f"Loaded checkpoint: {filename}")
                return data
            except Exception as e:
                print(f"Failed to load checkpoint {filename}: {e}")
        return None

    def load_data(self) -> tuple:
        """
        Load and preprocess raw subreddit dataset containing posts and
        comments by user, located in the configured source directories.
        Create checkpoints only if that configuration is set.

        This method attempts to reuse existing checkpoints to speed up load times.

        Returns:
            tuple(pd.DataFrame, pd.DataFrame): Dataframes with posts and comments.
        """

        def _load_jsonl_chunked(file_path: str, chunk_size: int = 100_000) -> pd.DataFrame:
            """
            Read JSONL line by line into pandas DataFrame in chunks with cleaning.
            This avoids memory spikes and handles malformed lines gracefully.

            Args:
                file_path (str): Path to the JSONL file.
                chunk_size (int, optional): Number of JSON objects per chunk.

            Returns:
                pd.DataFrame or None: Concatenated dataframe or None on failure.
            """
            print(f"Loading JSONL from '{file_path}'")
            chunks = []
            current_chunk = []
            line_count = 0

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())

                            # Clean and normalize text field proactively
                            if "text" in data:
                                # Replace deleted markers with empty text
                                if data["text"] == "[deleted]":
                                    data["text"] = ""
                                else:
                                    # Remove null bytes, empty lines and normalize newlines
                                    text = data["text"].replace("\x00", "")
                                    text = "\n".join(line for line in text.splitlines() if line.strip())
                                    data["text"] = text
                            else:
                                data["text"] = ""

                            current_chunk.append(data)
                            line_count += 1

                            # Flush chunk to DataFrame to keep memory bounded
                            if len(current_chunk) >= chunk_size:
                                df_chunk = pd.DataFrame(current_chunk)
                                chunks.append(df_chunk)
                                current_chunk = []
                                if line_count % 500_000 == 0:
                                    print(f"Processed {line_count:,} lines")

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
                        chunks.append(df_chunk)

                if chunks:
                    df = pd.concat(chunks, ignore_index=True)

                    # Final cleaning: ensure text column is str and remove problematic characters
                    if "text" in df.columns:
                        df["text"] = df["text"].fillna("").astype(str)
                        # Remove non-ascii characters to avoid plotting/serialization issues downstream
                        df["text"] = df["text"].apply(lambda x: x.encode("ascii", "ignore").decode("ascii") if x else "")

                    print("Creating CSV for faster future loading...")
                    csv_path = file_path.replace(".jsonl", ".csv")
                    df.to_csv(csv_path, index=False, encoding="utf-8", escapechar="\\", quoting=csv.QUOTE_ALL)
                    print(f"Saved loaded comments to '{csv_path}'")

                    return df

                return None

            except Exception as e:
                logger.error(f"Failed to load JSONL file: {e}")
                return None

        def _load_posts(file_name: str = "conversations.json") -> pd.DataFrame:
            """
            Load posts from a JSON file.

            Returns:
                pd.DataFrame: Dataframe of posts uniquely identified by their IDs.
            """

            posts_file = os.path.join(self.data_path, file_name)
            if not os.path.exists(posts_file):
                raise FileNotFoundError(f"Posts file not found: {posts_file}")

            with open(posts_file, "r", encoding="utf-8") as f:
                posts_data = json.load(f)

            df_posts = pd.DataFrame.from_dict(posts_data, orient="index")
            df_posts["id"] = df_posts.index
            df_posts = df_posts.reset_index(drop=True)
            return df_posts

        def _load_comments(file_name_without_extension: str = "utterances") -> pd.DataFrame:
            """
            Load comments from CSV or JSONL using chunked reading and robust text handling.
            Prefers CSV (faster) if present; otherwise reads JSONL and saves a CSV copy
            for future speedups.

            Returns:
                pd.DataFrame or None: Loaded comments dataframe or None on failure.
            """
            path = os.path.join(self.data_path, file_name_without_extension)
            csv_path = f"{path}.csv"
            jsonl_path = f"{path}.jsonl"

            df = None

            # If CSV exists, read in chunks to bound memory usage
            if os.path.exists(csv_path):
                print("Loading comments from CSV file...")
                try:
                    chunks = []
                    for chunk in pd.read_csv(
                        csv_path,
                        low_memory=False,
                        chunksize=100_000,
                        encoding="utf-8",
                        escapechar="\\",
                        quoting=csv.QUOTE_ALL
                    ):
                        # Ensure text column is string and missing values are handled
                        # TODO CHANGE THE TEXT PROCESSING TO BE SURE TO TEXT VARIABLE IS CLEAN AND READY FOR SENTIMENT ANALYSIS
                        if "text" in chunk.columns:
                            chunk["text"] = chunk["text"].fillna("").astype(str)
                        chunks.append(chunk)

                    df = pd.concat(chunks, ignore_index=True)

                except Exception as e:
                    print(f"CSV loading failed: {e}")

            # Fallback to JSONL chunked loading
            elif os.path.exists(jsonl_path):
                print("Loading comments from JSONL file...")
                try:
                    df = _load_jsonl_chunked(jsonl_path)

                except Exception as e:
                    print(f"JSONL loading failed: {e}")

            else:
                # Neither file found
                raise FileNotFoundError(f"No utterances file found in {self.data_path}")

            if df is None:
                raise ValueError("Failed to load comment data")

            return df

        print("Loading raw data...")
        df_posts = df_comments = None

        # Try to reuse cached preprocessed data
        if self.use_checkpoints:
            print("Attempting to reuse checkpoint...")
            checkpoint_data = self.load_checkpoint(self.checkpoint_loading_step)
            if checkpoint_data is not None:
                df_posts, df_comments = checkpoint_data
                print("Using cached data")

        if df_comments is None:
            try:
                print("Loading post data from raw files...")
                df_posts = _load_posts()

                print("Loading comment data from raw files...")
                df_comments = _load_comments()

                # Save a checkpoint for faster future runs
                if self.use_checkpoints:
                    self.save_checkpoint((df_posts, df_comments), self.checkpoint_loading_step)

            except Exception as e:
                print(f"Data loading failed: {e}")
                traceback.print_exc()

        print("Data loading complete")

        self.df_posts = df_posts
        self.df_comments = df_comments

        try:
            # Create global comment->user mapping for cross-period reply resolution
            self.global_comment_map = df_comments.set_index("id")["user"]
            print(f"Created global comment user map with {len(self.global_comment_map):,} entries")
        except Exception as e:
            self.global_comment_map = None
            print(f"Could not create global comment user map: {e}")

        non_empty_texts = (df_comments["text"] != "").sum()

        print(
            "Loaded:",
            f"    • Posts: {len(df_posts):,}",
            f"    • Comments: {len(df_comments):,}",
            f"    • Non-empty comments: {non_empty_texts:,}",
            f"    • Unique users: {df_comments['user'].nunique():,}",
            sep="\n",
        )

        aux_df_comments = df_comments[df_comments["text"] != ""]
        sample_size = min(5, len(aux_df_comments))
        print(
            "Sample of comments:",
            *[
                f"Comment {i} ({len(aux_df_comments['text'].iloc[i])} characters): {aux_df_comments['text'].iloc[i][:100]}..."
                for i in range(sample_size)
            ],
            sep="\n",
        )

        return df_posts, df_comments

    def preprocess_comments(self) -> pd.DataFrame:
        """
        Preprocess comments and posts datasets. Run sentiment analysis
        over comments and add basic temporal features. Create checkpoints
        only if that configuration is set.

        Returns:
            pd.DataFrame: Cleaned and enriched comments data.
        """

        print("Preprocessing comment data...")
        loaded_data = False

        # Try to reuse cached preprocessed data
        if self.use_checkpoints:
            print("Attempting to reuse checkpoint...")
            checkpoint_data = self.load_checkpoint(self.checkpoint_preprocessing_step)
            if checkpoint_data is not None:
                df_posts, df_comments = checkpoint_data
                loaded_data = True
                print("Using cached data")

        if not loaded_data:
            df_posts = self.df_posts
            df_comments = self.df_comments

            print("Parsing post timestamps")
            df_posts["datetime"] = pd.to_datetime(df_posts["timestamp"], unit="s", errors="coerce")

            print("Excluding comments with '[deleted]' or undefined users")
            df_comments = df_comments[
                (df_comments["user"] != "[deleted]")
                & df_comments["user"].notna()
                & df_comments["reply_to"].notna()
            ]

            print("Excluding '[removed]' comments")
            df_comments = df_comments[df_comments["text"] != "[removed]"]

            print("Running sentiment analysis (might take a few minutes)...")
            # Determine number of chunks based on dataset size and available workers
            n_chunks = min(self.n_workers, len(df_comments) // 10000 + 1)
            chunks = np.array_split(df_comments, n_chunks)

            print(f"    Processing {len(chunks)} chunks in parallel...")
            with Pool(self.n_workers) as pool:
                processed_chunks = pool.map(sentiment_analysis.process_sentiment_chunk, chunks)

            # Filter out any None chunks due to failures
            processed_chunks = [c for c in processed_chunks if c is not None]
            df_comments = pd.concat(processed_chunks, ignore_index=True)

            print("Creating additional temporal features...")

            df_comments["datetime"] = pd.to_datetime(df_comments["timestamp"], unit="s", errors="coerce")
            df_comments.dropna(subset=["datetime"], inplace=True)  # drop rows with invalid timestamps
            df_comments["week"] = df_comments["datetime"].dt.isocalendar().week
            df_comments["month"] = df_comments["datetime"].dt.month
            df_comments["year"] = df_comments["datetime"].dt.year
            df_comments["yearmonth"] = df_comments["datetime"].dt.strftime("%Y-%m")

            # Save a checkpoint for faster future runs
            if self.use_checkpoints:
                self.save_checkpoint((df_posts, df_comments), self.checkpoint_preprocessing_step)

        print(
            "Preprocessed:",
            f"    • Comments: {len(df_comments):,}",
            sep="\n",
        )

        aux_df_comments = df_comments[df_comments["text"] != ""]
        sample_size = min(5, len(aux_df_comments))
        print(
            "Sample of comments:",
            *[
                f"Comment {i} ({len(aux_df_comments['text'].iloc[i])} characters): {aux_df_comments['text'].iloc[i][:100]}..."
                for i in range(sample_size)
            ],
            sep="\n",
        )

        self.df_posts = df_posts
        self.df_comments = df_comments

        return df_comments

    def create_periodical_snapshots(
            self,
            temporal_unit: str = "month",
            min_gcc_size: int = 10_000,
            snapshots_per_year: int = 4,
            directed: bool = False,
    ) -> dict:
        """
        Construct temporal snapshots of the user-user interaction network
        for periods where a minimum giant component size is observed.

        Steps:
          - Scan data to estimate giant component (GCC) sizes for each snapshot period.
          - Select snapshot periods per year according to the provided rules.
          - Build weighed user-user interaction networks for the selected periods.
          - (Optional) Checkpoint produced data.

        Args:
            temporal_unit (str, optional): Column name in df_comments to use for
                period partitioning (default: "month").
            min_gcc_size (int, optional): Minimum number of nodes that must be observed
                in the giant component in order for it to be considered for network
                creation (default: 10_000).
            snapshots_per_year (int, optional): Number of network snapshots to create
                per year (default: 4).
            directed (bool, optional): If True, create directed graphs (nx.DiGraph);
                otherwise create undirected graphs (nx.Graph) (default: False).

        Returns:
            dict: A dictionary mapping periods to user-user networks.
        """

        def _build_snapshot_network_worker(args):
            """
            Worker that builds user-user network for a single period.

            Args:
                args (tuple): (period, period_df, global_comment_map, min_interactions, directed)

            Returns:
                tuple: (period, networks_dict)
            """
            period, period_df, global_comment_map, min_interactions, directed = args

            try:
                if period_df is None or len(period_df) == 0:
                    return period, nx.DiGraph() if directed else nx.Graph()

                # Map comment ids to users (prefer global map if provided)
                if global_comment_map is not None:
                    comment_to_user = global_comment_map
                else:
                    try:
                        comment_to_user = period_df.set_index("id")["user"]
                    except Exception:
                        comment_to_user = pd.Series(dtype=object)

                period_df["reply_to_user"] = period_df["reply_to"].map(comment_to_user)

                valid_replies = period_df[
                    (period_df["reply_to_user"].notna())
                    & (period_df["reply_to_user"] != "[deleted]")
                    & (period_df["user"] != period_df["reply_to_user"])
                ]

                if valid_replies.empty:
                    return period, nx.DiGraph() if directed else nx.Graph()

                # Build network
                U = nx.DiGraph() if directed else nx.Graph()

                interaction_data = defaultdict(lambda: {"count": 0, "sentiment": []})
                for _, row in valid_replies.iterrows():
                    pair = (row["user"], row["reply_to_user"])
                    interaction_data[pair]["count"] += 1
                    interaction_data[pair]["sentiment"].append(row["sentiment"])

                all_users = set()
                for s, t in interaction_data.keys():
                    all_users.add(s)
                    all_users.add(t)
                U.add_nodes_from(all_users)

                for (source, target), data in interaction_data.items():
                    if data["count"] >= min_interactions:
                        if U.has_edge(source, target):  # accounting for reverse order of nodes
                            U[source][target]["weight"] += data["count"]
                        else:
                            U.add_edge(source, target, weight=data["count"])

                return period, U

            except Exception as e:
                logger.error(f"build_period_network_worker failed for {period}: {e}", exc_info=True)
                return period, nx.DiGraph() if directed else nx.Graph()

        self.temporal_unit = temporal_unit
        max_date = 12 if temporal_unit == "month" else 52  # assume "weeks" otherwise
        period = snapshots_per_year % max_date if snapshots_per_year != max_date else max_date
        all_periods = list(range(1, max_date + 1, max_date // period))[:period]

        df_comments = self.df_comments[self.df_comments[temporal_unit].isin(all_periods)]
        df_comments["aux_partition_key"] = df_comments["year"].astype(str) + "#" + df_comments[temporal_unit].astype(str)
        snapshots = {}

        try:
            print("Starting parallel network construction...")

            print("Analyzing network giant component sizes across all requested snapshots...")
            candidate_periods = df_comments["aux_partition_key"].unique()
            n_periods = len(candidate_periods)

            print(f"    Scanning {n_periods} periods in parallel...")

            # Prepare args for worker: (period, period_df, global_comment_map)
            global_comment_map = self.global_comment_map
            period_args = [
                (timestamp, df_comments[df_comments["aux_partition_key"] == timestamp], global_comment_map)
                for timestamp in candidate_periods
            ]

            # Process periods in parallel using top-level worker to avoid pickling local functions
            period_gcc_sizes = {}
            with Pool(self.n_workers) as pool:
                for i, (period, gcc_size) in enumerate(pool.imap_unordered(_estimate_gcc_size_worker, period_args), 1):
                    period_gcc_sizes[period] = gcc_size
                    # print lightweight progress updates
                    if i % max(1, min(10, n_periods // 10)) == 0 or i == n_periods:
                        print(f"    Progress: {i}/{n_periods} periods analyzed...")

            gcc_stats = pd.DataFrame(
                {
                    "period": list(period_gcc_sizes.keys()),
                    "year": [p.split("-")[0] if "-" in p else str(p) for p in period_gcc_sizes.keys()],
                    "gcc_size": list(period_gcc_sizes.values()),
                }
            )

            # Only keep periods that meet the minimum size criterion
            periods_to_keep = gcc_stats[gcc_stats["gcc_size"] >= min_gcc_size]
            periods_to_keep.sort_values("gcc_size", ascending=False, inplace=True)

            print(
                "Giant component size analysis complete!",
                "Summary:",
                f"    • Periods analyzed: {n_periods}",
                f"    • Periods meeting minimum GCC node count (≥ {min_gcc_size:,}: {len(periods_to_keep)}",
                f"    • Largest GCC: {gcc_stats['gcc_size'].iloc[0]:,} nodes in period {gcc_stats['period'].iloc[0]}",
                f"    • Average GCC size: {gcc_stats['gcc_size'].mean():,.0f} nodes",
                sep="\n",
            )

            if len(periods_to_keep) == 0:
                print(
                    f"No periods meet the {min_gcc_size:,} node criterion! Top 10 periods by GCC size:",
                    *[f"    • {gcc_stats['period'].iloc[i]}: {gcc_stats['gcc_size'].iloc[i]:,} nodes" for i in range(10)],
                    sep="\n",
                )
                return {}

            print("Creating snapshots of the network for the selected periods...")

            selected_periods = periods_to_keep["period"].unique()
            n_selected_periods = len(selected_periods)

            # Prepare args for pool workers
            snapshot_build_args = []
            for period in selected_periods:
                # allow checkpoint loading to skip work
                snapshot_filename = f"user_network_{temporal_unit}_{period}.pkl"
                if directed:
                    snapshot_filename = f"directed_{snapshot_filename}"
                if self.use_checkpoints:
                    snapshot_data = self.load_checkpoint(snapshot_filename)
                    if snapshot_data is not None:
                        snapshots[period] = snapshot_data
                        continue
                snapshot_df = df_comments[df_comments["aux_partition_key"] == period]
                snapshot_build_args.append((period, snapshot_df, global_comment_map, 1, directed))

            if snapshot_build_args:
                for i, args in enumerate(snapshot_build_args, 1):
                    period, _, _, _, _ = args
                    networks = _build_snapshot_network_worker(args)
                    if networks and networks[1] and networks[1].number_of_nodes() > 0:
                        snapshots[period] = networks[1]
                    # lightweight progress
                    if i % max(1, n_selected_periods // 10) == 0 or i == n_selected_periods:
                        print(f"    Built networks for {i}/{n_selected_periods} selected periods...")
            else:
                print("No snapshots required building (all loaded from checkpoints)")

            if self.use_checkpoints:
                for period in snapshots.keys():
                    try:
                        snapshot_filename = f"user_network_{temporal_unit}_{period}.pkl"
                        if directed:
                            snapshot_filename = f"directed_{snapshot_filename}"
                        self.save_checkpoint(snapshots[period], snapshot_filename)
                    except Exception as e:
                        print(f"    Failed to save checkpoint for {period}: {e}")

            self.snapshots = snapshots
            return snapshots

        except Exception as e:
            print(f"Network construction failed: {e}")
            self.snapshots = {}
            return {}

    def detect_communities(
            self,
            synthetic: bool = False,
            synthetic_network: nx.Graph = None,
            synthetic_network_name: str = "synthetic_network",
            min_admissible_nodes: int = 20,
        ) -> dict:
        """
        Detect communities in the networks. Supports overlapping detection with BigClam
        or non-overlapping via the Louvain algorithm as per the analyzer configurations.

        If a synthetic network is provided, it will detect communities in that network.
        Otherwise, detect communities in the snapshots previously generated. Outputs
        are saved to the respective class attribute.

        Args:
            synthetic (bool, optional): If True, apply community detection algorithms
                to the synthetic network provided; otherwise, apply them to the observed
                snapshots (default: False).
            synthetic_network (nx.Graph, optional): A synthetic undirected network
                (default: None).
            synthetic_network_name (str, optional): Name to assign to the synthetic network
                (default: 'synthetic_network').
            min_admissible_nodes (int, optional): Minimum number of nodes that the network
                must contain in order to apply a community detection algorithm (default: 20).

        Returns:
            dict: Results mapping network name to detected community memberships.
        """

        def _detect_overlapping_communities(period, network):
            """Apply BigClam algorithm to detect communities."""

            try:
                # BigClam works best on undirected structure for community definition
                if network.is_directed():
                    network_for_algo = network.to_undirected()
                else:
                    network_for_algo = network

                n_nodes = network_for_algo.number_of_nodes()
                dimensions = max(2, min(20, int(np.sqrt(n_nodes) / 5)))

                bigclam = BigClam(
                    dimensions=dimensions,
                    iterations=1000,
                    seed=self.random_seed,
                )

                adj_matrix = nx.to_scipy_sparse_array(network_for_algo)
                bigclam.fit(adj_matrix)
                memberships = bigclam.get_memberships()

                communities = {}
                node_list = list(network_for_algo.nodes())
                for node_idx, comm_affiliations in enumerate(memberships):
                    if comm_affiliations and len(comm_affiliations) > 0:
                        node_name = node_list[node_idx]
                        communities[node_name] = list(comm_affiliations)

                # Validate result for reasonableness
                if communities:
                    all_community_ids = set()
                    for comm_affiliations in communities.values():
                        all_community_ids.update(comm_affiliations)

                    n_communities = len(all_community_ids)
                    overlapping_users = sum(1 for comm_affiliations in communities.values() if len(comm_affiliations) > 1)

                    if n_communities > 1 and n_communities < len(communities) * 0.8:
                        print(
                            f"BigCLAM successful for {period}:",
                            f"    • Detected {n_communities} communities",
                            f"    • Users in community overlaps: {overlapping_users}",
                            f"    • Overlap ratio: {overlapping_users * 100 / len(node_list):.1f} %",
                            sep="\n",
                        )

                return communities

            except Exception as e:
                print(f"BigCLAM algorithm failed for {period}: {e}")
                return {}

        def _detect_non_overlapping_communities(period, network):
            """Apply Louvain algorithm to detect communities."""

            try:
                # Louvain requires undirected graph
                if network.is_directed():
                    weighted_net = network.to_undirected()
                else:
                    weighted_net = network.copy()

                for _, _, d in weighted_net.edges(data=True):
                    if "weight" not in d:
                        d["weight"] = 1.0

                partition = community_louvain.best_partition(
                    weighted_net,
                    weight="weight",
                    random_state=self.random_seed
                )

                communities = {}
                for node, community_id in partition.items():
                    communities[node] = [community_id]

                n_communities = len(set(partition.values()))
                print(
                    f"Louvain successful for {period}:",
                    f"    Detected {n_communities} communities",
                    sep="\n",
                )
                return communities

            except Exception as e:
                print(f"Louvain algorithm failed for {period}: {e}")
                return {}

        print(
            "=" * 70,
            "COMMUNITY DETECTION",
            "=" * 70,
            f"Detection mode: {'Overlapping communities (BigClam)' if self.overlapping_communities else 'Non-overlapping communities (Louvain)'}",
            f"Target: {'Synthetic network ' + synthetic_network_name if synthetic else 'Observed network'}",
            sep="\n",
        )

        snapshots = {synthetic_network_name: synthetic_network} if synthetic else self.snapshots
        detection_function = _detect_overlapping_communities if self.overlapping_communities else _detect_non_overlapping_communities

        results = {}
        if self.use_checkpoints:
            checkpoint_data = self.load_checkpoint(self.community_memberships_filename)
            if checkpoint_data is not None:
                results = checkpoint_data

        for period, network in snapshots.items():

            n_nodes = network.number_of_nodes()

            if n_nodes < min_admissible_nodes:
                print(f"Network {period} does not have the minimum required number of nodes. Skipping community detection!")
                results[period] = {}
                continue

            if self.use_checkpoints and period in results.keys():
                print(f"Using checkpoint data for network {period}. Skipping community detection!")
                continue

            if network is None:
                print(f"No data for {period}!")
                continue

            results[period] = detection_function(period, network)

        if synthetic:
            self.detected_communities_synthetic_data = results
        else:
            self.detected_communities = results

        if self.use_checkpoints:
            self.save_checkpoint(results, self.community_memberships_filename)

        return results

    def analyze_evolution_dynamics(self):
        """
        [Deep Dive] Analyzes the stability of specific users and hubs over time.
        Robust version: handles both wrapped and unwrapped community data structures.
        """
        if not self.detected_communities or len(self.detected_communities) < 2:
            print("Need at least 2 periods with detected communities.")
            return

        print("\n" + "="*70)
        print("EVOLUTION DYNAMICS: HUBS & USER LOYALTY")
        print("="*70)

        # Ensure we only analyze periods present in both datasets
        available_periods = set(self.detected_communities.keys()) & set(self.snapshots.keys())
        sorted_periods = sorted(list(available_periods))

        if len(sorted_periods) < 2:
            print(f" Not enough common periods. Found: {sorted_periods}")
            return

        # --- PART A: HUB PERSISTENCE (Who stays on top?) ---
        print("\nHUB PERSISTENCE ANALYSIS")

        period_hubs = {}
        for period in sorted_periods:
            G = self.snapshots[period]
            degrees = dict(G.degree())
            if not degrees: continue

            # Threshold for top 1%
            threshold = np.percentile(list(degrees.values()), 99)
            hubs = {u for u, d in degrees.items() if d >= threshold}
            period_hubs[period] = hubs
            print(f"   • {period}: {len(hubs)} hubs (Degree >= {int(threshold)})")

        for i in range(len(sorted_periods) - 1):
            p1, p2 = sorted_periods[i], sorted_periods[i+1]
            if p1 not in period_hubs or p2 not in period_hubs: continue

            hubs1 = period_hubs[p1]
            hubs2 = period_hubs[p2]

            retained = hubs1.intersection(hubs2)
            retention_rate = len(retained) / len(hubs1) if hubs1 else 0

            print(f"   {p1} -> {p2}: {len(retained)} hubs survived ({retention_rate:.1%})")

        # Evergreen Hubs
        if period_hubs:
            evergreen_hubs = set.intersection(*period_hubs.values())
            print(f"\n   EVERGREEN HUBS (Present in all {len(period_hubs)} periods): {len(evergreen_hubs)}")
            if evergreen_hubs:
                print(f"      Sample: {list(evergreen_hubs)[:5]}")

        # --- PART B: USER COMMUNITY FIDELITY (Do they switch sides?) ---
        print("\nUSER COMMUNITY FIDELITY")

        user_fidelity_scores = []
        switch_counts = defaultdict(int)

        for i in range(len(sorted_periods) - 1):
            p1, p2 = sorted_periods[i], sorted_periods[i+1]

            # --- CORRECTION: Robust Data Extraction ---
            # Handles if data is stored as {user: [comm]} OR {'communities': {user: [comm]}}
            data1 = self.detected_communities[p1]
            comm_data1 = data1.get('communities', data1) if isinstance(data1, dict) and 'communities' in data1 else data1

            data2 = self.detected_communities[p2]
            comm_data2 = data2.get('communities', data2) if isinstance(data2, dict) and 'communities' in data2 else data2
            # ------------------------------------------

            # Invert to get {comm_id: set(members)}
            def get_groups(c_data):
                g = defaultdict(set)
                for u, c_ids in c_data.items():
                    if c_ids:
                        g[c_ids[0]].add(u)
                return g

            groups1 = get_groups(comm_data1)
            groups2 = get_groups(comm_data2)

            common_users = set(comm_data1.keys()) & set(comm_data2.keys())

            if not common_users:
                print(f" No common users between {p1} and {p2}")
                continue

            period_switch_count = 0

            for user in common_users:
                # Get community IDs
                c1 = comm_data1[user][0]
                c2 = comm_data2[user][0]

                members_t1 = groups1.get(c1, set())
                members_t2 = groups2.get(c2, set())

                # Jaccard Similarity on Neighbors
                intersection = len(members_t1 & members_t2)
                union = len(members_t1 | members_t2)

                # If union is 0 (isolated user), stability is technically undefined/0
                stability = intersection / union if union > 0 else 0

                user_fidelity_scores.append(stability)

                # Define a "Switch" as < 10% neighbour overlap
                if stability < 0.1:
                    period_switch_count += 1
                    switch_counts[user] += 1

            print(f"   {p1} -> {p2}: {len(common_users)} common users")
            if common_users:
                # Avg stability for THIS transition
                current_scores = user_fidelity_scores[-len(common_users):]
                print(f"    Avg Stability: {np.mean(current_scores):.3f}")
                print(f"    Switched Communities: {period_switch_count} ({period_switch_count/len(common_users):.1%})")

        # Global Stats
        if user_fidelity_scores:
            avg_fidelity = np.mean(user_fidelity_scores)
            print(f"\n   OVERALL USER METRICS:")
            print(f"      Global Average Fidelity: {avg_fidelity:.3f} (1.0 = Loyal)")

            frequent_switchers = {u: c for u, c in switch_counts.items() if c >= (len(sorted_periods) - 2)}
            print(f"      Frequent Switchers (>80% of times): {len(frequent_switchers)}")

    def run_network_analysis(
            self,
            synthetic: bool = False,
            synthetic_network: nx.Graph = None,
            synthetic_network_name: str = "synthetic_network",
        ) -> tuple:
        """
        Analyse and produce relevant metrics for the network snapshots
        and provided synthetic networks.

        If a synthetic network is provided, it will calculate metrics
        for that network. Otherwise, use the snapshots previously generated.
        Outputs are saved to the respective class attribute.

        Args:
            synthetic (bool, optional): If True, use the synthetic network provided;
                otherwise, use the observed snapshots (default: False).
            synthetic_network (nx.Graph, optional): A synthetic undirected network
                (default: None).
            synthetic_network_name (str, optional): Name to assign to the synthetic
                network (default: 'synthetic_network').
            min_admissible_nodes (int, optional): Minimum number of nodes that the network
                must contain in order to apply a community detection algorithm (default: 20).

        Returns:
            tuple: Dictionary mapping snapshot dates to pandas dataframes
                containing descriptive metrics calculated for the network
                as observed in the respective time period.
        """

        def _record_stat(name: str, metric_name: str, metric_value: float):
            """Utility function to add a row to the numeric metrics table."""
            numeric_metrics["network"].append(name)
            numeric_metrics["metric_name"].append(metric_name)
            numeric_metrics["metric_value"].append(metric_value)

        def _record_distribution_stat(name: str, network: nx.Graph):
            """Utility function to add data to the distribution metrics dictionary."""

            distribution_metrics[name] = {
                **distribution_metrics.get(name, {}),
                "degrees": metrics.get_node_degrees_distribution(network),
                "connected_component_sizes": metrics.get_connected_component_size_distribution(network),
                "clustering_coefficient": metrics.get_clustering_coefficient_distribution(network),
                "betweenness_centrality": metrics.get_betweenness_centrality_distribution(network),
            }

        def _record_community_sentiment_stat(period: str, df_comments: pd.DataFrame):
            """Utility function to add data to the distribution metrics dictionary."""

            user_count = df_comments.groupby("community_id", as_index=False)["user"].agg("nunique")
            median_sentiment = df_comments.groupby("community_id", as_index=False)["sentiment"].agg(np.median)
            mean_sentiment = df_comments.groupby("community_id", as_index=False)["sentiment"].agg(np.mean)
            stdev_sentiment = df_comments.groupby("community_id", as_index=False)["sentiment"].agg(np.std)

            distribution_metrics[period] = {
                **distribution_metrics.get(period, {}),
                "Total users per community": user_count,
                "Median sentiment per community": median_sentiment,
                "Mean sentiment per community": mean_sentiment,
                "Stdev sentiment per community": stdev_sentiment,
            }

        def _calculate_post_stats(period: str, community: dict):

            print("        Calculating stats based on comments dataframe")

            snapshot_period_components = period.split("#")
            year = int(snapshot_period_components[0])
            unit = int(snapshot_period_components[1])

            aux_df_comments = self.df_comments[
                (self.df_comments["year"] == year)
                & (self.df_comments[self.temporal_unit] == unit)
            ]

            # Basic statistics summary

            _record_stat(period, "Total comment count", len(aux_df_comments))
            _record_stat(period, "Unique users", aux_df_comments["user"].nunique())
            _record_stat(period, "Unique posts with comments", aux_df_comments["root"].nunique())
            _record_stat(period, "Total comments without text", len(aux_df_comments[aux_df_comments["text"].isna() | (aux_df_comments["text"] == "")]))

            # Sentiment analysis per community

            if not self.overlapping_communities:
                aux_df_comments["community_id"] = aux_df_comments["user"].apply(lambda x: community.get(x, [-1])[0])
                _record_community_sentiment_stat(period, aux_df_comments)

            # Filter out empty texts for the rest of this analysis
            aux_df_comments_copy = aux_df_comments[~aux_df_comments["text"].isna() & (aux_df_comments["text"] != "")].copy()

            # Reply analysis to understand pairwise interactions
            df_comments_with_replies = aux_df_comments_copy[aux_df_comments_copy["reply_to"].notna()]

            _record_stat(period, "Total comments that are replies", len(df_comments_with_replies))

            # Build mapping from comment id to user to analyze reply chains
            if len(df_comments_with_replies) > 0:
                comment_to_user = aux_df_comments.set_index("id")["user"]
                df_comments_with_replies["reply_to_user"] = df_comments_with_replies["reply_to"].map(comment_to_user)
                valid_replies = df_comments_with_replies[df_comments_with_replies["reply_to_user"].notna()]
                valid_replies = valid_replies[valid_replies["reply_to_user"] != "[deleted]"]
                valid_replies = valid_replies[valid_replies["user"] != valid_replies["reply_to_user"]]  # Remove self-replies

                _record_stat(period, "Valid user to user replies", len(valid_replies))
                _record_stat(period, "Unique user to user reply pairs", valid_replies[["user", "reply_to_user"]].drop_duplicates().shape[0])

                # Reply pattern statistics
                reply_counts = valid_replies.groupby(["user", "reply_to_user"]).size().reset_index(name="count")

                _record_stat(period, "Average user reply counts", reply_counts["count"].mean())
                _record_stat(period, "Maximum user reply counts", reply_counts["count"].max())

            user_post_counts = aux_df_comments.groupby("user")["root"].nunique()
            post_user_counts = aux_df_comments.groupby("root")["user"].nunique()

            _record_stat(period, "Median posts per user", np.median(user_post_counts))
            _record_stat(period, "Median users per post", np.median(post_user_counts))
            _record_stat(period, "Ratio of users commenting on multiple posts", (user_post_counts > 1).sum() * 100 / user_post_counts.sum())
            _record_stat(period, "Ratio of posts with multiple users commenting", (post_user_counts > 1).sum() * 100 / post_user_counts.sum())

        def _calculate_network_stats(period: str, network: nx.Graph, community: dict):

            print("        Calculating stats based on network")

            partition = {}
            for node, lst in community.items():
                for comm_id in lst:
                    if comm_id not in partition.keys():
                        partition[comm_id] = {node}
                    else:
                        partition[comm_id].add(node)
            partition = list(partition.values())

            _record_distribution_stat(period, network)
            _record_stat(period, "Total node count", metrics.get_node_count(network))
            _record_stat(period, "Total edge count", metrics.get_edge_count(network))
            _record_stat(period, "Density", metrics.get_density(network))
            _record_stat(period, "Assortativity", metrics.get_assortativity(network))
            _record_stat(period, "Community count", metrics.get_community_count(partition))
            _record_stat(period, "Community modularity", metrics.get_modularity(network, partition))
            _record_stat(period, "Average internal vs external edge ratio", metrics.get_mean_internal_edge_ratio(network, partition))

            print("        Calculating stats based on GCC")

            gcc = metrics.get_largest_connected_component(network)

            _record_distribution_stat(f"{period}-GCC", gcc)
            _record_stat(f"{period}-GCC", "Total node count", metrics.get_node_count(gcc))
            _record_stat(f"{period}-GCC", "Total edge count", metrics.get_edge_count(gcc))
            _record_stat(f"{period}-GCC", "Density", metrics.get_density(gcc))
            _record_stat(f"{period}-GCC", "Diameter", metrics.get_diameter(gcc))
            _record_stat(f"{period}-GCC", "Average shortest path", metrics.get_average_shortest_path(gcc))
            _record_stat(f"{period}-GCC", "Assortativity", metrics.get_assortativity(gcc))

            print("        Calculating stats based on largest community")

            comm_graph = metrics.get_largest_community(network, partition)

            _record_distribution_stat(f"{period}-COMM", comm_graph)
            _record_stat(f"{period}-COMM", "Total node count", metrics.get_node_count(comm_graph))
            _record_stat(f"{period}-COMM", "Total edge count", metrics.get_edge_count(comm_graph))
            _record_stat(f"{period}-COMM", "Density", metrics.get_density(comm_graph))
            _record_stat(f"{period}-COMM", "Assortativity", metrics.get_assortativity(comm_graph))

        print("Calculating relevant metrics...")

        snapshots = {synthetic_network_name: synthetic_network} if synthetic else self.snapshots
        communities = self.detected_communities_synthetic_data if synthetic else self.detected_communities

        numeric_metrics = self.numeric_metrics
        distribution_metrics = self.distribution_metrics

        if self.use_checkpoints:
            checkpoint_data = self.load_checkpoint(self.metrics_filename)
            if checkpoint_data is not None:
                numeric_metrics, distribution_metrics = checkpoint_data

        # Save simple network-level metrics
        for period, network in snapshots.items():

            print(f"    Processing period {period}...")

            if self.use_checkpoints and period in distribution_metrics.keys():
                continue

            if not synthetic:
                _calculate_post_stats(period, communities[period])

            _calculate_network_stats(period, network, communities[period])

            if self.use_checkpoints:
                self.save_checkpoint((numeric_metrics, distribution_metrics), self.metrics_filename)

        output_path = os.path.join(self.reports_dir, self.metrics_filename).replace(".pkl", ".csv")
        numeric_metrics = pd.DataFrame(numeric_metrics)
        numeric_metrics.to_csv(output_path, index=False)
        print(f"Exported numeric metrics to {output_path}")

        self.numeric_metrics = numeric_metrics
        self.distribution_metrics = distribution_metrics

        return numeric_metrics, distribution_metrics

################

    def calculate_stability_metrics(self, community_results, hub_evolution):
        """
        Calculate stability metrics in relation to a reference network.

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

            return stability_metrics

        except Exception as e:
            print(f"\n✗ Stability calculation failed: {e}\n")
            return {}

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


def _estimate_gcc_size_worker(args):
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
