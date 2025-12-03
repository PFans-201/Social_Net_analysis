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
            print("Attempting to reuse chackpoint...")
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

        Returns:
            dict: A dictionary mapping periods to user-user networks.
        """

        def _build_snapshot_network_worker(args):
            """
            Worker that builds undirected user-user network for a single period.

            Args:
                args (tuple): (period, period_df, global_comment_map, min_interactions)

            Returns:
                tuple: (period, networks_dict)
            """
            period, period_df, global_comment_map, min_interactions = args

            try:
                if period_df is None or len(period_df) == 0:
                    return period, nx.Graph()

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
                    return period, nx.Graph()

                # Build undirected user network by aggregating directed weights
                U = nx.Graph()

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
                return period, nx.Graph()

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

            selected_periods = gcc_stats["period"].unique()
            n_selected_periods = len(selected_periods)

            # Prepare args for pool workers
            snapshot_build_args = []
            for period in selected_periods:
                # allow checkpoint loading to skip work
                snapshot_filename = f"user_network_{temporal_unit}_{period}.pkl"
                if self.use_checkpoints:
                    snapshot_data = self.load_checkpoint(snapshot_filename)
                    if snapshot_data is not None:
                        snapshots[period] = {"user_network": snapshot_data[0], "network_stats": snapshot_data[1]}
                        continue
                snapshot_df = df_comments[df_comments["aux_partition_key"] == period]
                snapshot_build_args.append((period, snapshot_df, global_comment_map, 1))

            # Create snapshots in parallel using the top-level worker
            if snapshot_build_args:
                for i, args in enumerate(snapshot_build_args):
                    period, _, _, _ = args
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
                dimensions = max(2, min(20, int(np.sqrt(n_nodes) / 5)))

                bigclam = BigClam(
                    dimensions=dimensions,
                    iterations=1000,
                    seed=self.random_seed,
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
                "degrees": metrics.get_node_degrees_distribution(network),
                "connected_component_sizes": metrics.get_connected_component_size_distribution(network),
                "clustering_coefficient": metrics.get_clustering_coefficient_distribution(network),
                "betweenness_centrality": metrics.get_betweenness_centrality_distribution(network),
            }

        def _calculate_post_stats(period: str):

            snapshot_period_components = period.split("#")
            yearmonth = snapshot_period_components[0]
            unit = snapshot_period_components[1]

            aux_df_comments = self.df_comments[
                (self.df_comments["yearmonth"] == yearmonth)
                & (self.df_comments[self.temporal_unit] == unit)
            ]
            aux_df_posts = self.df_posts[
                (self.df_posts["datetime"].dt.strftime("%Y-%m") == yearmonth)
            ]

            # Basic statistics summary

            _record_stat(period, "Total comment count", len(aux_df_comments))
            _record_stat(period, "Unique users", aux_df_comments["user"].nunique())
            _record_stat(period, "Unique posts with comments", aux_df_comments["root"].nunique())
            _record_stat(period, "Total comments without text", len(aux_df_comments[aux_df_comments["text"].isna() | (aux_df_comments["text"] == "")]))

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

            self._create_temporal_plots(aux_df_posts, aux_df_comments)
            self._analyze_user_activity(aux_df_comments)
            self._analyze_post_engagement(aux_df_posts, aux_df_comments)
            self._analyze_reply_network(aux_df_comments)
            self._analyze_sentiment(aux_df_comments)

            user_post_counts = aux_df_comments.groupby("user")["root"].nunique()
            post_user_counts = aux_df_comments.groupby("root")["user"].nunique()

            _record_stat(period, "Median posts per user", np.median(user_post_counts))
            _record_stat(period, "Median users per post", np.median(post_user_counts))
            _record_stat(period, "Ratio of users commenting on multiple posts", (user_post_counts > 1).sum() * 100 / user_post_counts.sum())
            _record_stat(period, "Ratio of posts with multiple users commenting", (post_user_counts > 1).sum() * 100 / post_user_counts.sum())

        def _calculate_network_stats(period: str, network: nx.Graph, community: dict):

            gcc = metrics.get_largest_connected_component(network)

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

            _record_distribution_stat(f"{period}-GCC", gcc)
            _record_stat(period, "Total node count", metrics.get_node_count(gcc))
            _record_stat(period, "Total edge count", metrics.get_edge_count(gcc))
            _record_stat(period, "Density", metrics.get_density(gcc))
            _record_stat(period, "Diameter", metrics.get_diameter(gcc))
            _record_stat(period, "Average shortest path", metrics.get_average_shortest_path(gcc))
            _record_stat(period, "Assortativity", metrics.get_assortativity(gcc))

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
                _calculate_post_stats(period)

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

    def export_network_metrics(self):
        """
        Export detailed network metrics (on the GCC) for each period into a CSV.

        Args:
            snapshots (dict): Mapping period -> network dict created earlier.
            output_file (str): Filename for the CSV export.

        Returns:
            pd.DataFrame or None: DataFrame with exported metrics if any, otherwise None.
        """
        # TODO REWRITE THIS FUNCTION TO GET ALL METRICS (FOR THE WHOLE NETWORK AND GCC,
        # be sure to get as much metrics as possible)
        snapshots = self.snapshots

        network_metrics = []
        for period, networks in snapshots.items():
            stats = dict(networks.get('network_stats', {}))
            stats['period'] = period

            # Include directed network stats when available
            directed_net = networks.get('directed_reply_network')
            if directed_net and directed_net.number_of_nodes() > 0:
                stats['directed_nodes'] = directed_net.number_of_nodes()
                stats['directed_edges'] = directed_net.number_of_edges()
                stats['reciprocity'] = directed_net.graph.get('reciprocity', 0)

            # GCC metrics on undirected user network
            user_net = networks.get('user_network')
            if user_net and user_net.number_of_nodes() > 0 and user_net.number_of_edges() > 0:
                try:
                    gcc_nodes = max(nx.connected_components(user_net), key=len)
                    gcc = user_net.subgraph(gcc_nodes).copy()
                    stats['gcc_nodes'] = gcc.number_of_nodes()
                    stats['gcc_edges'] = gcc.number_of_edges()
                    stats['gcc_density'] = nx.density(gcc)

                    degs = [d for _, d in gcc.degree()]
                    stats['gcc_avg_degree'] = float(np.mean(degs)) if degs else 0.0
                    stats['gcc_median_degree'] = float(np.median(degs)) if degs else 0.0
                    stats['gcc_max_degree'] = int(max(degs)) if degs else 0
                    stats['gcc_min_degree'] = int(min(degs)) if degs else 0
                    stats['gcc_degree_std'] = float(np.std(degs)) if degs else 0.0

                    stats['gcc_avg_clustering'] = nx.average_clustering(gcc) if gcc.number_of_nodes() > 1 else 0.0
                    stats['gcc_transitivity'] = nx.transitivity(gcc) if gcc.number_of_nodes() > 2 else 0.0

                    # TODO REMOVE IF IT WORKS FOR THE WHOLE NETWORK
                    # Path metrics (sampled for large graphs)
                    if gcc.number_of_nodes() > 1000:
                        rng = np.random.default_rng(42)
                        sample = min(1000, gcc.number_of_nodes())
                        sampled_nodes = rng.choice(list(gcc.nodes()), size=sample, replace=False)
                        # This is compulationally expensive to calculate with 10k+ nodes in a network
                        pairs = itertools.islice(itertools.combinations(sampled_nodes, 2), 5000)
                        dists = []
                        for u, v in pairs:
                            try:
                                dists.append(nx.shortest_path_length(gcc, u, v))
                            except Exception:
                                continue
                        if dists:
                            stats['gcc_avg_path_length'] = float(np.mean(dists))
                            stats['gcc_diameter_estimate'] = int(max(dists))
                        stats['gcc_is_sampled'] = True
                    else:
                        stats['gcc_avg_path_length'] = nx.average_shortest_path_length(gcc)
                        stats['gcc_diameter'] = nx.diameter(gcc)
                        stats['gcc_is_sampled'] = False

                    # Centralization proxies and assortativity
                    dc = nx.degree_centrality(gcc)
                    stats['gcc_degree_centralization'] = float(max(dc.values())) if dc else 0.0
                    bc = nx.betweenness_centrality(gcc, k=min(500, gcc.number_of_nodes()))
                    stats['gcc_betweenness_centralization'] = float(max(bc.values())) if bc else 0.0
                    stats['gcc_assortativity'] = float(nx.degree_assortativity_coefficient(gcc)) if gcc.number_of_nodes() > 1 else 0.0
                    stats['gcc_edge_density'] = (2 * gcc.number_of_edges()) / (gcc.number_of_nodes() * (gcc.number_of_nodes() - 1)) if gcc.number_of_nodes() > 1 else 0.0

                    # Community summary via Louvain on GCC
                    part = community_louvain.best_partition(gcc, weight='weight')
                    stats['gcc_num_communities'] = int(len(set(part.values())))
                    stats['gcc_modularity'] = float(community_louvain.modularity(part, gcc))
                except Exception as e:
                    logger.warning(f"GCC metric calculation failed for {period}: {e}")

            network_metrics.append(stats)

        if network_metrics:
            network_report_output_path = os.path.join(self.reports_dir, self.network_metrics_filename)
            df_metrics = pd.DataFrame(network_metrics)
            cols = ['period'] + [c for c in df_metrics.columns if c != 'period']
            df_metrics = df_metrics[cols]
            df_metrics.to_csv(network_report_output_path, index=False)
            print(f"Exported report to '{network_report_output_path}'")

        else:
            print("No network metrics to export")

        return df_metrics

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

    def parallel_hub_analysis(self, community_results):
        """
        Analyze hubs (high-degree nodes) per period and optionally reuse checkpoints.

        Args:
            community_results (dict): Output of parallel_community_detection.
            use_checkpoints (bool): Whether to load/save hub analysis checkpoints.

        Returns:
            dict: Mapping period -> {'hubs': {user:deg}, 'degrees': {user:deg}}
        """

        def _analyze_hubs_worker(network_tuple):
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

        # TODO: ratio of number of in-edges > number of out-edges

        print("\n" + "="*70)
        print("⭐ PHASE 4: HUB ANALYSIS")
        print("="*70 + "\n")

        hub_evolution = {}

        # Prepare hub worker args (skip if checkpoint exists)
        hub_args = []
        for period, result in community_results.items():
            hub_filename = f"hubs_{period}.pkl"
            if self.use_checkpoints:
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
                for i, (period, hubs, degrees) in enumerate(pool.imap_unordered(_analyze_hubs_worker, hub_args), 1):
                    if hubs:
                        period_hubs = {'hubs': hubs, 'degrees': degrees}
                        hub_evolution[period] = period_hubs
                        if self.use_checkpoints:
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
