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

import logging
import traceback
import warnings

import networkx as nx

from src.helpers.network_analyzer import RedditNetworkAnalyzer


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


def run_analysis(
        data_path: str,
        snapshots_per_year: int = 4,
        min_giant_component_size: int = 10_000,
        overlapping_communities: bool = True,
        use_checkpoints: bool = True,
        checkpoint_dir: str = "./checkpoints",
        reports_dir: str = "./reports",
        synthetic_network: nx.Graph = None,
        synthetic_network_name: str = None,
        n_workers: int = 1,
    ) -> RedditNetworkAnalyzer:
    """
    Run the full Reddit network analysis pipeline.

    High-level phases:
      1) Load and preprocess raw data (with optional checkpointing)
      2) Construct temporal networks for selected periods
      3) Calculate descriptive metrics for the network (e.g. structure, stability)
      4) Detect communities (overlapping or otherwise)
      5) Hub analysis
      6) Generate reports, plots, and exports

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
        synthetic_network (nx.Graph, optional): If provided, compares
            the observed network to a synthetic network (default: None).
        synthetic_network_name (str, optional): Name to assign to the
            synthetic network (default: None).
        n_workers (int, optional): Number of workers to use for parallel
            processing (default: 1).

    Returns:
        RedditNetworkAnalyzer: The initialized analyzer instance.
    """

    print(
        "=" * 70,
        "REDDIT USER-USER INTERACTIONS NETWORK ANALYSIS",
        "=" * 70,
        "CONFIGURATION:",
        f"    • Snapshots per year: {snapshots_per_year}",
        f"    • Minimum GCC size: {min_giant_component_size} nodes",
        f"    • Overlapping community detection enabled: {'Yes (BigClam)' if overlapping_communities else 'No (Louvain)'}",
        f"    • Checkpoints enabled: {'Yes (' + checkpoint_dir + ')' if use_checkpoints else 'No'}",
        f"    • Parallel workers: {n_workers or 'auto'}",
        sep="\n",
    )

    try:

        analyzer = RedditNetworkAnalyzer(
            data_path=data_path,
            snapshots_per_year=snapshots_per_year,
            min_giant_component_size=min_giant_component_size,
            overlapping_communities=overlapping_communities,
            use_checkpoints=use_checkpoints,
            checkpoint_dir=checkpoint_dir,
            reports_dir=reports_dir,
            n_workers=n_workers,
        )

        # Phase 1: Load raw subreddit data
        analyzer.load_data()

        if analyzer.df_comments is None or len(analyzer.df_comments) == 0:
            print("Failed to load data!")
            return analyzer

        # Phase 2: Preprocess subreddit data
        analyzer.preprocess_comments()

        # Phase 3: Build user-user interaction networks for the selected periods
        analyzer.create_periodical_snapshots(
            temporal_unit="month",
            min_gcc_size=min_giant_component_size,
            snapshots_per_year=snapshots_per_year,
        )

        if not analyzer.snapshots:
            print("Failed to build networks!")
            return analyzer

        # Phase 4: Detect communities
        analyzer.detect_communities()

        if synthetic_network is not None:
            analyzer.detect_communities(
                synthetic=True,
                synthetic_network=synthetic_network,
                synthetic_network_name=synthetic_network_name,
            )

        # Phase 5: Generate descriptive metrics about networks
        analyzer.run_network_analysis()

        if synthetic_network is not None:
            analyzer.run_network_analysis(
                synthetic=True,
                synthetic_network=synthetic_network,
                synthetic_network_name=synthetic_network_name,
            )

        print(
            "=" * 70,
            "ANALYSIS COMPLETED SUCCESSFULLY",
            "=" * 70,
            f"Processed data: {analyzer.checkpoint_dir}",
            f"Reports: {analyzer.reports_dir}",
            sep="\n",
        )

    except Exception as e:
        print(
            "=" * 70,
            "ANALYSIS FAILED",
            "=" * 70,
            f"Error: {e}",
            sep="\n",
        )
        traceback.print_exc()

    finally:
        return analyzer


if __name__ == "__main__":

    # Example usage (update data_path as needed)
    analyzer, community_results, stability_metrics, network_metrics_df = run_analysis(
        data_path="../Documentaries.corpus",
        snapshots_per_year=2,
        min_giant_component_size=10_000,
        overlapping_communities=False,
        use_checkpoints=True,
        checkpoint_dir="checkpoints",
        reports_dir="reports",
        n_workers=1,
    )
