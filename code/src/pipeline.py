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
import matplotlib.pyplot as plt
import seaborn as sns


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

# Configure matplotlib and seaborn palletes
plt.style.use("default")
sns.set_palette("husl")


def run_analysis(
        data_path: str,
        snapshots_per_year: int = 4,
        min_giant_component_size: int = 10_000,
        overlapping_communities: bool = True,
        use_checkpoints: bool = True,
        checkpoint_dir: str = "./checkpoints",
        reports_dir: str = "./reports",
        synthetic_network: nx.Graph = None,
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
        synthetic_network (nx.Graph): If provided, compares the observed
            network to a synthetic network (default: None).
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
            )

        # Phase 5: Generate descriptive metrics about networks
        analyzer.run_network_analysis()

        analyzer.export_network_metrics(
            synthetic=synthetic_network is not None,
            synthetic_network=synthetic_network,
        )

        hub_evolution = analyzer.parallel_hub_analysis(community_results)
        stability_metrics = analyzer.calculate_stability_metrics(community_results, hub_evolution)
        if stability_metrics:
            analyzer.plot_stability_metrics(stability_metrics)

        # Export final results and generate report
        analyzer.export_results(community_results, hub_evolution, stability_metrics)
        analyzer.generate_report(stability_metrics)

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


# Usage example in Jupyter notebook:
def demo_visualizations(analyzer, community_results, hub_evolution, stability_metrics):
    """
    Demo helper showing how to call visualization functions interactively.

    Args:
        analyzer (RedditNetworkAnalyzer): An initialized analyzer instance.
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

    # Example usage (update data_path as needed)
    analyzer, community_results, stability_metrics, network_metrics_df = run_analysis(
        data_path="../Documentaries.corpus",
        checkpoint_dir="checkpoints",
        min_giant_component_size=10000,
        snapshots_per_year=6,
        overlapping_communities=True,
        use_checkpoints=True
    )

    # # MANUAL INSPECTION AND VISUALIZATION
    # analyzer.visualize_network_period('2018-03', communities, hubs)
    # analyzer.compare_period_transition('2018-03', '2018-04', communities, hubs)

    # # Each edge in the networks can be accessed via:
    # # snapshots['2018-03']['user_network'].edges(data=True)
    # # Example usage:
    # for u, v, data in G.edges(data=True):
    #     print(f"Edge {u} -> {v}:")
    #     print(f"  Interactions: {data['weight']}")
    #     print(f"  Average sentiment: {data['avg_sentiment']:.3f}")
    #     print(f"  Sentiment range: [{data['min_sentiment']:.3f}, {data['max_sentiment']:.3f}]")
    #     print(f"  Sentiment std: {data['sentiment_std']:.3f}")

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
