import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


# Configure color palletes
plt.style.use("default")
sns.set_palette("husl")


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


def create_temporal_plots(self, df_posts, df_comments):
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


def analyze_user_activity(self, df_comments):
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


def analyze_post_engagement(self, df_posts, df_comments):
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


def analyze_sentiment(self, df_comments):
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
