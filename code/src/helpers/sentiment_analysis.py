import logging

import pandas as pd
import networkx as nx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


logger = logging.getLogger(__name__)


def sentiment_analyzer():
    return SentimentIntensityAnalyzer()


def process_sentiment_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Worker function to compute VADER sentiment for a chunk of comments.
    Designed to be called via multiprocessing.Pool.map.

    Args:
        chunk (pd.DataFrame): Partition of the comments DataFrame.

    Returns:
        pd.DataFrame or None: Processed partition with 'sentiment' column or None on error.
    """

    logger.debug(f"Processing sentiment chunk of size {len(chunk)}")

    try:
        analyzer = SentimentIntensityAnalyzer()
        new_chunk = chunk.copy()
        new_chunk["sentiment"] = new_chunk["text"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
        logger.debug("Successfully processed chunk")
        return new_chunk
    except Exception as e:
        logger.error(f"Chunk processing failed: {e}", exc_info=True)
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
