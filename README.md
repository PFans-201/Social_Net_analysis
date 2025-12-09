# Social_Net_analysis
## Non-Random Patterns in Stochastic Social Network Evolution

Study of whether macroscopic stability emerges from microscopic stochasticity in the user-user interaction network obtained from the r/Documentaries subredit.

---

* **Course:** Complex Data Analysis 2025
* **Professor:** André Falcão
* **Group 5:**
 * Maria João Vicente (44489)
 * Pedro Fanica (54346)
 * Quentin Weiss (66292)

---

## Theoretical Foundation

We're investigating evidence of equilibrium dynamics in human social networks evolving in time - whether macroscopic stability emerges from microscopic stochasticity in the user-user interaction network obtained from the r/Documentaries subreddit.

## Research Questions

1. **Equilibrium dynamics:** Do user-user interaction networks in r/Documentaries exhibit stationary transition probabilities and satisfy detailed balance conditions despite continuous microscopic rewiring?
2. **Hub persistence:** To what extent do overlapping community/hubs (identified via BigCLAM) maintain their structural positions across temporal snapshots, and what role does bursty switching dynamics play in hub collapse?
3. **Community evolution:** Can we model community evolution using spatiotemporal graph Laplacians and detect non-random patterns in what appears to be stochastic community fragmentation?
4. **Predictive stability:** Given features derived from network motifs, sentiment analysis coherence, and temporal activity patterns, can we predict which communities will maintain structural integrity across time windows?
5. **Sentiment coevolution:** How does sentiment polarization (measured via VADER) co-evolve with community structure, and does sentiment homogeneity predict/is related to community stability?

## Dataset Choice

* This dataset was mentioned in the article:
  * S. Kumar, W.L. Hamilton, J. Leskovec, D. Jurafsky. [Community Interaction and Conflict on the Web](https://cs.stanford.edu/~srijan/pubs/conflict-paper-www18.pdf). World Wide Web Conference, 2018.
* This article was very reliant on sentiment analysis and the study reveals that a small group of reddit communities initiate most intercommunity conflicts, which form echo chambers, harm user activity, and can be predicted and mitigated using an LSTM-based early-warning model.
* From there we looked at [datasets in Reddit corpus by subreddit](https://convokit.cornell.edu/documentation/subreddit.html) to find the [full r/Documentaries dataset](https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/DnD35Clinic~-~DoesAnybodyElse/Documentaries.corpus.zip) (340 MB)

### Dataset useful information / metadata (from the source website)

* **Scope:** Timestamps from their creation up to **October 2018**.
* **Structure:**
  * Each **subreddit** has its own corpus (`subreddit-[name]`).
  * Each **post + comment thread** is a *conversation*.
  * Each **comment or post** is an *utterance*.
  * Each **user** is a *speaker* (identified by Reddit username).
* **Utterance-level fields:** ID, speaker, conversation ID, reply-to ID, timestamp, text, score, gilding info, stickied status, permalink, and author flair.
* **Conversation-level fields:** Title, number of comments, domain, subreddit name, gilding, stickied status, and author flair.
* **Corpus-level fields:** Subreddit name, total posts, total comments, and number of unique speakers.
* **Usage:** Can be downloaded as a zip file or loaded with `convokit`, and combined with other subreddit corpora for cross-community analysis.
* **Notes:**
  * Some subreddit data may be incomplete or contain broken thread links.
  * Large subreddits may have very large corpus files.
  * Speaker counts may be inflated due to duplicates in preprocessing.
  * Dataset is **beta** and subject to updates for completeness and data consistency.
