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

## Notes from the Professor

- Good choice of dataset, relevant and rich for temporal analysis and holistic modeling of dynamic social networks.
- FOCUS ON CLASS 7 MATERIAL: Dynamic social networks, influences by hubs in time:
  - Threshold (deterministic) vs independent cascade models (probabilistic)
  - Information diffusion models (markov chains, SIR/SIS?)
- Understanding a network by modeling the network creation process is a key aspect:
  - For example, our user-user interaction network has a pareto distribution of node degrees
  - Can we model this network generation process? Can we simulate similar networks with a Barabási-Albert (BA) extended model or other?
  - Could it be modeled as a stochastic block model (SBM) or an extension like the dynamic SBM?
- Consider also modeling how network evolves over time (social network dynamics).

## Theoretical Foundation

Based on the literature, we're investigating "Evidence of equilibrium dynamics in human social networks evolving in time" - whether macroscopic stability emerges from microscopic stochasticity in the user-user interaction network obtained from the r/Documentaries subreddit.

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

---

## References

* [Complex Network Modelling with Power-law Activating Patterns and Its Evolutionary Dynamics](https://arxiv.org/abs/2502.09768)
* [Multiresolution Analysis and Statistical Thresholding on Dynamic Networks](https://arxiv.org/abs/2506.01208)
* [Continuous-time Graph Representation with Sequential Survival Process](https://browse.arxiv.org/html/2312.13068v1)
* [Motif-Based Visual Analysis of Dynamic Networks](https://export.arxiv.org/pdf/2208.11932v1)
* [Evidence of equilibrium dynamics in human social networks evolving in time](https://arxiv.org/abs/2410.11635)
* [Clustering time-evolving networks using the spatiotemporal graph Laplacian](https://ui.adsabs.harvard.edu/abs/2025Chaos..35a3126T/abstract)
* [Bursty Switching Dynamics Promotes the Collapse of Network Topologies](https://arxiv.org/abs/2505.12417)
* [Triadic balance and network evolution in predictive models of signed networks](https://www.nature.com/articles/s41598-024-85078-5?error=cookies_not_supported&code=182e36d1-0ff2-467f-acc2-5ff7baeb2d0f)
* [Reliable Time Prediction in the Markov Stochastic Block Model](https://hal.science/hal-02536727v2/file/msbm.pdf)
* [Temporal Dynamics of Coordinated Online Behavior: Stability, Archetypes, and Influence | AI Research Paper Details](https://aimodels.fyi/papers/arxiv/temporal-dynamics-coordinated-online-behavior-stability-archetypes)
* [A two-stage model leveraging friendship network for community evolution prediction in interactive networks](https://arxiv.org/abs/2503.15788)
* [Random walk based snapshot clustering for detecting community dynamics in temporal networks](https://arxiv.org/abs/2412.12187)
* [Community Shaping in the Digital Age: A Temporal Fusion Framework for Analyzing Discourse Fragmentation in Online - - Social Networks | AI Research Paper Details](https://aimodels.fyi/papers/arxiv/community-shaping-digital-age-temporal-fusion-framework)
* [Benchmarking Evolutionary Community Detection Algorithms in Dynamic Networks](https://browse.arxiv.org/html/2312.13784v1)
* [Detection of dynamic communities in temporal networks with sparse data](https://appliednetsci.springeropen.com/articles/10.1007/s41109-024-00687-3)
* [Temporal Dynamics of Coordinated Online Behavior: Stability, Archetypes, and Influence](https://arxiv.org/abs/2301.06774)
* [Evidence of equilibrium dynamics in human social networks evolving in time](https://arxiv.org/abs/2410.11635)
* [Modeling the duality of content niches and user interactions on online social media platforms](https://www.research-collection.ethz.ch/handle/20.500.11850/743046)
* [Here Be Livestreams: Trade-offs in Creating Temporal Maps of Reddit](https://browse.arxiv.org/html/2309.14259v2)

### Other links

* https://github.com/benedekrozemberczki/karateclub
* https://cdlib.readthedocs.io/en/v0.2.5/reference/cd_algorithms/algs/cdlib.algorithms.big_clam.html
* https://robromijnders.github.io/bigclam/
* https://notebook.community/sanja7s/SR_Twitter/src_FIN/BigClam
* https://karateclub.readthedocs.io/en/latest/_modules/karateclub/community_detection/overlapping/bigclam.html
* https://www.bibsonomy.org/bibtex/23bb0e53d728e55799d56e8c6049575c1/jaeschke
* https://hackmd.io/@deadwing97/rJcVp9NpI
* https://paperswithcode.com/paper/overlapping-community-detection-at-scale-a
* https://blog.csdn.net/weixin_57643648/article/details/123652501
