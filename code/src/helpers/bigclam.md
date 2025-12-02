The **BigClam** algorithm is used for **overlapping community detection** in graphs. It uses block coordinate gradient ascent to learn a low-rank, non-negative matrix factorization that represents the strength of a node's affiliation to different communities, maximizing the likelihood of the observed network structure.

Here is a breakdown of its usage based on the provided documentation:

## BigClam usage

To use the `BigClam` algorithm, you generally follow three steps: initialize the model, fit it to a graph, and retrieve the results (embedding or memberships).

### 1\. Model Initialization

You create an instance of the `BigClam` class, specifying parameters to control the embedding process:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`dimensions`** | `int` | `8` | Number of embedding dimensions (which corresponds to the number of communities/clusters). |
| **`iterations`** | `int` | `50` | Number of training iterations (gradient ascent steps). |
| **`learning_rate`** | `float` | `0.005` | Gradient ascent learning rate. |
| **`seed`** | `int` | `42` | Random seed value for reproducibility. |

**Example Initialization:**

```python
from karateclub.community import BigClam
import networkx as nx

# Initialize the model with custom parameters
model = BigClam(dimensions=10, iterations=100, learning_rate=0.01)

# Or, use all default parameters
# model = BigClam()
```

-----

### 2\. Model Fitting

The model is trained using the `fit` method, which requires a **NetworkX graph** object.

| Argument | Type | Description |
| :--- | :--- | :--- |
| **`graph`** | `networkx.classes.graph.Graph` | The graph to be clustered. |

**Example Fitting:**

```python
# Create a sample NetworkX graph
graph = nx.gnm_random_graph(n=50, m=100, seed=42)

# Fit the model to the graph
model.fit(graph)
```

-----

### 3\. Retrieving Results

After fitting the model, the two primary outputs can be extracted: the node embeddings and the cluster memberships.

#### Node Embedding

The `get_embedding` method returns the factorization matrix, where each row is a node's embedding, and each column corresponds to a community.

  * **Return Type:** `numpy.array`
  * **Description:** The embedding of nodes. This matrix reflects the **node-community affiliation strengths**, where embedding[i, j] is the affiliation strength of node i to community j.

```python
embedding = model.get_embedding()
```

#### Cluster Memberships

The `get_memberships` method converts the affiliation strengths into discrete community assignments.

  * **Return Type:** `Dict[int, int]` (Dictionary of lists, as it is an overlapping method)
  * **Description:** Node cluster memberships. This is a dictionary where keys are the node IDs, and the value is a **list of cluster IDs** the node belongs to.

```python
memberships = model.get_memberships()
```
