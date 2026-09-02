import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", css_file="custom.css")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Session 2: Graph base methods

    The core idea is the data is structured, in traditional tabular data each row is assumed to be independent, but in some applications the information between points also has information. Graph based methods use the node connections as information.

    In this notebook:

    1. Toy dataset of clients of some mock company, connected by a continuous, multi-attribute interaction strength (not just "connected or not")
    2. Spectral clustering to group said clients
    3. Numpy implementation of spectral clustering to learn inner working
    4. Louvain community method
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import adjusted_rand_score

    return SpectralClustering, adjusted_rand_score, mo, np, nx, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Grouping clients by relationship structure

    A mid-size consultancy has ~35 client accounts. For every pair of clients,
    account managers and CRM data give us a **strength of relationship** —
    not just "connected or not" but *how* connected: how many projects they
    share, how much combined contract value flows between them, how often
    the teams communicate. We don't know in advance which "sector" each
    client belongs to — we only see this weighted interaction graph.

    The goal: **recover the natural groups of clients from graph structure
    alone**, using spectral graph clustering on a continuous, multi-attribute
    edge signal instead of a plain yes/no adjacency.

    1. Build a synthetic-but-realistic **weighted** client graph, where each
       edge carries several raw attributes that get combined into one
       real-valued interaction-strength weight (so we secretly know the
       "true" groups and can grade ourselves).
    2. Cluster it with a library (`scikit-learn`'s `SpectralClustering`),
       passing the continuous weight matrix straight in as the affinity.
    3. Re-implement the same algorithm from scratch with plain NumPy, so
       you can see it works internally.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Build the mock client interaction graph

    Three hidden sectors — **Retail**, **Manufacturing**, **Tech
    Services** — but this time every pair of clients gets a **continuous**
    interaction-strength edge weight in `[0, 1]`, built from three raw
    per-relationship signals:

    - **Shared projects** — count of joint engagements
    - **Contract value** — combined contract value, in $M
    - **Communication score** — how frequently the account teams talk, in `[0, 1]`

    Same-sector pairs tend to score higher on all three; cross-sector pairs
    tend to score lower — but it's noisy, and the mapping from "three raw
    numbers" to "one edge weight" is exactly the kind of design decision you
    have to make before handing a graph to spectral clustering or Louvain.
    """)
    return


@app.cell
def _(mo):
    separation_slider = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.85, label="Sector separation (signal strength)"
    )
    noise_slider = mo.ui.slider(
        0.0, 1.5, step=0.05, value=0.4, label="Edge-weight noise (std dev)"
    )
    mo.hstack([separation_slider, noise_slider])
    return noise_slider, separation_slider


@app.cell
def _(np, separation_slider):
    sector_names = ["Retail", "Manufacturing", "Tech Services"]
    group_sizes = [12, 10, 14]
    n_nodes = sum(group_sizes)
    block = np.repeat(np.arange(len(group_sizes)), group_sizes)
    same_sector = block[:, None] == block[None, :]

    sep = separation_slider.value
    same_lambda, diff_lambda_base = 5.0, 0.5
    same_mu, diff_mu_base = 1.0, -1.0
    contract_sigma = 0.4
    same_a, same_b = 6.0, 2.0
    diff_a_base, diff_b_base = 2.0, 6.0

    diff_lambda = same_lambda - sep * (same_lambda - diff_lambda_base)
    diff_mu = same_mu - sep * (same_mu - diff_mu_base)
    diff_a = same_a - sep * (same_a - diff_a_base)
    diff_b = same_b - sep * (same_b - diff_b_base)

    lam_matrix = np.where(same_sector, same_lambda, diff_lambda)
    mu_matrix = np.where(same_sector, same_mu, diff_mu)
    a_matrix = np.where(same_sector, same_a, diff_a)
    b_matrix = np.where(same_sector, same_b, diff_b)

    rng = np.random.default_rng(42)
    shared_projects = rng.poisson(lam_matrix).astype(float)
    contract_value = rng.lognormal(mean=mu_matrix, sigma=contract_sigma)
    comm_score = rng.beta(a_matrix, b_matrix)

    _iu = np.triu_indices(n_nodes, k=1)
    for _mat in (shared_projects, contract_value, comm_score):
        _mat[(_iu[1], _iu[0])] = _mat[_iu]
        np.fill_diagonal(_mat, 0.0)
    return (
        block,
        comm_score,
        contract_value,
        group_sizes,
        n_nodes,
        sector_names,
        shared_projects,
    )


@app.cell
def _(comm_score, contract_value, n_nodes, noise_slider, np, shared_projects):
    def _zscore(x, mask):
        vals = x[mask]
        mu, sigma = vals.mean(), vals.std()
        sigma = sigma if sigma > 0 else 1.0
        return (x - mu) / sigma

    _iu = np.triu_indices(n_nodes, k=1)
    _mask = np.zeros_like(shared_projects, dtype=bool)
    _mask[_iu] = True

    # compute the value of the interactions between clients
    z_projects = _zscore(shared_projects, _mask)
    z_contract = _zscore(np.log1p(contract_value), _mask)
    z_comm = _zscore(comm_score, _mask)

    # noise
    noise_rng = np.random.default_rng(7)
    noise = noise_rng.normal(0.0, noise_slider.value, size=shared_projects.shape)
    noise[(_iu[1], _iu[0])] = noise[_iu]
    np.fill_diagonal(noise, 0.0)

    # combine all the interaction scores as a single weight
    combined = 0.45 * z_projects + 0.35 * z_contract + 0.20 * z_comm + noise
    weight = 1.0 / (1.0 + np.exp(-combined))
    np.fill_diagonal(weight, 0.0)
    return (weight,)


@app.cell
def _(block, comm_score, contract_value, n_nodes, nx, shared_projects, weight):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        G.nodes[i]["block"] = int(block[i])
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            G.add_edge(
                i,
                j,
                weight=float(weight[i, j]),
                shared_projects=float(shared_projects[i, j]),
                contract_value=float(contract_value[i, j]),
                comm_score=float(comm_score[i, j]),
            )
    true_labels_dict = {node: G.nodes[node]["block"] for node in G.nodes}
    return G, true_labels_dict


@app.cell
def _(G, np, nx, true_labels_dict):
    adjacency = nx.to_numpy_array(G, weight="weight")
    true_labels = np.array([true_labels_dict[node] for node in G.nodes])
    return adjacency, true_labels


@app.cell(hide_code=True)
def _(G, mo):
    mo.md(f"""
    Generated a weighted interaction graph over **{G.number_of_nodes()} clients**.
    Every pair of clients has a real-valued interaction-strength edge weight in
    `[0, 1]` built from three underlying signals (shared projects, combined
    contract value, communication frequency) — **{sum(1 for _, _, d in G.edges(data=True) if d["weight"] >= 0.5)}**
    of the **{G.number_of_edges()}** pairs cross the 0.5 "strong interaction"
    threshold. Sector membership is hidden from the clustering algorithms —
    they only ever see edge weights like these.
    """)
    return


@app.cell
def _(block, comm_score, contract_value, n_nodes, np, plt, shared_projects):
    _iu = np.triu_indices(n_nodes, k=1)
    _same_mask = (block[:, None] == block[None, :])[_iu]

    fig_feat, axes_feat = plt.subplots(1, 3, figsize=(14, 3.5))
    for _ax, _values, _name in zip(
        axes_feat,
        [shared_projects[_iu], contract_value[_iu], comm_score[_iu]],
        ["Shared projects", "Combined contract value ($M)", "Communication score"],
    ):
        _ax.hist(_values[_same_mask], bins=20, alpha=0.6, label="same sector", color="tab:blue")
        _ax.hist(_values[~_same_mask], bins=20, alpha=0.6, label="different sector", color="tab:orange")
        _ax.set_title(_name, fontsize=10)
        _ax.legend(fontsize=8)
    plt.tight_layout()
    fig_feat
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each raw signal above only *weakly* separates same-sector from
    cross-sector pairs on its own — that's realistic, real signals are
    noisy. To turn these three numbers into one edge weight we:

    1. **Standardize** each raw signal (z-score) so they're on comparable
       scales.
    2. Take a **weighted sum** (0.45 · shared projects + 0.35 · contract
       value + 0.20 · communication score) plus some Gaussian noise
       (the noise slider above).
    3. Squash it through a **sigmoid** to land in `[0, 1]`.

    This is the key modeling step for graph algorithms: `SpectralClustering`,
    Louvain, and our NumPy version below never see the three raw attributes —
    they only ever see this single combined weight per edge. Whatever
    domain knowledge or feature engineering goes into building that scalar
    (or a learned model that produces it) is where you inject the
    information; the graph algorithm itself is agnostic to where the number
    came from.
    """)
    return


@app.cell
def _(G, nx, plt, sector_names, true_labels_dict):
    def draw_graph(labels_by_node, title, ax=None, seed=7, k=0.6, edge_threshold=0.5):
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(6, 5))
        pos = nx.spring_layout(G, seed=seed, k=k, weight="weight")
        colors = [labels_by_node[n] for n in G.nodes]
        color_dict = {0: 'red', 1: 'blue', 2:'yellow'}
        # nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.Set1, node_size=150, ax=ax)
        nx.draw_networkx_nodes(G, pos, node_color=[color_dict[c] for c in colors], node_size=150, ax=ax)

        strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= edge_threshold]
        edge_weights = [G[u][v]["weight"] for u, v in strong_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=strong_edges, width=[w * 2.5 for w in edge_weights], alpha=0.35, ax=ax,
            edge_color='white',
        )
        ax.set_title(title)
        ax.axis("off")
        return fig if own_fig else None

    fig_true = draw_graph(
        true_labels_dict,
        "Client graph — true (hidden) sectors: " + ", ".join(sector_names),
    )
    fig_true
    return (draw_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Library approach — `scikit-learn` `SpectralClustering`

    We hand `SpectralClustering` the weight matrix directly
    (`affinity="precomputed"`) and ask for 3 clusters. "Precomputed" just
    means "trust these numbers as pairwise similarity" — it doesn't care
    whether they're 0/1 or continuous, which is exactly what lets it
    consume our derived interaction-strength weights with no extra
    transformation. Internally it builds the graph Laplacian, takes its
    smallest eigenvectors, and runs k-means on the resulting embedding —
    exactly what we build by hand in step 3.
    """)
    return


@app.cell
def _(adjacency, plt):
    # explore numpy array representation of graph

    plt.matshow(adjacency)
    plt.colorbar()
    plt.title("Weighted adjacency")
    plt.show()
    return


@app.cell
def _(
    SpectralClustering,
    adjacency,
    adjusted_rand_score,
    group_sizes,
    true_labels,
):
    k = len(group_sizes)

    sklearn_model = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    library_labels = sklearn_model.fit_predict(adjacency)
    library_ari = adjusted_rand_score(true_labels, library_labels)

    print('Sklearn based ARI:', library_ari)
    return k, library_ari, library_labels


@app.cell(hide_code=True)
def _(library_ari, mo):
    mo.md(f"""
    **Adjusted Rand Index vs. true sectors: `{library_ari:.3f}`**
    (1.0 = perfect recovery, 0.0 = random guessing.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. From scratch — NumPy only

    Spectral clustering in four steps, no library beyond NumPy for the
    linear algebra — and nothing here changes whether `A` is a binary
    0/1 adjacency or, like ours, a continuous weight matrix:

    1. **Degree matrix** $D$ and **adjacency** $A$ → **normalized
       Laplacian** $L_{sym} = I - D^{-1/2} A D^{-1/2}$.
    2. **Eigendecomposition** of $L_{sym}$ — it's symmetric, so
       `np.linalg.eigh` gives real eigenvalues/eigenvectors sorted ascending.
    3. Take the $k$ eigenvectors with the **smallest** eigenvalues (the
       "smoothest" signals on the graph) and stack them as columns → an
       $n \times k$ embedding where clients in the same tightly-connected
       group land close together.
    4. Row-normalize that embedding and run **k-means**, implemented here
       with a plain NumPy loop (init centroids, assign, recompute, repeat).
    """)
    return


@app.cell
def _(adjacency, k, np):
    def normalized_laplacian(A):
        d = A.sum(axis=1)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    def kmeans_numpy(X, k, seed=42, n_iter=100):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        centroids = X[rng.choice(n, size=k, replace=False)]
        assignments = np.zeros(n, dtype=int)
        for _ in range(n_iter):
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            new_centroids = np.array(
                [
                    X[assignments == c].mean(axis=0) if np.any(assignments == c) else centroids[c]
                    for c in range(k)
                ]
            )
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        return assignments

    def spectral_clustering_numpy(A, k, seed=42):
        L_sym = normalized_laplacian(A)
        eigvals, eigvecs = np.linalg.eigh(L_sym)  # ascending order
        U = eigvecs[:, :k]
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        U_norm = U / norms
        return kmeans_numpy(U_norm, k, seed=seed), eigvals

    scratch_labels, eigvals = spectral_clustering_numpy(adjacency, k, seed=42)
    return (scratch_labels,)


@app.cell
def _(adjusted_rand_score, scratch_labels, true_labels):
    scratch_ari = adjusted_rand_score(true_labels, scratch_labels)
    print('From scratch ARI:', scratch_ari)
    return (scratch_ari,)


@app.cell(hide_code=True)
def _(mo, scratch_ari):
    mo.md(f"""
    **From-scratch NumPy ARI vs. true sectors: `{scratch_ari:.3f}`** —
    compare this to the `scikit-learn` score above; they should match
    closely, since it's the same algorithm.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Side-by-side: true sectors vs. recovered clusters
    """)
    return


@app.cell
def _(G, draw_graph, library_labels, plt, scratch_labels):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels_a = {n: int(library_labels[n]) for n in G.nodes}
    labels_b = {n: int(scratch_labels[n]) for n in G.nodes}
    draw_graph(labels_a, "scikit-learn SpectralClustering", ax=axes[0])
    draw_graph(labels_b, "From-scratch NumPy spectral clustering", ax=axes[1])
    plt.tight_layout()
    fig
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. A different algorithm — Louvain community detection

    Spectral clustering needed us to say "there are 3 clusters" up front.
    **Louvain** community detection doesn't: it greedily merges nodes into
    communities to maximize *modularity* — how much denser the connections
    are within a community than you'd expect from a random graph with the
    same degree distribution — and stops on its own once no merge improves
    the score further. With a weighted graph, "denser" means "higher total
    edge weight," so Louvain uses our interaction-strength scores directly.
    It's the standard go-to for very large real-world networks (social
    graphs, citation networks) precisely because you usually don't know `k`
    in advance.
    """)
    return


@app.cell
def _(G, adjusted_rand_score, np, nx, true_labels):
    communities = nx.community.louvain_communities(G, weight="weight", seed=42)

    louvain_labels_dict = {}
    for community_id, community_nodes in enumerate(communities):
        for node in community_nodes:
            louvain_labels_dict[node] = community_id
    louvain_labels = np.array([louvain_labels_dict[n] for n in G.nodes])

    modularity = nx.community.modularity(G, communities, weight="weight")
    louvain_ari = adjusted_rand_score(true_labels, louvain_labels)
    print('Louvain ARI', louvain_ari)
    return communities, louvain_ari, louvain_labels_dict, modularity


@app.cell(hide_code=True)
def _(communities, louvain_ari, mo, modularity):
    mo.md(f"""
    Louvain found **{len(communities)} communities** on its own — nobody
    told it there were 3 sectors.

    **Modularity: `{modularity:.3f}`** (higher is better; above roughly 0.3
    is usually considered strong community structure).

    **Adjusted Rand Index vs. true sectors: `{louvain_ari:.3f}`**
    """)
    return


@app.cell
def _(draw_graph, louvain_labels_dict):
    fig_louvain = draw_graph(
        louvain_labels_dict,
        "Louvain communities (k not specified)",
    )
    fig_louvain
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. All three methods, side by side
    """)
    return


@app.cell
def _(G, draw_graph, library_labels, louvain_labels_dict, plt, scratch_labels):
    fig_all, axes_all = plt.subplots(1, 3, figsize=(17, 5))
    labels_lib = {n: int(library_labels[n]) for n in G.nodes}
    labels_scratch = {n: int(scratch_labels[n]) for n in G.nodes}
    draw_graph(labels_lib, "Spectral (scikit-learn)", ax=axes_all[0])
    draw_graph(labels_scratch, "Spectral (from-scratch NumPy)", ax=axes_all[1])
    draw_graph(louvain_labels_dict, "Louvain (no k needed)", ax=axes_all[2])
    plt.tight_layout()
    fig_all
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the result

    * All three methods recover the same three clusters purely from *how
      strongly* clients interact — no separate client attributes, no
      manual labeling, just the combined edge weight.
    * The two spectral methods need `k` specified up front; Louvain
      discovers the number of communities on its own by optimizing
      weighted modularity. That's the main practical trade-off between the
      two families: spectral clustering gives you a reusable,
      well-understood embedding and fine control over `k`, while Louvain
      is more hands-off and scales comfortably to much larger graphs.
    * Drag the sliders in step 1 (lower the separation, raise the noise) —
      past a point the sectors blur together, the Adjusted Rand Index for
      all three methods drops toward 0, and Louvain will typically start
      finding a different number of communities than 3. That's the graph
      losing enough structure for "sector" to stop being a meaningful
      concept from connectivity alone.
    * Every algorithm here only ever sees **one number per edge** — the
      combined interaction-strength weight. The three raw signals (shared
      projects, contract value, communication score) never reach
      `SpectralClustering` or Louvain directly; turning multi-attribute
      edge data into that single scalar (here: z-score, weighted sum,
      sigmoid) is where the modeling judgment lives. Swap in a different
      combination — or a learned one, fit against a proxy outcome — and
      every downstream algorithm keeps working unchanged, because all it
      ever required was a valid weight matrix. (A model that consumes the
      raw multi-dimensional edge vectors directly, instead of a
      hand-built scalar, is a different family of tools — e.g. a GNN edge
      encoder — beyond what we cover here.)
    * In a real deployment you wouldn't have `true_labels` — you'd
      validate cluster quality with things like modularity (as computed
      above), silhouette score on the spectral embedding, or by having
      account managers sanity-check a few clusters by hand.
    """)
    return


if __name__ == "__main__":
    app.run()
