import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return F, mo, nn, np, nx, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classifying papers with a Graph Convolutional Network (GCN)

    The previous notebook clustered clients using *only* the graph
    structure — no node features, no labels, nothing but "who's connected
    to whom." A **Graph Convolutional Network** is a different animal:

    * It combines graph structure **with node features** — each node
      carries its own vector of information, not just an ID.
    * It's normally used **semi-supervised**: you know the true label for
      only a handful of nodes, and the model has to propagate that scant
      supervision through the graph to label everyone else.

    **Scenario:** a small citation network. Each node is a paper, each
    edge is a citation, and each paper has a "keyword frequency" feature
    vector. Papers cluster into three research topics — but we only know
    the topic for a few papers up front, the way a new citation database
    might have partial hand-labeling. Can the graph help us fill in the
    rest?

    **The GCN layer**, following Kipf & Welling's formulation, updates
    every node's representation by averaging its neighbors' (and its own)
    representation, then applying a learned linear transform and a
    non-linearity:

    $$H^{(l+1)} = \text{ReLU}\!\left(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

    where $\hat{A} = A + I$ adds a self-loop to every node (so a node's
    own features count too) and $\hat{D}$ is the degree matrix of
    $\hat{A}$. Stack two of these layers and you have a simple GCN.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Build a synthetic citation network

    Three hidden topics, sparse citations (papers cite a handful of
    others, mostly — but not only — within their own topic), and an
    8-dimensional feature vector per paper standing in for keyword
    frequencies. The sliders control how many papers have a known topic
    label, and how noisy those keyword features are.
    """)
    return


@app.cell
def _(mo):
    labels_slider = mo.ui.slider(1, 10, step=1, value=3, label="Labeled papers per topic")
    noise_slider = mo.ui.slider(0.5, 4.0, step=0.25, value=2.0, label="Feature noise (std dev)")
    mo.hstack([labels_slider, noise_slider])
    return labels_slider, noise_slider


@app.cell
def _(nx):
    topic_names = ["Computer Vision", "NLP", "Robotics"]
    group_sizes = [20, 20, 20]

    G = nx.random_partition_graph(group_sizes, 0.18, 0.02, seed=0)
    true_labels_dict = {n: G.nodes[n]["block"] for n in G.nodes}
    return G, group_sizes, topic_names, true_labels_dict


@app.cell
def _(G, np, true_labels_dict):
    feat_dim = 8

    true_labels = np.array([true_labels_dict[i] for i in range(G.number_of_nodes())])
    return feat_dim, true_labels


@app.cell
def _(feat_dim, np, noise_slider, true_labels):
    def make_features(labels, dim, noise_std, seed=0):
        rng = np.random.default_rng(seed)
        centers = rng.normal(0, 1.5, size=(3, dim))
        X = np.zeros((len(labels), dim))
        for i, lab in enumerate(labels):
            X[i] = centers[lab] + rng.normal(0, noise_std, size=dim)
        return X

    X = make_features(true_labels, feat_dim, noise_slider.value)
    return (X,)


@app.cell
def _(G, np, nx):
    def normalize_adjacency(graph):
        A = nx.to_numpy_array(graph)
        A_hat = A + np.eye(A.shape[0])
        degrees = A_hat.sum(axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    A_norm = normalize_adjacency(G)
    return (A_norm,)


@app.cell
def _(labels_slider, np, true_labels):
    def pick_labeled_nodes(labels, per_class, seed=1):
        rng = np.random.default_rng(seed)
        chosen = []
        for c in range(3):
            idx_c = np.where(labels == c)[0]
            k = min(per_class, len(idx_c))
            chosen.extend(rng.choice(idx_c, size=k, replace=False))
        mask = np.zeros(len(labels), dtype=bool)
        mask[chosen] = True
        return mask

    train_mask = pick_labeled_nodes(true_labels, labels_slider.value)
    return (train_mask,)


@app.cell(hide_code=True)
def _(G, feat_dim, mo, train_mask):
    mo.md(
        f"""
    Built a citation graph with **{G.number_of_nodes()} papers**,
    **{G.number_of_edges()} citations**, and **{feat_dim}**-dimensional
    keyword features per paper. Only **{int(train_mask.sum())} of
    {G.number_of_nodes()} papers** have a known topic label — the model
    only ever sees those during training, and has to infer the rest.
    """
    )
    return


@app.cell
def _(G, nx, plt, topic_names, train_mask, true_labels_dict):
    def draw_topic_graph(labels_by_node, title, ax=None, seed=7, k=0.4, mark_labeled=False):
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(6, 5))
        pos = nx.spring_layout(G, seed=seed, k=k)
        colors = [labels_by_node[n] for n in G.nodes]
        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
        if mark_labeled:
            labeled_nodes = [n for n in G.nodes if train_mask[n]]
            unlabeled_nodes = [n for n in G.nodes if not train_mask[n]]
            nx.draw_networkx_nodes(
                G, pos, nodelist=unlabeled_nodes,
                node_color=[labels_by_node[n] for n in unlabeled_nodes],
                cmap=plt.cm.Set2, node_size=180, ax=ax,
            )
            nx.draw_networkx_nodes(
                G, pos, nodelist=labeled_nodes,
                node_color=[labels_by_node[n] for n in labeled_nodes],
                cmap=plt.cm.Set2, node_size=260, edgecolors="black",
                linewidths=1.8, ax=ax,
            )
        else:
            nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.Set2, node_size=200, ax=ax)
        ax.set_title(title)
        ax.axis("off")
        return fig if own_fig else None

    fig_true = draw_topic_graph(
        true_labels_dict,
        "True topics — " + ", ".join(topic_names) + " (bold border = labeled paper)",
        mark_labeled=True,
    )
    fig_true
    return (draw_topic_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Library approach — PyTorch GCN vs. a plain MLP baseline

    Two tiny 2-layer networks, same architecture, same features, same
    handful of labeled papers to train on:

    * **GCN** — multiplies by the normalized adjacency at every layer, so
      each paper's representation is blended with its neighbors'.
    * **Plain MLP** — the same two linear layers, but *ignores the graph
      entirely* and classifies each paper from its own features alone.

    Both are trained only on the labeled papers; we then check both
    against the topics of the *unlabeled* papers, which we secretly know
    since this is a synthetic example.
    """)
    return


@app.cell
def _(A_norm, F, X, nn, torch, true_labels, train_mask):
    X_t = torch.tensor(X, dtype=torch.float32)
    A_t = torch.tensor(A_norm, dtype=torch.float32)
    y_t = torch.tensor(true_labels, dtype=torch.long)
    train_mask_t = torch.tensor(train_mask)

    class GCN(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.W1 = nn.Linear(in_dim, hidden_dim)
            self.W2 = nn.Linear(hidden_dim, out_dim)

        def forward(self, X, A):
            H = F.relu(A @ self.W1(X))
            return A @ self.W2(H)

    class MLPBaseline(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.W1 = nn.Linear(in_dim, hidden_dim)
            self.W2 = nn.Linear(hidden_dim, out_dim)

        def forward(self, X, A=None):
            H = F.relu(self.W1(X))
            return self.W2(H)

    def train_and_predict(model, epochs=200, lr=0.05):
        torch.manual_seed(0)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            out = model(X_t, A_t)
            loss = F.cross_entropy(out[train_mask_t], y_t[train_mask_t])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_t, A_t).argmax(dim=1)
        return pred.numpy()

    return GCN, MLPBaseline, train_and_predict


@app.cell
def _(GCN, MLPBaseline, feat_dim, train_and_predict):
    gcn_pred = train_and_predict(GCN(feat_dim, 8, 3))
    mlp_pred = train_and_predict(MLPBaseline(feat_dim, 8, 3))
    return gcn_pred, mlp_pred


@app.cell
def _(gcn_pred, mlp_pred, train_mask, true_labels):
    held_out = ~train_mask
    gcn_acc = (gcn_pred[held_out] == true_labels[held_out]).mean()
    mlp_acc = (mlp_pred[held_out] == true_labels[held_out]).mean()
    return gcn_acc, mlp_acc


@app.cell(hide_code=True)
def _(gcn_acc, mlp_acc, mo, train_mask):
    mo.md(
        f"""
    Checked against the **{int((~train_mask).sum())} unlabeled papers**:

    * **GCN accuracy: `{gcn_acc:.1%}`** — uses features *and* the citation graph
    * **Plain MLP accuracy: `{mlp_acc:.1%}`** — uses features only, ignoring the graph

    The gap is the graph doing real work: a paper's own keyword vector is
    noisy, but its neighbors are — thanks to the citation pattern we built
    in — usually the same topic, so averaging over them smooths out the
    noise. Try dragging the noise slider up in step 1 to widen the gap
    further, or turn it down until the MLP baseline catches up.
    """
    )
    return


@app.cell
def _(G, draw_topic_graph, gcn_pred, mlp_pred, plt, true_labels_dict):
    fig_compare, axes_compare = plt.subplots(1, 3, figsize=(17, 5))
    gcn_labels_dict = {n: int(gcn_pred[n]) for n in G.nodes}
    mlp_labels_dict = {n: int(mlp_pred[n]) for n in G.nodes}
    draw_topic_graph(true_labels_dict, "True topics", ax=axes_compare[0])
    draw_topic_graph(gcn_labels_dict, "GCN predictions", ax=axes_compare[1])
    draw_topic_graph(mlp_labels_dict, "MLP-only predictions", ax=axes_compare[2])
    plt.tight_layout()
    fig_compare
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. From scratch — NumPy only

    The GCN forward pass is just matrix multiplication, so backpropagation
    through it is ordinary chain-rule calculus — no different in kind from
    backprop through a plain MLP, just with an extra (constant) matrix
    multiply by $A_{norm} = \hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$ at each layer:

    $$Z^{(1)} = XW_1 + b_1, \quad H^{(1)}_{pre} = A_{norm} Z^{(1)}, \quad H^{(1)} = \text{ReLU}(H^{(1)}_{pre})$$
    $$Z^{(2)} = H^{(1)}W_2 + b_2, \quad \text{out} = A_{norm} Z^{(2)}, \quad \text{probs} = \text{softmax}(\text{out})$$

    Since $A_{norm}$ is a fixed matrix (not a learned parameter), the
    gradient just passes back through it via $A_{norm}^{\top}$ at each
    layer — everything else is the same softmax-cross-entropy and ReLU
    backprop as a standard 2-layer network.
    """)
    return


@app.cell
def _(A_norm, X, feat_dim, np, train_mask, true_labels):
    def one_hot(y, k):
        Y = np.zeros((len(y), k))
        Y[np.arange(len(y)), y] = 1
        return Y

    def softmax(Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def train_gcn_numpy(epochs=300, lr=0.1, hidden=8, classes=3, seed=1):
        rng = np.random.default_rng(seed)
        W1 = rng.normal(0, 0.5, size=(feat_dim, hidden))
        b1 = np.zeros(hidden)
        W2 = rng.normal(0, 0.5, size=(hidden, classes))
        b2 = np.zeros(classes)

        Y_onehot = one_hot(true_labels, classes)
        m = train_mask.sum()
        loss_history = []
        probs = None

        for _ in range(epochs):
            Z1 = X @ W1 + b1
            H1_pre = A_norm @ Z1
            H1 = np.maximum(H1_pre, 0)
            Z2 = H1 @ W2 + b2
            out_pre = A_norm @ Z2
            probs = softmax(out_pre)

            train_loss = -np.log(probs[train_mask, true_labels[train_mask]] + 1e-12).mean()
            loss_history.append(train_loss)

            dOut = (probs - Y_onehot) / m
            dOut[~train_mask] = 0.0

            dZ2 = A_norm.T @ dOut
            dW2 = H1.T @ dZ2
            db2 = dZ2.sum(axis=0)

            dH1 = dZ2 @ W2.T
            dH1_pre = dH1 * (H1_pre > 0)
            dZ1 = A_norm.T @ dH1_pre
            dW1 = X.T @ dZ1
            db1 = dZ1.sum(axis=0)

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        return probs.argmax(axis=1), loss_history

    scratch_pred, loss_history = train_gcn_numpy()
    return loss_history, scratch_pred


@app.cell
def _(loss_history, plt):
    fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
    ax_loss.plot(loss_history)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training loss (labeled papers only)")
    ax_loss.set_title("From-scratch NumPy GCN — training loss")
    fig_loss
    return


@app.cell
def _(scratch_pred, train_mask, true_labels):
    scratch_acc = (scratch_pred[~train_mask] == true_labels[~train_mask]).mean()
    return (scratch_acc,)


@app.cell(hide_code=True)
def _(mo, scratch_acc):
    mo.md(
        f"""
    **From-scratch NumPy GCN accuracy on unlabeled papers: `{scratch_acc:.1%}`** —
    both implementations run the exact same forward equations, so they
    should land in the same neighborhood as the PyTorch version above
    (small differences come from different random initializations and
    optimizers: plain gradient descent here vs. Adam there).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the result

    * The previous notebook's spectral clustering and Louvain community
      detection found groups using *only* topology, with no features and
      no labels at all — a fully unsupervised setting. This GCN sits in a
      different part of the graph ML landscape: it needs node features
      and a few known labels, but in exchange it can do something
      topology-only methods can't — *classify* new, specific categories
      rather than just discovering "some kind of grouping exists."
    * The GCN-vs-MLP gap in step 2 is the whole point of graph
      convolution: propagating information across edges lets a
      well-connected but ambiguous paper "borrow" signal from its
      neighbors, which a feature-only classifier can never do.
    * With very few labels (drag the slider in step 1 down to 1 per
      topic), the MLP baseline usually struggles badly, while the GCN
      typically holds up much better — this is the classic argument for
      GNNs in the low-label regime that citation-network benchmarks like
      Cora were originally built to demonstrate.
    * A real GCN library (e.g. PyTorch Geometric or DGL) would handle
      sparse adjacency matrices, mini-batching, and many more layer types
      — but the two-line propagation step in `GCN.forward` above is
      genuinely the core of it.
    """)
    return


if __name__ == "__main__":
    app.run()
