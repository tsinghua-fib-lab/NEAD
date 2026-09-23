import nd2py as nd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.metrics.snr_v4 import delta_snr


def auto_text_color(bgcolor):
    r, g, b = mcolors.to_rgb(bgcolor)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


def layout(node, tree, depth=0, x=0, spacing=1):
    children = tree[node]
    if not children:
        return {node: (x, -depth)}, x + spacing

    positions = {}
    child_x = x
    for child in children:
        child_pos, child_x = layout(child, tree, depth + 1, child_x, spacing)
        positions.update(child_pos)

    min_x = positions[children[0]][0]
    max_x = positions[children[-1]][0]
    positions[node] = ((min_x + max_x) / 2, -depth)
    return positions, child_x


def tree_plot(f, X, ax=None, fontsize=10, size=0.2):
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.CenteredNorm(2.18, halfrange=2.18)

    tree = {}
    labels = {}
    colors = {}
    digits = {}
    for node in f.iter_preorder():
        path = f.path_to(node)
        tree[path] = [path + (i,) for i in range(len(node.operands))]

        if isinstance(node, nd.Variable):
            name = node.name
        elif isinstance(node, nd.Number):
            name = f"{node.value:.2f}"
        else:
            name = node.__class__.__name__
        labels[path] = name
        digits_ = delta_snr(node, X, noise_level=1e-3, return_type="digits", eps=1e-3)
        digits[path] = digits_
        colors[path] = cmap(norm(digits_))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    pos, _ = layout(tuple(), tree)

    # edges
    for parent, children in tree.items():
        for child in children:
            x1, y1 = pos[parent]
            x2, y2 = pos[child]
            ax.plot([x1, x2], [y1, y2], color="black")

    # nodes
    for node, (x, y) in pos.items():
        circle = plt.Circle((x, y), size, color=colors[node], ec="none", zorder=100)
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            labels[node],
            ha="center",
            va="center",
            fontsize=fontsize,
            color=auto_text_color(colors[node]),
            zorder=101,
        )
        ax.text(
            x + size,
            y,
            f" {digits[node]:.1f}",
            ha="left",
            va="center",
            fontsize=fontsize,
            color="black",
            zorder=102,
        )

    ax.set_aspect("equal")
    ax.axis("off")
