import re

import matplotlib.pyplot as plt
import networkx as nx
import sympy
from matplotlib.patches import FancyBboxPatch

from .bipartite import get_bipartite, get_maximal_matching, get_min_vertex_cover
from .operators import SumOfProducts


def show_assigns(
    operators: SumOfProducts,
    W_assigns: list[list[int]],
    coef_site: list[int],
    scale: float = 0.8,
):
    G = nx.Graph()
    pos = {}
    ndim = operators.ndim
    group_nodes: list[set[str]] = [set() for _ in range(ndim)]
    height = 0
    max_height = 0
    coef_edge = []
    for i in range(ndim):
        height = 0
        for j, (assign, op) in enumerate(
            zip(W_assigns, operators, strict=True)
        ):
            name = f"${sympy.latex(op[i])}$[{assign[i]}]"
            if name in pos:
                pass
            else:
                pos[name] = (i + 1, height)
                height += 1
            G.add_node(name)
            group_nodes[i].add(name)
            if i < ndim - 1:
                edge = (
                    f"${sympy.latex(op[i])}$[{assign[i]}]",
                    f"${sympy.latex(op[i+1])}$[{assign[i+1]}]",
                )
                G.add_edge(*edge)
                if coef_site[j] == i + 1:
                    coef_edge.append(edge)
        max_height = max(max_height, height)

    # add terminal nodes
    name = r"$\hat{1}_{-1}$"
    G.add_node(name)
    pos[name] = (0, 0)
    for assign, op in zip(W_assigns, operators, strict=True):
        name_right = f"${sympy.latex(op[0])}$[{assign[0]}]"
        G.add_edge(name, name_right)

    name = r"$\hat{1}" + f"_{ndim}$"
    G.add_node(name)
    pos[name] = (ndim + 1, 0)
    for assign, op in zip(W_assigns, operators, strict=True):
        name_left = f"${sympy.latex(op[ndim-1])}$[{assign[ndim-1]}]"
        G.add_edge(name, name_left)

    for i in range(ndim):
        name = f"W{i}"
        G.add_node(name)
        pos[name] = (i + 1, -1)

    node_colors = []
    node_shapes = []
    for node in G.nodes():
        if re.match(r"W\d", node):
            node_colors.append("red")
            node_shapes.append("s")
        else:
            node_colors.append("lightblue")
            node_shapes.append("o")

    edge_colors = []
    for edge in G.edges():
        if edge in coef_edge:
            edge_colors.append("green")
        else:
            edge_colors.append("black")

    plt.figure(figsize=(ndim * scale + 2, max_height * scale))
    for node, shape in zip(G.nodes(), node_shapes, strict=False):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_color=[node_colors[list(G.nodes()).index(node)]],
            node_shape=shape,
            node_size=1000,
        )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=2,
    )

    nx.draw_networkx_labels(G, pos, font_size=8)

    for i in range(ndim):
        x_vals = [pos[node][0] for node in group_nodes[i]]
        y_vals = [pos[node][1] for node in group_nodes[i]]
        x_min, x_max = min(x_vals) - 0.2, max(x_vals) + 0.2
        y_min, y_max = min(y_vals) - 0.2, max(y_vals) + 0.2

        rect = FancyBboxPatch(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            boxstyle="round,pad=0.2",
            edgecolor="red",
            linewidth=2,
            facecolor="none",
        )
        plt.gca().add_patch(rect)
    plt.title("MPO assginment graph")
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def show_bipartite(U, V, E, retained_E=None):
    G = get_bipartite(U, V, E)
    G_latex = _renameG2latex(G)
    pos = {node: (0, i) for i, node in enumerate(U)}
    pos |= {node: (1, i) for i, node in enumerate(V)}
    pos_latex = _rename_pos2latex(pos)
    if retained_E is None:
        nx.draw(
            G_latex,
            pos_latex,
            with_labels=True,
            node_size=1000,
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
            edge_color="black",
            width=3,
        )
    else:
        retained_E = _renameE2latex(retained_E)
        E = retained_E.copy()
        nx.draw(
            G_latex,
            pos_latex,
            with_labels=True,
            node_size=1000,
            node_color="skyblue",
            font_size=10,
            font_weight="bold",
            edge_color="red",
            width=3,
            style="dashed",
        )
        for new_edge in retained_E:
            nx.draw_networkx_edges(
                G_latex,
                pos_latex,
                edgelist=[new_edge],
                edge_color="black",
                width=3,
            )
    plt.title("Bipartite graph")
    plt.show()
    return G, pos


def show_maximal_matching(
    G: nx.Graph, pos: dict[str, tuple[int, int]]
) -> dict[str, str]:
    M = get_maximal_matching(G)
    M_latex = {_node2latex(k): _node2latex(v) for k, v in M.items()}
    G_latex = _renameG2latex(G)
    pos_latex = _rename_pos2latex(pos)
    nx.draw(
        G_latex,
        pos_latex,
        with_labels=True,
        node_size=1000,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="black",
        width=3,
    )
    nx.draw_networkx_edges(
        G_latex, pos_latex, edgelist=M_latex.items(), edge_color="red", width=3
    )
    plt.title("max matching (red)")
    plt.show()
    return M


def show_min_vertex_cover(
    G: nx.Graph, pos: dict[str, tuple[int, int]], max_matching: dict[str, str]
) -> list[str]:
    C = get_min_vertex_cover(G, max_matching)
    G_latex = _renameG2latex(G)
    max_matching_latex = {
        _node2latex(k): _node2latex(v) for k, v in max_matching.items()
    }
    C_latex = [_node2latex(node) for node in C]
    pos_latex = _rename_pos2latex(pos)
    nx.draw(
        G_latex,
        pos_latex,
        with_labels=True,
        node_size=1000,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="black",
        width=3,
    )
    nx.draw_networkx_edges(
        G_latex,
        pos_latex,
        edgelist=max_matching_latex.items(),
        edge_color="red",
        width=3,
    )
    nx.draw_networkx_nodes(
        G_latex, pos_latex, nodelist=C_latex, node_color="green", node_size=1000
    )
    plt.title("minimum vertex cover (green) and max matching (red)")
    plt.show()
    return C


def _omit_eye_site(latex_symbol: str) -> str:
    latex = re.sub(r"\\hat\{1\}_[0-9]+", "", latex_symbol)
    if re.match(r"\$[ ]*\$", latex):
        if re.search(r"\\hat\{1\}_0", latex_symbol):
            return r"$\hat{1}_{\text{left}}$"
        else:
            return r"$\hat{1}_{\text{right}}$"
    return latex


def _node2latex(node: str) -> str:
    latex = f"${sympy.latex(sympy.simplify(node).expand())}$"
    return _omit_eye_site(latex)


def _renameG2latex(G: nx.Graph) -> nx.Graph:
    all_labels = G.nodes()
    mapping = dict(map(lambda x: (x, _node2latex(x)), all_labels))
    G_latex = nx.relabel_nodes(G, mapping)
    return G_latex


def _rename_pos2latex(
    pos: dict[str, tuple[int, int]],
) -> dict[str, tuple[int, int]]:
    pos_latex = {_node2latex(node): pos[node] for node in pos}
    return pos_latex


def _renameE2latex(E: list[tuple[str, str]]) -> list[tuple[str, str]]:
    E_latex = [(_node2latex(e[0]), _node2latex(e[1])) for e in E]
    return E_latex
