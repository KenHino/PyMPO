import networkx as nx
import sympy

import pympo

from .operators import SumOfProducts


def get_bipartite(U, V, E):
    G = nx.Graph()
    for u in U:
        G.add_node(u, bipartite=0)
    for v in V:
        G.add_node(v, bipartite=1)
    for e in E:
        G.add_edge(e[0], e[1])
    return G


def get_UVE(
    operator: SumOfProducts,
    W_assigns: list[list[int]],
    isite: int,
    Unew=None,
) -> tuple[list[str], list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    Usym = set()
    Vsym = set()
    Esym = set()
    E_assigns: list[tuple[str, str]] = [None for _ in range(operator.nops)]  # type: ignore
    ndim = operator.ndim
    for jop, prod_op in enumerate(operator.ops):
        if isite > 0:
            assert Unew is not None
            z_op = prod_op[isite]
            assign_left = W_assigns[jop][isite - 1]
            U_op = sympy.simplify(Unew[assign_left]) * z_op
        elif isite == 0:
            U_op = prod_op[0]
        else:
            raise ValueError("i must be greater than 0")
        V_op = prod_op[isite + 1 : ndim]
        Usym.add(U_op)
        Vsym.add(V_op)
        Esym.add((U_op, V_op))
        E_assigns[jop] = (sympy.srepr(U_op), sympy.srepr(V_op))
    U = [sympy.srepr(node) for node in Usym]
    V = [sympy.srepr(node) for node in Vsym]
    E = [(sympy.srepr(edge[0]), sympy.srepr(edge[1])) for edge in Esym]
    assert all([E_assign is not None for E_assign in E_assigns])
    return U, V, E, E_assigns


def get_maximal_matching(G: nx.Graph) -> dict[str, str]:
    assert nx.is_bipartite(G)
    # nx.bipartite.maximum_matching(G) cannot work when graph is disconnected
    # thus, before execution, we split the graph into connected components
    M = {}
    for component in nx.connected_components(G):
        G_sub = G.subgraph(component)
        assert nx.is_bipartite(G_sub)
        assert nx.is_connected(G_sub)
        M.update(nx.bipartite.maximum_matching(G_sub))
    return M


def get_min_vertex_cover(
    G: nx.Graph, max_matching: dict[str, str]
) -> list[str]:
    assert nx.is_bipartite(G)
    # nx.bipartite.to_vertex_cover(G, max_matching) cannot work when graph is disconnected
    # thus, before execution, we split the graph into connected components
    C = set()
    for component in nx.connected_components(G):
        G_sub = G.subgraph(component)
        assert nx.is_bipartite(G_sub)
        assert nx.is_connected(G_sub)
        M_sub = {u: v for u, v in max_matching.items() if u in G_sub}
        C_sub = nx.bipartite.to_vertex_cover(G_sub, M_sub)
        C.update(C_sub)

    return list(C)


def assign_core(
    *,
    min_vertex_cover,
    U,
    V,
    E,
    operators: SumOfProducts,
    isite,
    W_assigns,
    E_assigns,
    coef_site,
    visualize: bool = True,
):
    Unew: list[str] = []  # U[1..i]
    unique_ops = []
    for j, vertex in enumerate(min_vertex_cover):
        if vertex in U:
            Unew.append(vertex)
            # remove the edge connected to the vertex
            retained_E = []
            remove_E = []
            for edge in E:
                if vertex != edge[0]:
                    retained_E.append(edge)
                else:
                    remove_E.append(edge)
            represent_ops = {}
            for k, op in enumerate(operators.ops):
                if E_assigns[k] in remove_E:
                    opsite = op[isite]
                    W_assigns[k][isite] = j
                    if opsite not in represent_ops:
                        unique_ops.append(k)
                        represent_ops[opsite] = k
        else:
            assert vertex in V, f"{vertex=} is not in {V=}"
            retained_E = []
            remove_E = []
            vertex_U_concat = 0
            for edge in E:
                if vertex != edge[1]:
                    retained_E.append(edge)
                else:
                    remove_E.append(edge)
            represent_ops = {}
            for k, op in enumerate(operators.ops):
                if E_assigns[k] in remove_E:
                    W_assigns[k][isite] = j
                    opsite = op[isite]
                    if opsite not in represent_ops:
                        unique_ops.append(k)
                        represent_ops[opsite] = k
                    elif coef_site[k] >= isite:
                        unique_ops.append(k)

                    if coef_site[k] < isite:
                        vertex_U_concat += op[0 : isite + 1]
                    else:
                        coef_site[k] = isite
                        vertex_U_concat += op.coef * op[0 : isite + 1]
            Unew.append(sympy.srepr(vertex_U_concat))
        if visualize:
            pympo.visualize.show_bipartite(U, V, E, retained_E)
        E = retained_E
    if visualize:
        pympo.visualize.show_assigns(operators, W_assigns, coef_site)
    if isite == 0:
        left_dim = 1
    else:
        left_dim = max([assign[isite - 1] for assign in W_assigns]) + 1
    if isite == operators.ndim - 1:
        right_dim = 1
    else:
        right_dim = len(Unew)
        right_dim_debug = max([assign[isite] for assign in W_assigns]) + 1
        assert (
            right_dim == right_dim_debug
        ), f"{right_dim} != {right_dim_debug}, {[assign[isite] for assign in W_assigns]}"
    Wi = sympy.zeros(left_dim, right_dim)
    for k in unique_ops:
        op = operators.ops[k]
        if isite == 0:
            left_index = 0
        else:
            left_index = W_assigns[k][isite - 1]
        if isite == operators.ndim - 1:
            right_index = 0
        else:
            right_index = W_assigns[k][isite]
        # there are three cases
        # 1. W[a,b] = x_i
        # 2. W[a,b] += x_i
        # 3. W[a,b] += coef * x_i
        if coef_site[k] == isite:
            # case 3
            Wi[left_index, right_index] += op.coef * op[isite]
        elif coef_site[k] < isite:
            Wi[left_index, right_index] += op[isite]
        else:
            # case 1
            if Wi[left_index, right_index] == 0:
                Wi[left_index, right_index] = op[isite]
            else:
                assert (
                    Wi[left_index, right_index] == op[isite]
                ), f"{Wi[left_index, right_index]=} while {op[isite]=} when {coef_site[k]=}, {isite=}"
    return Unew, Wi
