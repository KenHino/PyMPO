import networkx as nx
import numpy as np
import sympy
from loguru import logger
from numpy.typing import NDArray

import pympo

from .operators import SumOfProducts


class AssignManager:
    def __init__(
        self,
        operator: SumOfProducts,
    ) -> None:
        self.operator = operator
        self.ndim = operator.ndim
        self.nops = operator.nops
        self.W_assigns: list[list[int]] = [
            [k for _ in range(self.ndim)] for k in range(self.nops)
        ]
        self.coef_site: list[int] = [self.ndim - 1 for _ in range(self.nops)]
        self.unique_ops: list[list[int]] = [
            [iops for iops in range(self.nops)] for _ in range(self.ndim)
        ]
        self.Wsym: list[sympy.Matrix] = self.reset_Wsym()

    def _get_bond_dim(self, isite: int) -> tuple[int, int]:
        if isite == 0:
            shape = (1, max([assign[0] for assign in self.W_assigns]) + 1)
        elif isite == self.ndim - 1:
            shape = (
                max([assign[isite - 1] for assign in self.W_assigns]) + 1,
                1,
            )
        else:
            shape = (
                max([assign[isite - 1] for assign in self.W_assigns]) + 1,
                max([assign[isite] for assign in self.W_assigns]) + 1,
            )
        return shape

    def _get_bond_index(self, isite: int, kops: int) -> tuple[int, int]:
        if isite == 0:
            left_index = 0
        else:
            left_index = self.W_assigns[kops][isite - 1]
        if isite == self.ndim - 1:
            right_index = 0
        else:
            right_index = self.W_assigns[kops][isite]
        return left_index, right_index

    def reset_core(self, isite: int) -> sympy.Matrix:
        shape = self._get_bond_dim(isite)
        core = sympy.zeros(*shape)
        for k in self.unique_ops[isite]:
            op = self.operator.ops[k]
            left_index, right_index = self._get_bond_index(isite, k)
            # there are three cases
            # 1. W[a,b] = x_i
            # 2. W[a,b] += x_i
            # 3. W[a,b] += coef * x_i
            if sympy.latex(op[isite]) == r"\hat{1}" + f"_{isite}":
                opisite = 1
            else:
                opisite = op[isite]
            if self.coef_site[k] == isite:
                # case 3
                core[left_index, right_index] += op.coef * opisite
            elif self.coef_site[k] < isite:
                core[left_index, right_index] += opisite
            else:
                # case 1
                if core[left_index, right_index] == 0:
                    core[left_index, right_index] = opisite
                else:
                    assert (
                        core[left_index, right_index] == opisite
                    ), f"{core[left_index, right_index]=} while {op[isite]=} when {self.coef_site[k]=}, {isite=}"
        return core

    def reset_Wsym(self) -> list[sympy.Matrix]:
        Wsym = []
        for isite in range(self.ndim):
            Wsym.append(self.reset_core(isite))
        return Wsym

    def _get_UVE(
        self,
        isite: int,
        Unew: list[str] | None = None,
        keep_symbol: bool = False,
    ) -> tuple[set[str], set[str], set[tuple[str, str]], list[tuple[str, str]]]:
        Uset = set()
        Vset = set()
        Eset = set()
        E_assigns: list[tuple[str, str]] = [None for _ in range(self.nops)]  # type: ignore
        ndim = self.ndim
        for jop, prod_op in enumerate(self.operator.ops):
            if isite > 0:
                assert Unew is not None
                z_op = prod_op[isite]
                assign_left = self.W_assigns[jop][isite - 1]
                # This may require too much time
                if keep_symbol:
                    U_op = sympy.simplify(Unew[assign_left]) * z_op
                else:
                    U_op = Unew[assign_left] + "*" + str(z_op)
            elif isite == 0:
                if keep_symbol:
                    U_op = prod_op[0]
                else:
                    U_op = str(prod_op[0])
            else:
                raise ValueError("i must be greater than 0")
            if keep_symbol:
                V_op = prod_op[isite + 1 : ndim]
            else:
                V_op = str(prod_op[isite + 1 : ndim])
            if keep_symbol:
                U_repr = sympy.srepr(U_op)
                V_repr = sympy.srepr(V_op)
            else:
                U_repr = str(U_op)
                V_repr = str(V_op)
            Uset.add(U_repr)
            Vset.add(V_repr)
            Eset.add((U_repr, V_repr))
            E_assigns[jop] = (U_repr, V_repr)
        # U = [sympy.srepr(node) for node in Usym]
        # V = [sympy.srepr(node) for node in Vsym]
        # E = [(sympy.srepr(edge[0]), sympy.srepr(edge[1])) for edge in Esym]
        # U = list(Uset)
        # V = list(Vset)
        # E = [(edge[0], edge[1]) for edge in Eset]
        assert all([E_assign is not None for E_assign in E_assigns])
        return Uset, Vset, Eset, E_assigns

    def _update(
        self,
        isite: int,
        Unew: list[str] | None = None,
        keep_symbol: bool = False,
    ) -> list[str]:
        U, V, E, E_assigns = self._get_UVE(isite, Unew, keep_symbol)
        G = get_bipartite(U, V, E)
        max_matching = get_maximal_matching(G)
        min_vertex_cover = get_min_vertex_cover(G, max_matching)
        Unew = []
        self.unique_ops[isite] = []
        update_coef_ops = []
        # ops_list = [(k, op) for k, op in enumerate(self.operator.ops)]
        for j, vertex in enumerate(min_vertex_cover):
            if vertex in U:
                # U.remove(vertex)
                Unew.append(vertex)
                retained_E = []
                remove_E = []
                for edge in E:
                    if vertex != edge[0]:
                        retained_E.append(edge)
                    else:
                        remove_E.append(edge)
                represent_ops = {}
                # remained_ops_list = []
                for k, op in enumerate(self.operator.ops):
                    # while ops_list:
                    # k, op = ops_list.pop()
                    if E_assigns[k] in remove_E:
                        opsite = op[isite]
                        self.W_assigns[k][isite] = j
                        if opsite not in represent_ops:
                            self.unique_ops[isite].append(k)
                            represent_ops[opsite] = k
                    # else:
                    # remained_ops_list.append((k, op))
            else:
                assert vertex in V, f"{vertex=} is not in {V=}"
                # V.remove(vertex)
                retained_E = []
                remove_E = []
                vertex_U_concat = 0
                for edge in E:
                    if vertex != edge[1]:
                        retained_E.append(edge)
                    else:
                        remove_E.append(edge)
                represent_ops = {}
                # remained_ops_list = []
                for k, op in enumerate(self.operator.ops):
                    # while ops_list:
                    # k, op = ops_list.pop()
                    if E_assigns[k] in remove_E:
                        self.W_assigns[k][isite] = j
                        opsite = op[isite]
                        if self.coef_site[k] >= isite:
                            # isiteでcoefが複数回登場する場合にバグを引き起こす
                            self.unique_ops[isite].append(k)
                        elif opsite not in represent_ops:
                            # isiteでcoefが必要な列と不要な列が両方ある場合にバグを引き起こす
                            self.unique_ops[isite].append(k)
                            represent_ops[opsite] = k
                        if self.coef_site[k] < isite:
                            vertex_U_concat += op[0 : isite + 1]
                        else:
                            update_coef_ops.append(k)
                            # self.coef_site[k] = isite
                            vertex_U_concat += op.coef * op[0 : isite + 1]
                    # else:
                    # remained_ops_list.append((k, op))
                Unew.append(sympy.srepr(vertex_U_concat))
            E = retained_E  # type: ignore
            # ops_list = remained_ops_list
            # U = set()
            # V = set()
            # for edge in E:
            #     U.add(edge[0])
            #     V.add(edge[1])
        # assert len(ops_list) == 0
        for k in update_coef_ops:
            self.coef_site[k] = isite
        return Unew

    def assign(self, keep_symbol: bool = False) -> list[sympy.Matrix]:
        Unew = None
        for isite in range(self.ndim):
            Unew = self._update(isite, Unew, keep_symbol)
            logger.info(f"assigned {isite=}/{self.ndim}")
        self.Wsym = self.reset_Wsym()
        # assert sympy.Mul(*self.Wsym).expand()[0] == self.operator.symbol
        return self.Wsym

    def show_graph(self) -> None:
        pympo.visualize.show_assigns(
            self.operator, self.W_assigns, self.coef_site
        )

    def numerical_mpo(
        self,
        dtype=np.complex128,
        subs: dict[sympy.Symbol, int | float | complex] | None = None,
    ) -> list[NDArray]:
        mpo = []
        n_basis_list = self.operator.nbasis_list
        is_diag_list = self.operator.is_diag_list
        coef_list = []
        for op in self.operator.ops:
            coef = op.coef
            if isinstance(coef, int | float | complex):
                coef_list.append(coef)
            elif isinstance(coef, sympy.Basic):
                _coef = coef.subs(subs)
                if dtype == np.complex128:
                    _coef = complex(_coef)
                elif dtype == np.float64:
                    _coef = float(_coef)
                else:
                    raise ValueError(f"{dtype=} is not supported")
                assert isinstance(
                    _coef, int | float | complex
                ), f"{_coef=} is not a number but {type(_coef)}"
                coef_list.append(_coef)
            else:
                raise ValueError(f"{coef=} is not a number")
        for isite in range(self.ndim):
            left_dim, right_dim = self._get_bond_dim(isite)
            n_basis = n_basis_list[isite]
            is_diag = is_diag_list[isite]
            if is_diag:
                core = np.zeros((left_dim, n_basis, right_dim), dtype=dtype)
            else:
                core = np.zeros(
                    (left_dim, n_basis, n_basis, right_dim), dtype=dtype
                )

            for k in self.unique_ops[isite]:
                left_index, right_index = self._get_bond_index(isite, k)
                opisite = self.operator.ops[k].get_site_value(
                    isite, n_basis=n_basis, is_diag=is_diag
                )
                if is_diag:
                    assert opisite.shape == (n_basis,)
                else:
                    assert opisite.shape == (n_basis, n_basis)
                if self.coef_site[k] == isite:
                    coef = coef_list[k]
                    assert isinstance(
                        coef, int | float | complex
                    ), f"{coef=} is not a number"
                    if is_diag:
                        core[left_index, :, right_index] += coef * opisite
                    else:
                        core[left_index, :, :, right_index] += coef * opisite
                elif self.coef_site[k] < isite:
                    if is_diag:
                        core[left_index, :, right_index] += opisite
                    else:
                        core[left_index, :, :, right_index] += opisite
                else:
                    if is_diag:
                        assert np.allclose(
                            core[left_index, :, right_index], 0
                        ) or np.allclose(
                            core[left_index, :, right_index], opisite
                        )
                        core[left_index, :, right_index] = opisite
                    else:
                        core[left_index, :, :, right_index] = opisite
                        assert np.allclose(
                            core[left_index, :, :, right_index], 0
                        ) or np.allclose(
                            core[left_index, :, :, right_index], opisite
                        )
            mpo.append(core)

        return mpo


def get_bipartite(
    U: set[str] | list[str],
    V: set[str] | list[str],
    E: set[tuple[str, str]] | list[tuple[str, str]],
) -> nx.Graph:
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
    Unew: list[str] | None = None,
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
    min_vertex_cover: list[str],
    U: list[str],
    V: list[str],
    E: list[tuple[str, str]],
    operators: SumOfProducts,
    isite: int,
    W_assigns: list[list[int]],
    E_assigns: list[tuple[str, str]],
    coef_site: list[int],
    visualize: bool = True,
) -> tuple[list[str], sympy.Matrix]:
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
        if sympy.latex(op[isite]) == r"\hat{1}" + f"_{isite}":
            opisite = 1
        else:
            opisite = op[isite]
        # there are three cases
        # 1. W[a,b] = x_i
        # 2. W[a,b] += x_i
        # 3. W[a,b] += coef * x_i
        if coef_site[k] == isite:
            # case 3
            Wi[left_index, right_index] += op.coef * opisite
        elif coef_site[k] < isite:
            Wi[left_index, right_index] += opisite
        else:
            # case 1
            if Wi[left_index, right_index] == 0:
                Wi[left_index, right_index] = opisite
            else:
                assert (
                    Wi[left_index, right_index] == opisite
                ), f"{Wi[left_index, right_index]=} while {op[isite]=} when {coef_site[k]=}, {isite=}"
    return Unew, Wi
