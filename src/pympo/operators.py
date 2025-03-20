from __future__ import annotations

import re
from bisect import bisect_left
from typing import Iterator, Sequence, overload

import numpy as np
import sympy
from numpy.typing import NDArray


class OpSite:
    """
    Represents an operator acting on a specific site in a quantum system.

    Attributes:
        symbol (sympy.Basic): The symbolic representation of the operator.
        isite (int): The site index on which the operator acts.
        value (NDArray | None): The numerical value of the operator, if available.
        isdiag (bool): Indicates if the operator is diagonal.

    Operator z_i acting on site i.
    """

    symbol: sympy.Basic
    isite: int
    value: NDArray | None
    isdiag: bool

    def __init__(
        self,
        symbol: sympy.Basic | str,
        isite: int,
        *,
        value: NDArray | None = None,
        isdiag: bool = False,
    ) -> None:
        if isinstance(symbol, sympy.Basic):
            self.symbol = symbol
        elif isinstance(symbol, str):
            self.symbol = sympy.Symbol(symbol, commutative=False)
        else:
            raise ValueError("Invalid type", type(symbol))
        self.isite = isite
        self.value = value
        if value is not None:
            self.isdiag = value.ndim == 1
        else:
            self.isdiag = isdiag

    def __repr__(self) -> str:
        retval = self.symbol
        assert isinstance(retval, sympy.Basic)
        return retval.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def __mul__(
        self,
        other: int | float | complex | sympy.Basic | OpSite | OpProductSite,
    ) -> OpSite | OpProductSite:
        if isinstance(other, int | float | complex | sympy.Basic):
            retval = OpProductSite([self]) * other
            assert isinstance(retval, OpProductSite)
            return retval
        elif isinstance(other, OpSite):
            if self.isite < other.isite:
                return OpProductSite([self, other])
            elif self.isite > other.isite:
                return OpProductSite([other, self])
            else:
                symbol = self.symbol * other.symbol
                isite = self.isite
                isdiag = self.isdiag and other.isdiag
                if self.value is not None and other.value is not None:
                    if isdiag:
                        value = self.value * other.value
                    else:
                        if self.value.ndim == 1:
                            value1 = np.diag(self.value)
                        else:
                            value1 = self.value
                        if other.value.ndim == 1:
                            value2 = np.diag(other.value)
                        else:
                            value2 = other.value
                        value = value1 @ value2
                else:
                    value = None
                return OpSite(symbol, isite, value=value, isdiag=isdiag)
        elif isinstance(other, OpProductSite):
            return OpProductSite([self] + other.ops)
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __rmul__(
        self,
        other: int | float | complex | sympy.Basic | OpSite | OpProductSite,
    ) -> OpSite | OpProductSite:
        if isinstance(other, int | float | complex | sympy.Basic):
            """ Commutative """
            return self.__mul__(other)
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __truediv__(
        self, other: int | float | complex | sympy.Basic
    ) -> OpSite | OpProductSite:
        if isinstance(other, int | float | complex | sympy.Basic):
            retval = self.__mul__(1 / other)
            return retval
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __rtruediv__(
        self, other: int | float | complex | sympy.Basic
    ) -> OpSite | OpProductSite:
        if isinstance(other, int | float | complex | sympy.Basic):
            return self.__truediv__(other)
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __add__(
        self,
        other: OpSite
        | OpProductSite
        | SumOfProducts
        | int
        | float
        | complex
        | sympy.Basic,
    ) -> OpSite | SumOfProducts:
        if isinstance(other, OpSite):
            return SumOfProducts([self, other])
        elif isinstance(other, OpProductSite):
            return SumOfProducts([self, other])
        elif isinstance(other, SumOfProducts):
            op_product = OpProductSite([self])
            other.ops.append(op_product)
            other.coefs.append(op_product.coef)
            other.symbols.append(op_product.symbol)
            return other
        elif isinstance(other, int | float | complex | sympy.Basic):
            if isinstance(self.value, np.ndarray):
                n_basis = self.value.shape[0]
            else:
                n_basis = None
            eye = get_eye_site(self.isite, n_basis=n_basis)
            return self + eye
        else:
            raise ValueError("Invalid type")

    def __sub__(
        self, other: OpSite | OpProductSite | SumOfProducts
    ) -> OpSite | SumOfProducts:
        if isinstance(other, OpSite | OpProductSite | SumOfProducts):
            return self + (-1) * other
        else:
            raise ValueError("Invalid type")


def get_eye_site(i: int, n_basis: int | None = None) -> OpSite:
    """
    Create an identity operator site.

    Parameters:
    -----------
    i (int): The index of the site.
    n_basis (int | None, optional): The number of basis states. If provided,
                                    an array of ones with length `n_basis` is created.
                                    Defaults to None.

    Returns:
    --------
    OpSite: An operator site with the identity operator.
    """
    value: NDArray | None = None
    if isinstance(n_basis, int):
        value = np.ones(n_basis)
    return OpSite(
        sympy.Symbol(r"\hat{1}_" + f"{i}"), i, value=value, isdiag=True
    )


def omit_eye_site(latex_symbol: str) -> str:
    r"""
    Args:
        latex_symbol (str): The latex symbol of the operator like
            $\hat{1}_0\hat{z}_1$.

    Returns:
        str: The latex symbol of the operator without
            the identity operator like $\hat{z}_1$.

    """
    latex = re.sub(r"\\hat\{1\}_[0-9]+", "", latex_symbol)
    if re.match(r"\$[ ]*\$", latex):
        if re.search(r"\\hat\{1\}_0", latex_symbol):
            return r"$\hat{1}_{\text{left}}$"
        else:
            return r"$\hat{1}_{\text{right}}$"
    return latex


class OpProductSite:
    """
    Represents a product of operators acting on multiple sites, such as z_i * z_j * z_k.

    Attributes:
        coef (int | float | complex | sympy.Basic): Coefficient of the operator product.
        symbol (sympy.Basic): Symbolic representation of the operator product.
        ops (list[OpSite]): List of operators in the product.
        sites (list[int]): List of site indices where the operators act.

    Product of operators acting on multiple sites like
    z_i * z_j * z_k

    """

    coef: int | float | complex | sympy.Basic
    symbol: sympy.Basic
    ops: list[OpSite]
    sites: list[int]

    def __init__(self, ops: list[OpSite]) -> None:
        argsrt = np.argsort([op.isite for op in ops])
        self.ops = [ops[i] for i in argsrt]
        self.sites = []
        self.symbol = 1
        self.coef = 1
        for op in self.ops:
            self.symbol *= op.symbol
            self.sites.append(op.isite)
        if self._is_duplicated():
            raise ValueError("Duplicate site index")
        if not self._is_sorted():
            raise ValueError("Site index is not sorted")
        self.left_product: list[sympy.Basic] = None  # type: ignore
        self.right_product: list[sympy.Basic] = None  # type: ignore

    def _set_left_product(self) -> None:
        self.left_product = [self.ops[0].symbol]
        k = 0
        for i in range(self.sites[0] + 1, self.sites[-1] + 1):
            if i in self.sites[1:]:
                k += 1
                self.left_product.append(
                    self.left_product[-1] * self.ops[k].symbol
                )
            else:
                self.left_product.append(self.left_product[-1])
        assert k == len(self.ops) - 1, f"{k=}, {len(self.ops)=}, {self.sites=}"

    def _set_right_product(self) -> None:
        self.right_product = [self.ops[-1].symbol]
        k = len(self.ops) - 1
        for i in range(self.sites[-1] - 1, self.sites[0] - 1, -1):
            if i in self.sites[:-1]:
                k -= 1
                self.right_product.append(
                    self.ops[k].symbol * self.right_product[-1]
                )
            else:
                self.right_product.append(self.right_product[-1])
        self.right_product = self.right_product[::-1]
        assert k == 0, f"{k=}, {self.sites=}"

    def replace(self, new_op: OpSite) -> None:
        """
        Replace an existing operator in the list with a new operator.

        Args:
            new_op (OpSite): The new operator to replace the existing one.

        Raises:
            AssertionError: If the site of the new operator is not found in the existing operators.

        Modifies:
            self.symbol: Updates the symbol by multiplying the symbols of all operators.
            self.ops: Replaces the operator at the matching site with the new operator.
        """
        self.symbol = 1
        is_replaced = False
        for i, op in enumerate(self.ops):
            if op.isite == new_op.isite:
                self.ops[i] = new_op
                is_replaced = True
            self.symbol *= self.ops[i].symbol
        assert is_replaced, f"{new_op.isite=} is not found in {self.sites=}"

    def __repr__(self) -> str:
        return " * ".join([op.__repr__() for op in self.ops])

    def __str__(self) -> str:
        return " * ".join([op.__str__() for op in self.ops])

    @overload
    def __mul__(
        self,
        other: int | float | complex | sympy.Basic | OpSite | OpProductSite,
    ) -> OpProductSite: ...

    @overload
    def __mul__(self, other: SumOfProducts) -> SumOfProducts: ...

    def __mul__(
        self,
        other: int
        | float
        | complex
        | sympy.Basic
        | OpSite
        | OpProductSite
        | SumOfProducts,
    ) -> OpProductSite | SumOfProducts:
        if isinstance(other, int | float | complex | sympy.Basic):
            self.coef *= other
            return self
        elif isinstance(other, OpSite):
            if other.isite in self.sites:
                idx = bisect_left(self.sites, other.isite)
                self.symbol *= other.symbol
                same_site_op = self.ops[idx]
                isdiag = same_site_op.isdiag and other.isdiag
                if same_site_op.value is not None and other.value is not None:
                    if isdiag:
                        value = same_site_op.value * other.value
                    else:
                        if same_site_op.value.ndim == 1:
                            value1 = np.diag(same_site_op.value)
                        else:
                            value1 = same_site_op.value
                        if other.value.ndim == 1:
                            value2 = np.diag(other.value)
                        else:
                            value2 = other.value
                        value = value1 @ value2
                else:
                    value = None
                site_symbol = same_site_op.symbol * other.symbol
                new_op = OpSite(
                    site_symbol,
                    same_site_op.isite,
                    value=value,
                    isdiag=isdiag,
                )
                self.ops[idx] = new_op
                return self
            else:
                idx = bisect_left(self.sites, other.isite)
                self.ops.insert(idx, other)
                return OpProductSite(self.ops)
        elif isinstance(other, OpProductSite):
            coef = self.coef * other.coef
            new_product = OpProductSite(self.ops)
            for op in other.ops:
                assert isinstance(op, OpSite)
                _new_product = new_product * op
                assert isinstance(_new_product, OpProductSite)
                new_product = _new_product
            new_product.coef = coef
            return new_product
        elif isinstance(other, SumOfProducts):
            ops = []
            opproduct1 = OpProductSite(self.ops)
            opproduct1.coef = self.coef
            for opproduct2 in other.ops:
                assert isinstance(opproduct2, OpProductSite)
                _new_product = opproduct1 * opproduct2
                assert isinstance(_new_product, OpProductSite)
                ops.append(_new_product)
            return SumOfProducts(ops)

        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __rmul__(
        self,
        other: int | float | complex | sympy.Basic | OpSite | OpProductSite,
    ) -> OpProductSite:
        if isinstance(other, int | float | complex | sympy.Basic):
            retval = self.__mul__(other)
        elif isinstance(other, OpSite | OpProductSite):
            retval = other.__mul__(self)  # type: ignore
        else:
            raise ValueError(f"Invalid type: {type(other)=}")
        assert isinstance(retval, OpProductSite)
        return retval

    def __truediv__(
        self, other: int | float | complex | sympy.Basic
    ) -> OpProductSite:
        if isinstance(other, int | float | complex | sympy.Basic):
            retval = self.__mul__(1 / other)
            assert isinstance(retval, OpProductSite)
            return retval
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __add__(
        self,
        other: OpSite
        | OpProductSite
        | SumOfProducts
        | int
        | float
        | complex
        | sympy.Basic,
    ) -> SumOfProducts:
        if isinstance(other, OpSite):
            return SumOfProducts([self, other])
        elif isinstance(other, OpProductSite):
            return SumOfProducts([self, other])
        elif isinstance(other, SumOfProducts):
            other.ops.append(self)
            other.coefs.append(self.coef)
            other.symbols.append(self.symbol)
            return other
        elif isinstance(other, int | float | complex | sympy.Basic):
            const = get_eye_site(self.sites[0]) * other
            return SumOfProducts([self, const])
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __radd__(
        self,
        other: OpSite
        | OpProductSite
        | SumOfProducts
        | int
        | float
        | complex
        | sympy.Basic,
    ) -> SumOfProducts:
        return self.__add__(other)

    def __sub__(
        self,
        other: OpSite
        | OpProductSite
        | SumOfProducts
        | int
        | float
        | complex
        | sympy.Basic,
    ) -> SumOfProducts:
        if isinstance(
            other,
            OpSite
            | OpProductSite
            | SumOfProducts
            | int
            | float
            | complex
            | sympy.Basic,
        ):
            return self + (-1) * other
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __rsub__(
        self,
        other: OpSite
        | OpProductSite
        | SumOfProducts
        | int
        | float
        | complex
        | sympy.Basic,
    ) -> SumOfProducts:
        return self.__sub__((-1) * other)

    def _is_duplicated(self) -> bool:
        return len(self.sites) != len(set(self.sites))

    def _is_sorted(self) -> bool:
        return self.sites == sorted(self.sites) and self.sites == [
            op.isite for op in self.ops
        ]

    def get_symbol_interval(
        self, start_site: int, end_site: int
    ) -> sympy.Basic:
        """
        Get the symbol of the operator acting on the sites between start_site and end_site.

        When the operator is symbol = z_1 * z_6,
        - get_symbol_interval(0, 3) returns 1_0 * z_1 * 1_2,
        - get_symbol_interval(3, 8) returns 1_3 * 1_4 * 1_5 * z_6 * 1_7.

        Args:
            start_site (int): The start site.
            end_site (int): The end site.

        Returns:
            sympy.Basic: The symbol of the operator acting on the sites between start_site and end_site.

        To Do:
            - Improve the performance of the function by memoization.
        """
        if start_site + 1 == end_site:
            if start_site in self.sites:
                return self.ops[bisect_left(self.sites, start_site)].symbol
            else:
                return get_eye_site(start_site).symbol

        if start_site <= self.sites[0]:
            case_left = 0
        elif start_site <= self.sites[-1]:
            case_left = 1
        else:
            case_left = 2

        if end_site <= self.sites[0]:
            case_right = 0
        elif end_site <= self.sites[-1]:
            case_right = 1
        else:
            case_right = 2

        match (case_left, case_right):
            case (0, 0):
                return get_eye_site(end_site).symbol
            case (0, 1):
                if self.left_product is None:
                    self._set_left_product()
                return self.left_product[end_site - self.sites[0]]
            case (0, 2):
                if self.left_product is None:
                    self._set_left_product()
                return self.left_product[-1]
            case (1, 1):
                return self._get_symbol_interval_intermidiate(
                    start_site, end_site
                )
            case (1, 2):
                if self.right_product is None:
                    self._set_right_product()
                return self.right_product[start_site - self.sites[0]]
            case (2, 2):
                return get_eye_site(start_site).symbol
            case _:
                raise ValueError(
                    f"{case_left=}, {case_right=}, {start_site=}, {end_site=}"
                )

    def _get_symbol_interval_intermidiate(
        self, start_site: int, end_site: int
    ) -> sympy.Basic:
        symbol = 1
        idx = bisect_left(self.sites, start_site)
        is_eye = True
        for i in range(start_site, end_site):
            if len(self.sites) > idx and self.sites[idx] == i:
                # If operator is 1_1 * 1_2 * 1_3 * z_4 * 1_5 ...
                # skip 1_1 * 1_2 * 1_3
                if is_eye:
                    symbol = self.ops[idx].symbol
                    is_eye = False
                else:
                    symbol *= self.ops[idx].symbol
                idx += 1
            else:
                if symbol == 1:
                    # To reduce cost for symbolic computation,
                    # identity operator is only used when symbol == 1.
                    symbol = get_eye_site(i).symbol
                else:
                    pass
        if symbol == 1:
            symbol = get_eye_site(start_site).symbol
        assert isinstance(symbol, sympy.Basic)
        # self.symbol_intervals[(start_site, end_site)] = symbol
        return symbol

    def __getitem__(self, key: int | slice) -> sympy.Basic:
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            if start is None:
                start = 0
            if stop is None:
                raise ValueError("Invalid slice. End index is not trivial.")
            return self.get_symbol_interval(start, stop)
        elif isinstance(key, int):
            return self.get_symbol_interval(key, key + 1)
        else:
            raise ValueError("Invalid type")

    def get_site_value(self, isite: int, n_basis: int, isdiag: bool) -> NDArray:
        """
        Get the value of the operator acting on the site isite.

        Args:
            isite (int): The site index.
            n_basis (int): The number of basis.

        Returns:
            NDArray: The value of the operator acting on the site isite.
        """
        idx = bisect_left(self.sites, isite)
        if len(self.sites) > idx and self.sites[idx] == isite:
            value = self.ops[idx].value
            assert isinstance(value, np.ndarray)
            if isdiag:
                assert value.shape == (
                    n_basis,
                ), f"{value.shape=} while {n_basis=}"
            elif value.ndim == 1:
                value = np.diag(value)
                assert value.shape == (
                    n_basis,
                    n_basis,
                ), f"{value.shape=} while {n_basis=}"
            return value
        else:
            if isdiag:
                return np.ones(n_basis)
            else:
                return np.eye(n_basis)


class SumOfProducts:
    """
    Sum of products of operators acting on multiple sites like
    z_i * z_j + z_k * z_l

    Args:
        ops (Sequence[OpProductSite | OpSite], optional): List of operator products. Defaults to [].

    Attributes:
        coefs (list[int | float | complex | sympy.Basic]): Coefficients of the operator products.
        ops (list[OpProductSite]): List of operator products.
        symbols (list[sympy.Basic]): List of symbolic representations of the operator products.

    """

    coefs: list[int | float | complex | sympy.Basic]
    ops: list[OpProductSite]
    symbols: list[sympy.Basic]

    def __init__(self, ops: Sequence[OpProductSite | OpSite] = []) -> None:
        self.coefs = []
        self.ops = []
        self.symbols = []
        for op in ops:
            if isinstance(op, OpSite):
                op = OpProductSite([op])
            assert isinstance(op, OpProductSite)
            self.ops.append(op)
            self.coefs.append(op.coef)
            self.symbols.append(op.symbol)

    def simplify(self) -> SumOfProducts:
        """
        Concatenate common operator such as
        q_i * a_j + q_i * a^dagger_j -> q_i * (a_j + a^dagger_j)

        Note that the computational complexity is O(n^2) where n is the number of operators.
        """
        skip_flag = [False] * len(self.ops)
        new_ops = []
        for i in range(len(self.ops)):
            if skip_flag[i]:
                continue
            for j in range(i + 1, len(self.ops)):
                if skip_flag[j]:
                    continue
                if self.ops[i].sites != self.ops[j].sites:
                    continue
                if self.ops[i].coef != self.ops[j].coef:
                    continue
                continue_flag = False
                op_i_common = None
                op_j_common = None
                for op_i, op_j in zip(
                    self.ops[i].ops, self.ops[j].ops, strict=True
                ):
                    if op_i.symbol != op_j.symbol:
                        if op_i_common is not None:
                            # When two operators are not common, skip the loop.
                            continue_flag = True
                            break
                        op_i_common = op_i
                        op_j_common = op_j
                if continue_flag or op_i_common is None:
                    # Either two or more operators are not common or no common operator is found.
                    continue
                assert isinstance(op_i_common, OpSite) and isinstance(
                    op_j_common, OpSite
                )
                assert op_j_common.isite == op_i_common.isite
                skip_flag[j] = True
                new_symbol = op_i_common.symbol + op_j_common.symbol
                new_isdiag = op_i_common.isdiag and op_j_common.isdiag
                if (
                    op_i_common.value is not None
                    and op_j_common.value is not None
                ):
                    if new_isdiag:
                        new_value = op_i_common.value + op_j_common.value
                    else:
                        if op_i_common.value.ndim == 1:
                            value1 = np.diag(op_i_common.value)
                        else:
                            value1 = op_i_common.value
                        if op_j_common.value.ndim == 1:
                            value2 = np.diag(op_j_common.value)
                        else:
                            value2 = op_j_common.value
                        new_value = value1 + value2
                else:
                    new_value = None
                new_op = OpSite(
                    new_symbol,
                    op_i_common.isite,
                    value=new_value,
                    isdiag=new_isdiag,
                )
                self.ops[i].replace(new_op)
            new_ops.append(self.ops[i])
        return SumOfProducts(new_ops)

    @property
    def symbol(self) -> sympy.Basic | int | float | complex:
        symbol = 0
        for i in range(len(self.ops)):
            symbol += self.ops[i].symbol * self.coefs[i]
        assert isinstance(
            symbol, sympy.Basic | int | float | complex
        ), f"{symbol=}"
        return symbol

    @property
    def ndim(self) -> int:
        max_ndim = 0
        for op in self.ops:
            max_ndim = max(max_ndim, max(op.sites) + 1)
        return max_ndim

    def get_unique_ops_site(self, i: int) -> set[OpSite]:
        unique_ops_set = set()
        for op in self.ops:
            op_i = op[i]
            unique_ops_set.add(op_i)
        return unique_ops_set

    @property
    def nops(self) -> int:
        return len(self.ops)

    @property
    def nbasis_list(self) -> list[int]:
        nbasis_list = [0] * self.ndim
        for opproduct in self.ops:
            for isite, opsite in zip(
                opproduct.sites, opproduct.ops, strict=True
            ):
                if opsite.value is None:
                    raise ValueError(f"Value at {opsite=} is not defined.")
                if nbasis_list[isite] == 0:
                    nbasis_list[isite] = opsite.value.shape[0]
                else:
                    if nbasis_list[isite] != opsite.value.shape[0]:
                        raise ValueError(
                            f"Number of basis at {isite=} is not consistent with {opsite=} and {nbasis_list[isite]=}"
                        )
        for i, nbasis in enumerate(nbasis_list):
            if nbasis == 0:
                raise ValueError(f"Number of basis at {i=} is ambiguous.")
        return nbasis_list

    @property
    def isdiag_list(self) -> list[bool]:
        isdiag_list = [True] * self.ndim
        for opproduct in self.ops:
            for isite, opsite in zip(
                opproduct.sites, opproduct.ops, strict=True
            ):
                isdiag_list[isite] &= opsite.isdiag
        return isdiag_list

    def __add__(
        self,
        other: OpSite
        | OpProductSite
        | SumOfProducts
        | int
        | float
        | complex
        | sympy.Basic,
    ) -> SumOfProducts:
        if isinstance(other, int | float | complex | sympy.Basic):
            op_product = OpProductSite([get_eye_site(self.ops[0].sites[0])])
            return self + op_product * other
        elif isinstance(other, OpSite):
            op_product = OpProductSite([other])
            return self + op_product
        elif isinstance(other, OpProductSite):
            self.ops.append(other)
            self.coefs.append(other.coef)
            self.symbols.append(other.symbol)
            return self
        elif isinstance(other, SumOfProducts):
            for i in range(len(other.ops)):
                self.ops.append(other.ops[i])
                self.coefs.append(other.coefs[i])
                self.symbols.append(other.symbols[i])
            return self
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __sub__(
        self, other: OpSite | OpProductSite | SumOfProducts
    ) -> SumOfProducts:
        if isinstance(other, OpSite | OpProductSite | SumOfProducts):
            return self + (-1) * other
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __mul__(
        self,
        other: int
        | float
        | complex
        | sympy.Basic
        | OpSite
        | OpProductSite
        | SumOfProducts,
    ) -> SumOfProducts:
        if isinstance(other, int | float | complex | sympy.Basic):
            for i in range(len(self.coefs)):
                self.ops[i] *= other
                self.coefs[i] = self.ops[i].coef
            return self
        elif isinstance(other, OpSite):
            for op in self.ops:
                op *= other
            return self
        elif isinstance(other, OpProductSite):
            for op, coef in zip(self.ops, self.coefs, strict=True):
                op *= other
                coef *= other.coef
            return self
        elif isinstance(other, SumOfProducts):
            raise NotImplementedError()
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __rmul__(
        self, other: int | float | complex | sympy.Basic
    ) -> SumOfProducts:
        if isinstance(other, int | float | complex | sympy.Basic):
            return self.__mul__(other)
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def to_mpo(self) -> list[NDArray]:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[OpProductSite]:
        return iter(self.ops)
