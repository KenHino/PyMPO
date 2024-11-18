from __future__ import annotations

import re
from bisect import bisect_left
from typing import Iterator

import numpy as np
import sympy
from numpy.typing import NDArray


class OpSite:
    """
    Operator z_i acting on site i.
    """

    symbol: sympy.Basic
    isite: int
    value: NDArray | None
    coef: int | float | complex | sympy.Basic
    isdiag: bool

    def __init__(
        self,
        symbol: sympy.Basic | str,
        isite: int,
        *,
        value: NDArray | None = None,
        coef: int | float | complex | sympy.Basic | str = 1,
        isdiag: bool = False,
    ) -> None:
        if isinstance(symbol, sympy.Basic):
            self.symbol = symbol
        elif isinstance(symbol, str):
            self.symbol = sympy.Symbol(symbol)
        else:
            raise ValueError("Invalid type", type(symbol))
        self.isite = isite
        self.value = value
        if isinstance(coef, str):
            self.coef = sympy.Basic(coef)
        self.coef = coef
        if value is not None:
            self.isdiag = value.ndim == 1
        else:
            self.isdiag = isdiag

    def __repr__(self) -> str:
        retval = self.symbol * self.coef
        assert isinstance(retval, sympy.Basic)
        return retval.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def __mul__(
        self,
        other: int | float | complex | sympy.Basic | OpSite | OpProductSite,
    ) -> OpSite | OpProductSite:
        if isinstance(other, int | float | complex | sympy.Basic):
            return OpSite(
                self.symbol,
                self.isite,
                value=self.value,
                coef=self.coef * other,
                isdiag=self.isdiag,
            )
        elif isinstance(other, OpSite):
            if self.isite != other.isite:
                return OpProductSite([self, other])
            else:
                raise NotImplementedError(
                    "Operator is not always commutative, thus multiplication of the same site is intentionally disabled."  # noqa E501
                )
        elif isinstance(other, OpProductSite):
            return OpProductSite([self] + other.ops)
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __rmul__(
        self,
        other: int | float | complex | sympy.Basic | OpSite | OpProductSite,
    ) -> OpSite | OpProductSite:
        return self.__mul__(other)

    def __truediv__(self, other: int | float | complex | sympy.Basic) -> OpSite:
        if isinstance(other, int | float | complex | sympy.Basic):
            retval = self.__mul__(1 / other)
            assert isinstance(retval, OpSite)
            return retval
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __rtruediv__(
        self, other: int | float | complex | sympy.Basic
    ) -> OpSite:
        return self.__truediv__(other)

    def __add__(
        self, other: OpSite | OpProductSite | SumOfProducts
    ) -> OpSite | SumOfProducts:
        if isinstance(other, OpSite):
            if self.isite != other.isite:
                return SumOfProducts([self, other])
            else:
                if self.value is not None or other.value is not None:
                    raise NotImplementedError()
                symbol = self.symbol * self.coef + other.symbol * other.coef
                isite = self.isite
                value = None
                isdiag = self.isdiag and other.isdiag
                return OpSite(symbol, isite, value=value, coef=1, isdiag=isdiag)
        elif isinstance(other, OpProductSite):
            return SumOfProducts([self, other])
        elif isinstance(other, SumOfProducts):
            op_product = OpProductSite([self])
            other.ops.append(op_product)
            other.coefs.append(op_product.coef)
            other.symbols.append(op_product.symbol)
            return other
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
    value: NDArray | None = None
    if isinstance(n_basis, int):
        value = np.ones(n_basis)
    return OpSite(
        sympy.Symbol(r"\hat{1}_" + f"{i}"), i, value=value, coef=1, isdiag=True
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
            self.coef *= op.coef
            op.coef = 1  # CAUTION original coef is set to 1
            self.symbol *= op.symbol
            self.sites.append(op.isite)
        if self._is_duplicated():
            raise ValueError("Duplicate site index")
        if not self._is_sorted():
            raise ValueError("Site index is not sorted")

    def __repr__(self) -> str:
        return " * ".join([op.__repr__() for op in self.ops])

    def __str__(self) -> str:
        return " * ".join([op.__str__() for op in self.ops])

    def __mul__(
        self,
        other: int | float | complex | sympy.Basic | OpSite | OpProductSite,
    ) -> OpProductSite:
        if isinstance(other, int | float | complex | sympy.Basic):
            self.coef *= other
            return self
        elif isinstance(other, OpSite):
            if other.isite in self.sites:
                raise ValueError("Duplicate site index")
            else:
                idx = bisect_left(self.sites, other.isite)
                self.coef *= other.coef
                self.symbol *= other.symbol
                other.coef = 1  # CAUTION original coef is set to 1
                self.ops.insert(idx, other)
                self.sites.insert(idx, other.isite)
                return self
        elif isinstance(other, OpProductSite):
            ops = self.ops + other.ops
            return OpProductSite(ops)
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __rmul__(
        self,
        other: int | float | complex | sympy.Basic | OpSite | OpProductSite,
    ) -> OpProductSite:
        return self.__mul__(other)

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
        self, other: OpSite | OpProductSite | SumOfProducts
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
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

    def __sub__(
        self, other: OpSite | OpProductSite | SumOfProducts
    ) -> SumOfProducts:
        if isinstance(other, OpSite | OpProductSite | SumOfProducts):
            return self + (-1) * other
        else:
            raise ValueError(f"Invalid type: {type(other)=}")

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
        symbol = 1
        idx = bisect_left(self.sites, start_site)
        is_eye = True
        for i in range(start_site, end_site):
            if len(self.sites) > idx and self.sites[idx] == i:
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
        assert isinstance(symbol, sympy.Basic) or symbol == 1
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

    def get_site_value(
        self, isite: int, n_basis: int, is_diag: bool
    ) -> NDArray:
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
            if is_diag:
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
            if is_diag:
                return np.ones(n_basis)
            else:
                return np.eye(n_basis)


class SumOfProducts:
    """
    Sum of products of operators acting on multiple sites like
    z_i * z_j + z_k * z_l

    """

    coefs: list[int | float | complex | sympy.Basic]
    ops: list[OpProductSite]
    symbols: list[sympy.Basic]

    def __init__(self, ops: list[OpProductSite | OpSite]) -> None:
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

    @property
    def symbol(self) -> sympy.Basic:
        symbol = 0
        for i in range(len(self.ops)):
            symbol += self.ops[i].symbol * self.coefs[i]
        assert isinstance(symbol, sympy.Basic)
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
    def is_diag_list(self) -> list[bool]:
        is_diag_list = [True] * self.ndim
        for opproduct in self.ops:
            for isite, opsite in zip(
                opproduct.sites, opproduct.ops, strict=True
            ):
                is_diag_list[isite] &= opsite.isdiag
        return is_diag_list

    def __add__(
        self, other: OpSite | OpProductSite | SumOfProducts
    ) -> SumOfProducts:
        if isinstance(other, OpSite):
            op_product = OpProductSite([other])
            self.ops.append(op_product)
            self.coefs.append(op_product.coef)
            self.symbols.append(op_product.symbol)
            return self
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
        self, other: int | float | complex | sympy.Basic
    ) -> SumOfProducts:
        for i in range(len(self.coefs)):
            self.coefs[i] *= other
            self.symbols[i] *= other
        return self

    def __rmul__(
        self, other: int | float | complex | sympy.Basic
    ) -> SumOfProducts:
        return self.__mul__(other)

    def to_mpo(self) -> list[NDArray]:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[OpProductSite]:
        return iter(self.ops)
