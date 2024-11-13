from __future__ import annotations

from bisect import bisect_left

import numpy as np
import sympy


class OpSite:
    """
    Operator z_i acting on site i.
    """

    symbol: sympy.Basic
    isite: int
    value: np.ndarray | None
    coef: int | float | complex | sympy.Basic = 1
    isdiag: bool = False

    def __init__(
        self,
        symbol: sympy.Basic | str,
        isite: int,
        *,
        value: np.ndarray | None = None,
        coef: int | float | complex | sympy.Basic | str = 1,
        isdiag: bool = False,
    ):
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
        self.isdiag = isdiag

    def __repr__(self):
        return (self.symbol * self.coef).__repr__()

    def __str__(self):
        return (self.symbol * self.coef).__str__()

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
            raise ValueError("Invalid type")

    def __rmul__(
        self,
        other: int | float | complex | sympy.Basic | OpSite | OpProductSite,
    ) -> OpSite | OpProductSite:
        return self.__mul__(other)

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


def get_eye_site(i: int, n_basis: int | None = None):
    value: np.ndarray | None = None
    if isinstance(n_basis, int):
        value = np.ones(n_basis)
    return OpSite(
        sympy.Basic(r"\hat{1}_" + f"{i}"), i, value=value, coef=1, isdiag=True
    )


class OpProductSite:
    """
    Product of operators acting on multiple sites like
    z_i * z_j * z_k

    """

    coef: int | float | complex | sympy.Basic = 1
    symbol: sympy.Basic = 1
    ops: list[OpSite]
    sites: list[int]

    def __init__(self, ops: list[OpSite]):
        argsrt = np.argsort([op.isite for op in ops])
        self.ops = [ops[i] for i in argsrt]
        self.sites = []
        for op in self.ops:
            self.coef *= op.coef
            op.coef = 1
            self.symbol *= op.symbol
            self.sites.append(op.isite)
        if self._is_duplicated():
            raise ValueError("Duplicate site index")
        if not self._is_sorted():
            raise ValueError("Site index is not sorted")

    def __repr__(self):
        return " * ".join([op.__repr__() for op in self.ops])

    def __str__(self):
        return " * ".join([op.__str__() for op in self.ops])

    def __mul__(
        self, other: int | float | complex | OpSite | OpProductSite
    ) -> OpProductSite:
        if isinstance(other, int | float | complex):
            self.coef *= other
            return self
        elif isinstance(other, OpSite):
            if other.isite in self.sites:
                raise ValueError("Duplicate site index")
            else:
                idx = bisect_left(self.sites, other.isite)
                self.coef *= other.coef
                self.symbol *= other.symbol
                other.coef = 1
                self.ops.insert(idx, other)
                self.sites.insert(idx, other.isite)
                return self
        elif isinstance(other, OpProductSite):
            ops = self.ops + other.ops
            return OpProductSite(ops)
        else:
            raise ValueError("Invalid type")

    def __rmul__(
        self, other: int | float | complex | OpSite | OpProductSite
    ) -> OpProductSite:
        return self.__mul__(other)

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
            raise ValueError("Invalid type")

    def __sub__(
        self, other: OpSite | OpProductSite | SumOfProducts
    ) -> SumOfProducts:
        if isinstance(other, OpSite | OpProductSite | SumOfProducts):
            return self + (-1) * other
        else:
            raise ValueError("Invalid type")

    def _is_duplicated(self):
        return len(self.sites) != len(set(self.sites))

    def _is_sorted(self):
        return self.sites == sorted(self.sites) and self.sites == [
            op.isite for op in self.ops
        ]


class SumOfProducts:
    """
    Sum of products of operators acting on multiple sites like
    z_i * z_j + z_k * z_l

    """

    coefs: list[int | float | complex | sympy.Basic]
    ops: list[OpProductSite]
    symbols: list[sympy.Basic]

    def __init__(self, ops: list[OpProductSite | OpSite]):
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
    def symbol(self):
        symbol = 0
        for i in range(len(self.ops)):
            symbol += self.ops[i].symbol * self.coefs[i]
        return symbol

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
            raise ValueError("Invalid type")

    def __sub__(
        self, other: OpSite | OpProductSite | SumOfProducts
    ) -> SumOfProducts:
        if isinstance(other, OpSite | OpProductSite | SumOfProducts):
            return self + (-1) * other
        else:
            raise ValueError("Invalid type")

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

    def to_mpo(self):
        # まずはJupyterで実装して移植する。
        raise NotImplementedError()
