import numpy as np
import sympy

from pympo.operators import OpProductSite, OpSite, SumOfProducts, get_eye_site


def test_opsite_initialization():
    symbol = sympy.Symbol("z")
    isite = 0
    value = np.array([[1, 0], [0, -1]])
    isdiag = True

    op = OpSite(symbol, isite, value=value, isdiag=isdiag)

    assert op.symbol == symbol
    assert op.isite == isite
    assert np.array_equal(op.value, value)
    assert not op.isdiag  # because value is not diagonal


def test_opsite_multiplication_with_scalar():
    symbol = sympy.Symbol("z")
    isite = 0
    op = OpSite(symbol, isite)

    result = op * 2

    assert result.coef == 2.0
    assert result.symbol == symbol


def test_opsite_multiplication_with_opsite_different_sites():
    symbol1 = sympy.Symbol("z_1")
    symbol2 = sympy.Symbol("z_2")
    op1 = OpSite(symbol1, 0)
    op2 = OpSite(symbol2, 1)

    # with pytest.raises(ValueError):
    op1 * op2


def test_opsite_multiplication_with_opproductsite():
    symbol1 = sympy.Symbol("sigma_1")
    symbol2 = sympy.Symbol("sigma_2")
    op1 = OpSite(symbol1, 0)
    op2 = OpSite(symbol2, 1)
    op_product = OpProductSite([op2])

    result = op1 * op_product
    assert len(result.ops) == 2
    assert result.symbol.simplify() == symbol1 * symbol2
    assert result.sites == [0, 1]


def test_opproductsite_initialization():
    symbol = sympy.Symbol("z")
    op1 = OpSite(symbol, 0)
    op2 = OpSite(symbol, 1)

    op_product = OpProductSite([op1, op2])

    assert len(op_product.ops) == 2
    assert op_product.sites == [0, 1]


def test_opproductsite_multiplication_with_scalar():
    symbol = sympy.Symbol("z")
    op1 = OpSite(symbol, 0)
    op_product = OpProductSite([op1])

    result = op_product * 2

    assert result.coef == 2.0


def test_opproductsite_multiplication_with_opsite():
    symbol1 = sympy.Symbol("z_1")
    symbol2 = sympy.Symbol("z_2")
    op1 = OpSite(symbol1, 0)
    op2 = OpSite(symbol2, 1)
    op_product = OpProductSite([op1])

    result = op_product * op2
    assert len(result.ops) == 2
    assert result.symbol.simplify() == symbol1 * symbol2
    assert result.sites == [0, 1]


def test_opproductsite_multiplication_with_opproductsite():
    symbol = sympy.Symbol("z")
    op1 = OpSite(symbol, 0)
    op2 = OpSite(symbol, 1)
    op_product1 = OpProductSite([op1])
    op_product2 = OpProductSite([op2])

    result = op_product1 * op_product2

    assert len(result.ops) == 2
    assert result.sites == [0, 1]


def test_opsite_creation():
    symbol = sympy.Symbol("X")
    opsite = OpSite(symbol, 0)
    assert opsite.symbol == symbol
    assert opsite.isite == 0


def test_opproductsite_creation():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 1)
    opproductsite = OpProductSite([opsite1, opsite2])
    assert opproductsite.sites == [0, 1]


def test_opproductsite_multiplication():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 1)
    opproductsite = OpProductSite([opsite1])
    result = opproductsite * opsite2
    assert isinstance(result, OpProductSite)
    assert result.sites == [0, 1]


def test_sumofproducts_creation():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 1)
    opproductsite1 = OpProductSite([opsite1])
    opproductsite2 = OpProductSite([opsite2])
    sumofproducts = SumOfProducts([opproductsite1, opproductsite2])
    assert sumofproducts.symbol == symbol1 + symbol2


def test_get_eye_site():
    eye_site = get_eye_site(0, 2)
    assert eye_site.value is not None
    assert np.array_equal(eye_site.value, np.ones(2))


def test_opsite_addition_with_opsite():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 0)
    result = opsite1 + opsite2
    assert isinstance(result, SumOfProducts)


def test_opsite_addition_with_opsite_different_sites():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 1)
    result = opsite1 + opsite2
    assert isinstance(result, SumOfProducts)


def test_opsite_addition_with_opproductsite():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 1)
    opproductsite = OpProductSite([opsite2])
    result = opsite1 + opproductsite
    assert isinstance(result, SumOfProducts)


def test_opproductsite_addition_with_opsite():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 1)
    opproductsite = OpProductSite([opsite1])
    result = opproductsite + opsite2
    assert isinstance(result, SumOfProducts)


def test_opproductsite_addition_with_opproductsite():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 1)
    opproductsite1 = OpProductSite([opsite1])
    opproductsite2 = OpProductSite([opsite2])
    result = opproductsite1 + opproductsite2
    assert isinstance(result, SumOfProducts)


def test_sumofproducts_addition_with_opsite():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 1)
    sumofproducts = SumOfProducts([opsite1])
    result = sumofproducts + opsite2
    assert isinstance(result, SumOfProducts)


def test_sumofproducts_addition_with_opproductsite():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 1)
    opproductsite = OpProductSite([opsite2])
    sumofproducts = SumOfProducts([opsite1])
    result = sumofproducts + opproductsite
    assert isinstance(result, SumOfProducts)


def test_sumofproducts_addition_with_sumofproducts():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    opsite1 = OpSite(symbol1, 0)
    opsite2 = OpSite(symbol2, 1)
    sumofproducts1 = SumOfProducts([opsite1])
    sumofproducts2 = SumOfProducts([opsite2])
    result = sumofproducts1 + sumofproducts2
    assert isinstance(result, SumOfProducts)


def test_sumofproducts_isdiag_list_all_diag():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    value1 = np.array([1, 0])
    value2 = np.array([0, 1])
    opsite1 = OpSite(symbol1, 0, value=value1, isdiag=True)
    opsite2 = OpSite(symbol2, 1, value=value2, isdiag=True)
    opproductsite1 = OpProductSite([opsite1])
    opproductsite2 = OpProductSite([opsite2])
    sumofproducts = SumOfProducts([opproductsite1, opproductsite2])
    assert sumofproducts.isdiag_list == [True, True]


def test_sumofproducts_isdiag_list_mixed():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    value1 = np.array([1, 0])
    value2 = np.array([[0, 1], [1, 0]])
    opsite1 = OpSite(symbol1, 0, value=value1, isdiag=True)
    opsite2 = OpSite(symbol2, 1, value=value2, isdiag=False)
    opproductsite1 = OpProductSite([opsite1])
    opproductsite2 = OpProductSite([opsite2])
    sumofproducts = SumOfProducts([opproductsite1, opproductsite2])
    assert sumofproducts.isdiag_list == [True, False]


def test_sumofproducts_isdiag_list_all_nondiag():
    symbol1 = sympy.Symbol("X")
    symbol2 = sympy.Symbol("Y")
    value1 = np.array([[1, 0], [0, 1]])
    value2 = np.array([[0, 1], [1, 0]])
    opsite1 = OpSite(symbol1, 0, value=value1, isdiag=False)
    opsite2 = OpSite(symbol2, 1, value=value2, isdiag=False)
    opproductsite1 = OpProductSite([opsite1])
    opproductsite2 = OpProductSite([opsite2])
    sumofproducts = SumOfProducts([opproductsite1, opproductsite2])
    assert sumofproducts.isdiag_list == [False, False]
