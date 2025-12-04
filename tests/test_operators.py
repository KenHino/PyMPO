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


def test_sumofproducts_to_mpo():
    """Test that to_mpo() returns a valid MPO representation."""
    # Create a simple 2-site operator: z_0 + z_1
    value = np.array([1.0, -1.0])  # Pauli Z eigenvalues
    opsite0 = OpSite("z_0", 0, value=value, isdiag=True)
    opsite1 = OpSite("z_1", 1, value=value, isdiag=True)
    sumofproducts = SumOfProducts([opsite0, opsite1])

    # Test to_mpo()
    mpo = sumofproducts.to_mpo()

    # Check that mpo is a list of NDArray
    assert isinstance(mpo, list)
    assert len(mpo) == 2  # 2 sites
    for core in mpo:
        assert isinstance(core, np.ndarray)
        # MPO cores should be 3-rank or 4-rank tensors
        assert core.ndim in (3, 4)


def test_sumofproducts_to_mpo_consistency():
    """Test that to_mpo() gives the same result as using AssignManager directly."""
    import pympo

    # Create a simple 2-site operator: z_0 * z_1
    value = np.array([1.0, -1.0])  # Pauli Z eigenvalues
    opsite0 = OpSite("z_0", 0, value=value, isdiag=True)
    opsite1 = OpSite("z_1", 1, value=value, isdiag=True)
    op_product = OpProductSite([opsite0, opsite1])
    sumofproducts = SumOfProducts([op_product])

    # Test to_mpo()
    mpo_from_to_mpo = sumofproducts.to_mpo()

    # Test using AssignManager directly
    am = pympo.AssignManager(sumofproducts)
    am.assign(keep_symbol=False)
    mpo_from_am = am.numerical_mpo(parallel=True)

    # Compare results
    assert len(mpo_from_to_mpo) == len(mpo_from_am)
    for core1, core2 in zip(mpo_from_to_mpo, mpo_from_am, strict=True):
        np.testing.assert_allclose(core1, core2, atol=1e-10)
