import numpy as np
from loguru import logger

from pympo.utils import (
    _validate_mpo,
    export_itensor_hdf5,
    export_npz,
    full_matrix,
    import_itensor_hdf5,
    import_npz,
    qr,
    sum_mpo,
    svd,
    to_tensor_train,
)


def test_util():
    mpo1 = [
        np.random.rand(1, 10, 2),
        np.random.rand(2, 10, 3),
        np.random.rand(3, 10, 10, 1),
    ]
    mpo2 = [
        np.random.rand(1, 10, 3),
        np.random.rand(3, 10, 10, 3),
        np.random.rand(3, 10, 10, 1),
    ]
    _validate_mpo(mpo1)
    _validate_mpo(mpo2)
    mpo = sum_mpo(mpo1, mpo2)
    _validate_mpo(mpo)
    export_npz(mpo, "test.npz")
    mpo = import_npz("test.npz")
    _validate_mpo(mpo)
    mpo = [
        np.random.rand(1, 3, 3, 2),
        np.random.rand(2, 3, 3, 2),
        np.random.rand(2, 3, 3, 1),
    ]
    _validate_mpo(mpo)
    import os

    test_dir = os.path.dirname(__file__)
    export_itensor_hdf5(mpo, os.path.join(test_dir, "test.h5"), "H")
    mpo_imported = import_itensor_hdf5(os.path.join(test_dir, "test.h5"), "H")
    for core1, core2 in zip(mpo, mpo_imported, strict=True):
        np.testing.assert_allclose(core1, core2, atol=1e-10)

    mpo = [
        np.random.rand(1, 10, 10, 8),
        np.random.rand(8, 4, 4, 3),
        np.random.rand(3, 5, 5, 1),
    ]
    mpo_qr = qr(mpo)
    mpo_svd = svd(mpo, eps=2e-02)

    full = full_matrix(mpo)
    full_qr = full_matrix(mpo_qr)
    full_svd = full_matrix(mpo_svd)
    np.testing.assert_allclose(full, full_qr, atol=1e-20)
    assert (
        np.mean(np.abs(full - full_svd)) < 1e-01
    ), f"The difference is {np.mean(np.abs(full - full_svd))}"

    mpo = to_tensor_train(full, [10, 4, 5])
    _validate_mpo(mpo)
    full_again = full_matrix(mpo)
    np.testing.assert_allclose(full, full_again, atol=1e-20)

    logger.info("All tests passed")
