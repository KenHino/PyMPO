from pathlib import Path
from typing import Any, Literal, TypeAlias, Union, cast

import h5py
import numpy as np
from loguru import logger
from numpy.typing import NDArray

# Define specific array shapes for different tensor ranks
Core3: TypeAlias = NDArray[
    Any
]  # 3-rank tensor with shape (left_bond, physical, right_bond)
Core4: TypeAlias = NDArray[
    Any
]  # 4-rank tensor with shape (left_bond, physical, physical, right_bond)
Core: TypeAlias = Union[Core3, Core4]  # Union of 3-rank and 4-rank tensors
Mpo: TypeAlias = list[Core]  # list of cores


def _validate_mpo(mpo: Mpo, support_3_rank: bool = True) -> list[int]:
    left_bd = 1
    bond_dims = [
        1,
    ]
    for i, core in enumerate(mpo):
        if not isinstance(core, np.ndarray) or core.dtype not in (
            np.float64,
            np.complex128,
        ):
            raise ValueError(
                f"Core {i} must be a numpy array with dtype float64 or complex128 but got {type(core)} with dtype {core.dtype}"
            )
        if core.ndim != 3 and core.ndim != 4:
            raise ValueError(
                f"Core {i} must be a 3-rank or 4-rank tensor but got {core.ndim}-rank tensor"
            )
        if core.ndim == 3 and not support_3_rank:
            raise ValueError(
                f"Core {i} must be a 4-rank tensor but got {core.ndim}-rank tensor"
            )
        if core.ndim == 4:
            if core.shape[1] != core.shape[2]:
                raise ValueError(
                    f"The bra and ket physical dimension is not consistent for core {i} with shape {core.shape}"
                )
            l, c, c, r = core.shape
            if l * c * c < r or l > c * c * r:
                raise ValueError(
                    f"The core {i} with shape {core.shape} has redundant bond dimension"
                )
        if core.ndim == 3:
            l, c, r = core.shape
            if l * c < r or l > c * r:
                raise ValueError(
                    f"The core {i} with shape {core.shape} has redundant bond dimension"
                )
        if l != left_bd:
            raise ValueError(
                f"The left bond dimension is not consistent for core {i} with shape {core.shape}. That should be {left_bd} but got {l}."
            )
        left_bd = r
        bond_dims.append(r)
    if left_bd != 1:
        raise ValueError(
            f"The right bond dimension of the final core is not 1 but {left_bd}"
        )
    return bond_dims


def fill_diag(core: Core) -> Core:
    if core.ndim == 3:
        l, c, r = core.shape
        tmp = np.zeros((l, c, c, r), dtype=core.dtype)
        i, j = np.diag_indices(c)
        tmp[:, i, j, :] = core[:, :, :]
        return tmp
    else:
        raise ValueError(
            f"The core is not a 3-rank tensor but {core.ndim}-rank tensor"
        )


def sum_mpo(mpo1: Mpo, mpo2: Mpo) -> Mpo:
    _validate_mpo(mpo1)
    _validate_mpo(mpo2)
    mpo = []
    for i, (core1, core2) in enumerate(zip(mpo1, mpo2, strict=True)):
        dtype = np.promote_types(core1.dtype, core2.dtype)
        if core1.ndim == 3:
            l1, c1, r1 = core1.shape
        else:
            l1, c1, c1, r1 = core1.shape
        if core2.ndim == 3:
            l2, c2, r2 = core2.shape
        else:
            l2, c2, c2, r2 = core2.shape
        assert c1 == c2
        if i == 0:
            # [[W1, W2]]
            assert l1 == l2 == 1
            if core1.ndim == 3 and core2.ndim == 3:
                l, c, r = 1, c1, r1 + r2
                core = np.zeros((l, c, r), dtype=dtype)
                core[0, :, :r1] = core1[0, :, :]
                core[0, :, r1:] = core2[0, :, :]
            elif core1.ndim == 4 and core2.ndim == 4:
                l, c, r = 1, c1, r1 + r2
                core = np.zeros((l, c, c, r), dtype=dtype)
                core[0, :, :, :r1] = core1[0, :, :, :]
                core[0, :, :, r1:] = core2[0, :, :, :]
            else:
                if core1.ndim == 3:
                    core1 = fill_diag(core1)
                if core2.ndim == 3:
                    core2 = fill_diag(core2)
                l, c, r = 1, c1, r1 + r2
                core = np.zeros((l, c, c, r), dtype=dtype)
                core[0, :, :, :r1] = core1[0, :, :, :]
                core[0, :, :, r1:] = core2[0, :, :, :]
        elif i == len(mpo1) - 1:
            # [[W1],
            #  [W2]]
            assert r1 == r2 == 1
            if core1.ndim == 3 and core2.ndim == 3:
                l, c, r = l1 + l2, c1, 1
                core = np.zeros((l, c, r), dtype=dtype)
                core[:l1, :, 0] = core1[:, :, 0]
                core[l1:, :, 0] = core2[:, :, 0]
            elif core1.ndim == 4 and core2.ndim == 4:
                l, c, r = l1 + l2, c1, 1
                core = np.zeros((l, c, c, r), dtype=dtype)
                core[:l1, :, :, 0] = core1[:, :, :, 0]
                core[l1:, :, :, 0] = core2[:, :, :, 0]
            else:
                if core1.ndim == 3:
                    core1 = fill_diag(core1)
                if core2.ndim == 3:
                    core2 = fill_diag(core2)
                l, c, r = l1 + l2, c1, 1
                core = np.zeros((l, c, c, r), dtype=dtype)
                core[:l1, :, :, 0] = core1[:, :, :, 0]
                core[l1:, :, :, 0] = core2[:, :, :, 0]
        else:
            # [[W1, 0],
            #  [0, W2]]
            l, c, r = l1 + l2, c1, r1 + r2
            if core1.ndim == 3 and core2.ndim == 3:
                core = np.zeros((l, c, r), dtype=dtype)
                core[:l1, :, :r1] = core1[:, :, :]
                core[l1:, :, r1:] = core2[:, :, :]
            elif core1.ndim == 4 and core2.ndim == 4:
                core = np.zeros((l, c, c, r), dtype=dtype)
                core[:l1, :, :, :r1] = core1[:, :, :, :]
                core[l1:, :, :, r1:] = core2[:, :, :, :]
            else:
                if core1.ndim == 3:
                    core1 = fill_diag(core1)
                if core2.ndim == 3:
                    core2 = fill_diag(core2)
                core = np.zeros((l, c, c, r), dtype=dtype)
                core[:l1, :, :, :r1] = core1[:, :, :, :]
                core[l1:, :, :, r1:] = core2[:, :, :, :]
        mpo.append(core)
    _validate_mpo(mpo)
    return mpo


def to_mps(tensor: np.ndarray) -> list[Core3]:
    mps = []
    ndim = tensor.ndim
    shape = tensor.shape
    norm = 1.0
    tensor = tensor[np.newaxis, ...]
    for i in range(ndim):
        q, r = np.linalg.qr(
            tensor.reshape(tensor.shape[0] * shape[i], -1), mode="reduced"
        )
        mps.append(cast(Core3, q.reshape(-1, shape[i], r.shape[0])))
        tensor = r
    assert r.shape == (
        1,
        1,
    ), f"The residual tensor is not a scalar but {r.shape}"
    norm = np.max(np.abs(r))
    mps[-1] *= np.sign(r[0, 0]).astype(np.float64)
    nth_root_norm = pow(norm, 1 / ndim)
    for core in mps:
        core *= nth_root_norm

    _validate_mpo(mps)
    return mps


def to_tensor_train(
    matrix_or_vector: np.ndarray, dims: list[int]
) -> list[Core]:
    if matrix_or_vector.ndim == 2:
        matrix = matrix_or_vector
        assert matrix.shape == (np.prod(dims), np.prod(dims))
        # Current shape: (d1 * d2 * d3 * ... * d1 * d2 * d3 * ...)
        tensor = matrix.reshape(*dims, *dims)
        # Current shape: (d1, d2, d3, ..., d1, d2, d3, ...)
        perm = [i for i in range(0, 2 * len(dims), 2)] + [
            i for i in range(1, 2 * len(dims), 2)
        ]
        tensor = tensor.transpose(np.argsort(perm))
        # Current shape: (d1, d1, d2, d2, d3, d3, ...)
        shape = []
        for d in dims:
            shape.append(d * d)
        tensor = tensor.reshape(*shape)
        # Current shape: (d1 * d1, d2 * d2, d3 * d3, ...)
        mps = to_mps(tensor)
        mpo = []
        for d, core in zip(dims, mps, strict=True):
            mpo.append(
                cast(Core4, core.reshape(core.shape[0], d, d, core.shape[2]))
            )
        return mpo
    elif matrix_or_vector.ndim == 1:
        vector = matrix_or_vector
        assert vector.shape == (np.prod(dims),)
        tensor = vector.reshape(*dims)
        mpo = to_mps(tensor)
        return mpo
    else:
        raise ValueError


def full_matrix(mpo: Mpo) -> np.ndarray:
    _validate_mpo(mpo)
    is_all_3_rank = all(core.ndim == 3 for core in mpo)
    if is_all_3_rank:
        vector = mpo[0]
        for core in mpo[1:]:
            vector = np.einsum("...i,ijk->...jk", vector, core)
        vector = vector[0, ..., 0]
        return vector.reshape(-1, order="F")
    else:
        matrix = mpo[0]
        size = matrix.shape[1]
        if matrix.ndim == 3:
            matrix = fill_diag(matrix)
        for core in mpo[1:]:
            if core.ndim == 3:
                core = fill_diag(core)
            size *= core.shape[1]
            matrix = np.einsum("...i,ijkl->...jkl", matrix, core)
        matrix = matrix[0, ..., 0]
        # Current shape: (d1, d1, d2, d2, d3, d3, ...)
        # Target shape: (d1, d2, d3, ..., d1, d2, d3, ...)
        perm = [i for i in range(0, 2 * len(mpo), 2)] + [
            i for i in range(1, 2 * len(mpo), 2)
        ]
        matrix = matrix.transpose(perm)
        matrix = matrix.reshape(size, size)
        return matrix


def qr(mpo: Mpo) -> Mpo:
    _validate_mpo(mpo)
    mpo_qr = []
    r = np.eye(1)
    eps = np.finfo(float).eps
    for core in mpo:
        if core.ndim == 4:
            i, j, k, l = core.shape
            h = r.shape[0]
            q, r = np.linalg.qr(
                np.einsum("hi,ijkl->hjkl", r, core).reshape(h * j * k, l),
                mode="reduced",
            )
        else:
            i, j, l = core.shape
            h = r.shape[0]
            q, r = np.linalg.qr(
                np.einsum("hi,ijl->hjl", r, core).reshape(h * j, l),
                mode="reduced",
            )
        max_r = np.max(np.abs(r))
        r /= max_r
        nonzero_indices = ~np.all(np.abs(r) < eps, axis=1)
        q = q[:, nonzero_indices] * max_r
        r = r[nonzero_indices, :]
        m = q.shape[-1]
        if core.ndim == 4:
            mpo_qr.append(cast(Core4, q.reshape(h, j, k, m)))
        else:
            mpo_qr.append(cast(Core3, q.reshape(h, j, m)))
    sign = np.sign(r[0, 0])
    mpo_qr[-1] *= sign
    r *= sign
    np.testing.assert_allclose(r, np.eye(1), atol=1e-10)
    _validate_mpo(mpo_qr)
    return mpo_qr


def svd(mpo: Mpo, eps: float, already_qr: bool = False) -> Mpo:
    assert 0.0 < eps < 1.0, "The truncation error must be between 0.0 and 1.0"
    if eps > 0.1:
        logger.warning(
            f"The truncation error is {eps}, which is large. The result may not be accurate."
        )
    if not already_qr:
        mpo = qr(mpo)
    else:
        mpo = [core.copy() for core in mpo]
        _validate_mpo(mpo)
    nsite = len(mpo)
    is_all_3_rank = all(core.ndim == 3 for core in mpo)
    for i in range(nsite - 1, 0, -1):
        if is_all_3_rank:
            l, c, r = mpo[i].shape
            twodot = np.tensordot(mpo[i - 1], mpo[i], axes=(-1, 0))
            twodot = twodot.reshape(-1, c * r)
        else:
            if mpo[i].ndim == 3:
                mpo[i] = fill_diag(mpo[i])
            if mpo[i - 1].ndim == 3:
                mpo[i - 1] = fill_diag(mpo[i - 1])
            l, c, c, r = mpo[i].shape
            twodot = np.tensordot(mpo[i - 1], mpo[i], axes=(-1, 0))
            twodot = twodot.reshape(-1, c * c * r)
        u, s, vh = np.linalg.svd(twodot, full_matrices=False)
        cumsum_s = np.cumsum(s)
        cumsum_s /= cumsum_s[-1]
        j = np.searchsorted(cumsum_s, 1 - eps, side="left")
        s = s[: j + 1]
        sqrt_norm = np.sqrt(np.max(s))
        u = u[:, : j + 1]
        us = u @ np.diag(s) / sqrt_norm
        vh = vh[: j + 1, :] * sqrt_norm
        if is_all_3_rank:
            l, c, r = mpo[i].shape
            mpo[i] = cast(Core3, vh.reshape(-1, c, r))
            l, c, r = mpo[i - 1].shape
            mpo[i - 1] = cast(Core3, us.reshape(l, c, -1))
        else:
            l, c, c, r = mpo[i].shape
            mpo[i] = cast(Core4, vh.reshape(-1, c, c, r))
            l, c, c, r = mpo[i - 1].shape
            mpo[i - 1] = cast(Core4, us.reshape(l, c, c, -1))
    _validate_mpo(mpo)
    return mpo


def _validate_filename(
    filename: Path | str, suffix: str, mode: Literal["r", "w"]
) -> Path:
    if not isinstance(filename, Path):
        filename = Path(filename)
    if filename.suffix == "":
        filename = filename.with_suffix(suffix)
    if mode == "r":
        if not filename.exists():
            raise FileNotFoundError(f"The file {filename} does not exist")
        if not filename.is_file():
            raise IsADirectoryError(f"The file {filename} is a directory")
    if not filename.suffix == suffix:
        raise ValueError(f"The file {filename} is not a {suffix} file")
    return filename


def export_npz(mpo: Mpo, filename: Path | str) -> None:
    """
    Export an MPO to a numpy .npz file.

    Args:
        mpo: The MPO to export.
        filename: The path to the .npz file.

    Returns:
        None
    """
    _validate_mpo(mpo)
    filename = _validate_filename(filename, ".npz", "w")
    data: dict[str, NDArray[np.float64 | np.complex128]] = {
        f"W{i}": core for i, core in enumerate(mpo)
    }
    np.savez(filename, allow_pickle=True, **data)
    logger.info(f"The MPO has been exported to {filename}")


def import_npz(filename: Path | str) -> Mpo:
    """
    Import an MPO from a numpy .npz file exported by `pympo.utils.export_npz`.

    Args:
        filename: The path to the .npz file.

    Returns:
        The MPO.
    """
    filename = _validate_filename(filename, ".npz", "r")
    data = np.load(filename)
    mpo = [cast(Core, data[f"W{i}"]) for i in range(len(data.files))]
    logger.info(f"The MPO has been imported from {filename}")
    _validate_mpo(mpo)
    return mpo


def export_itensor_hdf5(
    mpo: Mpo, filename: Path | str, name: str = "H"
) -> None:
    """
    EXPERIMENTAL and SUPPORT BOSON MPO ONLY

    Args:
        mpo (Mpo): The MPO to export.
        filename (Path | str): The path to the .h5 file.
        name (str): The name of the MPO in the .h5 file.

    One needs to prepare the hdf5 file in advance with ITensors.jl.

    ```julia
    using ITensors, ITensorMPS, HDF5
    N = 6
    sites = siteinds("Boson", N, dim=10)
    f = h5open("mpo.h5", "w")
    write(f, "H", randomMPO(sites))
    close(f)
    ```

    Then, one can use `pympo.utils.export_itensor_hdf5` to export the MPO.

    ```python
    export_itensor_hdf5(mpo, "mpo.h5", "H")
    ```

    Finally, one can use MPO in ITensor.jl.

    ```julia
    using ITensors, ITensorMPS, HDF5
    f = h5open("mpo.h5", "r")
    H = read(f, "H")
    close(f)
    sites = [siteinds(H)[i][2] for i in 1:length(H)]
    ```

    Returns:
        None

    See also:
        - https://itensor.github.io/ITensors.jl/dev/examples/ITensor.html#Write-and-Read-an-ITensor-to-Disk-with-HDF5

    ToDo:
        - support QN conservation MPO
    """
    bond_dims = _validate_mpo(mpo, support_3_rank=False)
    filename = _validate_filename(filename, ".h5", "r")
    f = h5py.File(filename, "r+")
    N = f[name]["length"][()]  # number of sites
    assert N == len(mpo), (
        f"The number of sites in the MPO is {len(mpo)} but {N} in the HDF5file"
    )

    for i in range(1, N + 1):
        site = f[name][f"MPO[{i}]"]
        n_index = site["inds"]["length"][()]
        data = mpo[i - 1]
        orig2tmp = [0, 1, 2, 3]
        for j in range(n_index):
            index = site["inds"][f"index_{j + 1}"]
            tags = index["tags"]["tags"][()]
            changed_index = orig2tmp.index(j)
            if tags == f"Link,l={i}".encode():
                index["dim"][()] = bond_dims[i]
                data = np.swapaxes(data, orig2tmp[3], j)
                orig2tmp[changed_index], orig2tmp[3] = (
                    orig2tmp[3],
                    orig2tmp[changed_index],
                )
            elif tags in [
                f"Boson,Site,n={i}".encode(),
                f"S=1/2,Site,n={i}".encode(),
            ]:
                assert (
                    index["dim"][()]
                    == mpo[i - 1].shape[1]
                    == mpo[i - 1].shape[2]
                ), (
                    f"The physical dimension of the MPO is {mpo[i - 1].shape[1]} but {index['dim'][()]} in the HDF5file"
                )
                if index["plev"][()] == 1:
                    data = np.swapaxes(data, orig2tmp[1], j)
                    orig2tmp[changed_index], orig2tmp[1] = (
                        orig2tmp[1],
                        orig2tmp[changed_index],
                    )
                else:
                    data = np.swapaxes(data, orig2tmp[2], j)
                    orig2tmp[changed_index], orig2tmp[2] = (
                        orig2tmp[2],
                        orig2tmp[changed_index],
                    )
            elif tags == f"Link,l={i - 1}".encode():
                index["dim"][()] = bond_dims[i - 1]
                data = np.swapaxes(data, orig2tmp[0], j)
                orig2tmp[changed_index], orig2tmp[0] = (
                    orig2tmp[0],
                    orig2tmp[changed_index],
                )
            else:
                raise ValueError(f"Unknown tag: {tags}")
        data_reshaped = data.reshape(-1, order="F")
        if "data" not in site["storage"]:
            raise ValueError(
                "No data in storage. use randomMPO[sites] instead of MPO[sites]"
            )
        del site["storage"]["data"]
        site["storage"].create_dataset("data", data=data_reshaped, chunks=True)
    f.close()
    logger.info(f"The MPO has been exported to {filename}")


def import_itensor_hdf5(filename: Path | str, name: str = "H") -> Mpo:
    """
    Import an MPO from a .h5 file exported by ITensor.jl HDF5 format.

    Args:
        filename (Path | str): The path to the .h5 file.
        name (str): The name of the MPO in the .h5 file.

    Returns:
        The MPO.

    One needs to prepare the hdf5 file in advance with ITensors.jl.

    ```julia
    using ITensors, ITensorMPS, HDF5
    N = 6
    sites = siteinds("Boson", N, dim=10)
    f = h5open("mpo.h5", "w")
    write(f, "H", randomMPO(sites))
    close(f)
    ```

    Then, one can use `pympo.utils.import_itensor_hdf5` to import the MPO.

    ```python
    mpo = import_itensor_hdf5("mpo.h5", "H")
    ```

    See also:
        - https://itensor.github.io/ITensors.jl/dev/examples/ITensor.html#Write-and-Read-an-ITensor-to-Disk-with-HDF5

    """
    filename = _validate_filename(filename, ".h5", "r")
    f = h5py.File(filename, "r+")
    N = f[name]["length"][()]  # number of sites
    mpo = []
    for i in range(1, N + 1):
        site = f[name][f"MPO[{i}]"]
        n_index = site["inds"]["length"][()]
        swap = []
        shape = []
        if i == 1:
            swap.append(0)
            shape.append(1)

        for j in range(n_index):
            index = site["inds"][f"index_{j + 1}"]
            tags = index["tags"]["tags"][()]
            if tags == f"Link,l={i}".encode():
                r = index["dim"][()]
                shape.append(r)
                swap.append(3)
            elif tags in [
                f"Boson,Site,n={i}".encode(),
                f"S=1/2,Site,n={i}".encode(),
            ]:
                if index["plev"][()] == 1:
                    swap.append(1)
                else:
                    swap.append(2)
                c = index["dim"][()]
                shape.append(c)
            elif tags == f"Link,l={i - 1}".encode():
                l = index["dim"][()]
                swap.append(0)
                shape.append(l)
            else:
                raise ValueError(f"Unknown tag: {tags}")
        if "data" not in site["storage"]:
            raise ValueError(
                "No data in storage. use randomMPO[sites] instead of MPO[sites]"
            )

        if i == N:
            swap.append(3)
            shape.append(1)
        data = np.array(site["storage"]["data"][()])
        data = data.reshape(tuple(shape), order="F").transpose(
            *np.argsort(swap)
        )
        mpo.append(cast(Core4, data))
    f.close()
    logger.info(f"The MPO has been imported from {filename}")
    _validate_mpo(mpo, support_3_rank=False)
    return mpo
