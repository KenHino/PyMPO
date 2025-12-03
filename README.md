[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Deploy static content to Pages](https://github.com/KenHino/PyMPO/actions/workflows/static.yml/badge.svg)](https://github.com/KenHino/PyMPO/actions/workflows/static.yml)
[![unittest-uv](https://github.com/KenHino/PyMPO/actions/workflows/unittest-uv.yml/badge.svg)](https://github.com/KenHino/PyMPO/actions/workflows/unittest-uv.yml)

# PyMPO
Automatic and symbolic construction of matrix product operator (MPO)

Python (Sympy) is employed as interface and Rust as part of backend.

> [!NOTE]
> PyMPO employs [maturin](https://github.com/PyO3/maturin) as a build backend. Thus, [cargo](https://github.com/rust-lang/cargo) 1.78+, package manager of Rust, is required in your envoronment. If you have'nt, follow [rustup installation](https://www.rust-lang.org/tools/install).

## Document
https://kenhino.github.io/PyMPO/example/autompo-sym.html

## Installation

Just using [`uv`](https://docs.astral.sh/uv/)
```bash
$ uv sync
```

or

using `pip`
```bash
$ pip install git+https://github.com/KenHino/PyMPO
```


## Reference
- [Jiajun Ren, Weitang Li, Tong Jiang, Zhigang Shuai; A general automatic method for optimal construction of matrix product operators using bipartite graph theory. J. Chem. Phys. 28 August 2020; 153 (8): 084118.](https://doi.org/10.1063/5.0018149)
