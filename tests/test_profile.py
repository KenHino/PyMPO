import itertools
import math
import os
import sys
from collections import defaultdict
from typing import Literal

import numpy as np
import pytest
import sympy
from loguru import logger

import pympo

sys.path.append(os.getcwd())
# file = "ch2o_potential.py"
file = "c10h12_local_potential.py"

try:
    match file:
        case "c10h12_local_potential.py":
            from c10h12_local_potential import k_orig
        case "ch2o_potential.py":
            from ch2o_potential import k_orig
except ModuleNotFoundError:
    os.system(
        f"wget https://raw.githubusercontent.com/QCLovers/PyTDSCF/refs/heads/main/pytdscf/potentials/{file}"
    )
finally:
    match file:
        case "c10h12_local_potential.py":
            from c10h12_local_potential import k_orig
        case "ch2o_potential.py":
            from ch2o_potential import k_orig
        case _:
            raise ValueError(f"Invalid file: {file}")
try:
    from discvar import HarmonicOscillator as HO
except ModuleNotFoundError:
    # git clone from https://github.com/KenHino/Discvar and add `discvar` to sys.path
    # uv pip install discvar
    os.system("uv pip install git+https://github.com/KenHino/Discvar.git")
finally:
    from discvar import HarmonicOscillator as HO


cutoff = 1.0e-09
au_in_cm1 = 2.194746e5
N = 10

active_modes = sorted(list(set(itertools.chain.from_iterable(k_orig.keys()))))
index = {}
for i, mode in enumerate(active_modes):
    index[mode] = i
logger.info(f"{index=}")
f = len(active_modes)
k_new = {}
for key, value in k_orig.items():
    new_key = tuple([index[k] for k in key])
    if abs(value) > cutoff:
        k_new[new_key] = value
logger.info(f"{len(k_new)=}")

freqs = [math.sqrt(k_new[(k, k)]) * au_in_cm1 for k in range(f)]
logger.info(f"{freqs=}")

nprims = [N] * f  # Number of primitive basis
dvr_prims = [
    # HO(ngrid=nprim, omega=omega, units='cm-1')
    HO(nprim, omega)
    for nprim, omega in zip(nprims, freqs, strict=True)
]

dq2 = [prim.get_2nd_derivative_matrix_dvr() for prim in dvr_prims]
q_scale = 10
# q1 = [prim.get_pos_rep_matrix() / q_scale for prim in dvr_prims]
# q2 = [ints @ ints for ints in q1]
# q3 = [ints @ ints @ ints for ints in q1]
# q4 = [ints @ ints @ ints @ ints for ints in q1]
q1 = [np.array(prim.get_grids()) for prim in dvr_prims]
q2 = [ints**2 for ints in q1]
q3 = [ints**3 for ints in q1]
q4 = [ints**4 for ints in q1]
qn = [q1, q2, q3, q4]


@pytest.mark.parametrize("backend", ["py", "rs"])
def test_profile(backend: Literal["py", "rs"]):
    pympo.config.backend = backend
    logger.info(f"{pympo.config.backend=}")
    kinetic_sop = pympo.SumOfProducts([])
    for isite in range(f):
        kinetic_sop -= (
            1
            / 2
            * pympo.OpSite(
                r"\frac{\partial}{\partial q^2_{" + f"{isite}" + r"}}",
                isite,
                value=dq2[isite],
            )
        )
    potential_sop = pympo.SumOfProducts([])
    coef_symbol = {}
    for key, coef in k_new.items():
        cnt_site = defaultdict(int)
        for isite in key:
            cnt_site[isite] += 1
        op = 1
        coef_sym = sympy.Symbol(f"k_{key}")
        for isite, order in cnt_site.items():
            coef /= math.factorial(order)
            op *= pympo.OpSite(
                f"q^{order}" + r"_{" + f"{isite}" + r"}",
                isite,
                value=qn[order - 1][isite],
            )
        op *= coef_sym
        coef_symbol[coef_sym] = coef * (q_scale ** len(key))
        potential_sop += op
    hamiltonian_sop = pympo.SumOfProducts([])
    hamiltonian_sop += kinetic_sop
    hamiltonian_sop += potential_sop
    hamiltonian_sop = hamiltonian_sop.simplify()
    am_hamiltonian = pympo.AssignManager(hamiltonian_sop)
    am_hamiltonian.assign(keep_symbol=False)
    _ = am_hamiltonian.numerical_mpo(subs=coef_symbol)
    # assert sympy.Mul(*am_hamiltonian.Wsym)[0].expand() == hamiltonian_sop.symbol.expand()


if __name__ == "__main__":
    test_profile(backend="rs")
    # import cProfile
    # import pstats

    # profiler = cProfile.Profile()
    # profiler.run('test_profile()')
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')
    # stats.print_stats(30)  # 上位30行を表示
