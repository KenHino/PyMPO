#!/usr/bin/env python
# coding: utf-8

# # Construct MPO of radical pair system by [RadicalPy](https://github.com/Spin-Chemistry-Labs/radicalpy)
#
# In this tutorial, one constructs MPO of radical pair system (two electron spins and a couple of nuclear spins under magnetic field) by using RadicalPy library.

# In[1]:


import radicalpy


# In[2]:


import numpy as np
from sympy import Symbol
from pympo import (
    AssignManager,
    OpSite,
    SumOfProducts,
)
import matplotlib.pyplot as plt

import radicalpy as rp
from radicalpy.simulation import State


# ## Total Hamiltonian
# $$
# \hat{H}_{\text{total}} =
# \hat{H}_{\text{Z}} + \hat{H}_{\text{H}} + \hat{H}_{\text{J}} + \hat{H}_{\text{D}}
# $$

# ## Define systems

# In[3]:


flavin = rp.simulation.Molecule.fromisotopes(isotopes=["1H"], hfcs=[4.0e-01])
Z = rp.simulation.Molecule.fromisotopes(isotopes=["1H"], hfcs=[5.0e-01])
sim = rp.simulation.HilbertSimulation([flavin, Z])
sim


# Now, one defines matrix product state (MPS) in the following order
#
# (nuclei in `flavin`) $\to$ (electronic states $\{|T_{+}\rangle, |T_{0}\rangle, |S\rangle, |T_{-}\rangle\}$) $\to$ (neclei in `Z`)

# ## Extract one particle operator
#
# RadicalPy provides variety of spin operators such as
#
# - $\hat{s}_x, \hat{s}_y, \hat{s}_z$ for radical singlet-triplet basis
# - $\hat{I}_x, \hat{I}_y, \hat{I}_z$ for nuclear Zeeman basis

# In[4]:


# Clear nuclei temporally
_nuclei_tmp0 = sim.molecules[0].nuclei
_nuclei_tmp1 = sim.molecules[1].nuclei
sim.molecules[0].nuclei = []
sim.molecules[1].nuclei = []

ST = np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        )

sx_1 = sim.spin_operator(0, 'x') # for Zeeman basis
sy_1 = sim.spin_operator(0, 'y') # for Zeeman basis
sz_1 = sim.spin_operator(0, 'z') # for Zeeman basis
sx_2 = sim.spin_operator(1, 'x') # for Zeeman basis
sy_2 = sim.spin_operator(1, 'y') # for Zeeman basis
sz_2 = sim.spin_operator(1, 'z') # for Zeeman basis

plt.imshow((sx_1 @ sx_2 + sy_1 @ sy_2 + sz_1 @ sz_2).real, cmap='bwr', vmin=-1.0, vmax=1.0)
plt.title(r'$\hat{S}_1\cdot \hat{S}_2$ (Singlet Triplet basis)')
plt.colorbar()
plt.xticks([0, 1, 2, 3], [r'$|T_+\rangle$', r'$|T_0\rangle$', r'$|S\rangle$', r'$|T_-\rangle$'])
plt.yticks([0, 1, 2, 3], [r'$\langle T_+|$', r'$\langle T_0 |$', r'$\langle S|$', r'$\langle T_- |$'])
plt.show()

# Reverse nuclei
sim.molecules[0].nuclei = _nuclei_tmp0
sim.molecules[1].nuclei = _nuclei_tmp1

sim.particles

for i, op1 in enumerate([sx_1, sy_1, sz_1, sx_2, sy_2, sz_2]):
    for j, op2 in enumerate([sx_1, sy_1, sz_1, sx_2, sy_2, sz_2]):
        try:
            np.testing.assert_allclose(op1 @ op2, op2 @ op1, atol=1e-14)
        except:
            print(i,j)


# ## Define `OpSite` and coefficients

# In[5]:


# Coefficient
SCALE = 1.0e-07

A = {}
J = 0.0
B = np.array((0.0, 0.0, 0.0))
gamma = [p.gamma_mT for p in sim.particles]

g_ele_sym = [Symbol(r"\gamma_e^{(" + f"{i+1}"+")}") for i in range(len(sim.radicals))]
g_nuc_sym = [[Symbol(r"\gamma_n^{" + f"{(i+1,j+1)}"+"}") for j in range(len(sim.molecules[i].nuclei))] for i in range(len(sim.radicals))]

subs = {}
for i, ge in enumerate(g_ele_sym):
    subs[ge] = sim.radicals[i].gamma_mT * SCALE
    for j, gn in enumerate(g_nuc_sym[i]):
        subs[gn] = sim.molecules[i].nuclei[j].gamma_mT * SCALE

D = np.zeros((3, 3))

# Isotropic
for i in range(len(sim.radicals)):
    for j, nuc in enumerate(sim.molecules[i].nuclei):
        A[(i,j)] = np.eye(3) * nuc.hfc.isotropic

subs, A


# In[6]:


Sx_ops = []
Sy_ops = []
Sz_ops = []

I1x_ops = []
I1y_ops = []
I1z_ops = []

I2x_ops = []
I2y_ops = []
I2z_ops = []

for j, nuc in enumerate(sim.molecules[0].nuclei):
    I1x_ops.append(OpSite(r"\hat{I}_x^{" + f"{(1, j+1)}" + "}", j, value=nuc.pauli['x']))
    I1y_ops.append(OpSite(r"\hat{I}_y^{" + f"{(1, j+1)}" + "}", j, value=nuc.pauli['y']))
    I1z_ops.append(OpSite(r"\hat{I}_z^{" + f"{(1, j+1)}" + "}", j, value=nuc.pauli['z']))

ele_site = len(sim.molecules[0].nuclei)
for j, nuc in enumerate(sim.molecules[1].nuclei):
    I2x_ops.append(OpSite(r"\hat{I}_x^{" + f"{(2, j+1)}" + "}", ele_site + 1 + j, value=nuc.pauli['x']))
    I2y_ops.append(OpSite(r"\hat{I}_y^{" + f"{(2, j+1)}" + "}", ele_site + 1 + j, value=nuc.pauli['y']))
    I2z_ops.append(OpSite(r"\hat{I}_z^{" + f"{(2, j+1)}" + "}", ele_site + 1 + j, value=nuc.pauli['z']))

Ix_ops = [I1x_ops, I2x_ops]
Iy_ops = [I1y_ops, I2y_ops]
Iz_ops = [I1z_ops, I2z_ops]

S1S2_op = OpSite(r"\hat{S}_1\cdot\hat{S}_2", ele_site, value=(sx_1 @ sx_2 + sy_1 @ sy_2 + sz_1 @ sz_2))
E_op = OpSite(r"\hat{E}", ele_site, value=np.eye(*sx_1.shape))

Sx_ops.append(OpSite(r"\hat{S}_x^{(1)}", ele_site, value=sx_1))
Sy_ops.append(OpSite(r"\hat{S}_y^{(1)}", ele_site, value=sy_1))
Sz_ops.append(OpSite(r"\hat{S}_z^{(1)}", ele_site, value=sz_1))
Sx_ops.append(OpSite(r"\hat{S}_x^{(2)}", ele_site, value=sx_2))
Sy_ops.append(OpSite(r"\hat{S}_y^{(2)}", ele_site, value=sy_2))
Sz_ops.append(OpSite(r"\hat{S}_z^{(2)}", ele_site, value=sz_2))

Sr_ops = [Sx_ops, Sy_ops, Sz_ops]
Ir_ops = [Ix_ops, Iy_ops, Iz_ops]


# ## Hyperfine coupling Hamiltonian
# $$
# \hat{H}_{\text{H}} = \sum_i \sum_j \hat{S}_i\cdot A_{ij}\cdot \hat{I}_{ij}
# = \sum_i \sum_j \sum_{r\in\{x, y, z\}} A_{ij}\hat{S}_{r}^{(i)}\hat{I}_{r}^{(ij)}
# $$

# In[7]:


hyperfine = SumOfProducts()

xyz = "xyz"

for i in range(len(sim.radicals)):
    for j in range(len(sim.molecules[i].nuclei)):
        for k, Sr_op in enumerate(Sr_ops):
            for l, Ir_op in enumerate(Ir_ops):
                if A[(i,j)][k,l] == 0.0:
                    continue
                else:
                    print(i, j, k, l, A[(i, j)][k, l])
                Asym = Symbol("A^{" + f"{(i+1,j+1)}" + "}_{" + f"{xyz[k]}" + f"{xyz[l]}" + "}")
                subs[Asym] = A[(i,j)][k, l].item()
                hyperfine += Asym * g_ele_sym[i] * Sr_op[i] * Ir_op[i][j]


# In[8]:


hyperfine = hyperfine.simplify()
hyperfine.symbol


# In[9]:


am = AssignManager(hyperfine)


# In[10]:


_ = am.assign()


# In[11]:




# In[12]:


mpo = am.numerical_mpo(subs=subs)


# ## PyTDSCF (Tensor Network Simulation)

# In[13]:


import pytdscf


# In[14]:


from pytdscf import BasInfo, Model, Simulator, units, Exciton
from pytdscf.dvr_operator_cls import TensorOperator
from pytdscf.hamiltonian_cls import TensorHamiltonian


# In[15]:


backend = "numpy"
m = 30
Δt = 1.0e-11 / SCALE


# In[16]:


basis = []
for nuc in sim.molecules[0].nuclei:
    basis.append(Exciton(nstate=nuc.multiplicity))
basis.append(Exciton(nstate=4))
for nuc in sim.molecules[1].nuclei:
    basis.append(Exciton(nstate=nuc.multiplicity))
basinfo = BasInfo([basis], spf_info=None)

nsite = len(basis)


# In[17]:


op_dict = {tuple([(isite, isite) for isite in range(nsite)]): TensorOperator(mpo=mpo)}
H = TensorHamiltonian(nsite, potential=[[op_dict]], kinetic=None, backend=backend)


# In[18]:


density_sum = None
from itertools import product
from scipy.stats import qmc

# --- example: two discrete variables ---
#   X  in {0, 1}
#   Y  in {0, 1, 2}
engine = qmc.Sobol(d=nsite-1, scramble=True)
u = engine.random(n=2**3)                 # points in [0,1)^2
pairs = np.column_stack([np.floor(u[:,i] * basis[i + int(ele_site <= i)].nstate).astype(int) for i in range(nsite-1)])

print(pairs)
qmc_lists = []

for i, init_nuc_spins in enumerate(pairs):

    operators = {"hamiltonian": H}
    model = Model(basinfo=basinfo, operators=operators)
    model.m_aux_max = m
    model.init_HartreeProduct = [[]]

    for isite in range(nsite):
        if isite == ele_site:
            model.init_HartreeProduct[0].append([0, 0, 1, 0]) # Singlet
        else:
            # model.init_HartreeProduct[0].append((np.ones(basis[isite].nstate) / basis[isite].nstate).tolist())
            weights = np.zeros(basis[isite].nstate)
            weights[init_nuc_spins[isite - int(ele_site <= isite)]] = 1.0
            model.init_HartreeProduct[0].append(weights.tolist()) # all down

    print(model.init_HartreeProduct)


    # In[19]:


    jobname = f"radicalpair_{i}"
    simulator = Simulator(jobname=jobname, model=model, backend=backend)
    import time
    start = time.time()
    ener, wf = simulator.propagate(
        reduced_density=(
            [(ele_site, ele_site)],
            10,
        ),
        maxstep=1000, stepsize=Δt)#.(maxstep=100, stepsize=Δt)
    end = time.time()
    print(end-start)


    # In[20]:


    print(ener, wf)


    # In[21]:


    print(wf.norm())


    # In[22]:


    import netCDF4 as nc

    with nc.Dataset(f"{jobname}_prop/reduced_density.nc", "r") as file:
        density_data_real = file.variables[f"rho_({ele_site}, {ele_site})_0"][:]["real"]
        density_data_imag = file.variables[f"rho_({ele_site}, {ele_site})_0"][:]["imag"]
        time_data = file.variables["time"][:]


    # In[23]:


    density_data = np.array(density_data_real) + 1.0j * np.array(density_data_imag)
    time_data = np.array(time_data)

    if density_sum is None:
        density_sum = density_data
    else:
        density_sum += density_data


from pytdscf.util.anim_density_matrix import get_anim

density_data = density_sum / 4

fig, anim = get_anim(
    density_data,
    time_data,
    title="Reduced density matrix",
    time_unit="fs",
    save_gif=True,
    gif_filename="complex_matrix.gif",
    row_names=[r"$|T+\rangle$", r"$|T0\rangle$", r"$|S\rangle$", r"$|T-\rangle$"],
    col_names=[r"$\langle T+|$", r"$\langle T0|$", r"$\langle S|$", r"$\langle T-|$"],
)


# In[24]:




# In[25]:

plt.clf()

plt.plot(time_data, density_data[:, 0, 0].real, label='T+')
plt.plot(time_data, density_data[:, 1, 1].real, label='T0')
plt.plot(time_data, density_data[:, 2, 2].real, label='S')
plt.plot(time_data, density_data[:, 3, 3].real, label='T-')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Popultation')
plt.savefig("population.png")
plt.show()


# In[ ]:
