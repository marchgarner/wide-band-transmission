#This code was tested using Python 3.6.4, GPAW 1.3.0, and ASE 3.16.0

from ase import Atoms
from ase.io import read, write
#from ase.visualize import view
from ase.dft.kpoints import monkhorst_pack

from gpaw import GPAW, FermiDirac
from gpaw import setup_paths
from gpaw.lcao.tools import get_lcao_hamiltonian, get_lead_lcao_hamiltonian
from gpaw.lcao.tools import dump_hamiltonian_parallel, get_bfi2

import pickle as pickle

import numpy as np
from numpy import ascontiguousarray as asc

from utilities_wideband import calc_trans, plot_transmission, identify_and_align,plot_basis

#Arguments can be put directly in when running the code. By default it reads the junction file "hh_junc.traj" and runs the program with a dzp basis.
"""ARGPARSE"""
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--path',
    default='.',
    help='path to data folder')
parser.add_argument(
    '--xyzname',
    default='hh_junc.traj',
    help='name of the xyz file')
parser.add_argument('--basis',
                    default='dzp',
                    help='basis (sz, dzp, ...)')
parser.add_argument('--ef',
                    default=0.,
                    help='fermi')
args = parser.parse_args()
ef = float(args.ef)

import os
basis = args.basis
path = os.path.abspath(args.path) + "/"
xyzname = args.xyzname


#First half of the script dumps the Hamiltonian of the central region.
"""
Constants
"""
xc = 'PBE'
FDwidth = 0.1
kpts = (1, 1, 1)
mode = 'lcao'
h = 0.20
vacuum = 4
basis_full = {'H': 'sz', 'C': basis, 'S': basis, 'N': basis, 'Si': basis,'Ge': basis, 'B': basis, 'O': basis, 'F': basis, 'Cl': basis, 'P': basis}

"""
Read molecule
"""
molecule = read(path + xyzname)

"""
Identify end atoms and align according to z-direction

atoms the furthers from one another
"""
#Note, this is not automated, hydrogen electrodes have same number in the molecule file always.
atoms = identify_and_align(molecule)
#view(atoms)
"""
Run and converge calculation
"""
calc = GPAW(h=h,
            xc=xc,
            basis=basis_full,
            occupations=FermiDirac(width=FDwidth),
            kpts=kpts,
            mode=mode,
            symmetry={'point_group': False, 'time_reversal': False})
atoms.set_calculator(calc)
atoms.get_potential_energy()  # Converge everything!
Ef = atoms.calc.get_fermi_level()

wfs = calc.wfs
kpt = monkhorst_pack((1, 1, 1))

basename = "basis_{0}__xc_{1}__h_{2}__fdwithd_{3}__kpts_{4}__mode_{5}__vacuum_{6}__".format(
    basis, xc, h, FDwidth, kpts, mode, vacuum)

dump_hamiltonian_parallel(path + 'hamiltonian', atoms, direction='z')

#clean up and save some memory
del atoms
del calc
del wfs
del kpt
del basename
del Ef

#calculate the transmission using Hamiltonian that was dumped.

#effective coupling into dihydrogen electrodes
gamma=1e0 

#Energy grid to calculate transmission over
estart=-3
eend=3
es=3e-2

# constants
eV2au = 1/27.211
au2eV = 27.211
au2A = 0.529177249

#Load Hamiltonian
fname = "hamiltonian0.pckl"

#Calculation is performed in atomic units
estart *= eV2au
eend *= eV2au
es *= eV2au
gamma *= eV2au

H_ao, S_ao = pickle.load(open(path+fname, 'rb'))
H_ao = H_ao[0, 0]
S_ao = S_ao[0]
n = len(H_ao)

H_ao = H_ao *eV2au
GamL = np.zeros([n,n])
GamR = np.zeros([n,n])
GamL[0,0] = gamma
GamR[n-1,n-1] = gamma

print("Calculating transmission")
e_grid = np.arange(estart, eend, es)
Gamma_L = [GamL for en in range(len(e_grid))]
Gamma_R = [GamR for en in range(len(e_grid))]
Gamma_L = np.swapaxes(Gamma_L, 0, 2)
Gamma_R = np.swapaxes(Gamma_R, 0, 2)

###Test memory usage###
#import psutil
#print("before transmission",psutil.virtual_memory())

Gr = []

#Make retarded gf array
for en in range(len(e_grid)):
	#print('calculating retarded-GF, step', en, " with memory",psutil.virtual_memory())
	Gr.append(np.linalg.inv(e_grid[en]*S_ao-H_ao+(1j/2.)*(Gamma_L[:, :, en]+Gamma_R[:, :, en])))
	
Gr = np.asarray(Gr)

Gr = np.swapaxes(Gr, 0, 2)
#print("Before transmission",psutil.virtual_memory())
trans = calc_trans(e_grid, Gr, Gamma_L, Gamma_R)
plot_transmission(e_grid*27.211399, trans, path+"trans.png")
np.save(path+'transmission.npy',[e_grid*27.211399,trans]) 
print("transmission done")

