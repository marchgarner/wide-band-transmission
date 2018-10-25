#This code was tested using Python 3.6.4, GPAW 1.3.0, and ASE 3.16.0

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write
#from ase.visualize import view

from gpaw import GPAW, FermiDirac
from gpaw import setup_paths
from gpaw.lcao.tools import get_lcao_hamiltonian, get_lead_lcao_hamiltonian
from gpaw.lcao.tools import dump_hamiltonian_parallel, get_bfi2

from numpy import ascontiguousarray as asc

import pickle
#from tqdm import tqdm

def plot_basis(atoms, phi_xG, ns, folder_name='./basis'):
    """
    r: coefficients of atomcentered basis functions
    atoms: Atoms-object 
    ns: indices of bfs functions to plot. 
    """
    # for n, phi in zip(ns, phi_xG.take(ns, axis=0)):
    n=0
    for phi in phi_xG:
        # print "writing %d of %d" %(n, len(ns)), 
        write('%s/%d.cube' % (folder_name,n), atoms, data=phi)
        n += 1


def distance(pos1, pos2):
    dis = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] -
                                            pos2[1])**2 + (pos1[2] - pos2[2])**2)
    return dis


def distance_matrix(pos):
    dM = np.array([[distance(pos[i], pos[j]) for i in range(pos.shape[0])]
           for j in range(pos.shape[0])])
    #sns.heatmap(dM)
    #plt.show()
    return dM



##We now always use junction with same numbered H-electrodes to align
def identify_and_align(molecule):
    vacuum = 4

	#Use distance matrix, this only works for well-behaved linear molecules, so we do not do this.
    pos = molecule.get_positions()

    dM = distance_matrix(pos)
    m = np.unravel_index(np.argmax(dM, axis=None), dM.shape)

    endatom1, endatom2 = m

    #electrode
    #sI = endatom1
    #eI = endatom2
    sI = -1
    eI = -3
    
    #alternative alignment atoms
    
    altatom1 = -1
    altatom2 = -3

    po = (molecule[altatom1].position)
    lo = (molecule[altatom2].position)

    v = lo - po
    z = [0, 0, 1]
    molecule.rotate(v, z)


    molecule.center(vacuum=vacuum)

    elec1 = Atoms('H', positions=[molecule[sI].position])
    elec2 = Atoms('H', positions=[molecule[eI].position])
    
    del molecule[eI]
    del molecule[sI]


    atoms = elec1 + molecule + elec2

    atoms.center(vacuum=vacuum)
    atoms.set_pbc([1, 1, 1])

    #view(atoms)
    return atoms

def plot_transmission(energy_grid, trans, save_name):
    """
    plots the transmission
    """
    plt.plot(energy_grid, trans)
    plt.yscale('log')
    plt.ylabel(r'Transmission')
    plt.xlabel(r'E-E$_F$ (eV)')
    plt.savefig(save_name)
    plt.close()

def calc_trans(energy_grid, gret, gamma_left, gamma_right):
    """
    Landauer Transmission
    """
    trans = np.array([np.dot(np.dot(np.dot(\
                    gamma_left[:, :, en], gret[:, :, en]),\
                    gamma_right[:, :, en]), gret[:, :, en].T.conj())\
                    for en in range(len(energy_grid))])
    trans_trace = np.array([trans[en, :, :].trace() for en in range(len(energy_grid))])
    return trans_trace
