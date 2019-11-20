
"""

Calculate and plot the eigenvalues of the hamiltonian for a magnetic dipole in a magnetic field subject to a time dependent periodic
perturbation, in this case a circularly polarized field (Rabi ploblem)
or a linearly polarized field, using Floquet-Fourier approach.

Author: Carla Borja Espinosa
Date: Julio 2018

Example run line in terminal: 
$ python3 MagneticDipole.py 3

"""

import sympy as sp
import numpy as np
from numpy import linalg as la
from sympy.abc import t
import matplotlib.pyplot as plt
from sympy.printing.str import StrPrinter
from sys import argv

class Hamiltonian(object):
    """Class to calculate and plot the eigenvalues"""
    def __init__(self):
        """Variables"""
        self.harm = int(argv[1])    #number of harmonics
        self.T = 2*sp.pi    #period of the perturbation
        self.g = 0.5    #amplitude of the perturbation
        self.h = 1    #Plank constant
        self.wo = sp.Symbol('wo',real=True)  #magentic dipole frequency
        self.delta = self.h*self.wo    
        self.w = 2*sp.pi/self.T    #frequency of the perturbation
        
    def FreeHamiltonian(self):
        """Define the free evolution hamiltonian"""
        sz = sp.Matrix([[1,0],[0,-1]])
        self.Ho = (-self.delta/2)*sz

    def TDPLinearizedField(self):
        """Define the time dependent hamiltonian for the linearly
           polarized field perturbation"""
        sx = sp.Matrix([[0,1],[1,0]])
        self.Vt = self.g*sp.cos(self.w*t)*sx
        
    def TDPCircularizedField(self):
        """Define the time dependent hamiltonian for the circularly
           polarized field perturbation"""
        m = sp.Matrix([[0,self.g*sp.exp(sp.I*self.w*t)],[self.g*sp.exp
                         (-sp.I*self.w*t),0]])
        self.Vt = -self.h/2*m
        
    def TotalHamiltonian(self):
        """Defines the total hamiltonian"""
        self.H = self.Ho + self.Vt
        self.size = self.H.shape[1]

    def NonDiagonalElements(self,n,m):
        """Calculate the non diagonal elements of the time-independent 
           Floquet hamiltonian"""
        Hnm = sp.zeros(self.size,self.size)
        for i in range(0,self.size):
            for j in range(0,self.size):
                Hnm[i,j] = 1/self.T*sp.integrate(
                           self.Vt[i,j]*sp.exp(-sp.I*(m-n)*
                           self.w*t),(t,0,self.T),conds='none')
        return(Hnm)

    def DiagonalElements(self,n):
        """Calculate the diagonal elements of the time-independent 
           Floquet hamiltonian"""
        return(self.Ho+n*self.w*sp.eye(self.size)) 

    def FHamiltonian(self,n):
        """Set up the time-independent Floquet hamiltonian of the 
           system defined by the total hamiltonian"""
        Hf = sp.Matrix([])
        ndep = self.NonDiagonalElements(0,1)
        nden = self.NonDiagonalElements(1,0)
        for i in range(-n,n+1):
            fila = sp.Matrix([])
            for j in range(-n,n+1):
                if j==i:
                    fila = fila.row_join(self.DiagonalElements(i))
                elif j==i+1:
                    fila = fila.row_join(ndep)
                elif j==i-1:
                    fila = fila.row_join(nden)
                else:
                    fila = fila.row_join(sp.Matrix([[0,0],[0,0]]))
            Hf = Hf.col_join(fila) 
        Hf.simplify()
        printer = StrPrinter()
        print(Hf.table(printer,align='center'))
        self.Hf = Hf
        
    def EigenvalsPlot(self):
        """Calculate and plot the eigenvalues of the time-independent 
           Floquet hamiltonian"""
        Hfwo = sp.lambdify('wo',self.Hf)
        wo_vals = np.arange(-4,4.1,0.1)
        size_wo = np.size(wo_vals)
        size_aut = self.Hf.shape[1]
        graph = np.empty([size_wo,size_aut])
        for i in range(0,size_wo):
            energies,vectors = la.eig(Hfwo(wo_vals[i]))
            idx = np.argsort(energies)
            energies = energies[idx]
            graph[i] = energies
        for i in range(0,size_aut):
            plt.plot(wo_vals,graph[:,i],color='g')
        plt.show() 
        #plt.ylim(self.h*self.w/2, self.h*self.w/2)    #Brillouin Zone
        #plt.show()

def main():
    """Command Line Executable"""
    hamil = Hamiltonian()
    hamil.FreeHamiltonian()
    hamil.TDPLinearizedField()
    #hamil.TDPCircularizedField()
    hamil.TotalHamiltonian()
    hamil.FHamiltonian(hamil.harm)
    hamil.EigenvalsPlot()

main()
