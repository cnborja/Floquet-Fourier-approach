
"""
Calculate and plot the quasienergy spectra from the bulk hamiltoninan
of the Su-Schrieffer-Heeger (SSH) tigh biding model where the hoppings 
are modulated in time.

For detailed information about the system consult the article:
Dal Lago, V., Atala, M., & Torres, L. F. (2015). Floquet topological 
transitions in a driven one-dimensional topological insulator. 
Physical Review A, 92(2), 023624.

This code reproduce the results in FIG. 2. from Dal Lago, et al. 2015 

Author: Carla Borja Espinosa
Date: Febrero 2019

Example command line in terminal to run the program: 
$ python3 0.5 0.5 0 1.5 1 

"""

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from sympy.printing.str import StrPrinter
from sys import argv
from sympy.abc import t

class Hamiltonian(object):
    """Define the total hamiltonian for the system"""
    def __init__(self):
        """Variables""" 
        self.v = float(argv[1])    #intracell hoping
        self.w = float(argv[2])    #extracell hoping
        self.V = float(argv[3])    #driving amplitude
        self.T = float(argv[4])*sp.pi    #period of the perturbation
        self.harm = int(argv[5])    #number of harmonics   
        self.G = 2*sp.pi/self.T
        self.k = sp.Symbol('k',real=True)
        """Functions"""
        self.FreeHamiltonian()
        self.ModulatedHamiltonian()
        self.FHamiltonian(self.harm)
        self.EigenvalsPlot()

    def FreeHamiltonian(self):
        """Define the free evolution hamiltonian"""
        self.Ho = -sp.Matrix([[0,self.v+self.w*sp.exp(-sp.I*self.k)],[self.v+self.w*sp.exp(sp.I*self.k),0]])
        self.size = self.Ho.shape[1]

    def ModulatedHamiltonian(self):
        """Define the hamiltonian for the time-dependent periodic 
           perturbation"""
        m = 2*self.V*sp.cos(self.G*t) #modulation of hoppings
        self.Vt = sp.Matrix([[0,m-m*sp.exp(-sp.I*self.k)],
                            [m-m*sp.exp(sp.I*self.k),0]])

    def NonDiagonalElements(self,n=0,m=1):
        """Calculate the non diagonal elements of the time-independent 
           Floquet hamiltonian"""
        Hnm = sp.zeros(self.size,self.size)
        for i in range(0,self.size):
            for j in range(0,self.size):
                Hnm[i,j] = 1/self.T*sp.integrate(self.Vt[i,j]*sp.exp(-sp.I*(m-n)*self.G*t),(t,0,self.T),conds='none')
        return(Hnm)

    def DiagonalElements(self,n):
        """Calculate the diagonal elements of the time-independent 
           Floquet hamiltonian"""
        return(self.Ho+n*self.G*sp.eye(2)) 

    def FHamiltonian(self,n):
        """Set up the time-independent Floquet hamiltonian"""
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
        #printer = StrPrinter()
        #print(Hf.table(printer,align='center'))
        self.Hf = Hf

    def EigenvalsPlot(self):
        """Calculate and plot the eigenvalues of the time-independent 
           Floquet hamiltonian"""
        Hfk = sp.lambdify('k',self.Hf)
        k_vals = np.arange(-np.pi,np.pi,0.001)
        size_k = np.size(k_vals)
        size_aut = self.Hf.shape[1]
        graph = np.empty([size_k,size_aut]) 
        for i in range(0,size_k):
            energies,vectors = la.eig(Hfk(k_vals[i]))
            energies = np.sort(np.real(energies))
            graph[i] = energies
        for i in range(0,size_aut):
            plt.plot(k_vals,graph[:,i],'g')
        #plt.ylim(-2,2)
        plt.show()
        #plt.ylim(self.G/2, self.G/2) #Brillouin Zone
        #plt.show()  

"""Command Line Executable"""
Hamiltonian()
