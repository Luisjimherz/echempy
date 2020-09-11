"""
@author: Luis Fernando Jimenez-Hernandez, BSc. Technology
contact: luisfernandojhz@comunidad.unam.mx
Last update: august 31th, 2020

-*- coding: utf-8 -*-
This module is in development phase. It is aimed to contain methods and functions
for electrochemically simulation under different numerical and experimental conditions.

Current status:
    It can solve diffusional cyclic voltammetry for one-electron transfer processes
    by finite differences numerical solution of Fick's diffusion laws.
    
Theoretical considerations:
    The module is suitable for simulating diffusional cyclic voltammetry from an heterogeneous Nersntian reaction
    (A <-> B) where both species have equal diffusion coefficients, considering a single electron E-mechanism.
    The electrode is considered to be a planar macroelectrode, and both migration and convection were neglected
    assuming a strongly supported media within an hydrostatic system.
    
    Systems in Nernstian equillibrium can be simulated fast, but inexactly, through explicit
    finite differences approach by using the fast() method.
"""

import numpy as np
import matplotlib.pyplot as plt


class ElectroSystem:
    def __init__(self):
        self.voltage = []  # List of voltage values
        self.time = []  # List of time values
        self.current = []  # List of current values
        self.concentration = []  # List of concentration values
        self.params = {'Experimental': {},
                       'Kinetics': {},
                       'Accuracy': {}}
        self.__plot_kind__ = {'vlt': ('Voltage', 'Current'),
                              'cvlt': ('Voltage', 'Concentration'),
                              'amp': ('Time', 'Current'),
                              'camp': ('Time', 'Concentration')}

    def setdata(self):
        self.__data__ = {'vlt': (self.voltage, self.current),
                         'cvlt': (self.voltage, self.concentration),
                         'amp': (self.time, self.current),
                         'camp': (self.time, self.concentration)}

    def display(self, plot=None):
        """display() method makes a plot of either Current/Concentration vs Voltage/Time"""
        plot = self.experiment_id if plot is None else plot
        try:
            x_header, y_header = self.__plot_kind__[plot]
            x_column, y_column = self.__data__[plot]
        except KeyError:
            raise KeyError('Not a valid plotting request')
        except AttributeError:
            raise AttributeError('You must perform a simulation prior to displaying a graph')
        plt.plot(x_column, y_column)
        plt.title(self.experiment)
        plt.xlabel(x_header)
        plt.ylabel(y_header)
        plt.show()

    def sendto(self, filename, kind=None):
        """Export simulated results into a csv file"""
        kind = self.experiment_id if kind is None else kind
        try:
            x_header, y_header = self.__plot_kind__[kind]
            x_column, y_column = self.__data__[kind]
        except KeyError:
            raise KeyError('exp parameter should match either vlt, cvlt, camp or camp')
        except AttributeError:
            raise AttributeError('You must perform a simulation prior to displaying a graph')
        with open(filename, 'w', newline='') as file:
            file.write('Experiment:, %s' % self.experiment)
            file.write('\nParameters:\n')
            for ky in self.params.keys():
                file.write('   %s:,' % ky)
                for sbky in self.params[ky].keys():
                    file.write("%s: %s | " % (sbky, self.params[ky][sbky]))
                file.write('\n')
            file.write('\n')
            for column in (x_header, y_header):
                file.write('%s,' % (column))
            file.write('\n')
            for x, y in zip(x_column, y_column):
                file.write('%s,%s,\n' % (x, y))


class Voltammogram(ElectroSystem):
    def __init__(self):
        ElectroSystem.__init__(self)
        self.experiment_id = 'vlt'


class Chronoamperogram(ElectroSystem):
    def __init__(self):
        ElectroSystem.__init__(self)
        for item in ('vlt', 'cvlt'):
            self.__plot_kind__.pop(item)
        self.experiment_id = 'amp'


class CyclicVoltammetry(Voltammogram):
    def __init__(self, ei, es, scanr):
        Voltammogram.__init__(self)
        self.experiment = 'Cyclic Voltammetry'
        self.params['Experimental'] = {'ei': ei, 'es': es, 'scanr': scanr}

    def fast(self, DX=1e-1, DT=1e-6):
        """
        It solves the dimensionless diffusional cyclic voltammetry of an oxidized species (A) of a reversible system
        under the following considerations:
            1. E-mechanism A (oxidized) <-> B (reduced)
            2. One-electron transfer process
            3. Reaction at Nernstian equilibrium
            4. Driven by mass transport
            5. Migration and convection neglectable
            6. Planar macroelectrode as working electrode
            7. Diffusion of species in one spatial dimension
            8. Equal diffusion coefficients
        Numerical method: Explicit approach of finite differences with equidistant spatial, and temporal grids
        (Individual solution of concentration at each moment through an iterative algorithm)

        Parameters:
        ei: (Experimental) Starting potential
        es: (Experimental) Switching potential
        scanr: (Experimental) Potential's scan rate
        DX: (Accuracy) Length between spatial points
        DT: (Accuracy) Length between time points
        """
        self.params['Kinetics'] = {'Model': 'Nernst'}
        self.params['Accuracy'] = {'DX': DX, 'DT': DT}
        ei = self.params['Experimental']['ei']
        es = self.params['Experimental']['es']
        scanr = self.params['Experimental']['scanr']
        time = 2 * np.abs(es - ei) / scanr  # Experiment's lasting time
        cell_len = 6 * np.sqrt(time)  # Cell's length
        n = int(cell_len / DX)  # Number of points to simulate in spacial mesh
        m = int(time / DT)  # Number of points to simulate in temporal mesh
        De = 2 * np.abs(es - ei) / m  # Potential step
        lmbd = DT / DX ** 2  # Lambda parameter for numeric method
        Concentration_old = np.ones(n)  # Concentration of species A at k-1
        Concentration_new = np.ones(n)  # Concentration of species A at k
        self.current = np.zeros(m)  # Simulated current over time
        self.voltage = np.zeros(m)  # Applied voltage over time
        self.concentration = np.ones(m)  # Concentration (species A) near the electrode over time
        self.time = [t for t in range(m)]
        self.voltage[-1] = ei  # if an error try Voltage[0] y range(1, m)
        for k in range(m):  # Numerical Solution
            if k < m / 2:
                self.voltage[k] = self.voltage[k - 1] - De
            else:
                self.voltage[k] = self.voltage[k - 1] + De
            Concentration_new[0] = 1 / (1 + np.exp(-self.voltage[k]))
            for i in range(1, n - 1):
                Concentration_new[i] = Concentration_old[i] + lmbd * (Concentration_old[i + 1] - 2 * Concentration_old[i] + Concentration_old[i - 1])
            self.concentration[k] = Concentration_new[0]
            self.current[k] = -(-Concentration_new[2] + 4 * Concentration_new[1] - 3 * Concentration_new[0]) / (2 * DX)
            Concentration_old = Concentration_new
        # Output
        self.setdata()
        return self

    def nernst(self, DX=1e-3, DT=1e-6, omega=1.1):
        """
        It solves the dimensionless diffusional cyclic voltammetry of an oxidized species (A) of a reversible system
        under the following considerations:
            1. E-mechanism A (oxidized) <-> B (reduced)
            2. One-electron transfer process
            3. Reaction at Nernstian equilibrium
            4. Driven by mass transport
            5. Migration and convection neglectable
            6. Planar macroelectrode as working electrode
            7. Diffusion of species in one spatial dimension
            8. Equal diffusion coefficients
        Numerical method: Implicit approach of finite differences with either equidistant or expansive spatial, grid and
         an equidistant time grid ( Solution of PDE -> Linear system through an LU Decomposition algorithm
         (Thomas algorithm))

        Parameters:
        ei: (Experimental) Starting potential
        es: (Experimental) Switching potential
        scanr: (Experimental) Potential's scan rate
        DX: (Accuracy) Length between spatial points
        DT: (Accuracy) Length between time points
        grid : (Method) Sets either 'expansive' or 'equidistant' spatial mesh (Default: 'expansive')
        """
        self.params['Kinetics'] = {'Model': 'Nernst'}
        self.params['Accuracy'] = {'DX': DX, 'omega x': omega, 'DT': DT}
        ei = self.params['Experimental']['ei']
        es = self.params['Experimental']['es']
        scanr = self.params['Experimental']['scanr']
        time = 2 * np.abs(es - ei) / scanr  # Experiment's lasting time
        cell_len = 6 * np.sqrt(time)  # Cell's length
        m = int(time / DT)  # Temporal mesh
        De = 2 * np.abs(es - ei) / m  # Potential step
        if omega == 1:  # Spatial mesh
            n = int(cell_len / DX)
            lmbd = DT / DX ** 2  # Lambda parameter for numeric method
            # Thomas coefficients
            alpha = [-lmbd for s in range(n)]
            beta = [1.0 + (2.0 * lmbd) for s in range(n)]
            gamma = [-lmbd for s in range(n)]
            Gamma = np.zeros(n)  # Modified gamma coefficient
            for i in range(1, n - 1):
                Gamma[i] = gamma[i] / (beta[i] - Gamma[i - 1] * alpha[i])
        else:
            h = DX
            Spatial_points = [0]
            while Spatial_points[-1] < cell_len:
                Spatial_points.append(Spatial_points[-1] + h)
                h *= omega
            n = len(Spatial_points)
            alpha = np.zeros(n)  # Thomas coefficients
            beta = np.zeros(n)
            gamma = np.zeros(n)
            for i in range(1, n - 1):
                DX_m = Spatial_points[i] - Spatial_points[i - 1]
                DX_p = Spatial_points[i + 1] - Spatial_points[i]
                alpha[i] = - (2 * DT) / (DX_m * (DX_m + DX_p))
                gamma[i] = - (2 * DT) / (DX_p * (DX_m + DX_p))
                beta[i] = 1 - alpha[i] - gamma[i]
            Gamma = np.zeros(n)  # Modified gamma coefficient
            for i in range(1, n - 1):
                Gamma[i] = gamma[i] / (beta[i] - Gamma[i - 1] * alpha[i])
        Concentration_space = np.ones(n)  # Concentration of species A through space
        self.current = np.zeros(m)
        self.voltage = np.zeros(m)
        self.concentration = np.ones(m)  # Concentration of species A through time
        self.time = [t for t in range(m)]
        self.voltage[-1] = ei  # if there is an error try Voltage[0] and range(1, m)
        for k in range(m):
            if k < m / 2:
                self.voltage[k] = self.voltage[k-1] - De
            else:
                self.voltage[k] = self.voltage[k-1] + De
            # Forward swept
            Concentration_space[0] = 1 / (1 + np.exp(-self.voltage[k]))
            for i in range(1, n - 1):
                Concentration_space[i] = (Concentration_space[i] - Concentration_space[i - 1] * alpha[i]) / (beta[i] - Gamma[i - 1] * alpha[i])
            # Back substitution
            for i in range(n - 2, -1, -1):
                Concentration_space[i] = Concentration_space[i] - Gamma[i] * Concentration_space[i + 1]
            self.current[k] = -(-Concentration_space[2] + 4 * Concentration_space[1] - 3 * Concentration_space[0]) / (2 * DX)
            self.concentration[k] = Concentration_space[0]
        # Output
        self.setdata()
        return self

    def butlervolmer(self, a=0.5, k0=1e8, DX=1e-3, DT=1e-6, omega=1.1):
        """
        It solves the dimensionless diffusional cyclic voltammetry of an oxidized species (A) of reversible, quasi-reversible
        and non-reversible systems, under the following considerations :
           1. E-mechanism A (oxidized) <-> B (reduced)
           2. One-electron transfer process
           3. Reaction kinetics described by Butler-Volmer model
           4. Driven by mass transport
           5. Migration and convection neglectable
           6. Planar macroelectrode as working electrode
           7. Diffusion of species in one spatial dimension
           8. Equal diffusion coefficients
       Numerical method: Implicit approach of finite differences with an expansive spatial, and an equidistant temporal
        grids ( Solution of PDE -> Linear system through an LU Decomposition algorithm (Thomas algorithm))

       Parameters:
       ei: (Experimental) Starting potential
       es: (Experimental) Switching potential
       scanr: (Experimental) Potential's scan rate
       a: (Kinetic) Symmetry factor
       k0: (Kinetic) Charge transfer coefficient
       DX: (Accuracy) Length between spatial points
       DT: (Accuracy) Length between time points
       omega: (Accuracy) Expansive factor for spatial grid. If one, then an equistant spatial grid is used.
       """
        self.params['Kinetics'] = {'Model': 'Butler Volmer', 'alpha': a, 'K0': k0}
        self.params['Accuracy'] = {'DX': DX, 'omega x': omega, 'DT': DT}
        ei = self.params['Experimental']['ei']
        es = self.params['Experimental']['es']
        scanr = self.params['Experimental']['scanr']
        time = 2 * np.abs(es - ei) / scanr  # Experiment's lasting time
        cell_len = 6 * np.sqrt(time)  # Cell's length
        m = int(time / DT)  # Temporal mesh
        De = 2 * np.abs(es - ei) / m  # Potential step
        h = DX
        Spatial_points = [0]  # Spatial mesh
        while Spatial_points[-1] < cell_len:
            Spatial_points.append(Spatial_points[-1] + h)
            h *= omega
        n = len(Spatial_points)
        alpha, beta, gamma = np.zeros(n), np.zeros(n), np.zeros(n)  # Thomas coefficients
        for i in range(1, n - 1):
            DX_m = Spatial_points[i] - Spatial_points[i - 1]
            DX_p = Spatial_points[i + 1] - Spatial_points[i]
            alpha[i] = - (2 * DT) / (DX_m * (DX_m + DX_p))
            gamma[i] = - (2 * DT) / (DX_p * (DX_m + DX_p))
            beta[i] = 1 - alpha[i] - gamma[i]
        Gamma = np.zeros(n)  # Modified gamma coefficient
        Concentration_space = np.ones(n)  # Concentration of species A through space
        self.current = np.zeros(m)
        self.voltage = np.zeros(m)
        self.concentration = np.ones(m)  # Concentration of species A through time
        self.time = [t for t in range(m)]
        self.voltage[-1] = ei  # if there is an error try Voltage[0] and range(1, m)
        for k in range(m):
            if k < m / 2:
                self.voltage[k] = self.voltage[k - 1] - De
            else:
                self.voltage[k] = self.voltage[k - 1] + De
            # Forward swept
            beta[0] = 1 + (DX * np.exp(-a * self.voltage[k]) * k0 * (1 + np.exp(self.voltage[k])))
            Gamma[0] = -1 / beta[0]
            for i in range(1, n - 1):
                Gamma[i] = gamma[i] / (beta[i] - Gamma[i - 1] * alpha[i])
            Concentration_space[0] = (DX * np.exp(-a * self.voltage[k]) * k0 * np.exp(self.voltage[k])) / beta[0]
            for i in range(1, n-1):
                Concentration_space[i] = (Concentration_space[i] - Concentration_space[i - 1] * alpha[i]) / (beta[i] - Gamma[i - 1] * alpha[i])
            # Back substitution
            for i in range(n - 2, -1, -1):
                Concentration_space[i] = Concentration_space[i] - Gamma[i] * Concentration_space[i+1]
            self.current[k] = -(Concentration_space[1] - Concentration_space[0]) / Spatial_points[1] - Spatial_points[0]
            self.concentration[k] = Concentration_space[0]
        # Output
        self.setdata()
        return self


class Chronoamperometry(Chronoamperogram):
    def __init__(self, e, time):
        Chronoamperogram.__init__(self)
        self.experiment = 'Chronoamperometry'
        self.params['Experimental'] = {'e': e, 'Time': time}

    def butlervolmer(self, a=0.5, k0=1e8, DT=1e-12, DX=1e-6, omega_x=1.1, omega_t=1.1):
        """
        It solves the dimensionless diffusional chronoamperometry of an oxidized species (A) of reversible, quasi-reversible
        and non-reversible systems, under the following considerations :
           1. E-mechanism A (oxidized) <-> B (reduced)
           2. One-electron transfer process
           3. Reaction kinetics described by Butler-Volmer model
           4. Driven by mass transport
           5. Migration and convection neglectable
           6. Planar macroelectrode as working electrode
           7. Diffusion of species in one spatial dimension
           8. Equal diffusion coefficients
       Numerical method: Implicit approach of finite differences with expansive spatial, and temporal grids ( Solution of
       PDE -> Linear system through an LU Decomposition algorithm (Thomas algorithm))

       Parameters:
       e: (Experimental) Applied potential
       Time: (Experimental) Experiment's lasting
       a: (Kinetic) Symmetry factor
       k0: (Kinetic) Charge transfer coefficient
       DX: (Accuracy) Length between spatial points
       omega_x: (Accuracy) Expansive factor for spatial grid
       DT: (Accuracy) Length between time points
       omega_t: (Accuracy) Expansive factor for time grid
       """
        self.params['Kinetics'] = {'Model': 'Butler Volmer', 'alpha': a, 'K0': k0}
        self.params['Accuracy'] = {'DX': DX, 'omega x': omega_x, 'DT': DT, 'omega_t': omega_t}
        e = self.params['Experimental']['e']
        Time = self.params['Experimental']['Time']
        self.time = [0]  # Temporal grid
        g = DT
        while self.time[-1] <= Time:
            self.time.append(self.time[-1] + g)
            g *= omega_t
        m = len(self.time)
        Spatial_points = [0]  # Spatial grid
        h = DX
        while Spatial_points[-1] < 6 * np.sqrt(Time):  # Diffusion_layer's length
            Spatial_points.append(Spatial_points[-1] + h)
            h = h * omega_x
        n = len(Spatial_points)
        # Thomas Coefficients
        alpha, beta, gamma = np.zeros(n), np.zeros(n), np.zeros(n)  # Thomas coefficients
        for k in range(0, m - 1):
            for i in range(1, n - 1):
                DX_m = Spatial_points[i] - Spatial_points[i - 1]
                DX_p = Spatial_points[i + 1] - Spatial_points[i]
                DT_m = self.time[k] - self.time[k - 1]
                DT_p = self.time[k + 1] - self.time[k]
                DT = DT_m * (DT_m + DT_p)
                DX = DX_m * (DX_m + DX_p)
                alpha[i] = - 2 * DT / DX
                gamma[i] = - 2 * DT / DX
                beta[i] = 1 - alpha[i] - gamma[i]
        Gamma = np.zeros(n)  # Modified gamma coefficient
        Concentration_space = np.zeros(n)  # Concentration of species A through space
        self.voltage = [e for t in range(m)]
        self.current = np.zeros(m)
        self.concentration = np.zeros(m)  # Concentration of species A through time
        for k in range(m):
            # Forward sweep
            beta[0] = 1 + (DX * np.exp(-(a * e)) * k0 * (1 + np.exp(e)))
            Gamma[0] = -1 / beta[0]  # Modified gamma coefficients initialization
            for i in range(1, n - 1):
                Gamma[i] = gamma[i] / (beta[i] - Gamma[i - 1] * alpha[i])
            Concentration_space[0] = (DX * np.exp(-(a * e)) * k0 * np.exp(e)) / beta[0]
            for i in range(1, n - 1):
                Concentration_space[i] = (Concentration_space[i] - Concentration_space[i - 1] * alpha[i]) / (beta[i] - Gamma[i - 1] * alpha[i])
            # Back substitution
            for i in range(n - 2, -1, -1):
                Concentration_space[i] = Concentration_space[i] - Gamma[i] * Concentration_space[i + 1]
            self.current[k] = -(Concentration_space[1] - Concentration_space[0]) / (Spatial_points[1] - Spatial_points[0])
            self.concentration[k] = Concentration_space[0]
        # Output
        self.setdata()
        return self
