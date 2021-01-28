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
        self.voltage = None  # Container for voltage values
        self.time = None   # container for time values
        self.current = None  # Container for electric current values
        self.concentration = None  # Container for concentration values
        self.params = {'Experimental': {},
                       'Kinetics': {},
                       'Accuracy': {}}
        self.experiment = None
        self._experiment_format = None
        self._format = {'vlt': ('Voltage', 'Current'),
                        'cvlt': ('Voltage', 'Concentration'),
                        'amp': ('Time', 'Current'),
                        'camp': ('Time', 'Concentration')}
        self.__data = None

    def setdata(self):
        self.__data = {'vlt': (self.voltage, self.current),
                       'cvlt': (self.voltage, self.concentration),
                       'amp': (self.time, self.current),
                       'camp': (self.time, self.concentration)}

    def display(self, plot=None):
        """Makes a plot of either Current/Concentration vs Voltage/Time"""
        plot = self._experiment_format if plot is None else plot
        try:
            x_header, y_header = self._format[plot]
            x_column, y_column = self.__data[plot]
        except KeyError:
            raise KeyError('Not a valid plotting request')
        except AttributeError:
            raise AttributeError('No experiment have been run yet')
        plt.plot(x_column, y_column)
        plt.title(self.experiment)
        plt.xlabel(x_header)
        plt.ylabel(y_header)
        plt.show()

    def sendto(self, filename, kind=None):
        """Export simulated results into a csv file"""
        kind = self._experiment_format if kind is None else kind
        try:
            x_header, y_header = self._format[kind]
            x_column, y_column = self.__data[kind]
        except KeyError:
            raise KeyError('Not a valid exporting format')
        except AttributeError:
            raise AttributeError('No experiment have been run yet')
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
                file.write('%s,' % column)
            file.write('\n')
            for x, y in zip(x_column, y_column):
                file.write('%s,%s,\n' % (x, y))


class Voltammogram(ElectroSystem):
    def __init__(self):
        ElectroSystem.__init__(self)
        self._experiment_format = 'vlt'


class Chronoamperogram(ElectroSystem):
    def __init__(self):
        ElectroSystem.__init__(self)
        for item in ('vlt', 'cvlt'):  # Remove useful plot/export formats
            self._format.pop(item)
        self._experiment_format = 'amp'


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
        time = 2 * np.abs(es - ei) / scanr
        cell_len = 6 * np.sqrt(time)  # Maximum diffusion layer size
        n = int(cell_len / DX)  # Spatial grid
        m = int(time / DT)  # Time grid
        De = 2 * np.abs(es - ei) / m  # Potential step
        lmbd = DT / DX ** 2  # Lambda parameter for numeric method
        con_old = np.ones(n)  # Former concentration
        con_new = np.ones(n)  # Current concentration
        self.current = np.zeros(m)
        self.voltage = np.zeros(m)
        self.concentration = np.ones(m)
        self.time = [t for t in range(m)]
        self.voltage[-1] = ei
        for k in range(m):  # Numerical Solution
            self.voltage[k] = self.voltage[k - 1] - De if k < m / 2 else self.voltage[k - 1] + De
            con_new[0] = 1 / (1 + np.exp(-self.voltage[k]))
            for i in range(1, n - 1):
                con_new[i] = con_old[i] + lmbd * (con_old[i + 1] - 2 * con_old[i] + con_old[i - 1])
            self.concentration[k] = con_new[0]
            self.current[k] = -(-con_new[2] + 4 * con_new[1] - 3 * con_new[0]) / (2 * DX)
            con_old = con_new
        self.setdata()  # Output
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
        time = 2 * np.abs(es - ei) / scanr
        cell_len = 6 * np.sqrt(time)  # Maximum diffusion layer's size
        m = int(time / DT)  # Temporal grid
        De = 2 * np.abs(es - ei) / m  # Potential step
        if omega == 1:  # Spatial grid
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
            alpha = np.zeros(n)
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
        con = np.ones(n)  # Spatial changes in concentration
        self.current = np.zeros(m)
        self.voltage = np.zeros(m)
        self.concentration = np.ones(m)
        self.time = [t for t in range(m)]
        self.voltage[-1] = ei
        for k in range(m):
            self.voltage[k] = self.voltage[k - 1] - De if k < m / 2 else self.voltage[k - 1] + De
            # Forward swept
            con[0] = 1 / (1 + np.exp(-self.voltage[k]))
            for i in range(1, n - 1):
                con[i] = (con[i] - con[i - 1] * alpha[i]) / (beta[i] - Gamma[i - 1] * alpha[i])
            # Back substitution
            for i in range(n - 2, -1, -1):
                con[i] = con[i] - Gamma[i] * con[i + 1]
            self.current[k] = -(-con[2] + 4 * con[1] - 3 * con[0]) / (2 * DX)
            self.concentration[k] = con[0]
        self.setdata()  # Output
        return self

    def butlervolmer(self, a=0.5, k0=1e8, DX=1e-3, DT=1e-6, omega=1.1):
        """
        It solves the dimensionless diffusional cyclic voltammetry of an oxidized species (A) of reversible, and
        non-reversible systems, under the following considerations :
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
        time = 2 * np.abs(es - ei) / scanr
        cell_len = 6 * np.sqrt(time)  # Maximum diffusion layer's s9ze
        m = int(time / DT)  # Temporal grid
        De = 2 * np.abs(es - ei) / m  # Potential step
        h = DX
        Spatial_points = [0]  # Spatial grid
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
        con = np.ones(n)  # Spatial changes in concentration
        self.current = np.zeros(m)
        self.voltage = np.zeros(m)
        self.concentration = np.ones(m)
        self.time = [t for t in range(m)]
        self.voltage[-1] = ei
        for k in range(m):
            self.voltage[k] = self.voltage[k - 1] - De if k < m / 2 else self.voltage[k - 1] + De
            # Forward swept
            beta[0] = 1 + (DX * np.exp(-a * self.voltage[k]) * k0 * (1 + np.exp(self.voltage[k])))
            Gamma[0] = -1 / beta[0]
            for i in range(1, n - 1):
                Gamma[i] = gamma[i] / (beta[i] - Gamma[i - 1] * alpha[i])
            con[0] = (DX * np.exp(-a * self.voltage[k]) * k0 * np.exp(self.voltage[k])) / beta[0]
            for i in range(1, n-1):
                con[i] = (con[i] - con[i - 1] * alpha[i]) / (beta[i] - Gamma[i - 1] * alpha[i])
            # Back substitution
            for i in range(n - 2, -1, -1):
                con[i] = con[i] - Gamma[i] * con[i+1]
            self.current[k] = -(con[1] - con[0]) / Spatial_points[1] - Spatial_points[0]
            self.concentration[k] = con[0]
        self.setdata()  # Output
        return self


class Chronoamperometry(Chronoamperogram):
    def __init__(self, e, time):
        Chronoamperogram.__init__(self)
        self.experiment = 'Chronoamperometry'
        self.params['Experimental'] = {'e': e, 'Time': time}

    def butlervolmer(self, a=0.5, k0=1e8, DT=1e-12, DX=1e-6, omega_x=1.1, omega_t=1.1):
        """
        It solves the dimensionless diffusional chronoamperometry of an oxidized species (A) of reversible, and
        non-reversible systems, under the following considerations :
           1. E-mechanism A (oxidized) <-> B (reduced)
           2. One-electron transfer process
           3. Reaction kinetics described by Butler-Volmer model
           4. Driven by mass transport
           5. Migration and convection neglectable
           6. Planar macroelectrode as working electrode
           7. Diffusion of species in one spatial dimension
           8. Equal diffusion coefficients
       Numerical method: Implicit approach of finite differences with expansive spatial, and temporal grids (Solution of
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
        while Spatial_points[-1] < 6 * np.sqrt(Time):  # Maximum Diffusion_layer's length
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
        con = np.zeros(n)  # Spatial changes in concentration
        self.voltage = [e for t in range(m)]
        self.current = np.zeros(m)
        self.concentration = np.zeros(m)
        for k in range(m):
            # Forward sweep
            beta[0] = 1 + (DX * np.exp(-(a * e)) * k0 * (1 + np.exp(e)))
            Gamma[0] = -1 / beta[0]  # Modified gamma coefficients initialization
            for i in range(1, n - 1):
                Gamma[i] = gamma[i] / (beta[i] - Gamma[i - 1] * alpha[i])
            con[0] = (DX * np.exp(-(a * e)) * k0 * np.exp(e)) / beta[0]
            for i in range(1, n - 1):
                con[i] = (con[i] - con[i - 1] * alpha[i]) / (beta[i] - Gamma[i - 1] * alpha[i])
            # Back substitution
            for i in range(n - 2, -1, -1):
                con[i] = con[i] - Gamma[i] * con[i + 1]
            self.current[k] = -(con[1] - con[0]) / (Spatial_points[1] - Spatial_points[0])
            self.concentration[k] = con[0]
        self.setdata()  # Output
        return self
