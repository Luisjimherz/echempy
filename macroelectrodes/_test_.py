import echempy as ec

my_test = ec.Ciclyc_voltammetry().nernst(20, -20, 1e3)
my_test.plot_vlt()