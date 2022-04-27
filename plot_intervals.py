import numpy as np
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description="Plot the confidence intervals as a function of number of events.")
parser.add_argument("--interval-file", type=str, help="Location of the interval data file.")
parser.add_argument("--plot-fname", type=str, help="Filename for the plot.")
args = parser.parse_args()

intervals = np.loadtxt(args.interval_file)
x = np.arange(1, intervals.shape[0] + 1)

plt.figure(figsize=(8, 5))
plt.fill_between(x, intervals[:,0], intervals[:,1], color="red", alpha=0.1)
plt.fill_between(x, intervals[:,2], intervals[:,3], color="red", alpha=0.1)
plt.fill_between(x, intervals[:,4], intervals[:,5], color="red", alpha=0.1)
plt.savefig(args.plot_fname)
