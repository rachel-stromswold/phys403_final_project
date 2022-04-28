import numpy as np
import matplotlib.pyplot as plt
import argparse

PRIOR_RANGE = [20., 140.]

parser = argparse.ArgumentParser(description="it plots the posterior")
parser.add_argument("--posterior-file", help="the posterior file")
parser.add_argument("--save-location", help="the location to save")
args = parser.parse_args()

def calc_mean(h_0, p):
    return np.sum(h_0 * p) * (h_0[1] - h_0[0])

dat = np.loadtxt(args.posterior_file)

h_0 = np.linspace(PRIOR_RANGE[0], PRIOR_RANGE[1], dat.shape[1])

for i in range(dat.shape[0]):
    dat[i] /= (np.sum(dat[i]) * (h_0[1] - h_0[0]))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 15))

for i in range(dat.shape[0]):
    ax1.plot(h_0, dat[i], color="blue", linewidth=0.7, alpha=0.3)

ax2.plot(h_0, np.prod(dat, axis=0))

ax3.hist([calc_mean(h_0, dat[i]) for i in range(dat.shape[0])])

plt.savefig(args.save_location)
