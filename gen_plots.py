import numpy as np
import matplotlib.pyplot as plt
import argparse

H0_INJECTED = 70.
PRIOR_RANGE = [20., 140.]

LC = "red"

parser = argparse.ArgumentParser(description="Generate our plots for the paper")
parser.add_argument("--posterior-file", help="hhhhhhhhhhhhhh")
parser.add_argument("--output-directory", help="lasdfl;jkadf")
args = parser.parse_args()

dat = np.loadtxt(args.posterior_file)

def norm(p, dh):
    return np.sum(p) * dh

def mean(h, p, dh):
    return np.sum(h * p) * dh

def var(h, p, dh):
    mu = mean(h, p, dh)
    return np.sum((h - mu)**2 * p) * dh

p = np.ones(dat.shape[1])
h = np.linspace(PRIOR_RANGE[0], PRIOR_RANGE[1], dat.shape[1])
dh = h[1] - h[0]

mu = np.empty(dat.shape[0])
std = np.empty(dat.shape[0])

for i in range(dat.shape[0]):
    p *= dat[i]
    p /= norm(p, dh)
    mu[i] = mean(h, p, dh)
    std[i] = np.sqrt(var(h, p, dh))

plt.figure(figsize=(8, 5))
plt.plot(std, color=LC)
plt.xlabel("Number of events", fontsize=14)
plt.ylabel("$\sigma_{H_0}$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=14)
plt.ylim(0., 35.)
plt.axhline(5)
plt.show()
#plt.savefig(args.output_directory + "std.pdf")

plt.figure(figsize=(8, 5))
plt.plot(mu - H0_INJECTED, color=LC)
plt.xlabel("Number of events", fontsize=14)
plt.ylabel("$\langle H_0 \\rangle - H_{0,{\\rm true}}$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=14)
plt.savefig(args.output_directory + "diff.pdf")

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
for i in range(dat.shape[0]):
    ax1.plot(h, dat[i] / norm(dat[i], dh), color=LC, linewidth=0.5, alpha=0.5)
ax2.plot(h, p, color=LC)
ax2.set_xlabel("$H_0$ [km s$^{-1}$ Mpc$^{-1}]$", fontsize=14)
ax1.set_ylabel("$p(H_0 | D)$", fontsize=14)
ax2.set_ylabel("$p(H_0 | D)$ (cumulative)", fontsize=14)
ax2.axvline(mu[-1], linestyle="--", color="black", linewidth=1.)
ax2.axvline(mu[-1] - std[-1], linestyle="--", color="black", linewidth=1.)
ax2.axvline(mu[-1] + std[-1], linestyle="--", color="black", linewidth=1.)
ax2.axvline(H0_INJECTED, color="blue")
plt.tight_layout()
plt.savefig(args.output_directory + "posterior.pdf")
