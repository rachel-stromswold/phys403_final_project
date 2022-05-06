import numpy as np
import matplotlib.pyplot as plt
import argparse
import configparser

config = configparser.ConfigParser()
config.read('params.conf')
H0_INJECTED = float(config['simulation']['H_0_true'])
PRIOR_RANGE = [float(val) for val in config['physical']['H_0_prior'].split(sep=',')]

LC1 = "red"
LC2 = "blue"
ALPH = 0.3

parser = argparse.ArgumentParser(description="Generate our plots for the paper")
parser.add_argument("--posterior-files", nargs='+', help="The file containing information on posterior distributions as a function of number of detections. This file should be generated by a call to est_hubble.")
parser.add_argument("--compare-files", nargs='+', help="Another set of posterior files to plot on the same axes for comparison.")
parser.add_argument("--posterior-label", type=str, help="The label to use for the files specified by --posterior-files")
parser.add_argument("--compare-label", type=str, help="The label to use for the files specified by --compare-files")
parser.add_argument("--output-prefix", help="The directory to which plots should be saved")
parser.add_argument('--save-all', action='store_true', default=False, help="If specified, then plots of the posterior distribution on all data realizations will be saved. Otherwise, only the first is saved")
parser.add_argument('--desired-error', type=float, help="If specified, perform an estimate of how many detections are needed to get within the desired error.", default=5.0)
args = parser.parse_args()

def norm(p, dh):
    return np.sum(p) * dh

def mean(h, p, dh):
    return np.sum(h * p) * dh

def var(h, p, dh):
    mu = mean(h, p, dh)
    return np.sum((h - mu)**2 * p) * dh

def average_instances(file_list, label):
    n_realizations = len(file_list)
    #keep track of the averages and standard deviations for each entry. This might end up being a ragged array, hence the list of numpy arrays
    mus = [np.empty(0) for i in range(n_realizations)]
    stds = [np.empty(0) for i in range(n_realizations)]
    min_n_its = 0
    #load the data from the specified files
    for i, fname in enumerate(file_list):
        dat = np.loadtxt(fname)

        if i == 0 or dat.shape[0] < min_n_its:
            min_n_its = dat.shape[0]
        #initialize the arrays for the current realization
        mus[i] = np.empty(dat.shape[0])
        stds[i] = np.empty(dat.shape[0])

        #initialize arrays to store the pdf
        p = np.ones(dat.shape[1])
        h = np.linspace(PRIOR_RANGE[0], PRIOR_RANGE[1], dat.shape[1])
        dh = h[1] - h[0]

        #compute the mean and standard deviation for each number of detections in the given realization
        for j in range(dat.shape[0]):
            p *= dat[j]
            p /= norm(p, dh)
            mus[i][j] = mean(h, p, dh)
            stds[i][j] = np.sqrt(var(h, p, dh))

        #save the plot of the posterior distribution
        if i == 0 or args.save_all:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 6))
            for j in range(dat.shape[0]):
                ax1.plot(h, dat[j] / norm(dat[j], dh), color=LC1, linewidth=0.5, alpha=0.5)
            ax2.plot(h, p, color=LC1)
            ax2.set_xlabel("$H_0$ [km s$^{-1}$ Mpc$^{-1}]$", fontsize=14)
            ax1.set_ylabel("$p(H_0 | D)$", fontsize=14)
            ax2.set_ylabel("$p(H_0 | D)$ (cumulative)", fontsize=14)
            ax2.axvline(mus[i][-1], linestyle="--", color="black", linewidth=1.)
            ax2.axvline(mus[i][-1] - stds[i][-1], linestyle="--", color="black", linewidth=1.)
            ax2.axvline(mus[i][-1] + stds[i][-1], linestyle="--", color="black", linewidth=1.)
            ax2.axvline(H0_INJECTED, color="blue")
            plt.tight_layout()
            out_name = "posterior_{}.pdf".format(i)
            if label is not None:
                out_name = "posterior_{}_{}.pdf".format(label.replace(" ", "_"), i)
            plt.savefig(args.output_prefix + out_name)
            plt.clf()

    if min_n_its == 0:
        raise ValueError('At least one supplied posterior file must be nonempty')

    n_obs = -1
    n_obs_lo = -1
    n_obs_hi = -1
    #average the means and standard deviations over each instance
    mu = np.zeros(min_n_its)
    std = np.zeros(min_n_its)
    mu_std = np.zeros(min_n_its)
    std_std = np.zeros(min_n_its)
    for j in range(min_n_its):
        #compute averages
        for row in mus:
            mu[j] += row[j]
        mu[j] /= n_realizations
        for row in stds:
            std[j] += row[j]
        std[j] /= n_realizations
        #avoid division by zero
        if n_realizations > 1:
            #compute variances
            for row in mus:
                mu_std[j] += (mu[j] - row[j])**2
            mu_std[j] = np.sqrt( mu_std[j] / (n_realizations-1) )
            for row in stds:
                std_std[j] += (std[j] - row[j])**2
            std_std[j] = np.sqrt( std_std[j] / (n_realizations-1) )
        #if we're below the desired uncertainty threshold, report it
        if (std[j] < args.desired_error) and (n_obs < 0):
            n_obs = j
        if (std[j] - std_std[j] < args.desired_error) and (n_obs_lo < 0):
            n_obs_lo = j
        if (std[j] + std_std[j] < args.desired_error) and (n_obs_hi < 0):
            n_obs_hi = j
    #check to see if any events were below the desired error threshold
    if n_obs >= 0 and n_obs_lo >= 0 and n_obs_hi >= 0:
        print('It will take {} + {} - {} detections to confine H_0 to {} km s^-1 Mpc^-1 using {}'.format(n_obs, n_obs_hi-n_obs, n_obs-n_obs_lo, args.desired_error, label))

    return mu, std, mu_std, std_std, min_n_its

mu_post, std_post, mu_std_post, std_std_post, min_n_its_post = average_instances(args.posterior_files, args.posterior_label)
if (args.compare_files is not None) and (len(args.compare_files) > 0):
    mu_comp, std_comp, mu_std_comp, std_std_comp, min_n_its_comp = average_instances(args.compare_files, args.compare_label)

#plot the standard deviations
plt.figure(figsize=(8, 5))
plt.plot(std_post, color="red", label=args.posterior_label)
plt.fill_between([i for i in range(min_n_its_post)], std_post-std_std_post, std_post+std_std_post, color=LC1, alpha=ALPH)
if (args.compare_files is not None) and (len(args.compare_files) > 0):
    plt.plot(std_comp, color=LC2, label=args.compare_label)
    plt.fill_between([i for i in range(min_n_its_comp)], std_comp-std_std_comp, std_comp+std_std_comp, color=LC2, alpha=ALPH)
plt.xlabel("Number of events", fontsize=14)
plt.ylabel("$\sigma_{H_0}$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=14)
plt.ylim(0., 35.)
#only add the legend if labels have been provided
if (args.posterior_label is not None) and (args.compare_label is not None):
    plt.legend()
plt.savefig(args.output_prefix + "std.pdf")

#plot the biases
plt.figure(figsize=(8, 5))
plt.plot(mu_post - H0_INJECTED, color=LC1, label=args.posterior_label)
plt.fill_between([i for i in range(min_n_its_post)], mu_post-H0_INJECTED-mu_std_post, mu_post-H0_INJECTED+mu_std_post, color=LC1, alpha=ALPH)
if (args.compare_files is not None) and (len(args.compare_files) > 0):
    plt.plot(mu_comp - H0_INJECTED, color=LC2, label=args.compare_label)
    plt.fill_between([i for i in range(min_n_its_comp)], mu_comp-H0_INJECTED-mu_std_comp, mu_comp-H0_INJECTED+mu_std_comp, color=LC2, alpha=ALPH)
plt.xlabel("Number of events", fontsize=14)
plt.ylabel("$\langle H_0 \\rangle - H_{0,{\\rm true}}$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=14)
#only add the legend if labels have been provided
if (args.posterior_label is not None) and (args.compare_label is not None):
    plt.legend()
plt.savefig(args.output_prefix + "diff.pdf")

#plot the correlations between standard deviation and expectation value
plt.figure(figsize=(5, 5))
plt.scatter(mu_post, std_post, s=4., color=LC1, label=args.posterior_label)
if (args.compare_files is not None) and (len(args.compare_files) > 0):
    plt.scatter(mu_comp, std_comp, s=4., color=LC2, label=args.compare_label)
plt.xlabel("$\langle H_0 \\rangle$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=14)
plt.ylabel("$\sigma_{H_0}$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=14)
#only add the legend if labels have been provided
if (args.posterior_label is not None) and (args.compare_label is not None):
    plt.legend()
plt.savefig(args.output_prefix + "correlation.pdf")
