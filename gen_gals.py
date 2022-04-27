import math
import random
from datetime import datetime
import numpy as np
import h5py

import argparse
import configparser
import matplotlib.pyplot as plt

DIR_NAME = 'sim_events'
SKYLOC_SCALE = 1000
MPC_PER_GPC = 1000

POISSON_CUTOFF_LAMBDA = 20

config = configparser.ConfigParser()
config.read('params.conf')
GW_HIST_FNAME = config['simulation']['GW_hist_fname']
CONFINE_THRESHOLD = float(config['simulation']['min_event_volume'])
H0_TRUE = float(config['simulation']['H_0_true'])
C_L = float(config['physical']['light_speed'])
CLUST_DENSITY = float(config['physical']['cluster_density'])
GAL_DENSITY = float(config['physical']['galaxy_density'])
CRIT_GAL_N = int(config['physical']['crit_n_gal'])
R_200_SCALE = float(config['physical']['scale_const'])
R_SCALE_POW = float(config['physical']['scale_pow'])
GW_LOC_RANGE = [float(val) for val in config['simulation']['GW_dist_range'].split(sep=',')]
GW_LOC_SCALE = float(config['simulation']['GW_dist_err_scale'])
GW_DERR_SIG = float(config['simulation']['GW_dist_err_sig'])
SKY_ANGLE_RANGE = [int(val) for val in config['simulation']['skyloc_range'].split(sep=',')]
MIN_GAL_DIST = float(config['simulation']['min_gal_dist'])
VEL_DISP_MEAN = float(config['physical']['vel_disp_ex'])
VEL_DISP_ERR = float(config['physical']['vel_disp_er'])
Z_MEAS_ERR = float(config['simulation']['z_er'])
if len(GW_LOC_RANGE) != 2 or len(SKY_ANGLE_RANGE) != 2:
    raise ValueError("Ranges must have two elements.")

#the dimensionless Hubble constant H_0 / (100 km s^-1 Mpc^-1)
LITTLE_H = H0_TRUE / 100

#override configuration file with command line arguments if supplied
parser = argparse.ArgumentParser(description='Query sky-surveys for redshift data corresponding to a gravitational-wave detection.')
parser.add_argument('--density', type=str, nargs='+', help='Density of galaxies in cluster (Mpc^-3). Defaults to 5e-6.', default=GAL_DENSITY)
args = parser.parse_args()
GAL_DENSITY = args.density

class MCMC_Walker:
    def __init__(self, x0, prior_range, likelihood, step_fact):
        '''Constructor for a single MCMC walker
        x0: array-like starting location, this should be selected randomly and assigned
        prior range: this should be a list or array of tuples that specifies the valid range of the prior distribution for each parameter. The number of tuples must be the same as the length of the array x0 supplied. If prior_dist is not specified, a uniform prior is used by default.
        likelihood: The functional form for the likelihood p(D|x). This should accept the array x as an argument.
        step_sigma: The proposal matrix is taken to be an n-dimensional Gaussian with mean x and standard deviation step_sigma.
        prior_dist: Prior distribution for p(x).
        '''
        assert len(x0) == len(prior_range)
        self.n = len(x0)
        self.x = x0[:]
        self.prior_range = prior_range[:]
        self.history = []
        self.likelihood = likelihood
        self.step_fact = step_fact
        #store the likelihood of the current value of x for future steps
        self.p_t0 = likelihood(x0)

        self.time_consts = np.zeros(self.n)

    def in_prior(self, pt):
        for i in range(self.n):
            if pt[i] < self.prior_range[i][0] or pt[i] > self.prior_range[i][1]:
                return False
        return True

    def step(self):
        '''Take a single step by making a proposal and calculating the acceptance likelihood.
        '''
        self.history.append( [self.x, self.p_t0] )

        #generate a point from the proposal matrix by generating an n dimensional Guassian and normalizing
        prop = np.random.normal(0, 1, size=self.n)
        prop = prop / math.sqrt(sum(np.square(prop)))
        rad = random.uniform(0, self.step_fact)
        for i, ra in zip(range(self.n), self.prior_range):
            prop[i] = rad*prop[i]*(ra[1] - ra[0]) + self.x[i]

        if self.in_prior(prop):
            #calculate the likelihood of the new point
            p_t1 = self.likelihood(prop)

            #this corresponds to alpha < 1
            if p_t1 < self.p_t0:
                if random.random() < p_t1 / self.p_t0:
                    self.x = prop
                    self.p_t0 = p_t1
            else:
                self.x = prop
                self.p_t0 = p_t1

    def calc_autocorrelation(self, max_k=np.inf, zero_stop=True):
        '''Calculate the autocorrelation function of this walker and return the time constant.
        max_k: maximum value of k to retain when performing the calculation
        zero_stop: If set to True, the autocorrelation function is only computed up to (but not including) the value of k such that rho(k) < 0.
        returns: a tuple containing the computed time constants and the autocorrelation function
        '''
        #fix max_k to be at most the number of points we have sampled to avoid overflows
        if max_k > len(self.history):
            max_k = len(self.history)

        #calculate the mean values for each coordinate in x
        x_bar = np.zeros(self.n)
        for pt in self.history[:max_k]:
            for j, xi in enumerate(pt[0]):
                x_bar[j] += xi

        #divide each sum by the total number of points
        for j in range(self.n):
            x_bar[j] /= max_k

        #calculate variances
        x_var = np.zeros(self.n)
        for pt in self.history[:max_k]:
            for j, xi in enumerate(pt[0]):
                x_var[j] += (xi - x_bar[j])**2

        #now we need to iterate over all lag values k between zero and max_k
        autocorrs = np.zeros((self.n, max_k))
        for j in range(self.n):
            for k in range(max_k):
                for i in range(max_k-k):
                    autocorrs[j][k] += (self.history[i][0][j] - x_bar[j])*(self.history[i+k][0][j] - x_bar[j])
                autocorrs[j][k] /= x_var[j]
                if zero_stop and autocorrs[j][k] < 0:
                    max_k = k
                    break

        #to compute the time constant, we only fit to lags before the autocorrelation goes negative so that the logarithm is defined
        for j in range(self.n):
            cross_ind = max_k
            #we only need to find the point of zero crossing if we didn't already find it while calculating the autocorrelation
            if not zero_stop:
                for k, corr in enumerate(autocorrs[j]):
                    if corr < 0:
                        cross_ind = k
                        break
            from scipy.optimize import curve_fit
            opt, cov = curve_fit(lambda x, a, b: a - b*x, [k for k in range(cross_ind)], np.log(autocorrs[j][:cross_ind]))
            self.time_consts[j] = 1.0 / opt[1]

        return self.time_consts, autocorrs

    def get_samples_thin(self, thin_ratio=0.5):
        '''Fetch a list of samples from the thinned chain.
        thin_ratio: The thinning factor is taken to be thin_ratio*time_const where time_const is the time constant generated by a call to calc_autocorrelation()
        returns: A list of MCMC samples
        '''
        for tc in self.time_consts:
            if tc == 0:
                self.calc_autocorrelation()
            break

        #since this problem is multi-dimensional, we use the worst case scenario as the time constant
        tc = max(self.time_consts)
        thin_fact = int(tc*thin_ratio)
        return [self.history[i][0] for i in range(int(3*tc), len(self.history), thin_fact)]

def det(dim, vert_list):
    '''Helper function for calc_vol which computes the determinant of the matrix specified by the iterable of vertices.
    '''
    #recursion inside of recursion. Let's F*** that callstack up!
    if dim == 1:
        return vert_list[0]
    elif dim == 2:
        return vert_list[0][0]*vert_list[1][1] - vert_list[0][1]*vert_list[1][0]
    elif dim == 3:
        return vert_list[0][0]*vert_list[1][1]*vert_list[2][2] + vert_list[1][0]*vert_list[2][1]*vert_list[0][2] + vert_list[2][0]*vert_list[0][1]*vert_list[1][2] - vert_list[0][0]*vert_list[2][1]*vert_list[1][2] - vert_list[1][0]*vert_list[0][1]*vert_list[2][2] - vert_list[2][0]*vert_list[1][1]*vert_list[0][2]
    mult = 1
    part_sum = 0
    for i in range(dim):
        part_sum += mult*vert_list[i][0]*det(dim-1, [vert_list[j + (j-i+dim)//dim][1:] for j in range(dim-1)])
        mult *= -1
    return part_sum

def calc_vol(dim, vert_lst):
    '''Helper function for sample_GW_Events. Calculate the volume For a set of vertices specified by vert_lst.
    vert_list: list of vertices. Each vertex should have dim number elements.
    dim: dimensionality of each vertex
    returns: the volume of the cell or -1 if it is invalid
    '''
    #base cases
    if len(vert_lst) < dim+1:
        return 0

    #if we reach this point in execution then this isn't a base case
    return abs( det(dim, [[v[i]-vert_lst[0][i] for i in range(dim)] for v in vert_lst[1:dim+1]]) )/math.factorial(dim) \
            + calc_vol(dim, vert_lst[1:])

def sample_GW_events(hist_fname, n_events):
    '''Sample GW events using an MCMC model which is "trained" on historic data specified in localizations.txt
    hist_fname: filename in csv format describing posterior distributions for historic events. The csv should have four colums. The first describes the expectation on the luminosity distance posterior, the second and third the upper and lower uncertainties on the distance respectively and the fourth column should give sky angle localizations. Distances should be in units Gpc and solid angles in deg^2
    returns: a list of tuples for each event containing three elements, the measured luminosity distance, the uncertainty on luminosity distance and the solid angle confining sky location
    '''
    #import os.path
    from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
    import pickle

    #load historic event data.
    dat = np.loadtxt(hist_fname, delimiter=',')
    #the units deg^2 are larger by several orders of magnitude which causes Voronoi tesselations to break. scale these down to the same order of magnitude as everything else. We're approximating all distance posteriors as Gaussian anyway, so we restrict the upper and lower uncertainties to be the same.
    dat = np.array([[d[0], (d[1]+d[2])/2, d[3]/SKYLOC_SCALE] for d in dat])
    dist_range = [min(dat[:,0]), max(dat[:,0])]
    sky_range = [min(dat[:,3]), max(dat[:,3])]
    #dat = np.array([ dat[:,0], dat[:,3]/1000 ]).transpose()
    dim = len(dat[0])
    #generate the voronoi tesselation of the historic data points and calculate the volume of each
    vor = Voronoi(dat)
    #setup the kd-tree to quickly find nearest neighbors
    tree = KDTree(dat)

    #calculate the reciprocal of the volume of each Voronoi cell. This gives an estimate of the PDF within that cell.
    rec_vols = np.zeros(len(vor.regions))
    for i, reg in enumerate(vor.regions):
        valid = True
        for j in reg:
            if j < 0:
                valid = False
        if valid:
            vol = calc_vol(dim, [vor.vertices[j] for j in reg])
            rec_vols[i] = 0.0 if vol == 0 else 1/vol

    #we estimate the probability of a point by the reciprocal of the volume of its corresponding region in the Voronoi tesselation
    def est_prob(pt):
        near = tree.query(pt)[1]
        ind = vor.point_region[near]
        return rec_vols[ind]

    #set up MCMC walkers
    walk = MCMC_Walker(np.random.rand(dim)*6.0, [dist_range, dist_range, sky_range], est_prob, 2.0)
    for i in range(10000):
        walk.step()

    #sample data points. We scaled skylocations down, so we have to scale them back up
    samps = [(MPC_PER_GPC*s[0], MPC_PER_GPC*s[1], SKYLOC_SCALE*s[2]) for s in walk.get_samples_thin()]
    #make pretty pictures
    '''
    x_vals = [s[0] for s in samps]
    y_vals = [s[2] for s in samps]
    plt.scatter(x_vals, y_vals)
    plt.scatter(dat[:,0], dat[:,2], color='orange')
    plt.show()'''
    return samps[:n_events]

def sample_GW_events_uniform(dist_range, dist_er_scale, dist_er_sigma, skyloc_range, n_events):
    '''Samples from a uniform population of potential GW events.
    dist_range: a tuple containing minimal and maximal values for measured luminosity distance
    dist_er_scale: the average error on an event is taken to be dist_er_scale*dist where dist is the sampled luminosity distance
    dist_er_sigma: the observed uncertainty on luminosity distance is taken to be Gaussian with this specified uncertainty
    skyloc_range: a tuple containing minimal and maximal values for sky localization solid angle
    n_events: number of desired events
    returns: a list of tuples for each event containing three elements, the measured luminosity distance, the uncertainty on luminosity distance and the solid angle confining sky location
    '''
    ret = []
    for i in range(n_events):
        dist = random.uniform(dist_range[0], dist_range[1])
        dist_err = random.gauss(dist*dist_er_scale, dist_er_sigma)
        solid_angle = random.uniform(skyloc_range[0], skyloc_range[1])
        ret.append( (dist, dist_err, solid_angle) )
    return ret

#sample_GW_events()

def soliddeg_to_solidrad(solid_deg):
    '''Convert solid angle given in degrees^2 to solid angle in radians^2
    '''
    return solid_deg*(math.pi*math.pi)/(180*180)

def get_GW_event_vol(solid_angle, r_min, r_max):
    '''Calculate the volume contained in the solid angle between r_min and r_max.
    '''
    return solid_angle*( r_max**3 - r_min**3 ) / 3

def gen_poisson(mean):
    '''Sample from a Poisson distribution
    '''
    #for small lambdas we sample from a Poisson distribution. Above a sufficiently large lambda, we take the distribution to be Gaussian
    if lmbda < POISSON_CUTOFF_LAMBDA:
        u = random.uniform(0.0, 1.0)

        const = math.exp(-mean)
        p_s = 0.0
        n = 0
        while True:
            p_s += math.pow(mean, n)*const / factorial(n)
            if p_s >= u:
                return n
            n += 1
    else:
        return int( random.gauss(lmbda, math.sqrt(lmbda)) )

def gen_cluster_uniform(solid_angle, d_l, d_l_err):
    '''Generate (part) of a galaxy cluster with an area parameterized by the solid angle of possible sky locations and an estimate on the luminosity distance d_l. The volume is taken by assuming that (d_l-d_l_err, d_l+d_l_err) describes some confidence interval for luminosity distance.
    '''
    r_min = max(d_l-d_l_err, MIN_GAL_DIST)
    r_max = d_l+d_l_err

    #there are a few different ways we can partition up the space, but we'll give the declanation a range between -asin(sqrt(omega)/4) and asin(sqrt(omega)/4) and the right ascension a range between -sqrt(omega) and sqrt(omega)
    sqrt_omega = math.sqrt( soliddeg_to_solidrad(solid_angle) )
    theta_r = math.asin(sqrt_omega/4)
    phi_r = sqrt_omega
    lmbda = GAL_DENSITY*get_GW_event_vol(solid_angle, r_min, r_max)

    #comoving velocity dispersion for this particular cluster. We want to make sure this is positive, although negative values have roughly 0.3% chance to occur
    vel_sigma = -1.0
    while vel_sigma < 0:
        vel_sigma = random.gauss(VEL_DISP_MEAN, VEL_DISP_ERR)

    #approximate a poisson distribution with a Poisson distrubution mu=lambda sigma=sqrt(lambda)
    n_gals = gen_poisson(lmbda)
    loc_arr = np.zeros(shape=(5, n_gals)) #colunms are (RA, DEC, z, z_err) respectively
    print("Creating cluster with %d galaxies" % n_gals)

    #we need to ensure that galaxies are uniformly sampled. Note that p(r) = 3r^2/(r_max^3 - r_min^3) so F^-1(F)=(F*(r_max^3-r_min^3))^(1/3)
    r_facts = np.random.uniform(0, 1.0, n_gals)
    vol_per_angle = (r_max**3 - r_min**3)
    r_gals = np.cbrt(r_facts*vol_per_angle + r_min**3)

    #generate the (radial components) of peculiar velocity for each galaxy
    pec_vels = np.random.normal(0.0, vel_sigma, n_gals)
    #calculate corresponding redshifts for each galaxy
    for i in range(n_gals):
        #calculate the velocity using Hubble's law + peculiar motion
        vel = H0_TRUE*r_gals[i] + pec_vels[i]
        #store RA and DEC
        loc_arr[0, i] = random.uniform(-phi_r, phi_r)
        loc_arr[1, i] = random.uniform(-theta_r, theta_r)
        loc_arr[2, i] = r_gals[i]
        #for radial motion z ~ v/c
        loc_arr[3, i] = vel / C_L
        loc_arr[4, i] = Z_MEAS_ERR

    return loc_arr

def gen_clusters(solid_angle, d_l, d_l_err):
    '''This is a more physically realistic model for how clusters are distributed. We appeal to the Schechter mass function to describe how the number of galaxies in a cluster is distributed with parameters taken from Hansen et al.
    '''
    r_min = max(d_l-d_l_err, MIN_GAL_DIST)
    r_max = d_l+d_l_err

    #there are a few different ways we can partition up the space, but we'll give the declanation a range between -asin(sqrt(omega)/4) and asin(sqrt(omega)/4) and the right ascension a range between -sqrt(omega) and sqrt(omega)
    sqrt_omega = math.sqrt( soliddeg_to_solidrad(solid_angle) )
    theta_r = math.asin(sqrt_omega/4)
    phi_r = sqrt_omega
    lmbda = CLUST_DENSITY*get_GW_event_vol(solid_angle, r_min, r_max)

    n_clusters = gen_poisson(lmbda)
    n_gals_arr = np.random.chisquare(CRIT_GAL_N, size=n_clusters)

    #we need to ensure that galaxies are uniformly sampled. Note that p(r) = 3r^2/(r_max^3 - r_min^3) so F^-1(F)=(F*(r_max^3-r_min^3))^(1/3)
    r_facts = np.random.uniform(0, 1.0, n_clusters)
    vol_per_angle = (r_max**3 - r_min**3)
    dist_clusts = np.cbrt(r_facts*vol_per_angle + r_min**3)

    #colunms are (RA, DEC, z, z_err) respectively
    loc_arr = [[] for i in range(5)]
    print("Creating cluster with %d galaxies" % n_gals)

    for dist, n_gals in zip(dist_clusts, n_gals_arr):
        clust_phi = random.uniform(-phi_r, phi_r)
        clust_theta = random.uniform(-theta_r, theta_r)

        #The typical radius goes as a power law in the number of galaxies
        r_200 = R_200_SCALE*(n_gals**R_SCALE_POW)
        gals_cent_r = np.random.normal(0.0, r_200, n_gals)

        #generate the (radial components) of peculiar velocity for each galaxy
        pec_vels = np.random.normal(0.0, vel_sigma, n_gals)
        #calculate corresponding redshifts for each galaxy
        for i in range(n_gals):
            #calculate the velocity using Hubble's law + peculiar motion
            vel = H0_TRUE*r_gals[i] + pec_vels[i]
            #store RA and DEC
            loc_arr[0, i] = random.uniform(-phi_r, phi_r)
            loc_arr[1, i] = random.uniform(-theta_r, theta_r)
            loc_arr[2, i] = r_gals[i]
            #for radial motion z ~ v/c
            loc_arr[3, i] = vel / C_L
            loc_arr[4, i] = Z_MEAS_ERR

    return loc_arr

def trim_events(samples):
    '''Realistically, we wouldn't even bother performing analysis on events which have poor localizations. We thus find the best confined events (typically closer ones) and return only those.
    returns: the number of sufficiently confined samples and the total number of samples generated
    '''
    ret = []
    for i, s in enumerate(samples):
        #calculate the volume of the event. If it is below some threshold, then add it to the return array
        vol = get_GW_event_vol(s[2], s[0]-s[1], s[0]+s[1])
        if vol < CONFINE_THRESHOLD:
            ret.append(s)
    return ret, len(samples)

def make_samples(n_events):
    #events = sample_GW_events_uniform(GW_LOC_RANGE, GW_LOC_SCALE, GW_DERR_SIG, SKY_ANGLE_RANGE, n_events)
    events = trim_events( sample_GW_events(GW_HIST_FNAME, n_events) )
    for i, ev in enumerate(events[0]):
        rshift_fname = DIR_NAME + ( '/ev_{}_rshifts.h5'.format(i) )
        #just by eyeballing data from the GWTC-2 and GWTC-3 papers, Gaussian errors on distance are roughly one third the distance.
        dist = ev[0]
        dist_err = ev[1]
        #take the 2*sigma confidence interval
        dist_lo = dist - dist_err*2
        dist_hi = dist + dist_err*2

        #TODO: uniformity on sky angle is almost certainly a highly unrealistic assumption
        solid_angle = ev[2]

        locs = gen_cluster(solid_angle, dist, 2*dist_err)
        print("saving cluster to " + rshift_fname)
        #write the list of potential galaxies and most importantly their redshifts (with random blinding factor) to a file
        with h5py.File(rshift_fname, 'w') as f:
            #write the distance information
            dist_post = f.create_group("distance_posterior")
            dset1 = dist_post.create_dataset("expectation", (1,), dtype='f')
            dset1[0] = dist#*blind_fact
            dset2 = dist_post.create_dataset("std_error", (1,), dtype='f')
            dset2[0] = dist_err#*blind_fact
            dset3 = dist_post.create_dataset("omega", (1,), dtype='f')
            dset3[0] = solid_angle#*blind_fact
            dset4 = dist_post.create_dataset("conf_interval", (2,), dtype='f')
            dset4[0] = dist_lo#*blind_fact
            dset4[1] = dist_hi#*blind_fact
            #write the redshifts for matching galaxies
            rshift_grp = f.create_group('redshifts')
            rshift_grp['ra'] = np.array(locs[0])
            rshift_grp['dec'] = np.array(locs[1])
            rshift_grp['r'] = np.array(locs[2])
            rshift_grp['z'] = np.array(locs[3])
            rshift_grp['z_err'] = np.array(locs[4])

make_samples(100)
