import math
import random
from datetime import datetime
import numpy as np
import h5py

import argparse
import configparser
import matplotlib.pyplot as plt

from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree
import pickle

N_WALKERS = 3

DIR_NAME = 'sim_events'
SKYLOC_SCALE = 1000
MPC_PER_GPC = 1000

POISSON_CUTOFF_LAMBDA = 20
PI_BY_2_QUADROOT = (2 / math.pi)**0.25

config = configparser.ConfigParser()
config.read('params.conf')
GW_HIST_FNAME = config['simulation']['GW_hist_fname']
CONFINE_THRESHOLD_LO = float(config['simulation']['min_event_volume'])
CONFINE_THRESHOLD_HI = float(config['simulation']['max_event_volume'])
H0_TRUE = float(config['simulation']['H_0_true'])
C_L = float(config['physical']['light_speed'])
G = float(config['physical']['grav_const'])
CLUST_DENSITY = float(config['physical']['cluster_density'])
GAL_DENSITY = float(config['physical']['galaxy_density'])
CRIT_GAL_N = float(config['physical']['crit_n_gal'])
CLUST_MASS_LAMBDA_0 = float(config['physical']['mass_lambda_0'])
CLUST_MASS_SCALE = float(config['physical']['mass_scale_rich'])
CLUST_MASS_ALPHA = float(config['physical']['mass_alpha'])
CLUST_MASS_VAR = float(config['physical']['mass_gauss_err'])
GAMMA_SHAPE_N = float(config['physical']['n_alpha'])
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
MCMC_THIN = int(config['simulation']['MCMC_thin'])
MCMC_STEP = float(config['simulation']['MCMC_step'])
if len(GW_LOC_RANGE) != 2 or len(SKY_ANGLE_RANGE) != 2:
    raise ValueError("Ranges must have two elements.")

#the dimensionless Hubble constant H_0 / (100 km s^-1 Mpc^-1)
LITTLE_H = H0_TRUE / 100

#override configuration file with command line arguments if supplied
parser = argparse.ArgumentParser(description='Query sky-surveys for redshift data corresponding to a gravitational-wave detection.')
parser.add_argument('--density', type=str, nargs='+', help='Density of galaxies in cluster (Mpc^-3). Defaults to 5e-6.', default=GAL_DENSITY)
parser.add_argument('--n-events-generate', type=int, help='Number of galaxy catalogs to generate.', default=100)
parser.add_argument('--volume-range', type=float, nargs='+', help='Simulated detections which have a localization volume outside of this range will not be considered for analysis.', default=(CONFINE_THRESHOLD_LO, CONFINE_THRESHOLD_HI))
parser.add_argument('--out-directory', type=str, help='Location to save generated values to', default=DIR_NAME)
args = parser.parse_args()
GAL_DENSITY = args.density
CONFINE_THRESHOLD_LO, CONFINE_THRESHOLD_HI = args.volume_range
N_EVENTS = args.n_events_generate
DIR_NAME = args.out_directory

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

def sample_GW_events(hist_fname, n_events):
    '''Sample GW events using an MCMC model which is "trained" on historic data specified in localizations.txt
    hist_fname: filename in csv format describing posterior distributions for historic events. The csv should have four colums. The first describes the expectation on the luminosity distance posterior, the second and third the upper and lower uncertainties on the distance respectively and the fourth column should give sky angle localizations. Distances should be in units Gpc and solid angles in deg^2
    returns: a list of tuples for each event containing three elements, the measured luminosity distance, the uncertainty on luminosity distance and the solid angle confining sky location
    '''
    dat = np.loadtxt(hist_fname, delimiter=',')
    dat = np.array([[d[0], (d[1]+d[2])/2, d[3]/SKYLOC_SCALE] for d in dat])
    min_vals = [min(dat[:,col]) for col in range(3)]
    kern = stats.gaussian_kde(dat.transpose())
    samps = kern.resample(n_events).transpose()
    ret = []
    #we reject all points which aren't strictly positive or have an uncertainty greater than the mean
    for s in samps:
        if s[0] > min_vals[0] and s[1] > min_vals[1] and s[2] > min_vals[2] and s[0] > s[1]:
            ret.append([MPC_PER_GPC*s[0], MPC_PER_GPC*s[1], SKYLOC_SCALE*s[2]])

    return ret

def soliddeg_to_solidrad(solid_deg):
    '''Convert solid angle given in degrees^2 to solid angle in radians^2
    '''
    return solid_deg*(math.pi*math.pi)/(180*180)

def get_GW_event_vol(solid_angle, r_min, r_max):
    '''Calculate the volume contained in the solid angle between r_min and r_max.
    '''
    return solid_angle*( r_max**3 - r_min**3 ) / 3

def gen_poisson(mean):
    '''Sample from a Poisson distribution with mean lmbda
    '''
    #for small lambdas we sample from a Poisson distribution. Above a sufficiently large lambda, we take the distribution to be Gaussian
    if mean < POISSON_CUTOFF_LAMBDA:
        u = random.uniform(0.0, 1.0)

        const = math.exp(-mean)
        p_s = 0.0
        n = 0
        while True:
            p_s += math.pow(mean, n)*const / math.factorial(n)
            if p_s >= u:
                return n
            n += 1
    else:
        return int( random.gauss(mean, math.sqrt(mean)) )

def gen_cluster_uniform(solid_angle, d_l, d_l_err):
    '''Generate (part) of a galaxy cluster with an area parameterized by the solid angle of possible sky locations and an estimate on the luminosity distance d_l. The volume is taken by assuming that (d_l-d_l_err, d_l+d_l_err) describes some confidence interval for luminosity distance.
    '''
    r_min = max(d_l-d_l_err, MIN_GAL_DIST)
    r_max = d_l+d_l_err

    #there are a few different ways we can partition up the space, but we'll give the declanation a range between -asin(sqrt(omega)/4) and asin(sqrt(omega)/4) and the right ascension a range between -sqrt(omega) and sqrt(omega)
    sqrt_omega = math.sqrt( soliddeg_to_solidrad(solid_angle) )
    theta_r = math.asin(sqrt_omega/4)
    phi_r = sqrt_omega
    lmbda = GAL_DENSITY*get_GW_event_vol( soliddeg_to_solidrad(solid_angle), r_min, r_max )

    #comoving velocity dispersion for this particular cluster. We want to make sure this is positive, although negative values have roughly 0.3% chance to occur
    vel_sigma = -1.0
    while vel_sigma < 0:
        vel_sigma = random.gauss(VEL_DISP_MEAN, VEL_DISP_ERR)

    #approximate a poisson distribution with a Poisson distrubution mu=lambda sigma=sqrt(lambda)
    n_gals = gen_poisson(lmbda)
    loc_arr = np.zeros(shape=(5, n_gals)) #colunms are (RA, DEC, z, z_err) respectively

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
    #colunms are (RA, DEC, z, z_err) respectively
    loc_arr = [[] for i in range(5)]

    r_min = max(d_l-d_l_err, MIN_GAL_DIST)
    r_max = d_l+d_l_err
    vol = get_GW_event_vol(soliddeg_to_solidrad(solid_angle), r_min, r_max)

    #there are a few different ways we can partition up the space, but we'll give the declanation a range between -asin(sqrt(omega)/4) and asin(sqrt(omega)/4) and the right ascension a range between -sqrt(omega) and sqrt(omega)
    sqrt_omega = math.sqrt( soliddeg_to_solidrad(solid_angle) )
    theta_r = math.asin(sqrt_omega/4)
    phi_r = sqrt_omega
    lmbda = CLUST_DENSITY*vol

    n_clusters = gen_poisson(lmbda)
    print("lmbda = {}".format(lmbda))
    n_gals_arr = np.random.gamma(GAMMA_SHAPE_N, CRIT_GAL_N/(GAMMA_SHAPE_N-1), size=n_clusters).astype(int)

    #we need to find the total mass of the cluster to calculate its velocity dispersion
    '''clust_masses = CLUST_MASS_SCALE + CLUST_MASS_ALPHA*np.log(n_gals_arr/CLUST_MASS_LAMBDA_0)
    #apply a random scatter (eq 15 from Simet et al.) and exponentiate since we use log masses
    clust_scats = np.random.normal(0, 1, size=n_clusters)
    clust_masses = np.exp( clust_masses + np.sqrt(CLUST_MASS_VAR + CLUST_MASS_ALPHA**2/n_gals_arr)*clust_scats )'''

    #we take mass to be proportional to the number of clusters. The likelihood that a cluster is the host should be proportional to its mass
    n_gals_tot = sum(n_gals_arr)
    #to make probability proportional to number, sample uniformly between 0 and 1. Then add up each element from the sorted mass until we exceed the randomly selected fractional number
    gal_sel = random.uniform(0, n_gals_tot)
    part_sum = 0.0
    for i, n in enumerate(n_gals_arr):
        part_sum += n
        if part_sum >= gal_sel:
            #we take the host galaxy to be the first in the cluster
            tmp = n_gals_arr[0]
            n_gals_arr[0] = n_gals_arr[i]
            n_gals_arr[i] = tmp
            break

    #we need to ensure that galaxies are uniformly sampled. Note that p(r) = 3r^2/(r_max^3 - r_min^3) so F^-1(F)=(F*(r_max^3-r_min^3))^(1/3)
    r_facts = np.random.uniform(0, 1.0, n_clusters)
    vol_per_angle = (r_max**3 - r_min**3)
    dist_clusts = np.cbrt(r_facts*vol_per_angle + r_min**3)
    #generate central sky locations for each cluster
    clust_phis = np.random.uniform(-phi_r, phi_r, n_clusters)
    clust_thetas = np.random.uniform(-theta_r, theta_r, n_clusters)

    #get the center of galaxy in Cartesian coordinates
    x_cents = dist_clusts*np.sin(clust_thetas)*np.cos(clust_phis)
    y_cents = dist_clusts*np.sin(clust_thetas)*np.sin(clust_phis)
    z_cents = dist_clusts*np.cos(clust_thetas)

    #we need to make sure there is a host event near the center of the confidence interval, we'll make this a gaussian. To get a rough estimate of its width, we'll make it a two sigma event for the galaxy to be outside of our interval. If we consider our volume to be a sphere then we should make sigma=(3V/4)^{1/3}/2
    host_loc = np.random.normal(0.0, 0.5*(0.75*vol/math.pi)**(1/3), 3)
    x_cents[0] = host_loc[0] + d_l #the center of the distribution is on the x axis
    y_cents[0] = host_loc[1]
    z_cents[0] = host_loc[2]

    print("Creating space with %d clusters" % n_clusters)
    print( "\tn_gals={}, angle={}, dist={}, vol={}".format(n_gals_tot, solid_angle, d_l, vol) )

    for x_cent, y_cent, z_cent, n_gals in zip(x_cents, y_cents, z_cents, n_gals_arr):
        #The typical radius goes as a power law in the number of galaxies. We take the distance of the galaxies to follow a three dimensional Gaussian about the center. From this we can derive that if velocities are Gaussian distributed, they should have mean 0 and variance r_200*sqrt(pi/2)
        r_200 = R_200_SCALE*(n_gals**R_SCALE_POW)
        gals_cent_r = np.random.normal(0, r_200, 3*n_gals)

        #generate the (radial components) of peculiar velocity for each galaxy. We don't multiply by three because we only care about the radial component
        #print(G*clust_mass*r_200*PI_BY_2_QUADROOT)
        pec_vels = np.random.normal(0.0, math.sqrt(G*r_200*PI_BY_2_QUADROOT), n_gals)
        #calculate corresponding redshifts for each galaxy
        for i in range(n_gals):
            #get the location of the galaxy in absolute coordinates
            x = x_cent + gals_cent_r[3*i]
            y = y_cent + gals_cent_r[3*i+1]
            z = z_cent + gals_cent_r[3*i+2]

            #store RA and DEC
            ra = np.arctan2(y, x)
            dec = np.pi/2 - np.arctan2(z, x**2 + y**2)
            r = np.sqrt(x**2 + y**2 + z**2)
            #calculate the velocity using Hubble's law + peculiar motion. We only want the radial component, hence the factor 1/sqrt(3)
            vel = H0_TRUE*r + pec_vels[i]

            #it is possible that a cluster galaxy isn't in the region of space we're interested in. We must check this
            if (ra > -phi_r and ra < phi_r) and (dec > -theta_r and dec < theta_r) and (r > r_min and r < r_max):
                loc_arr[0].append(ra*180/math.pi)
                loc_arr[1].append(dec*180/math.pi)
                loc_arr[2].append(r)
                #for radial motion z ~ v/c
                loc_arr[3].append(vel / C_L)
                loc_arr[4].append(Z_MEAS_ERR)

    return loc_arr

def trim_events(samples):
    '''Realistically, we wouldn't even bother performing analysis on events which have poor localizations. We thus find the best confined events (typically closer ones) and return only those.
    returns: the number of sufficiently confined samples and the total number of samples generated
    '''
    ret = []
    for i, s in enumerate(samples):
        #calculate the volume of the event. If it is below some threshold, then add it to the return array
        vol = get_GW_event_vol(soliddeg_to_solidrad(s[2]), s[0]-s[1], s[0]+s[1])
        if vol > CONFINE_THRESHOLD_LO and vol < CONFINE_THRESHOLD_HI:
            ret.append(s)
    return ret, len(samples)

def make_samples(n_events):
    #events = sample_GW_events_uniform(GW_LOC_RANGE, GW_LOC_SCALE, GW_DERR_SIG, SKY_ANGLE_RANGE, n_events)
    events = []
    total_n_events = 0
    while len(events) < n_events:
        res = trim_events(sample_GW_events(GW_HIST_FNAME, n_events))
        events += res[0]
        total_n_events += res[1]
    events = events[:n_events]
    #events = trim_events( sample_GW_events(GW_HIST_FNAME, n_events) )
    for i, ev in enumerate(events):
        rshift_fname = DIR_NAME + ( '/ev_{}_rshifts.h5'.format(i) )
        #just by eyeballing data from the GWTC-2 and GWTC-3 papers, Gaussian errors on distance are roughly one third the distance.
        dist = ev[0]
        dist_err = ev[1]
        #take the 2*sigma confidence interval
        dist_lo = dist - dist_err*2
        dist_hi = dist + dist_err*2

        #TODO: uniformity on sky angle is almost certainly a highly unrealistic assumption
        solid_angle = ev[2]

        locs = gen_clusters(solid_angle, dist, 2*dist_err)
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
    print( "{} out of {} ({} %) events were sufficiently localized".format(len(events), total_n_events, 100*len(events)/total_n_events) )

make_samples(N_EVENTS)
