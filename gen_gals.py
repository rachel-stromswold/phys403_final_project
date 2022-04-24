import math
import random
from datetime import datetime
import numpy as np
import h5py

import argparse
import configparser

DIR_NAME = 'sim_events'

config = configparser.ConfigParser()
config.read('params.conf')
H0_TRUE = float(config['simulation']['H_0_true'])
C_L = float(config['physical']['light_speed'])
GAL_DENSITY = float(config['physical']['galaxy_density'])
GW_LOC_RANGE = [float(val) for val in config['simulation']['GW_dist_range'].split(sep=',')]
GW_DERR_SIG = float(config['simulation']['GW_dist_err_sig'])
SKY_ANGLE_RANGE = [int(val) for val in config['simulation']['skyloc_range'].split(sep=',')]
MIN_GAL_DIST = float(config['simulation']['min_gal_dist'])
VEL_DISP_MEAN = float(config['physical']['vel_disp_ex'])
VEL_DISP_ERR = float(config['physical']['vel_disp_er'])
Z_MEAS_ERR = float(config['simulation']['z_er'])
if len(GW_LOC_RANGE) != 2 or len(SKY_ANGLE_RANGE) != 2:
    raise ValueError("Ranges must have two elements.")

#override configuration file with command line arguments if supplied
parser = argparse.ArgumentParser(description='Query sky-surveys for redshift data corresponding to a gravitational-wave detection.')
parser.add_argument('--density', type=str, nargs='+', help='Density of galaxies in cluster (Mpc^-3). Defaults to 5e-6.', default=GAL_DENSITY)
args = parser.parse_args()
GAL_DENSITY = args.density

def soliddeg_to_solidrad(solid_deg):
    '''Convert solid angle given in degrees^2 to solid angle in radians^2
    '''
    return solid_deg*(math.pi*math.pi)/(180*180)

def gen_cluster(solid_angle, d_l, d_l_err):
    '''Generate (part) of a galaxy cluster with an area parameterized by the solid angle of possible sky locations and an estimate on the luminosity distance d_l. The volume is taken by assuming that (d_l-d_l_err, d_l+d_l_err) describes some confidence interval for luminosity distance.
    '''
    r_min = max(d_l-d_l_err, MIN_GAL_DIST)
    r_max = d_l+d_l_err

    #generate a cone around the azimuthal angle which has the appropriate solid angle and use this for our volume calculation
    theta_r = math.acos( 1 - soliddeg_to_solidrad(solid_angle)/(2*math.pi) )
    lmbda = GAL_DENSITY*solid_angle*( r_max**3 - r_min**3 ) / 3 #volume*number density

    #comoving velocity dispersion for this particular cluster. We want to make sure this is positive, although negative values have roughly 0.3% chance to occur
    vel_sigma = -1.0
    while vel_sigma < 0:
        vel_sigma = random.gauss(VEL_DISP_MEAN, VEL_DISP_ERR)

    #approximate a poisson distribution with a Poisson distrubution mu=lambda sigma=sqrt(lambda)
    n_gals = int( random.gauss(lmbda, math.sqrt(lmbda)) )
    loc_arr = np.zeros(shape=(4, n_gals)) #colunms are (RA, DEC, z, z_err) respectively
    print("Creating cluster with %d galaxies" % n_gals)

    #randomly generate locations for each galaxy
    thetas = np.random.uniform(0.0, theta_r, n_gals)
    phis = np.random.uniform(0.0, 2*math.pi, n_gals)
    r_gals = np.random.uniform(r_min, r_max, n_gals)
    #generate the (radial components) of peculiar velocity for each galaxy
    pec_vels = np.random.normal(0.0, vel_sigma, n_gals)
    #calculate corresponding redshifts for each galaxy
    for i in range(n_gals):
        #calculate the velocity using Hubble's law + peculiar motion
        vel = H0_TRUE*r_gals[i] + pec_vels[i]
        #for radial motion z ~ v/c
        loc_arr[2, i] = vel / C_L
        loc_arr[3, i] = Z_MEAS_ERR
        #TODO: add data for sky location

    return loc_arr

def make_samples(n_events):
    for i in range(n_events):
        rshift_fname = DIR_NAME + ( '/ev_{}_rshifts.h5'.format(i) )
        #just by eyeballing data from the GWTC-2 and GWTC-3 papers, Gaussian errors on distance are roughly one third the distance.
        dist = random.uniform(GW_LOC_RANGE[0], GW_LOC_RANGE[1])
        dist_err = random.gauss(dist/3, GW_DERR_SIG)
        #take the 2*sigma confidence interval
        dist_lo = dist - dist_err*2
        dist_hi = dist + dist_err*2

        #TODO: uniformity on sky angle is almost certainly a highly unrealistic assumption
        solid_angle = random.uniform(SKY_ANGLE_RANGE[0], SKY_ANGLE_RANGE[1])

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
            dset3 = dist_post.create_dataset("conf_interval", (2,), dtype='f')
            dset3[0] = dist_lo#*blind_fact
            dset3[1] = dist_hi#*blind_fact
            #write the redshifts for matching galaxies
            rshift_grp = f.create_group('redshifts')
            rshift_grp['z'] = np.array(locs[2])
            rshift_grp['z_err'] = np.array(locs[3])

make_samples(3)
