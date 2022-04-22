from astroquery.vizier import VizierClass
from astropy.coordinates import SkyCoord
from ligo.skymap.io import read_sky_map
from ligo.skymap import postprocess
import pesummary.io
from pesummary.gw.fetch import fetch_open_samples
from gwosc.datasets import find_datasets
from mocpy import MOC

from astropy import units as u
from astroquery.xmatch import XMatch

import h5py

import os
import numpy as np
from datetime import datetime
import random
import math

#constants that we'll need
C_L = 2.99792458e5      # km / s (speed of light in vacuum)
PRIOR_RANGE = (20, 140) # km s^-1 Mpc^-1 (range of prior distribution for Hubble's constant)
DIST_CONF_LEVEL = 0.9
SKYLOC_CONF_LEVEL = 0.9
N_SURVEY_ROWS = 1000000 # SDSS contains a lot of data (most of which we don't need)

envs = find_datasets(type="event", match="GW")

#TODO: I'm not sure which waveform template is best for GW200115, we'll need to look into that further or ommit it
EVENT_LIST = ['GW200202_154313-v1', 'GW200115_042309-v2', 'GW200208_222617-v1', 'GW190814_211039-v3']
WVFRM_LIST = ['C01:IMRPhenomXPHM', 'C01:IMRPhenomNSBH:HighSpin', 'C01:IMRPhenomXPHM', 'C01:SEOBNRv4PHM']

#we apply a random shift factor to the redshifts to blind our analysis
blind_fact = random.uniform(0.8, 1.2)
#we need to be able to unshift the data after we're done and repeat our analysis. No peeking!
with open("secret.txt", 'a') as file:
    file.write("Scaling factor for events {} (Generated at {}): {}".format(EVENT_LIST, datetime.now(), blind_fact))

#fetch a list of events which are saved locally
saved_events = os.listdir('GW_Events')

def search_local(name):
    '''Check to see if the event specified by name has a matching .h5 file which is saved locally so that we don't need to download it again.
    returns: the name of the file or an empty string if no matching event was found
    '''
    for ev in saved_events:
        loc = ev.find(name[0:6])
        if loc >= 0 and ev[-3:] == '.h5':
            return ev
    #if we reach this point in execution, no matching file was found
    return ''

def get_conf_interval(samples, conf_level=0.9):
    '''From an array-like of posterior samples, produce the central confidence interval
    samples: posterior samples from which we generate interval
    conf_level: desired confidence level
    returns: a tuple mean, variance and upper and lower bounds for the confidence interval
    '''
    arr = np.sort(np.asarray(samples))
    n_samps = arr.size
    ex = 0.0
    var = 0.0
    #find the expectation value
    for x in arr:
        ex += x / n_samps
    #find the variance
    for x in arr:
        var += (x-ex)**2 / (n_samps-1)

    #find the index of the upper and lower limits
    low_i = int( n_samps*(1.0 - conf_level)/2 )
    up_i =  int( n_samps*(0.5 + conf_level/2) )
    return ex, var, arr[low_i], arr[up_i]

for ev, wf in zip(EVENT_LIST, WVFRM_LIST):
    rshift_fname = 'GW_Events/' + ev + '_rshifts.h5'
    if not os.path.exists(rshift_fname):
        #check if we have the fits file for the event, create it if we don't
        ev_fname = search_local(ev)

        if ev_fname == '':
            #download the GW data, this is in an unhelpful format so we save it to a temporary directory and then convert it
            data = fetch_open_samples(ev, read_file=True, outdir='/tmp')
        else:
            data = pesummary.io.read('GW_Events/' + ev_fname)
        
        #find the skymap from the GW data
        skymap = data.skymap[wf]
        uniq = np.array([i for i in range(len(skymap))])
        ligo_sky = MOC.from_valued_healpix_cells(uniq, skymap, cumul_to=SKYLOC_CONF_LEVEL)

        #read from the vizier SDSS galaxy catalog
        print("Now finding galaxies in confidence interval for " + ev + ". This could take a while.")
        sdss_match = ligo_sky.query_vizier_table('V/147/sdss12', max_rows=N_SURVEY_ROWS)

        #now we need to convert our distance cut for the event into a redshift cut using extreme values (z= d_L*H_0/c e.g. the max value z may take is d_L,max*H_0,max/c)
        dist_ex, dist_var, dist_lo, dist_hi = get_conf_interval(data.samples_dict[wf]["luminosity_distance"])
        z_lo = dist_lo*PRIOR_RANGE[0] / C_L
        z_hi = dist_hi*PRIOR_RANGE[1] / C_L

        #write the list of potential galaxies and most importantly their redshifts (with random blinding factor) to a file
        print("saving to file")
        with h5py.File(rshift_fname, 'w') as f:
            #write the distance information
            dist_post = f.create_group("distance_posterior")
            dist_dset = dist_post.create_group("GW_distance")
            dset1 = dist_dset.create_dataset("expectation", (1,), dtype='f')
            dset1[0] = dist_ex*blind_fact
            dset2 = dist_dset.create_dataset("std_error", (1,), dtype='f')
            dset2[0] = math.sqrt(dist_var)*blind_fact
            dset3 = dist_dset.create_dataset("conf_interval", (2,), dtype='f')
            dset3[0] = dist_lo*blind_fact
            dset3[1] = dist_hi*blind_fact
            #write the redshifts for matching galaxies
            match_gals = [[], []]
            for row in sdss_match:
                #check that the object is actually a galaxy (type 3) and that it is in the 90% distance range
                if row['class'] == 3 and row['zsp'] > z_lo and row['zsp'] < z_hi:
                    match_gals[0].append(row['zsp'])
                    match_gals[1].append(row['e_zsp'])
            rshift_grp = f.create_group('redshifts')
            rshift_grp['z'] = np.array(match_gals[0])
            rshift_grp['z_err'] = np.array(match_gals[0])

        print("finished saving" + rshift_fname)
