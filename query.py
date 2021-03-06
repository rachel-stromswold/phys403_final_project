from astroquery.vizier import VizierClass
from astropy.coordinates import SkyCoord
from ligo.skymap.io import read_sky_map
from ligo.skymap import postprocess
import pesummary.io
from pesummary.gw.fetch import fetch_open_samples
from gwosc.datasets import find_datasets
import mocpy

from astropy import units as u
from astroquery.xmatch import XMatch

import h5py

import argparse
import os
import numpy as np
from datetime import datetime
import random
import math

DIR_NAME = 'GW_events'
C_L = 2.99792458e5      # km / s (speed of light in vacuum)

#parse commandline arguments
parser = argparse.ArgumentParser(description='Query sky-surveys for redshift data corresponding to a gravitational-wave detection.')
parser.add_argument('events', type=str, nargs='+',
        help='Which GW events to fetch data for')
        #default=['GW200202_154313-v1', 'GW200115_042309-v2', 'GW200208_222617-v1', 'GW190814_211039-v3'])
#TODO: I'm not sure which waveform template is best for GW200115, we'll need to look into that further or ommit it
parser.add_argument('waveforms', type=str, nargs='+',
        help='Different posterior distributions are generated using different template waveforms.')
        #default=['C01:IMRPhenomXPHM', 'C01:IMRPhenomNSBH:HighSpin', 'C01:IMRPhenomXPHM', 'C01:SEOBNRv4PHM'])
parser.add_argument('--hub_prior_min', type=int, nargs='?',
        help='The minimum bound for the prior on H_0 (km s^-1 Mpc^-1). Defaults to 20',
        default=20)
parser.add_argument('--hub_prior_max', type=int, nargs='?',
        help='The minimum bound for the prior on H_0 (km s^-1 Mpc^-1). Defaults to 140.',
        default=140)
parser.add_argument('--dist_conf', type=float, nargs='?',
        help='The desired confidence level for the distance estimate. Defaults to 0.9',
        default=0.9)
parser.add_argument('--sky_conf', type=float, nargs='?',
        help='The desired confidence level for the sky location. Defaults to 0.9',
        default=0.9)
parser.add_argument('--max_entries', type=int, nargs='?',
        help='the upper limit for the number of entries which may be returned from a survey. Defaults to 1000000',
        default=1000000)
args = parser.parse_args()
EVENT_LIST = args.events
WVFRM_LIST = args.waveforms
PRIOR_RANGE = (args.hub_prior_min, args.hub_prior_max)
DIST_CONF_LEVEL = args.dist_conf
SKYLOC_CONF_LEVEL = args.sky_conf
N_SURVEY_ROWS = args.max_entries

print("Fetching data using\n\tevents={} waveforms={}\n\tH_0 in {}\n\td_l confidence={}\n\tsky confidence={}".format(EVENT_LIST, WVFRM_LIST, PRIOR_RANGE, DIST_CONF_LEVEL, SKYLOC_CONF_LEVEL))

envs = find_datasets(type="event", match="GW")

#we apply a random shift factor to the redshifts to blind our analysis
blind_fact = random.uniform(0.8, 1.2)
#we need to be able to unshift the data after we're done and repeat our analysis. No peeking!
with open("secret.txt", 'a') as file:
    file.write("Scaling factor for events {} (Generated at {}): {}".format(EVENT_LIST, datetime.now(), blind_fact))

#fetch a list of events which are saved locally
saved_events = os.listdir(DIR_NAME)

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
    rshift_fname = DIR_NAME + '/' + ev + '_rshifts.h5'
    if not os.path.exists(rshift_fname):
        #check if we have the fits file for the event, create it if we don't
        ev_fname = search_local(ev)

        if ev_fname == '':
            #download the GW data, this is in an unhelpful format so we save it to a temporary directory and then convert it
            data = fetch_open_samples(ev, read_file=True, outdir='/tmp')
        else:
            data = pesummary.io.read(DIR_NAME + ev_fname)
        
        #find the skymap from the GW data
        skymap = data.skymap[wf]
        uniq = np.array([i for i in range(len(skymap))])
        ligo_sky = mocpy.MOC.from_valued_healpix_cells(uniq, skymap, cumul_to=SKYLOC_CONF_LEVEL)

        #read from the vizier SDSS galaxy catalog
        print("Now finding galaxies in confidence interval for " + ev + ". This could take a while.")

        #lookup the info from the table
        from io import BytesIO
        from astropy.io.votable import parse_single_table
        import requests

        #now we need to convert our distance cut for the event into a redshift cut using extreme values (z= d_L*H_0/c e.g. the max value z may take is d_L,max*H_0,max/c)
        dist_ex, dist_var, dist_lo, dist_hi = get_conf_interval(data.samples_dict[wf]["luminosity_distance"])
        z_lo = dist_lo*PRIOR_RANGE[0] / C_L
        z_hi = dist_hi*PRIOR_RANGE[1] / C_L

        moc_file = BytesIO()
        moc_fits = ligo_sky.serialize(format='fits')
        moc_fits.writeto(moc_file)

        r = requests.post('http://cdsxmatch.u-strasbg.fr/QueryCat/QueryCat',
                          data={'mode': 'mocfile',
                                'catName': 'V/147/sdss12',
                                'format': 'votable',
                                'limit': N_SURVEY_ROWS,
                                'filter': 'class==3 && (zsp>{} && zsp<{})'.format(z_lo, z_hi)},
                          files={'moc': moc_file.getvalue()},
                          headers={'User-Agent': 'MOCPy'},
                          stream=True)

        votable = BytesIO()
        votable.write(r.content)

        sdss_match = parse_single_table(votable).to_table()
        print("found {} matching galaxies for the {} confidence interval".format(len(sdss_match), SKYLOC_CONF_LEVEL))
        #sdss_match = ligo_sky.query_vizier_table('V/147/sdss12', max_rows=N_SURVEY_ROWS)

        #write the list of potential galaxies and most importantly their redshifts (with random blinding factor) to a file
        print("saving to file")
        with h5py.File(rshift_fname, 'w') as f:
            #write the distance information
            dist_post = f.create_group("distance_posterior")
            dset1 = dist_post.create_dataset("expectation", (1,), dtype='f')
            dset1[0] = dist_ex*blind_fact
            dset2 = dist_post.create_dataset("std_error", (1,), dtype='f')
            dset2[0] = math.sqrt(dist_var)*blind_fact
            dset3 = dist_post.create_dataset("conf_interval", (2,), dtype='f')
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
            rshift_grp['z_err'] = np.array(match_gals[1])

        print("finished saving to " + rshift_fname)
