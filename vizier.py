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

import os
import numpy as np

#
PRIOR_RANGE = (20, 140) # km s^-1 Mpc^-1 (range of prior distribution for Hubble's constant)

CONF_LEVEL = 0.4

envs = find_datasets(type="event", match="GW")

#TODO: I'm not sure which waveform template is best for GW200115, we'll need to look into that further or ommit it
EVENT_LIST = ['GW200202_154313-v1', 'GW200115_042309-v2', 'GW200208_222617-v1', 'GW190814_211039-v3']
WVFRM_LIST = ['C01:IMRPhenomXPHM', 'C01:IMRPhenomNSBH:HighSpin', 'C01:IMRPhenomXPHM', 'C01:SEOBNRv4PHM']


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

for ev, wf in zip(EVENT_LIST, WVFRM_LIST):
    #check if we have the fits file for the event, create it if we don't
    ev_fname = search_local(ev)
    fits_fname = 'GW_Events/' + ev + '.fits'
    #if not os.path.exists(fits_fname):
    if ev_fname == '':
        #download the GW data, this is in an unhelpful format so we save it to a temporary directory and then convert it
        data = fetch_open_samples(ev, read_file=True, outdir='/tmp')
    else:
        data = pesummary.io.read('GW_Events/' + ev_fname)
    
    skymap = data.skymap[wf]
    #cls = postprocess.find_greedy_credible_levels(skymap)
    #load the LIGO data skymap and crossmatch with the glade catalog
    uniq = np.array([i for i in range(len(skymap))])
    ligo_sky = MOC.from_valued_healpix_cells(uniq, skymap, cumul_to=CONF_LEVEL)

    #read from the vizier SDSS galaxy catalog
    print("Now finding galaxies in confidence interval")
    sdss_match = ligo_sky.query_vizier_table('V/147/sdss12', max_rows=100000)

    #write the list of potential galaxies and most importantly their redshifts (with random blinding factor) to a file
    with open('GW_Events/' + ev + '.txt', 'w') as file:
        file.write("#ra dec z zErr\n")
        for row in res:
            file.write( "{} {} {} {}\n".format(row['ra'], row['dec'], row['z']*blind_fact, row['zErr']*blind_fact) )

    #we need to be able to unshift the data after we're done and repeat our analysis. No peeking!
    with open("secret.txt", 'a') as file:
        file.write("Scaling factor for {} (Generated at {}): {}".format(gal_list_fname, datetime.now(), blind_fact))

    print("finished saving to file")
