from astroquery.vizier import VizierClass
from astropy.coordinates import SkyCoord
from ligo.skymap.io import read_sky_map
from ligo.skymap import postprocess
from pesummary.gw.fetch import fetch_open_samples
from gwosc.datasets import find_datasets

from astropy import units as u
from astroquery.xmatch import XMatch

import os

envs = find_datasets(type="event", match="GW")

#TODO: I'm not sure which waveform template is best for GW200115, we'll need to look into that further or ommit it
EVENT_LIST = ['GW200202_154313-v1', 'GW200115_042309-v2', 'GW200208_222617-v1', 'GW190814_211039-v3']
WVFRM_LIST = ['C01:IMRPhenomXPHM', 'C01:IMRPhenomNSBH:HighSpin', 'C01:IMRPhenomXPHM', 'C01:SEOBNRv4PHM']

#fetch a list of events which are saved locally
saved_events = os.listdir('GW_Events')

for ev, wf in zip(EVENT_LIST, WVFRM_LIST):
    #check if we have the fits file for the event, create it if we don't
    fits_fname = 'GW_Events/' + ev + '.fits'
    if not os.path.exists(fits_fname):
        #download the GW data, this is in an unhelpful format so we save it to a temporary directory and then convert it
        data = fetch_open_samples(ev, read_file=True, outdir='/tmp')
        skymap = data.skymap[wf]
        #save a .fits file which we can apply 
        skymap.save_to_file(fits_fname)
    
    #load the LIGO data skymap and crossmatch with the glade catalog
    ligo_sky = read_sky_map(fits_fname, moc=True)
    #Read GLADE skymap data 
    vizier = VizierClass(row_limit=-1, columns=['GWGC', 'RA_ICRS', 'DE_ICRS', 'zsp', 'e_zsp'])
    cat, = vizier.get_catalogs('V/147/sdss12')
    coordinates = SkyCoord(cat['RA_ICRS'], cat['DE_ICRS'], cat['Dist'])

    result = crossmatch(ligo_sky, coordinates)
    print(result)
    print(cat[result.searched_prob_vol < 0.9])
