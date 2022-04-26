import h5py
import numpy as np
#f = h5py.File("DESY3_GOLD_2_2.1_DNF.h5", "r")
with h5py.File("DESY3_GOLD_2_2.1_DNF.h5", "r") as f:
    print("H5 data sets:")
    print(list(f))
    
    
def make_samples():#(n_events):
    #events = sample_GW_events_uniform(GW_LOC_RANGE, GW_LOC_SCALE, GW_DERR_SIG, SKY_ANGLE_RANGE, n_events)
    #events = trim_events( sample_GW_events(GW_HIST_FNAME, n_events) )
    #for i, ev in enumerate(events[0]):
        #rshift_fname = DIR_NAME + ( '/ev_{}_rshifts.h5'.format(i) )
    rshift_fname = 'GW170814.h5'
        #just by eyeballing data from the GWTC-2 and GWTC-3 papers, Gaussian errors on distance are roughly one third the distance.
    dist = 0.5047#ev[0]
    dist_err = 0.0919#ev[1]
        #take the 2*sigma confidence interval
    dist_lo = dist - dist_err*2
    dist_hi = dist + dist_err*2

        #TODO: uniformity on sky angle is almost certainly a highly unrealistic assumption
    solid_angle = 61.66#ev[2]

        #locs = gen_cluster(solid_angle, dist, 2*dist_err)
    DES_file = h5py.File("DESY3_GOLD_2_2.1_DNF.h5", "r")
        #locs = 
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
            #rshift_grp['ra'] = np.array(locs[0])
            #rshift_grp['dec'] = np.array(locs[1])
            #rshift_grp['r'] = np.array(locs[2])
        rshift_grp['z'] = np.array(DES_file['catalog/unsheared/zmean_sof'])
        rshift_grp['z_err'] = np.array(DES_file['catalog/unsheared/zmc_sof'])
