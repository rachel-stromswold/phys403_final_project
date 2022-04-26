import math
import random
from datetime import datetime
import numpy as np
import h5py

import argparse
import configparser
import matplotlib.pyplot as plt

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

class Tree_kd:
    '''A kd tree for hierarchically organinzing points.
    '''
    def _split_pts(self, subpts, ax=0):
        '''A helper method used for recursively generating the kd tree.
        '''
        n_pts = subpts.shape[0]
        if n_pts > 0:
            subpts_new = subpts[subpts[:, ax].argsort()]
            #To predictably handle degenerate coordinates we take the lowest value for the median
            split_ind = n_pts//2
            split_val = subpts_new[split_ind, ax]
            while split_ind > 0 and subpts_new[split_ind-1, ax] == split_val:
                split_ind -= 1

            #iteratively switch sort axis
            #ax = (ax+1) % self.dim

            #use the first available index to store this node
            cur_ind = self._n_used
            self._n_used += 1
            self.nodes[cur_ind][0] = subpts_new[split_ind]

            #give points less than the split to the left child and the rest to the right child
            self.nodes[cur_ind][1] = self._split_pts(subpts_new[:split_ind], ax=(ax+1) % self.dim)
            self.nodes[cur_ind][2] = self._split_pts(subpts_new[split_ind + 1:], ax=(ax+1) % self.dim)

            return cur_ind
        return -1

    def __init__(self, points):
        self.dim = len(points[0])
        ax = 0
        #we initialize an empty list which stores the node. The first element specifies the index of the point vertex and the next two elements specify children indices in the self.nodes_list. self.n_used is used when generating the tree.
        self.nodes = [[[], -1, -1] for pt in points]
        self._n_used = 0
        self._split_pts(points)
        self.pt_inds = [0 for pt in points]
        #we need to figure out the index each where each point is stored.
        #TODO: this is an O(n^2) operation and by not being lazy can be generated at the same time as _split_pts() (which would be a LOT faster)
        for i, pt in enumerate(points):
            for j in range(len(self.nodes)):
                equiv = True
                for k, el in enumerate(pt):
                    if self.nodes[j][0][k] != el:
                        equiv = False
                        break
                if equiv:
                    self.pt_inds[i] = j

    def _distsq(self, pt, i):
        '''Helper function to find the distance between a point and the ith node
        '''
        return sum( [(pt[j] - self.nodes[i][0][j])**2 for j in range(self.dim)] )

    def find_nearest(self, pt, root=0):
        '''Iterate through the kd-tree to find the point closest to pt.
        returns: the point and its corresponding index in the set from which the kd-tree was generated.
        '''
        #figure out the axis and setup information for tracking
        ax = root % self.dim
        best_pt = self.nodes[root][0]
        best_distsq = np.inf
        best_ind = 0
        cur_ind = root
        branches = []
        nodes_visited = [root]

        #iterate until we reach a leaf specified by a -1
        while cur_ind >= 0:
            #update the best found distance
            n_dist = self._distsq(pt, cur_ind)
            if n_dist < best_distsq:
                best_distsq = n_dist
                best_pt = self.nodes[cur_ind][0]
                best_ind = cur_ind

            #find the appropriate branch
            if pt[ax] < self.nodes[cur_ind][0][ax]:
                nodes_visited.append(self.nodes[cur_ind][1])
                branches.append(-1) #this is less than the branch so store a minus sign
            else:
                nodes_visited.append(self.nodes[cur_ind][2])
                branches.append(1)
            cur_ind = nodes_visited[-1]

            #update the sorting axis
            ax = (ax+1) % self.dim

        #move back up the tree to check neighbors
        i = len(nodes_visited)-2
        best_dist = math.sqrt(best_distsq)
        while i >= 0:
            ax = i % self.dim
            brnch = branches[i]
            #check if the r=best node distance hypersphere intersects the current dividing plane. If it does, then we need to check that tree
            if brnch*(pt[ax] - self.nodes[nodes_visited[i]][0][ax]) < best_dist:
                #check the subtree found by taking the other branch and find the distance of its nearest neighbor
                ob_near = self.find_nearest(pt, root=self.nodes[nodes_visited[i]][1 + (1-brnch)//2])[1]
                ob_distsq = sum( [(pt[j] - ob_near[j])**2 for j in range(self.dim)] )
                if ob_distsq < best_distsq:
                    best_pt = ob_near
                    best_distsq = ob_distsq
                    best_dist = math.sqrt(best_distsq)
                    best_ind = cur_ind

            i -= 1
        return self.pt_inds[best_ind], best_pt

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

def sample_GW_events():
    '''Sample GW events using an MCMC model which is "trained" on historic data specified in localizations.txt
    returns: a list of simulated GW events. Each entry is a tuple with the elements (distance_expectation, std_error, sky_localization)
    '''
    #import os.path
    from scipy.spatial import Voronoi, voronoi_plot_2d
    import pickle

    #if not os.path.isfile('data_misc/voronoi.pckl'):
    dat = np.loadtxt("misc_data/localizations.txt", delimiter=' ')
    dat = np.array([dat[:,0], dat[:,3]/1000]).transpose()
    dim = len(dat[0])
    #generate the voronoi tesselation of the historic data points and calculate the volume of each
    vor = Voronoi(dat)
    rec_vols = np.zeros(len(vor.regions))
    for i, reg in enumerate(vor.regions):
        valid = True
        for j in reg:
            if j < 0:
                valid = False
        if valid:
            vol = calc_vol(dim, [vor.vertices[j] for j in reg])
            #rec_vols[i] = 0.0 if vol == 0 else 1/vol
            rec_vols[i] = vol

    #setup the kd-tree to quickly find nearest neighbors
    tree = Tree_kd(dat)
    #we estimate the probability of a point by the reciprocal of the volume of its corresponding region in the Voronoi tesselation
    def est_prob(pt):
        near = tree.find_nearest(pt)[0]
        ind = vor.point_region[near]
        return rec_vols[ind]

    voronoi_plot_2d(vor, show_points=True, show_vertices=True)
    plt.show()

    #data = [[est_prob([j*0.06, k*100]) for j in range(100)] for k in range(100)]
    data = [[vor.point_region[tree.find_nearest([j*0.06, k*100])[0]] for j in range(100)] for k in range(100)]
    plt.imshow(data)
    plt.show()

sample_GW_events()

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
    '''theta_r = math.acos( 1 - soliddeg_to_solidrad(solid_angle)/(2*math.pi) )
    lmbda = GAL_DENSITY*solid_angle*( r_max**3 - r_min**3 ) / 3''' #volume*number density
    #there are a few different ways we can partition up the space, but we'll give the declanation a range between -asin(sqrt(omega)/4) and asin(sqrt(omega)/4) and the right ascension a range between -sqrt(omega) and sqrt(omega)
    sqrt_omega = math.sqrt( soliddeg_to_solidrad(solid_angle) )
    theta_r = math.asin(sqrt_omega/4)
    phi_r = sqrt_omega
    lmbda = GAL_DENSITY*solid_angle*( r_max**3 - r_min**3 ) / 3

    #comoving velocity dispersion for this particular cluster. We want to make sure this is positive, although negative values have roughly 0.3% chance to occur
    vel_sigma = -1.0
    while vel_sigma < 0:
        vel_sigma = random.gauss(VEL_DISP_MEAN, VEL_DISP_ERR)

    #approximate a poisson distribution with a Poisson distrubution mu=lambda sigma=sqrt(lambda)
    n_gals = int( random.gauss(lmbda, math.sqrt(lmbda)) )
    loc_arr = np.zeros(shape=(5, n_gals)) #colunms are (RA, DEC, z, z_err) respectively
    print("Creating cluster with %d galaxies" % n_gals)

    #randomly generate locations for each galaxy
    '''thetas = np.random.uniform(0.0, theta_r, n_gals)
    phis = np.random.uniform(0.0, 2*math.pi, n_gals)'''

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

make_samples(3)
