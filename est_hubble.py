import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import integrate
import h5py

import argparse
import configparser

#setup parallelization
import multiprocessing as mp
N_CORES = mp.cpu_count()
#np.seterr(all='raise')

#read configuration
config = configparser.ConfigParser()
config.read('params.conf')
PRIOR_RANGE = [float(val) for val in config['physical']['H_0_prior'].split(sep=',')]
OMEGA_M = float(config['physical']['omega_matter'])
OMEGA_A = float(config['physical']['omega_lambda'])
C_L = float(config['physical']['light_speed'])
C_L_SQ = C_L*C_L
SAMPLE_TYPE = config['analysis']['type'].strip()
EVENT_LIST = [val.strip() for val in config['analysis']['events'].split(sep=',')]
if len(PRIOR_RANGE) != 2:
    raise ValueError("Ranges must have two elements.")
if len(EVENT_LIST) == 0:
    raise ValueError("At least one event must be supplied.")

#override configuration file with command line arguments if supplied
parser = argparse.ArgumentParser(description='Estimate the Hubble constant based on a GW event volume and a corresponding skymap.')
parser.add_argument('--type', type=str, nargs='?', help='Type of events to sample. Accepcted values are GW_events for real events and sim_events for simulated events. Defaults to {}.'.format(SAMPLE_TYPE), default=SAMPLE_TYPE)
args = parser.parse_args()
SAMPLE_TYPE = args.type

def integrand_em(z_i, z_mu, z_err_sq, h_0):
    '''integrand for Bayes' theorem
    z_i: the term being integrated over, the redshift of the ith galaxy
    z_mu: expected value of z based on EM observation
    z_err_sq: variance on z (taken to be Gaussian)
    sig_cz_sq: the square error of c*z_i
    gw_dist_mu: expected value of GW distance
    gw_dist_var: variance of GW distance
    h_0: Hubble's constant
    '''
    #we need to marginalize over dzi and solid angles (see Soares-Santos et al)
    c_z = C_L*z_i
    try:
        dV = c_z*c_z / (h_0*h_0*h_0*math.sqrt(OMEGA_M*(1+z_i)**3 + OMEGA_A))
    except ValueError:
        dV = 1
    dV = 1
    #compute marginal posteriors on the GW data and EM data
    marg_em = math.exp( (z_i - z_mu)*(z_mu - z_i) / (2*z_err_sq) )
    return marg_em*dV

def integrand(z_i, z_mu, z_err_sq, h_0, gw_dist_mu, gw_dist_var):
    '''integrand for Bayes' theorem
    z_i: the term being integrated over, the redshift of the ith galaxy
    z_mu: expected value of z based on EM observation
    z_err_sq: variance on z (taken to be Gaussian)
    sig_cz_sq: the square error of c*z_i
    gw_dist_mu: expected value of GW distance
    gw_dist_var: variance of GW distance
    h_0: Hubble's constant
    '''
    #we need to marginalize over dzi and solid angles (see Soares-Santos et al)
    c_z = C_L*z_i
    try:
        dV = c_z*c_z / (h_0*h_0*h_0*math.sqrt(OMEGA_M*(1+z_i)**3 + OMEGA_A))
    except ValueError:
        dV = 1
    dV = 1
    #compute marginal posteriors on the GW data and EM data
    marg_gw = math.exp( (h_0*gw_dist_mu - c_z)*(c_z - h_0*gw_dist_mu) / (2*gw_dist_var*h_0*h_0) )
    marg_em = math.exp( (z_i - z_mu)*(z_mu - z_i) / (2*z_err_sq) )
    return marg_gw*marg_em*dV

def p_part_gals(z, z_err, h_vals, z_min, z_max, gw_dist_mu, gw_dist_var):
    '''Helper function which computes the sum of a list of galaxies with specified redshifts
    z: iterable
    z_err: iterable, must have the same length as z
    h_vals: list of H_0 values from the prior distribution
    z_min: lower bound for integration over z
    z_max: upper bound for integration over z 
    gw_dist_mu: expected value of GW distance
    gw_dist_var: variance of GW distance
    returns: the probability distribution for H_0 considering only the current set of galaxies
    '''
    part_pdf = np.zeros(len(h_vals))
    for mu_z, sig_z in zip(z, z_err): 
        var_z = sig_z*sig_z #error on the redshift of galaxy i
        #some galaxies have zero catalogued redshift and error, we ignore these to avoid divisions by zero
        if var_z > 0:
            #iterate over the hubble constant prior
            for i, h in enumerate(h_vals):
                #integrate over possible redshifts
                p_em = integrate.quad( integrand_em, z_min, z_max, args=(mu_z, var_z, h) )[0]
                p_gw = integrate.quad( integrand, z_min, z_max, args=(mu_z, var_z, h, gw_dist_mu, gw_dist_var) )[0]
                part_pdf[i] += p_gw/(sig_z*p_em)
    return part_pdf

class Posterior_PDF:
    '''A utility class which stores a posterior pdf for a fixed vm and arbitrary distribution for H_0. The prior distribution for d is always assumed to be uniform.
    '''
    def __init__(self, h_range, num=50):
        '''Generates a posterior pdf for the Hubble data using the observed data vm and vm_err and the prior distribution for d
        h_range: a tuple specifying the range of values which H_0 may assume. We take this as a uniform prior
        num: number of values of d to sample when generating the pdf
        '''
        self.h_list = np.linspace(h_range[0], h_range[1], num=num)
        self.h_space = (h_range[1] - h_range[0]) / num
        self.p_list = np.ones(len(self.h_list))

    def guess_index(self, h):
        '''Helper function which finds the index in h_list which has the value closest to x assuming that lst is increasing'''
        h_ind_guess = int(h / self.h_space)
        list_len = len(self.h_list)
        #ensure that the index is between 0 and the length of the list
        if h_ind_guess < 1:
            h_ind_guess = 1
        if h_ind_guess >= list_len - 1:
            h_ind_guess = list_len - 2
        #look at adjacent points to find the best value
        err = abs(self.d_list[h_ind_guess] - h)
        best_ind = h_ind_guess
        if abs(self.d_list[h_ind_guess - 1] - h) < err:
            best_ind = h_ind_guess - 1
            err = abs(self.d_list[h_ind_guess - 1] - h)
        if abs(self.d_list[h_ind_guess + 1] - h) < err:
            best_ind = h_ind_guess + 1
        return best_ind

    def integrate(self, a=-1, b=-1):
        '''Finds the integral int_a^b p(d) dd
        d_list: iterable of distances, 
        p_list: iterable of probabilities associated with each value of d. This must have the same length as d_list
        a: lower bound or -1 to integrate from the lowest possible value for H_0
        b: upper bound or -1 to integrate up to the highest possible value for H_0
        returns: approximation of integral in that value
        '''
        a_ind = self.guess_index(a) if (a >= 0) else 0
        b_ind = self.guess_index(b) if (b >= 0) else len(h_list)-1

        #use the trapezoid rule
        int_p = 0.0
        for i in range(a_ind+1, b_ind):
            int_p += (self.h_list[i] - self.h_list[i-1])*(self.p_list[i] + self.p_list[i-1])/2
        return int_p

    def ex_value(self):
        '''Find the expectation value for H_0'''
        int_p = 0.0
        for i in range(1, len(self.h_list)):
            int_p += (self.h_list[i] - self.h_list[i-1])*(self.h_list[i]*self.p_list[i] + self.h_list[i-1]*self.p_list[i-1])/2
        return int_p

    def find_confidence_central(self, conf):
        '''Use a greedy algorithm to find the central conf% confidence interval
        conf: a number between 0 and 1 specifying the desired confidence interval
        returns: a tuple with three values. The first two values give lower and upper bounds respectively, the last value gives the actual probability contained
        '''
        int_p_low = 0.0
        found_lower = False
        lower = 0.0
        bounds = []

        #use the trapezoid rule
        int_p = 0.0
        for i in range(1, len(self.h_list)):
            int_p += (self.h_list[i] - self.h_list[i-1])*(self.p_list[i] + self.p_list[i-1])/2
            #if we've covered half of the confidence interval, then set the lower bound
            if not found_lower and int_p >= (1.0-conf)/2:
                lower = self.h_list[i]
                int_p_low = int_p
                found_lower = True
            if int_p >= 1.0 - conf/2:
                return lower, self.h_list[i], int_p-int_p_low
        return lower, self.h_list[-1], int_p-int_p_low

    def find_confidence(self, conf, tolerance=0.01, max_iters=10):
        '''find the minimum width conf confidence interval for the approximate probability distribution function specified by d_list and p_list
        conf: confidence interval to estimate
        tolerance: we stop trying to refine our interval if we have a confidence that is within tolerance a default 1% tolerance is used
        max_iters: maximum number of iterations to perform, even if we didn't achieve the desired tolerance
        '''
        center = self.ex_value()
        #these store the maximum probability heights which find_interval_iter is allowed to explore
        center_val = self.p_list[self.guess_index(center)]

        #we recursively guess a "height" on our pdf to consider. The pdf is searched to find values of d such that p(d)=height. Call these two values d1 and d2. We then integrate to find the probability between d1 and d2. If this value is greater (less) than conf then we take a higher (lower) height until we are within the specified tolerance or max_iter iterations have been performed
        def find_interval_iter(height, height_lower, height_upper, n_visited_heights):
            h_lo = self.h_list[0]
            h_hi = self.h_list[-1]
            #scan to find d1 and d2
            for i, p in enumerate(self.p_list):
                if p > height:
                    #we are just being carefull not to index out of bounds
                    if i == 0:
                        h_lo = height*(self.h_list[1] - self.h_list[0]) / self.p_list[1] #linear interpolation
                    elif self.p_list[i-1] < height:
                        #linear interpolation
                        h_lo = self.h_list[i-1] + (height - self.p_list[i-1])*(self.h_list[i] - self.h_list[i-1]) / (p - self.p_list[i-1])
                if i > 0 and p < height and self.p_list[i-1] > height:
                    #linear interpolation
                    d_hi = self.h_list[i-1] + (height - self.p_list[i-1])*(self.h_list[i] - self.h_list[i-1]) / (p - self.p_list[i-1])
            int_val = self.integrate(h_lo, h_hi)
            #check for convergence
            if abs(int_val - conf) < tolerance or n_visited_heights > max_iters:
                return (h_lo, h_hi, int_val)

            #explore a larger window of values by looking at a lower height halfway between the current height and the best lower height already visited
            if int_val < conf:
                height_upper = height
                return find_interval_iter( (height + height_lower)/2, height_lower, height, n_visited_heights+1)
            #similar but for a smaller window
            else:
                height_lower = height
                return find_interval_iter( (height + height_upper)/2, height, height_upper, n_visited_heights+1)

        return find_interval_iter(center_val/2, 0, center_val, 0)

    def normalize(self):
        '''given a pdf with the specified support (taken to be a subset of R), return its normalized value. The support is unaltered.
        '''
        #use the trapezoid rule
        int_p = 0.0
        for i in range( 1, len(support) ):
            int_p += (self.h_list[i] - self.h_list[i-1])*(self.p_list[i] + self.p_list[i-1])/2
        self.p_list /= int_p

    def add_event(self, rshift_fname):
        galaxies = h5py.File(rshift_fname, 'r')

        #figure out the range for our z-cut
        ev_d_range = galaxies['distance_posterior']['conf_interval']
        z_min = ev_d_range[0]*PRIOR_RANGE[0] / C_L
        z_max = ev_d_range[1]*PRIOR_RANGE[1] / C_L

        gw_dist_mu = galaxies['distance_posterior']['expectation'][0]
        gw_dist_var = (galaxies['distance_posterior']['std_error'][0])**2

        #posterier from Nair et al
        rshifts = galaxies['redshifts']

        #make sure that we have redshifts and errors for each galaxy
        assert len(rshifts['z']) == len(rshifts['z_err'])
        print("Found %d matching galaxies" % len(rshifts['z']))

        #allocate different parts of the job to different cores

        bsiz = len(rshifts['z']) // N_CORES
        rem = len(rshifts['z']) % N_CORES

        #we need to convert the two separate arrays for z and z_err into an array of arguments passed to p_one_gal.
        # ;)
        arg_list = [([z for z in rshifts['z'][i*(bsiz+1)+(rem-i)*((i-rem+N_CORES)//N_CORES):(i+1)*bsiz+i+1+(rem-i-1)*((i-rem+N_CORES)//N_CORES)]], [z for z in rshifts['z_err'][i*(bsiz+1)+(rem-i)*((i-rem+N_CORES)//N_CORES):(i+1)*bsiz+i+1+(rem-i-1)*((i-rem+N_CORES)//N_CORES)]], self.h_list, z_min, z_max, gw_dist_mu, gw_dist_var) for i in range(N_CORES)]

        #create a temporary pdf for this event. We need to take the product over multiple events to get the final result
        tmp_pdf = np.zeros(len(self.p_list))
        pool = mp.Pool(processes=N_CORES)
        res = pool.starmap(p_part_gals, arg_list)

        #sum together each result from the pool
        for r in res:
            for i in range(len(tmp_pdf)):
                tmp_pdf[i] += r[i]

        #move the new data into the pdf
        self.p_list = tmp_pdf*self.p_list
        galaxies.close()

pdf = Posterior_PDF(PRIOR_RANGE)
for ev in EVENT_LIST:
    #load the list of potential galaxies
    rshift_fname = SAMPLE_TYPE + '/' + ev + '_rshifts.h5'
    pdf.add_event(rshift_fname)

plt.plot(pdf.h_list, pdf.p_list)
plt.xlabel(r'$H_0$ km s$^{-1}$ Mpc$^{-1}$')
plt.ylabel(r'$p(H_0 | D)$')
plt.title(r'Posterior distribution for $H_0$')
plt.show()
#plt.savefig('hub.svg')
