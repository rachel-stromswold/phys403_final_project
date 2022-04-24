import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import integrate
import h5py

#np.seterr(all='raise')

C_L = 2.99792458e5      # km / s (speed of light in vacuum)
C_L_SQ = C_L*C_L        # km^2 s^-2 (speed of light squared)
PRIOR_RANGE = (20, 140) # km s^-1 Mpc^-1 (range of prior distribution for Hubble's constant)

OMEGA_M = 0.3           #unitless
OMEGA_A = 0.7

#these hardcoded events come from GW200202
#TODO: read these from the hdf5 file automatically
DIST_GW_MU = 410        # Mpc
DIST_GW_VR = 150        # Mpc (variance of the GW observation

#we need to integrate over potential redshifts for each galaxy. To make this feasible we'll limit the range to some multiple of the standard deviation
N_REDSHIFT_SIGS = 2

#EVENT_LIST = ['GW200202_154313-v1', 'GW200115_042309-v2', 'GW200208_222617-v1', 'GW190814_211039-v3']
SAMPLE_TYPE = 'sim_events'
EVENT_LIST = ['ev_0', 'ev_1', 'ev_2']

#prior range on the hubble constant
h_vals = np.linspace(PRIOR_RANGE[0], PRIOR_RANGE[1])
pdf = np.ones(len(h_vals))

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

def normalize_pdf(support, pdf):
    '''given a pdf with the specified support (taken to be a subset of R), return its normalized value. The support is unaltered.
    '''
    #use the trapezoid rule
    int_p = 0.0
    for i in range( 1, len(support) ):
        int_p += (support[i] - support[i-1])*(pdf[i] + pdf[i-1])/2
    return np.asarray(pdf) / int_p

for ev in EVENT_LIST:
    #create a temporary pdf for this event. We need to take the product over multiple events to get the final result
    tmp_pdf = np.zeros(len(pdf))

    #load the list of potential galaxies
    rshift_fname = SAMPLE_TYPE + '/' + ev + '_rshifts.h5'
    galaxies = h5py.File(rshift_fname, 'r')

    #figure out the range for our z-cut
    ev_d_range = galaxies['distance_posterior']['conf_interval']
    z_min = ev_d_range[0]*PRIOR_RANGE[0] / C_L
    z_max = ev_d_range[1]*PRIOR_RANGE[1] / C_L

    gw_dist_mu = galaxies['distance_posterior']['expectation'][0]
    gw_dist_var = (galaxies['distance_posterior']['std_error'][0])**2
    #posterier from Nair et al
    rshifts = galaxies['redshifts']
    print("Found %d matching galaxies" % len(rshifts['z']))
    for mu_z, sig_z in zip(rshifts['z'], rshifts['z_err']):
        var_z = sig_z*sig_z #error on the redshift of galaxy i

        #some galaxies have zero catalogued redshift and error, we ignore these to avoid divisions by zero
        if var_z > 0:
            #iterate over the hubble constant prior
            for i, h in enumerate(h_vals):
                #integrate over possible redshifts
                p_em = integrate.quad( integrand_em, z_min, z_max, args=(mu_z, var_z, h) )[0]
                tmp_pdf[i] += pdf[i]*integrate.quad( integrand, z_min, z_max, args=(mu_z, var_z, h, gw_dist_mu, gw_dist_var) )[0]/(sig_z*p_em)

    #move the new data into the pdf
    pdf = tmp_pdf

plt.plot(h_vals, pdf)
plt.xlabel(r'$H_0$ km s$^{-1}$ Mpc$^{-1}$')
plt.ylabel(r'$p(H_0 | D)$')
plt.title(r'Posterior distribution for $H_0$')
#plt.show()
plt.savefig('hub.svg')
