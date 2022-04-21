import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import integrate

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

#load the list of potential galaxies
gal_list_fname = 'GW_Events/GW200202_gals.txt'
galaxies = np.loadtxt(gal_list_fname, delimiter=' ')

#prior range on the hubble constant
h_vals = np.linspace(PRIOR_RANGE[0], PRIOR_RANGE[1])
pdf = np.zeros(len(h_vals))

def integrand(z_i, z_mu, z_err_sq, h_0):
    '''integrand for Bayes' theorem
    z_i: the term being integrated over, the redshift of the ith galaxy
    z_mu: expected value of z based on EM observation
    z_err_sq: variance on z (taken to be Gaussian)
    sig_cz_sq: the square error of c*z_i
    h_0: Hubble's constant
    '''
    #we need to marginalize over dzi and solid angles (see Soares-Santos et al)
    c_z = C_L*z_i
    try:
        dV = c_z*c_z / (h_0*h_0*h_0*math.sqrt(OMEGA_M*(1+z_i)**3 + OMEGA_A))
    except ValueError:
        dV = 1
    #compute marginal posteriors on the GW data and EM data
    marg_gw = math.exp( (h_0*DIST_GW_MU - c_z)*(c_z - h_0*DIST_GW_MU) / (2*DIST_GW_VR*h_0*h_0) )
    marg_em = math.exp( (z_i - z_mu)*(z_mu - z_i) / (2*z_err_sq) )
    return marg_gw*marg_em*dV

#posterier from Nair et al
for gal in galaxies:
    mu_z = gal[2]
    sig_z = gal[3] #error on the redshift of galaxy i
    var_z = gal[3]*gal[3] #error on the redshift of galaxy i

    #some galaxies have zero catalogued redshift and error, we ignore these to avoid divisions by zero
    if var_z > 0:
        #iterate over the hubble constant prior
        for i, h in enumerate(h_vals):
            #integrate over possible redshifts
            pdf[i] += integrate.quad(integrand, gal[2]-N_REDSHIFT_SIGS*gal[3], gal[2]+N_REDSHIFT_SIGS*gal[3], args=(mu_z, var_z, h,))[0]/sig_z

plt.plot(h_vals, pdf)
plt.xlabel(r'$H_0$ km s$^{-1}$ Mpc$^{-1}$')
plt.ylabel(r'$p(H_0 | D)$')
plt.title(r'Posterior distribution for $H_0$')
plt.show()
#plt.savefig('hub.svg')
