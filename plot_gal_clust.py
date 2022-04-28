import argparse
import configparser
import h5py
import math
import numpy as np
import matplotlib.pyplot as plt

#read the file name from argv
parser = argparse.ArgumentParser(description='Plot a galaxy cluster for visualization and debugging purposes.')
parser.add_argument('--fname', type=str, help='Hdf5 file specifying saved locations.')
args = parser.parse_args()

#read the hubble constant from the configuration file
config = configparser.ConfigParser()
config.read('params.conf')
H_0 = float(config['simulation']['H_0_true'])
C_L = float(config['physical']['light_speed'])

#load the user-specified file
galaxies = h5py.File(args.fname, 'r')
rshifts = galaxies['redshifts']
n_gals = len(rshifts['z'])

#plot the redshifts as a function of distance along with the expected value from the Hubble constant
plt.scatter(rshifts['r'], rshifts['z'])
r_vals = np.linspace(galaxies['distance_posterior']['conf_interval'][0], galaxies['distance_posterior']['conf_interval'][1])
plt.plot(r_vals, H_0*r_vals/C_L, color='black')
plt.show()

#data points for the 3d scatter plot
xs = np.zeros(n_gals)
ys = np.zeros(n_gals)
zs = np.zeros(n_gals)

#convert spherical to Cartesian coordinates
for i in range(n_gals):
    r = rshifts['r'][i]
    theta = (1 - rshifts['dec'][i]/90)*math.pi/2
    phi = rshifts['ra'][i]*math.pi/180
    xs[i] = r*math.sin(theta)*math.cos(phi)
    ys[i] = r*math.sin(theta)*math.sin(phi)
    zs[i] = r*math.cos(theta)

galaxies.close()

#plot the cluster
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim3d(0, 1.2*r_vals[-1])
#ax.set_ylim3d(0, r_vals[1])
#ax.set_zlim3d(1e17,1e19)

ax.scatter(xs,ys,zs)
plt.show()
