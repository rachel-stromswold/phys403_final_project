import numpy as np
import matplotlib.pyplot as plt

dat = np.transpose( np.loadtxt("localizations.txt", delimiter=' ') )

avg_dist_err = (dat[1] + dat[2])*0.5

plt.scatter(dat[0], dat[-1])
plt.show()
