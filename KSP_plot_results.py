from __future__ import division
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.ticker as ticker
from matplotlib import colors as mcolors
#===============================================================================
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


# evenly sampled time at 200ms intervals
t = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

t1 = np.array([140, 133, 117, 95, 67, 40, 23])

t2 = np.array([7, 13, 24, 36, 55, 60, 85])

plt.annotate('Trade-off point', xy=(0.53, 65), xytext=(0.55, 80),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )

# red dashes, blue squares and green triangles
plt.plot(t, t1, '-bs', t, t2, '-g^')
plt.xlabel('Successful transmission probability for the D2D network')
plt.ylabel('Throughput (Mb/s)')
plt.grid(True)
plt.legend(['WiFi throughput', 'D2D throughout']) #<-- give a legend
plt.savefig('KSP_2017.eps', format='eps', dpi=400)
plt.show()

# 0.1
# 0.2
# 0.3
# 0.4
# 0.5
# 0.6
# 0.7
# WiFi
# throughput(Mpbs)
# 140
# 133
# 117
# 95
# 67
# 40
# 23
# D2D
# throughput(Mpbs)
# 7
# 13
# 24
# 36
# 55
# 60
# 85
