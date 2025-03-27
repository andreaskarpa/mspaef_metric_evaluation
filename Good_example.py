#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:27:54 2024

@author: andreaskarpasitis
"""

import matplotlib.pyplot as plt
import warnings
import matplotlib as mpl

mpl.rcParams["font.family"] = "Arial" #"Ani"
warnings.filterwarnings("ignore")

x = [0.1 ,0.4 ,0.7 ,1 ,1.3 ,1.6 ,1.9]
y1 = [0.95, 0.85, 0.74, 0.72, 0.79, 0.86, 0.91]
y2 = [0.86, 0.73, 0.65, 0.61, 0.65, 0.73, 0.86]
y3 = [0.75, 0.69, 0.53, 0.49, 0.5, 0.53, 0.56]

y4 = [0.42, 0.46, 0.32, 0.26, 0.29, 0.32, 0.35]
y5 = [0.25, 0.18, 0.15, 0.19, 0.23, 0.26, 0.27]

fig = plt.figure(figsize=(6,4))
ax = plt.axes()
ax.plot(x,y1,'o-',color='black')
ax.plot(x,y2,'o-',color='black')
ax.plot(x,y3,'o-',color='black')
ax.plot(x,y4,'o-',color='blue')
ax.plot(x,y5,'o-',color='blue')
ax.set_ylim(0,1)
ax.set_xlim(-0.2,2)
ax.set_xlabel("$\lambda$", fontsize=15)
ax.set_ylabel("Metric value", fontsize=15)
ax.text(-0.06, 0.94, 'a', color='black',fontsize=15)
ax.text(-0.06, 0.84, 'b', color='black',fontsize=15)
ax.text(-0.06, 0.74, 'c', color='black',fontsize=15)
ax.text(-0.06, 0.41, 'd', color='blue',fontsize=15)
ax.text(-0.06, 0.25, 'e', color='blue',fontsize=15)

ax.grid()
plt.savefig('/path/to/somewhere/Good_example.pdf', dpi=300)