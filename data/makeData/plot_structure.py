import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import sys, pdb
import pandas as pd
from scipy.spatial.distance import cdist
satSize = 200
satMarker = 'o'
satColor = 'b'
atomSize = 150
atomMarker =  '.'
atomColor = 'g'
lineSize = 1
surfacePoints = 25
surfaceColor = 'k'
surfaceAlpha = 0.2

surface = 'grid'  # sphere, grid
legends = False

def plot_structure(satPath, clusterPath):

    #satPath = './FCC_sats11/normvaluesSatellitePositions_FCC_11sat_1pLC.csv'
    satDf = pd.read_csv(satPath,index_col=0,engine='python')

    satsX = satDf['x'].values
    satsY = satDf['y'].values
    satsZ = satDf['z'].values

    #clusterPath = './FCC_sats11/FCC/FCC_h_4_k_4_l_7_atom_Zn_lc_2.4442.xyz'
    atomX = []
    atomY = []
    atomZ = []
    f = open(clusterPath, "r")
    for idx, line in enumerate(f):
        if idx < 2:
            continue
        else:
            _, x, y, z = line.split()
            atomX.append(float(x))
            atomY.append(float(y))
            atomZ.append(float(z))

    atomX = np.array(atomX)# - np.array(atomX).mean()# - 0.00000
    atomY = np.array(atomY)# - np.array(atomY).mean()# - 6.91324
    atomZ = np.array(atomZ)# - np.array(atomZ).mean()# - 6.91324
    #pdb.set_trace()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(satsX,satsY,satsZ, marker=satMarker, s=satSize, c=satColor, label='Satellite')
    ax.scatter(atomX,atomY,atomZ, marker=atomMarker, s=atomSize, c=atomColor, label='Atom')
    # Hide grid lines
    ax.grid(False)
    #plt.legend()
    # Hide axes ticks
    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])
    #ax.set_axis_off()
    # Make background transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    for i in range(len(satsX)):
        if i == 0:
            ax.plot(np.array([atomX[0],satsX[i]]),np.array([atomY[0],satsY[i]]),np.array([atomZ[0],satsZ[i]]),c='r', ls='--',label='Distance', linewidth=lineSize)
        else:
            ax.plot(np.array([atomX[0],satsX[i]]),np.array([atomY[0],satsY[i]]),np.array([atomZ[0],satsZ[i]]),c='r', ls='--', linewidth=lineSize)

    XA = np.array([0,0,0])
    XB = np.array([satsX[0], satsY[0],satsZ[0]])
    print(XA, XB)
    maxDist = np.linalg.norm(XB-XA)
    print(maxDist)

    u = np.linspace(0, 2 * np.pi, surfacePoints)
    v = np.linspace(0, np.pi, surfacePoints)
    X = maxDist * np.outer(np.cos(u), np.sin(v))
    Y = maxDist * np.outer(np.sin(u), np.sin(v))
    Z = maxDist * np.outer(np.ones(np.size(u)), np.cos(v))

    if surface == 'sphere':
        ax.plot_surface(X, Y, Z, color = surfaceColor,rstride=4, cstride=4, alpha=surfaceAlpha)
    elif surface == 'grid':
        ax.plot_wireframe(X,Y,Z, color = surfaceColor,alpha=surfaceAlpha)
    #ax.plot(np.array([0,satsX[1]]),np.array([0,satsY[1]]),np.array([0,satsZ[1]]),c='r', ls='--')
    #ax.plot(np.array([0,satsX[2]]),np.array([0,satsY[2]]),np.array([0,satsZ[2]]),c='r', ls='--')
    if legends:
        plt.legend()

        lgnd = plt.legend(loc="upper right", scatterpoints=1, fontsize=10)
        for handle in lgnd.legendHandles:
            try:
                handle.set_sizes([60.0])
            except:
                pass

    plt.tight_layout()
    #plt.savefig('Sat_rep_Alllinesddd.pdf', dpi=400)

    plt.show()