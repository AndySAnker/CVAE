import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


n = 100
r = 1
generateFixed = True


def sampleSphere(n, r=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Generating fixed %d points on a sphere centered at the origin" % (n))
    alpha = 4.0*np.pi*r*r/n
    d = np.sqrt(alpha)
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alpha/d_nu
    count = 0
    points = []
    for m in range (0,m_nu):
        nu = np.pi*(m+0.5)/m_nu
        m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
        for n in range (0,m_phi):
            phi = 2*np.pi*n/m_phi
            xp = r*np.sin(nu)*np.cos(phi)
            yp = r*np.sin(nu)*np.sin(phi)
            zp = r*np.cos(nu)
            count = count +1
            points.append([xp, yp, zp])

    return np.array(points)