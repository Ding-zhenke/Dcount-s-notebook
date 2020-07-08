import numpy as np
from mayavi import mlab
x, y,z = np.mgrid[-2:3, -2:3, -2:3]
r = np.sqrt(x**2+y**2+z**4)
u= y * np.sin(r) / (r + 0.001)
V=-x*np.sin(r)/(r+0.001)
W = np.zeros_like(z)
obj = mlab.quiver3d(x, y, z, u, V, W, line_width=3, scale_factor=1)
mlab.show( )