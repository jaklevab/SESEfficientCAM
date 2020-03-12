'''
Generate grid of points within a PySAL geometry

Author: Dani Arribas-Bel <@darribas>

Copyright (c) 2014, Dani Arribas-Bel
All rights reserved.

LICENSE
-------

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* The name of the author may not be used to endorse or
  promote products derived from this software without specific prior written
  permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

import time
import pysal as ps
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def kde_map(xys, ax=None, cmap=plt.cm.YlOrRd, alpha=1, **kwds):
    '''
    Quickly create a KDE map from a set of point coordinates
    ...

    Arguments
    ---------
    xys         : ndarray
                  Nx2 array with the original XY locations
    ax          : matplotlib.axes.AxesSubplot
                  [Optional. Default=None] 
    cmap        : plt.cmap
                  [Optional. Default: plt.cm.YlOrRd] Matplotlib colormap
    alpha       : float
                  [Optional. Default: 1] Transparency for the surface map
    **kwds      : dictionary
                  Additional arguments to be passed to `kde_grid`
    '''
    # KDE computation
    xyz, gdim = kde_grid(xys, **kwds)
    x = xyz[:, 0].reshape(gdim)
    y = xyz[:, 1].reshape(gdim)
    z = xyz[:, 2].reshape(gdim)
    levels = np.linspace(0, z.max(), 25)
    # Plotting
    display=False
    if not ax:
        f, ax = plt.subplots(1)
        display = True
    ax.contourf(x, y, z, levels=levels, cmap=cmap, alpha=alpha)
    if display:
        plt.show()
    return ax
def kde_grid(xys, bw=None, shp_link=None, spaced=0.01, ax=None, est=None,
        verbose=False):
    '''
    Generate a KDE map of `xys`, potentially within the boundaries of `shp_link`. 

    ...

    Arguments
    ---------
    xys         : ndarray
                  Nx2 array with the original XY locations
    bw          : float
                  [Optional, default: None] Bandwith for the kernel. If left
                  blank, it is chosen by cross-validation as shown in the
                  following blog post:
                  http://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
    shp_link    : str
                  [Optional, default: None] Path to shapefile to use for
                  clipping the grid. If nothing is passed a grid on the
                  dimensions of the bounding box of the points is used.
    spaced      : float
                  [Optional, default: 0.01] Share in the range [0, 1] that
                  should be left between points on each axis
    ax          : matplotlib axis object
                  [Optional, default: None] If passed, the map is embedded into `ax`,
                  otherwise a brand new figure is created
    est         : KernelDensity
                  [Optional, default: None] Kernel density estimator in
                  scikit-learn. If none is passed, it initializes a new one
                  with the following parameters:

                        * metric='euclidean'
                        * kernel='gaussian'
                        * algorithm='ball_tree'

    verbose     : Boolean
                  [Optional. Default=False] Switch to turn on info printouts

    Returns
    -------
    kde         : ndarray
                  Nx3 array with the XY locations on the clipped grid and
                  their estimated density
    gdim        : tuple
                  Dimensions of the grid
    '''
    # Fitting
    if not est:
        kde = KernelDensity(metric='euclidean',
                            kernel='gaussian', algorithm='ball_tree')
    if not bw:
        ti = time.time()
        gs = GridSearchCV(kde, \
                {'bandwidth': np.linspace(0.1, 1.0, 30)}, \
                cv=3)
        cv = gs.fit(xys)
        bw = cv.best_params_['bandwidth']
        tf = time.time()
        if verbose:
            print ("Finding optimal bandwith (%i-fold cross-validation): ")%cv.cv, \
                    ("%.2f secs")%(tf-ti)
    kde.bandwidth = bw
    _ = kde.fit(xys)
    # Grid
    t0 = time.time()
    if shp_link:
        gXY, gdim = grid_clipped(shp_link, spaced)
        zi = np.where(gXY[:, 2] == 1)
        zXY = gXY[zi, :2]
    else:
        minX, minY = xys.min(axis=0)
        maxX, maxY = xys.max(axis=0)
        bbox = [minX, minY, maxX, maxY]
        gXY, gdim = grid_full(bbox, spaced)
        zi = np.arange(gXY.shape[0])
        zXY = gXY
    t1 = time.time()
    if verbose:
        print ("Creating grid: %.2f secs"%(t1-t0))
    # Evaluate
    z = np.exp(kde.score_samples(zXY))
    t2 = time.time()
    if verbose:
        print ("Evaluating kernel in grid points: %.2f secs"%(t2-t1))
    # Plug into grid
    zg = -9999 + np.zeros(gXY.shape[0])
    zg[zi] = z
    xyz = np.hstack((gXY[:, :2], zg[:, None]))
    return xyz, gdim

def grid_full(bbox, spaced):
    '''
    Generate a grid of points on the bounding box of `bbox`
    ...

    Arguments
    ---------
    bbox            : list
                      Bounding box for the grid in a sequence of minX, minY,
                      maxX, maxY
    spaced          : float
                      Share in the range [0, 1] that should be left between
                      points on each axis

    Returns
    -------
    xys     : ndarray
              Nx2 array with XY coordinates of the entire grid
    gdim    : tuple
              Dimensions of the grid
    '''
    difX = (bbox[2] - bbox[0]) * spaced
    xs = np.arange(bbox[0], bbox[2]+difX, difX)
    difY = (bbox[3] - bbox[1]) * spaced
    ys = np.arange(bbox[1], bbox[3]+difY, difY)
    X, Y = np.meshgrid(xs, ys)
    xys = np.vstack((X.ravel(), Y.ravel())).T
    return xys, X.shape

def grid_clipped(shp_link, spaced):
    '''
    Generate a grid of points on the bounding box of `shp_link` and mark those
    within its boundaries
    ...

    Arguments
    ---------
    shp_link        : str
                      Path to polygon shapefile
    spaced          : float
                      Share in the range [0, 1] that should be left between
                      points on each axis

    Returns
    -------
    xys     : ndarray
              Nx3 array with XY coordinates of the entire grid and 1 if the
              point is inside, 0 otherwise
    gdim    : tuple
              Dimensions of the grid
    '''
    shp = ps.open(shp_link)
    difX = (shp.bbox[2] - shp.bbox[0]) * spaced
    xs = np.arange(shp.bbox[0], shp.bbox[2]+difX, difX)
    difY = (shp.bbox[3] - shp.bbox[1]) * spaced
    ys = np.arange(shp.bbox[1], shp.bbox[3]+difY, difY)
    X, Y = np.meshgrid(xs, ys)
    xys = np.vstack((X.ravel(), Y.ravel(), np.zeros(Y.ravel().shape))).T
    corr = np.array(pip_xy_shp_multi(xys[:, :2], shp_link, empty=-9))
    corr[np.where(corr=='out')] = -9
    xys[:, 2] = corr.astype(int)
    xys[np.where(xys[:, 2] != -9), 2] = 1
    xys[np.where(xys[:, 2] == -9), 2] = 0
    return xys, X.shape

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from pysal.contrib.viz import mapping as maps

    link = ps.examples.get_path('taz.shp')
    link = ps.examples.get_path('columbus.shp')
    shp = ps.open(link)
    pts = np.array([p.centroid for p in ps.open(link)])

    xyz, gdim = kde_map(pts, shp_link=link, spaced=0.03)

    x = xyz[:, 0].reshape(gdim)
    y = xyz[:, 1].reshape(gdim)
    z = xyz[:, 2].reshape(gdim)
    levels = np.linspace(0, z.max(), 25)

    f = plt.figure()
    base = maps.map_poly_shp(ps.open(link))
    base.set_facecolor('none')
    
    ax = maps.setup_ax([base])

    cont = plt.contourf(x, y, z, levels=levels, cmap=plt.cm.bone)
    cont.set_alpha(0.5)
    cents = plt.scatter(pts[:, 0], pts[:, 1])

    plt.show()
