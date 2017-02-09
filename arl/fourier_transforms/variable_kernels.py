# Bojan Nikolic <b.nikolic@mrao.cam.ac.uk>
#
# Synthesise and Image interferometer data
"""Convolutional gridding support functions

All functions that involve convolutional gridding are kept here.
"""

from __future__ import division

from astropy.constants import c

from arl.fourier_transforms.convolutional_gridding import *

from functools import lru_cache

log = logging.getLogger(__name__)


def standard_kernel_lambda(vis, shape, oversampling=8, support=3):
    """Return a lambda function to calculate the standard visibility kernel

    This is only required for testing speed versus fixed_kernel_grid.

    :param vis: visibility
    :param shape: tuple with 2D shape of grid
    :param oversampling: Oversampling factor
    :param support: Support of kernel
    :returns: Function to look up gridding kernel
    """
    sk = anti_aliasing_calculate(shape, oversampling, support)[1]
    return lambda row, chan=0: sk


def w_kernel_lambda(vis, shape, fov, oversampling=4, wstep=100.0, npixel_kernel=16, cache_size=10000):
    """Return a lambda function to calculate the w term visibility kernel

    This function is called once. It uses an LRU cache to hold the convolution kernels. As a result,
    initially progress is slow as the cache is filled. Then it speeds up.

    :param vis: visibility
    :param shape: tuple with 2D shape of grid
    :param fov: Field of view in radians
    :param oversampling: Oversampling factor
    :param support: Support of kernel
    :param wstep: Step in w between cached functions
    :param cache_size: Size of cache in items
    :returns: Function to look up gridding kernel as function of row, and cache
    """
    wmax = numpy.max(numpy.abs(vis.w)) * numpy.max(vis.frequency) / c.value
    log.debug("w_kernel_lambda: Maximum w = %f wavelengths" % (wmax))
    
    warray = numpy.array(vis.w)
    karray = numpy.array(vis.frequency) / (c.value * wstep)
    
    @lru_cache(maxsize=None)
    def cached_on_w(w_integral):
        npixel_kernel_scaled = max(8, int(round(npixel_kernel*abs(w_integral*wstep)/wmax)))
        result = w_kernel(field_of_view=fov, w=wstep * w_integral, npixel_farfield=shape[0],
                        npixel_kernel=npixel_kernel_scaled, kernel_oversampling=oversampling)
        return result
    
    # The lambda function has arguments row and chan so any gridding function can only depend on those
    # parameters. Eventually we could extend that to include polarisation.
    return lambda row, chan=0: cached_on_w(int(round(warray[row] * karray[chan]))), cached_on_w


def variable_kernel_degrid(kernel_function, vshape, uvgrid, uv, uvscale, vmap):
    """Convolutional degridding with frequency and polarisation independent

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernel_function: Function to return oversampled convolution kernel for given row
    :param uvgrid:   The uv plane to de-grid from
    :param uv: fractional uv coordinates in range[-0.5,0.5[
    :param uvscale: scaling for each channel
    :param viscoords: list of visibility coordinates to use for kernel_function
    :param vmap: Function to may image channels to visibility channels
    :returns: Array of visibilities.
    """
    inchan, inpol, ny, nx = uvgrid.shape
    nvis, vnchan, vnpol = vshape
    assert vnpol == inpol, "Number of polarizations must be the same"
    vis = numpy.zeros(vshape, dtype='complex')
    wt = numpy.zeros(vshape)

    # Initialise the kernels.
    kmap = []
    for row in range(nvis):
        krow = []
        for vchan in range(vnchan):
            krow.append(kernel_function(row, vchan))
        kmap.append(krow)

    for row in range(nvis):
        krow = kmap[row]
        for vchan in range(vnchan):
            kernel = numpy.conjugate(krow[vchan])
            ichan = vmap(vchan)
            kernel_oversampling, _, gh, gw = kernel.shape
            y, yf = frac_coord(nx, kernel_oversampling, uvscale[1] * uv[row, 1])
            x, xf = frac_coord(ny, kernel_oversampling, uvscale[0] * uv[row, 0])
            slicey = slice(y - gh // 2, y + (gh + 1) // 2)
            slicex = slice(x - gw // 2, x + (gw + 1) // 2)
            for vpol in range(vnpol):
                vis[row, vchan, vpol] = numpy.sum(uvgrid[ichan, vpol, slicey, slicex] * kernel[yf, xf, :, :])
                wt[row, vchan, vpol] = numpy.sum(kernel[yf, xf, :, :].real)
    vis[numpy.where(wt > 0)] = vis[numpy.where(wt > 0)] / wt[numpy.where(wt > 0)]
    vis[numpy.where(wt < 0)] = 0.0
    return numpy.array(vis)


def variable_kernel_grid(kernel_function, uvgrid, uv, uvscale, vis, visweights, vmap):
    """Grid after convolving with frequency and polarisation independent gcf

    Takes into account fractional `uv` coordinate values where the GCF
    is oversampled

    :param kernel_function: Function to return oversampled convolution kernel for given row
    :param uvgrid: Grid to add to
    :param uv: UVW positions
    :param vis: CompressedVisibility values
    :param vis: CompressedVisibility weights
    :param vmap: Function to map image channel to visibility channel
    """
    
    inchan, inpol, ny, nx = uvgrid.shape
    nvis, vnchan, vnpol = vis.shape
    assert vnpol == inpol, "Number of polarizations must be the same"
    sumwt = numpy.zeros([inchan, inpol])
    
    # Initialise the kernels.
    kmap=[]
    for row in range(nvis):
        krow = []
        for vchan in range(vnchan):
            krow.append(kernel_function(row, vchan))
        kmap.append(krow)

    for row in range(nvis):
        krow = kmap[row]
        for vchan in range(vnchan):
            kernel = krow[vchan]
            ichan = vmap(vchan)
            kernel_oversampling, _, gh, gw = kernel.shape
            y, yf = frac_coord(nx, kernel_oversampling, uvscale[1] * uv[row, 1])
            x, xf = frac_coord(ny, kernel_oversampling, uvscale[0] * uv[row, 0])
            slicey = slice(y - gh // 2, y + (gh + 1) // 2)
            slicex = slice(x - gw // 2, x + (gw + 1) // 2)
            for vpol in range(vnpol):
                viswt = vis[row, vchan, vpol] * visweights[row, vchan, vpol]
                uvgrid[ichan, vpol, slicey, slicex] += kernel[yf, xf, :, :] * viswt
                sumwt[ichan, vpol] += numpy.sum(kernel[yf, xf, :, :].real) * visweights[row, vchan, vpol]

    return uvgrid, sumwt
