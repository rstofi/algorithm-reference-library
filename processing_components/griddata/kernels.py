"""
Functions that define and manipulate kernels

"""
import logging

import numpy

from data_models.memory_data_models import Image
from processing_library.fourier_transforms.convolutional_gridding import coordinates, grdsf
from processing_library.image.operations import copy_image, create_w_term_like, pad_image, fft_image
from processing_library.image.operations import create_image_from_array
from processing_components.griddata.convolution_functions import create_convolutionfunction_from_image
from processing_components.image.operations import reproject_image, create_empty_image_like

log = logging.getLogger(__name__)


def create_box_convolutionfunction(im, oversampling=1, support=1):
    """ Fill a box car function into a ConvolutionFunction

    Also returns the griddata correction function as an image

    :param im: Image template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as ConvolutionFunction
    """
    assert isinstance(im, Image)
    cf = create_convolutionfunction_from_image(im, oversampling=1, support=4)
    
    nchan, npol, _, _ = im.shape
    
    cf.data[...] = 0.0 + 0.0j
    cf.data[..., 2, 2] = 1.0 + 0.0j
    
    # Now calculate the griddata correction function as an image with the same coordinates as the image
    # which is necessary so that the correction function can be applied directly to the image
    nchan, npol, ny, nx = im.data.shape
    nu = numpy.abs(coordinates(nx))
    
    gcf1d = numpy.sinc(nu)
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf = 1.0 / gcf
    
    gcf_data = numpy.zeros_like(im.data)
    gcf_data[...] = gcf[numpy.newaxis, numpy.newaxis, ...]
    gcf_image = create_image_from_array(gcf_data, cf.projection_wcs, im.polarisation_frame)
    
    return gcf_image, cf


def create_pswf_convolutionfunction(im, oversampling=8, support=6):
    """ Fill an Anti-Aliasing filter into a ConvolutionFunction

    Fill the Prolate Spheroidal Wave Function into a GriData with the specified oversampling. Only the inner
    non-zero part is retained

    Also returns the griddata correction function as an image

    :param im: Image template
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as ConvolutionFunction
    """
    assert isinstance(im, Image), im
    # Calculate the convolution kernel. We oversample in u,v space by the factor oversampling
    cf = create_convolutionfunction_from_image(im, oversampling=oversampling, support=support)
    
    kernel = numpy.zeros([oversampling, support])
    for grid in range(support):
        for subsample in range(oversampling):
            nu = ((grid - support // 2) - (subsample - oversampling // 2) / oversampling)
            kernel[subsample, grid] = grdsf([nu / (support // 2)])[1]
    
    kernel /= numpy.sum(numpy.real(kernel[oversampling // 2, :]))
    
    nchan, npol, _, _ = im.shape
    
    cf.data = numpy.zeros([nchan, npol, 1, oversampling, oversampling, support, support]).astype('complex')
    for y in range(oversampling):
        for x in range(oversampling):
            cf.data[:, :, 0, y, x, :, :] = numpy.outer(kernel[y, :], kernel[x, :])[numpy.newaxis, numpy.newaxis, ...]
    norm = numpy.sum(numpy.real(cf.data[0, 0, 0, 0, 0, :, :]))
    cf.data /= norm
    
    # Now calculate the griddata correction function as an image with the same coordinates as the image
    # which is necessary so that the correction function can be applied directly to the image
    nchan, npol, ny, nx = im.data.shape
    nu = numpy.abs(2.0 * coordinates(nx))
    gcf1d = grdsf(nu)[0]
    gcf = numpy.outer(gcf1d, gcf1d)
    gcf[gcf > 0.0] = gcf.max() / gcf[gcf > 0.0]
    
    gcf_data = numpy.zeros_like(im.data)
    gcf_data[...] = gcf[numpy.newaxis, numpy.newaxis, ...]
    gcf_image = create_image_from_array(gcf_data, cf.projection_wcs, im.polarisation_frame)
    
    return gcf_image, cf


def create_awterm_convolutionfunction(im, make_pb=None, nw=1, wstep=1e15, oversampling=8, support=6, use_aaf=True,
                                      maxsupport=512, **kwargs):
    """ Fill AW projection kernel into a GridData.

    :param im: Image template
    :param make_pb: Function to make the primary beam model image (hint: use a partial)
    :param nw: Number of w planes
    :param wstep: Step in w (wavelengths)
    :param oversampling: Oversampling of the convolution function in uv space
    :return: griddata correction Image, griddata kernel as GridData
    """
    d2r = numpy.pi / 180.0
    
    # We only need the griddata correction function for the PSWF so we make
    # it for the shape of the image
    nchan, npol, ony, onx = im.data.shape
    
    assert isinstance(im, Image)
    # Calculate the template convolution kernel.
    cf = create_convolutionfunction_from_image(im, oversampling=oversampling, support=support)
    
    cf_shape = list(cf.data.shape)
    cf_shape[2] = nw
    cf.data = numpy.zeros(cf_shape).astype('complex')
    
    cf.grid_wcs.wcs.crpix[4] = nw // 2 + 1.0
    cf.grid_wcs.wcs.cdelt[4] = wstep
    cf.grid_wcs.wcs.ctype[4] = 'WW'
    if numpy.abs(wstep) > 0.0:
        w_list = cf.grid_wcs.sub([5]).wcs_pix2world(range(nw), 0)[0]
    else:
        w_list = [0.0]

    assert isinstance(oversampling, int)
    assert oversampling > 0

    nx = max(maxsupport, 2 * oversampling * support)
    ny = max(maxsupport, 2 * oversampling * support)
    
    qnx = nx // oversampling
    qny = ny // oversampling

    cf.data[...] = 0.0

    subim = copy_image(im)
    ccell = onx * numpy.abs(d2r * subim.wcs.wcs.cdelt[0]) / qnx

    subim.data = numpy.zeros([nchan, npol, qny, qnx])
    subim.wcs.wcs.cdelt[0] = -ccell / d2r
    subim.wcs.wcs.cdelt[1] = +ccell / d2r
    subim.wcs.wcs.crpix[0] = qnx // 2 + 1.0
    subim.wcs.wcs.crpix[1] = qny // 2 + 1.0

    if use_aaf:
        this_pswf_gcf, _ = create_pswf_convolutionfunction(subim, oversampling=1, support=6)
        norm = 1.0 / this_pswf_gcf.data
    else:
        norm = 1.0
    
    if make_pb is not None:
        pb = make_pb(subim)
        rpb, footprint = reproject_image(pb, subim.wcs, shape=subim.shape)
        rpb.data[footprint.data < 1e-6] = 0.0
        norm *= rpb.data

    # We might need to work with a larger image
    padded_shape = [nchan, npol, ny, nx]
    thisplane = copy_image(subim)
    thisplane.data = numpy.zeros(thisplane.shape, dtype='complex')
    for z, w in enumerate(w_list):
        thisplane.data[...] = 0.0 + 0.0j
        thisplane = create_w_term_like(thisplane, w, dopol=True)
        thisplane.data *= norm
        paddedplane = pad_image(thisplane, padded_shape)
        paddedplane = fft_image(paddedplane)
        
        ycen, xcen = ny // 2, nx // 2
        for y in range(oversampling):
            ybeg = y + ycen + (support * oversampling) // 2 - oversampling // 2
            yend = y + ycen - (support * oversampling) // 2 - oversampling // 2
            # vv = range(ybeg, yend, -oversampling)
            for x in range(oversampling):
                xbeg = x + xcen + (support * oversampling) // 2 - oversampling // 2
                xend = x + xcen - (support * oversampling) // 2 - oversampling // 2

                # uu = range(xbeg, xend, -oversampling)
                cf.data[..., z, y, x, :, :] = paddedplane.data[..., ybeg:yend:-oversampling, xbeg:xend:-oversampling]
                # for chan in range(nchan):
                #     for pol in range(npol):
                #         cf.data[chan, pol, z, y, x, :, :] = paddedplane.data[chan, pol, :, :][vv, :][:, uu]

    cf.data /= numpy.sum(numpy.real(cf.data[0, 0, nw // 2, oversampling // 2, oversampling // 2, :, :]))
    cf.data = numpy.conjugate(cf.data)

    #====================================
    #Use ASKAPSoft routine to crop the support size
    crop_ASKAPSOft_like = True;
    if crop_ASKAPSOft_like:
        #Hardcode the cellsize: 1 / FOV
        #uv_cellsize = 57.3;#N=1200 pixel and pixelsize is 3 arcseconds
        #uv_cellsize = 43.97;#N=1600 pixel and pixelsize is 3 arcseconds
        #uv_cellsize = 114.6;#N=1800 pixel with 1 arcsecond pixelsize
        #uv_cellsize = 57.3;#N=1800 pixel with 2 arcsecond pixelsize
        #uv_cellsize = 1145.91509915;#N=1800 pixxel with 0.1 arcsecond pixelsize

        #Get from **kwargs
        if kwargs is None:
            #Safe solution works for baselines up to > 100km and result in small kernels
            uv_cellsize = 1145.91509915;#N=1800 pixxel with 0.1 arcsecond pixelsize
            
        if 'UVcellsize' in kwargs.keys():
            uv_cellsize = kwargs['UVcellsize'];

        #print(uv_cellsize);

        #Cutoff param in ASKAPSoft hardcoded as well
        ASKAPSoft_cutof = 0.1;

        wTheta_list = numpy.zeros(len(w_list));
        for i in range(0,len(w_list)):
            if w_list[i] == 0:
                wTheta_list[i] = 0.9;#This is due to the future if statements cause if it is small, the kernel will be 3 which is a clear cutoff
            else:
                wTheta_list[i] =  numpy.fabs(w_list[i]) / (uv_cellsize * uv_cellsize);

        kernel_size_list = [];

        #We rounded the kernels according to conventional rounding rules
        for i in range(0,len(wTheta_list)):
            #if wTheta_list[i] < 1:
            if wTheta_list[i] < 1:#Change to ASKAPSoft
                kernel_size_list.append(int(3.));
            elif ASKAPSoft_cutof < 0.01:
                kernel_size_list.append(int(6 + 1.14*wTheta_list[i]));
            else:
                kernel_size_list.append(int(numpy.sqrt( 49 + wTheta_list[i] * wTheta_list[i])));

        log.info('W-kernel w-terms:');
        log.info(w_list);
        log.info('Corresponding w-kernel sizes:');
        log.info(kernel_size_list);

        print(numpy.unique(kernel_size_list));
        #print(kernel_size_list);

        crop_list = [];
        #another rounding according to conventional rounding rules
        for i in range(0,len(kernel_size_list)):
            if support - kernel_size_list[i] <= 0:
                crop_list.append(int(0));
            else:
                crop_list.append(int((support - kernel_size_list[i]) / 2));

        #Crop original suppor    
        for i in range(0,nw):
            if crop_list[i] != 0:
                cf.data[0,0,i,:,:,0:crop_list[i],:] = 0;
                cf.data[0,0,i,:,:,-crop_list[i]:,:] = 0;
                cf.data[0,0,i,:,:,:,0:crop_list[i]] = 0;
                cf.data[0,0,i,:,:,:,-crop_list[i]:] = 0;
            else:
                pass;


            #Plot
            #import matplotlib.pyplot as plt
            #cf.data[0,0,i,0,0,...][cf.data[0,0,i,0,0,...] != 0.] = 1+0.j;
            #plt.imshow(numpy.real(cf.data[0,0,i,0,0,...]))

            #plt.show(block=True)
            #plt.close();

    #====================================

    if use_aaf:
        pswf_gcf, _ = create_pswf_convolutionfunction(im, oversampling=1, support=6)
    else:
        pswf_gcf = create_empty_image_like(im)
        pswf_gcf.data[...] = 1.0
    
    return pswf_gcf, cf
