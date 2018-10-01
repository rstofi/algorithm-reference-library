# coding: utf-8

# # Pipeline processing using Dask

import numpy

from data_models.parameters import arl_path

results_dir = arl_path('test_results')
dask_dir = arl_path('test_results/dask-work-space')

from data_models.polarisation import PolarisationFrame
from wrappers.arlexecute.visibility.base import create_visibility_from_ms, create_visibility_from_rows
from wrappers.arlexecute.visibility.operations import append_visibility, convert_visibility_to_stokes
from wrappers.arlexecute.visibility.vis_select import vis_select_uvrange

from wrappers.arlexecute.image.deconvolution import deconvolve_cube, restore_cube
from wrappers.arlexecute.image.operations import export_image_to_fits, qa_image
from wrappers.arlexecute.image.gather_scatter import image_gather_channels
from wrappers.arlexecute.imaging.base import create_image_from_visibility
from wrappers.arlexecute.imaging.base import advise_wide_field

from workflows.arlexecute.imaging.imaging_arlexecute import invert_list_arlexecute_workflow
from workflows.serial.imaging.imaging_serial import invert_list_serial_workflow

from wrappers.arlexecute.execution_support.arlexecute import arlexecute

import logging

import argparse

def init_logging():
    logging.basicConfig(filename='%s/dprepb-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Benchmark pipelines in numpy and dask')
    parser.add_argument('--use_dask', type=str, default='True', help='Use Dask?')
    parser.add_argument('--serial_invert', type=str, default='True',
                        help='Use serial invert?')
    parser.add_argument('--nworkers', type=int, default=4, help='Number of workers')
    parser.add_argument('--npixel', type=int, default=1024, help='Number of pixels per axis')
    parser.add_argument('--context', dest='context', default='2d', help='Context: 2d|timeslice|wstack')

    args = parser.parse_args()
    
    log = logging.getLogger()
    logging.info("Starting Imaging pipeline")
    
    arlexecute.set_client(use_dask=args.use_dask=='True',
                          threads_per_worker=1,
                          memory_limit=8589934592,
                          n_workers=args.nworkers,
                          local_dir=dask_dir)
    print(arlexecute.client)
    arlexecute.run(init_logging)
    
    nchan = 40
    uvmax = 450.0
    nfreqwin = 2
    centre = 0
    cellsize = 0.0001
    npixel = args.npixel
    # This is about 9 pixels and causes the astropy.convolve function to take forever. Need to do
    # by FFT
    psfwidth = (((8.0 / 2.35482004503) / 60.0) * numpy.pi / 180.0) / cellsize
    psfwidth = 1.0
    
    context = args.context
    if context == 'wstack':
        vis_slices = 45
        print('wstack processing')
    elif context == 'timeslice':
        print('timeslice processing')
        vis_slices = 2
    else:
        print('2d processing')
        context = '2d'
        vis_slices = 1

    input_vis = [arl_path('data/vis/sim-1.ms'), arl_path('data/vis/sim-2.ms')]
    
    def load_ms(c):
        v1 = create_visibility_from_ms(input_vis[0], channum=[c])[0]
        v2 = create_visibility_from_ms(input_vis[1], channum=[c])[0]
        vf = append_visibility(v1, v2)
        vf = convert_visibility_to_stokes(vf)
        vf.configuration.diameter[...] = 35.0
        rows = vis_select_uvrange(vf, 0.0, uvmax=uvmax)
        return create_visibility_from_rows(vf, rows)
    
    
    # Load data from previous simulation
    vis_list = [arlexecute.execute(load_ms)(c) for c in range(nchan)]
    
    print('Reading visibilities')
    vis_list = arlexecute.persist(vis_list)
    
    # The vis data are on the workers so we run the advice function on the workers
    # without transfering the data back to the host.
    advice_list = [arlexecute.execute(advise_wide_field)(v, guard_band_image=8.0, delA=0.02,
                                                         wprojection_planes=1)
                   for _, v in enumerate(vis_list)]
    advice_list = arlexecute.compute(advice_list, sync=True)
    print(advice_list[0])
    
    pol_frame = PolarisationFrame("stokesIQUV")
    
    model_list = [arlexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                   polarisation_frame=pol_frame)
                  for v in vis_list]
    
    model_list = arlexecute.persist(model_list)
    
    if args.serial_invert == 'True':
        print("Invert is serial")
        dirty_list = [arlexecute.execute(invert_list_serial_workflow)([vis_list[i]],
                                                                      template_model_imagelist=[model_list[i]],
                                                                      context=context,
                                                                      vis_slices=vis_slices)[0]
                      for i in range(nchan)]
        psf_list = [arlexecute.execute(invert_list_serial_workflow)([vis_list[i]],
                                                                    template_model_imagelist=[model_list[i]],
                                                                    context=context, dopsf=True,
                                                                    vis_slices=vis_slices)[0]
                    for i in range(nchan)]
    else:
        print("Invert is parallel")
        dirty_list = invert_list_arlexecute_workflow(vis_list, template_model_imagelist=model_list, context=context,
                                                     vis_slices=vis_slices)
        psf_list = invert_list_arlexecute_workflow(vis_list, template_model_imagelist=model_list, context=context,
                                                   dopsf=True, vis_slices=vis_slices)
    
    
    def deconvolve(d, p, m):
        c, resid = deconvolve_cube(d[0], p[0], m, threshold=0.01, fracthresh=0.01, window_shape='quarter',
                                   niter=1000, gain=0.1, algorithm='hogbom-complex')
        r = restore_cube(c, p[0], resid)
        return r
    
    
    log.info('About assemble cubes and restore each frequency')
    restored_list = [arlexecute.execute(deconvolve)(dirty_list[c], psf_list[c], model_list[c])
                     for c in range(nchan)]
    restored_list = arlexecute.compute(restored_list, sync=True)
    restored_cube = image_gather_channels(restored_list)

    print(qa_image(restored_cube, context='CLEAN restored cube'))
    export_image_to_fits(restored_cube, '%s/dprepb_arlexecute_clean_restored_cube.fits' % (results_dir))
    
    try:
        arlexecute.close()
    except:
        pass
