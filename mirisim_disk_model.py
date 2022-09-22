#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thu Jun  9 10:08:29 2022: Creation

@author: echoquet from jupyter notebook J. Lensenring


Here we create the basics for a MIRI simulation to observe the HD 141569 system with the FQPM. 
This includes simulating the stellar source behind the center of the phase mask, 
the off-axis stellar companions, and a debris disk model that crosses the mask's quadrant boundaries.


Final outputs will be detector-sampled slope images (counts/sec).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from time import time

import webbpsf_ext
from webbpsf_ext import image_manip, setup_logging, coords
from webbpsf_ext import miri_filter
from webbpsf_ext.coords import jwst_point, plotAxes    #class to setup pointing info
from webbpsf_ext.image_manip import pad_or_cut_to_size

plt.rcParams['image.origin'] = 'lower'
plt.rcParams["image.cmap"] = 'gist_heat'#'hot'#'copper'

# TODO: make sure the central star is removed from the model if flag in dictionary is false
# TODO: Check that the flux matches the simulation (photometry is preserved)
# Simplify the code to keep the smaller FOV instead of full detector size
# add support for mcfost (?) or convert into a functionto include in another simulation code


#%% Functions

def generate_grid_psf(inst):
    """
    Generates a grid of PSF along the edges of the 4QPM mask, with a log density (denser close the edges)
    This is needed to properly convolved extended objects.

    """
    # Create grid locations for array of PSFs to generate
    apname = inst.psf_coeff_header['APERNAME']
    siaf_ap = inst.siaf[apname]

    # Mask Offset grid positions in arcsec
    xyoff_half = 10**(np.linspace(-2,1,10))
    xoff = yoff = np.concatenate([-1*xyoff_half[::-1],[0],xyoff_half])
    xgrid_off, ygrid_off = np.meshgrid(xoff, yoff)
    xgrid_off, ygrid_off = xgrid_off.flatten(), ygrid_off.flatten()

    # Rotation of ~5deg of the coronagraphic mask
    field_rot = 0 if inst._rotation is None else inst._rotation

    # Science positions in detector pixels
    xoff_sci_asec, yoff_sci_asec = coords.xy_rot(-1*xgrid_off, -1*ygrid_off, -1*field_rot)
    xgrid = xoff_sci_asec / siaf_ap.XSciScale + siaf_ap.XSciRef
    ygrid = yoff_sci_asec / siaf_ap.YSciScale + siaf_ap.YSciRef


    # Now, create all PSFs, one for each (xgrid, ygrid) location
    # Only need to do this once. Can be used for multiple dither positions.
    hdul_psfs = inst.calc_psf_from_coeff(coord_vals=(xgrid, ygrid), coord_frame='sci', return_oversample=True)
    
    return hdul_psfs



def make_spec(name=None, sptype=None, flux=None, flux_units=None, bp_ref=None, **kwargs):
    """
    Create pysynphot stellar spectrum from input dictionary properties.
    """

    from webbpsf_ext import stellar_spectrum
    
    # Renormalization arguments
    renorm_args = (flux, flux_units, bp_ref)
    
    # Create spectrum
    sp = stellar_spectrum(sptype, *renorm_args, **kwargs)
    if name is not None:
        sp.name = name
    
    return sp


def generate_star_psf(star_params, tel_point, inst, shape_new):
    '''
    Here we define the stellar atmosphere parameters for HD 141569, including spectral type, 
    optional values for (Teff, log_g, metallicity), normalization flux and bandpass, 
    as well as RA and Dec.
    Then Computes the PSF, including any offset/dither, using the coefficients.
    It includes geometric distortions based on SIAF info. 
    shape_new is the oversampled shape.
    The output image is shifted to place the central star at the center of the image despite dither/pointing error offsets.
    '''

    # Get `sci` position of center in units of detector pixels (center of mask)
    siaf_ap = tel_point.siaf_ap_obs
    x_cen, y_cen = siaf_ap.reference_point('sci')
    
    # Get `sci` position of the star, including offsets, errors, and dither  
    coord_obj = (star_params['RA_obj'], star_params['Dec_obj'])
    x_star, y_star = tel_point.radec_to_frame(coord_obj, frame_out='sci')
    
    # Get the corresponding shift from center (in regular detector pixel unit)
    x_star_off, y_star_off = (x_star-x_cen, y_star-y_cen)
    
    
    # Create PSF with oversampling (included in `inst`)
    # hdul = inst.calc_psf_from_coeff(sp=sp_star, coord_vals=(x_star,y_star), coord_frame='sci')
    sp_star = make_spec(**star_params)
    psf_image = inst.calc_psf_from_coeff(sp=sp_star, coord_vals=(x_star,y_star), coord_frame='sci', return_hdul=False)

    # Get oversampled pixel shifts
    osamp = inst.oversample
    star_off_oversamp = (y_star_off * osamp, x_star_off * osamp)
    
    # Crop or Expand the PSF to full frame and offset to proper position
    psf_image_full = pad_or_cut_to_size(psf_image, shape_new, offset_vals=star_off_oversamp)
    
  
    # Make new HDUList with the star
    # hdul_full = fits.HDUList(fits.PrimaryHDU(data=psf_image_full, header=hdul[0].header))
    
    return psf_image_full




## time to make the disk mode
## it's fine to use the same star & PSF grids if we want several disk models
## of course the disk model needs updating
def add_disk_into_model(disk_params, hdul_psfs, tel_point, inst, shape_new, star_params, display=True):
    '''
    Properly including extended objects is a little more complicated than for point sources. 
    First, we need properly format the input model to a pixel binning and flux units 
    appropriate for the simulations (ie., pixels should be equal to oversampled PSFs with 
    flux units of counts/sec). Then, the image needs to be rotated relative to the 'idl' 
    coordinate plane and subsequently shifted for any pointing offsets. 
    
    disk_params: input model must be a filename or HDUlist. 
                 For model purpose, it is advised to not include the star and set cen_star=False
    
    '''
    
    siaf_ap = tel_point.siaf_ap_obs
    x_cen, y_cen = siaf_ap.reference_point('sci')
    osamp = inst.oversample
    
    
    # Step 1: resample to the inst.sampling and rescale to inst.bandpass. Converts to photon/s. 
    # Input must be a filename or HDUlist
    # if cen_star==True, the flux of the brightest pixel (assuming it is the star) is saved and put in the center pixel (nx//2, ny//2)
    sp_star = make_spec(**star_params)
    hdul_disk_model = image_manip.make_disk_image(inst, disk_params, sp_star=sp_star)
    print('Input disk model shape: {}'.format(hdul_disk_model[0].data.shape))

    if display:
        fig1, ax1 = plt.subplots(1,1)
        fig1.suptitle('disk model input')
        extent = 0.5 * np.array([-1,1,-1,1]) * hdul_disk_model[0].data.shape[0] * inst.pixelscale/osamp
        ax1.imshow(hdul_disk_model[0].data, extent=extent, vmin=0,vmax=20)
        fig1.tight_layout()
        plt.show()

    
    # Step 2: Apply the pointing parameters: 
    #         Rotate to telescope orientation, and shift with dither/pointing error offsets  
    # Rotation necessary to go from sky coordinates to 'idl' frame   
    # Dither position & pointing errors in arcsec
    rotate_to_idl = -1*(tel_point.siaf_ap_obs.V3IdlYAngle + tel_point.pos_ang) 
    delx, dely = tel_point.position_offsets_act[0]
    hdul_out = image_manip.rotate_shift_image(hdul_disk_model, angle=rotate_to_idl,
                                              delx_asec=delx, dely_asec=dely)

    if display:
        fig1, ax1 = plt.subplots(1,1)
        fig1.suptitle('disk model rotated and shifted')
        extent = 0.5 * np.array([-1,1,-1,1]) * hdul_out[0].data.shape[0] * inst.pixelscale/osamp
        ax1.imshow(hdul_out[0].data, vmin=0,vmax=20, extent=extent)
        fig1.tight_layout()
        plt.show()

    # Step 3: Distort image on 'sci' coordinate grid. 
    # Done Around normal sampling sci_cen (oversampling is applied afterward)
    im_sci, xsci_im, ysci_im = image_manip.distort_image(hdul_out, ext=0, to_frame='sci', return_coords=True)
    hdul_disk_model_sci = fits.HDUList(fits.PrimaryHDU(data=im_sci, header=hdul_out[0].header))

    if display:
        fig1, ax1 = plt.subplots(1,1)
        fig1.suptitle('disk model distorted')
        extent = 0.5 * np.array([-1,1,-1,1]) * im_sci.shape[0] * inst.pixelscale/osamp
        ax1.imshow(im_sci, vmin=0,vmax=20, extent=extent)
        fig1.tight_layout()
        plt.show()


    # Step 4: Convolve the image with spatially-variant PSF.
    # Get X and Y indices corresponding to the center of aperture reference with the same sampling as disk model hdul
    # Note: if XIND_REF or XCEN is not set, input_image.shape / 2 is assumed, with is off by 1.5pixel
    xref, yref = siaf_ap.reference_point('sci')
    hdul_disk_model_sci[0].header['XIND_REF'] = (xref*osamp, "x index of aperture reference")
    hdul_disk_model_sci[0].header['YIND_REF'] = (yref*osamp, "y index of aperture reference")
    hdul_disk_model_sci[0].header['CFRAME'] = 'sci'

    # Convolve image
    im_conv = image_manip.convolve_image(hdul_disk_model_sci, hdul_psfs, output_sampling=osamp)
    print('\nConvolved disk shape: {}'.format(im_conv.shape)) #433x433

    if display:
        fig1, ax1 = plt.subplots(1,1)
        fig1.suptitle('Convolved disk')
        extent = 0.5 * np.array([-1,1,-1,1]) * im_sci.shape[0] * inst.pixelscale/osamp
        ax1.imshow(im_conv, vmin=0,vmax=20, extent=extent)
        fig1.tight_layout()
        plt.show()
    
    # Step 5: Crop or Expand the PSF to full frame and offset to proper position
    # Get `sci` position of the star, including offsets, errors, and dither  
    coord_obj = (star_params['RA_obj'], star_params['Dec_obj'])
    x_star, y_star = tel_point.radec_to_frame(coord_obj, frame_out='sci')
    
    # Get the corresponding shift from center (in regular detector pixel unit)
    x_star_off, y_star_off = (x_star-x_cen, y_star-y_cen)
    star_off_oversamp = (y_star_off * osamp, x_star_off * osamp)
    im_conv_resized = pad_or_cut_to_size(im_conv, shape_new, offset_vals=star_off_oversamp)
    
    return im_conv_resized



def quick_ref_psf(idl_coord, inst, out_shape, star_params=None):
    """
    Create a quick reference PSF for subtraction of the science target.
    """
    
    # Observed SIAF aperture
    siaf_ap = tel_point.siaf_ap_obs
    
    # Location of observation
    xidl, yidl = idl_coord
    
    # Get offset in SCI pixels
    xsci_off, ysci_off = np.array(siaf_ap.convert(xidl, yidl, 'idl', 'sci')) - \
                         np.array(siaf_ap.reference_point('sci'))
    
    # Get oversampled pixels offests
    osamp = inst.oversample
    xsci_off_over, ysci_off_over = np.array([xsci_off, ysci_off]) * osamp
    yx_offset = (ysci_off_over, xsci_off_over)
    
    # Create PSF
    prev_log = webbpsf_ext.conf.logging_level
    setup_logging('WARN', verbose=False)
    sp_star = make_spec(**star_params)
    hdul_psf_ref = inst.calc_psf_from_coeff(sp=sp_star, coord_vals=(xidl, yidl), coord_frame='idl')
    setup_logging(prev_log, verbose=False)

    im_psf = pad_or_cut_to_size(hdul_psf_ref[0].data, out_shape, offset_vals=yx_offset)

    return im_psf




def resize_list_limits(len1, len2, cent1=None):
    """
    Computes the inner and outer indices to resize a list to a given length.
    The returned indices ensure that the boudaries of the list are not exceeded.
    If the list is croped, the returned indices are centered on the input list center.

    Parameters
    ----------
    len1 : INT
        Length of the input list.
    len2 : INT
        Length of the resized list.

    Returns
    -------
    inner, outer: indices in the input list to resize it.

    """
    if cent1 is None:
        cent1 = len1//2
    inner = np.max([cent1 - len2//2, 0])
    outer = np.min([cent1 + len2//2 + len2%2, len1])
    return inner, outer
    
    
def resize(cube1, dim2, cent=[None,None]):
    """ Resize a cube of images of arbitrary dimensions: crop it or adds zeros on the edges.
    
    Parameters
    ----------
    cube1 : ND numpy array wherre the last two dimensions correspond to the images size.
    dim2 : list of two elements with the new shape of the array.
        
    Returns
    -------
    cube2 : ND numpy array with the last two dimensions resized to the requested shape.
    
    """  
    dims = np.array(cube1.shape)
    dim1 = dims[-2:]
    x1i, x1f = resize_list_limits(dim1[0], dim2[0], cent1=cent[0])
    y1i, y1f = resize_list_limits(dim1[1], dim2[1], cent1=cent[1])
    
    cube2 = np.zeros(np.concatenate((dims[:-2], dim2)))
    x2i, x2f = resize_list_limits(dim2[0], x1f-x1i)
    y2i, y2f = resize_list_limits(dim2[1], y1f-y1i)
    
    cube2[..., x2i:x2f, y2i:y2f] = cube1[..., x1i:x1f, y1i:y1f]
    
    return cube2




    #%% Figurre Jarron
    # fig, axes = plt.subplots(1,3, figsize=(12,4.5))

    # ############################
    # # Plot raw image
    # ax = axes[0]

    # im = im_sci
    # mn = np.median(im)
    # std = np.std(im)
    # vmin = 0
    # vmax = mn+10*std

    # xsize_asec = siaf_ap.XSciSize * siaf_ap.XSciScale
    # ysize_asec = siaf_ap.YSciSize * siaf_ap.YSciScale
    # extent = [-1*xsize_asec/2, xsize_asec/2, -1*ysize_asec/2, ysize_asec/2]
    # norm = LogNorm(vmin=im.max()/1e5, vmax=im.max())
    # ax.imshow(im, extent=extent, norm=norm)

    # ax.set_title("Raw Image (log scale)")

    # ax.set_xlabel('XSci (arcsec)')
    # ax.set_ylabel('YSci (arcsec)')
    # plotAxes(ax, angle=-1*siaf_ap.V3SciYAngle)

    # ############################
    # # Basic PSF subtraction
    # # Subtract a near-perfect reference PSF
    # ax = axes[1]
    # norm = LogNorm(vmin=imdiff.max()/1e5, vmax=imdiff.max())
    # ax.imshow(imdiff, extent=extent, norm=norm, cmap='magma')

    # ax.set_title("PSF Subtracted (log scale)")

    # ax.set_xlabel('XSci (arcsec)')
    # ax.set_ylabel('YSci (arcsec)')
    # plotAxes(ax, angle=-1*siaf_ap.V3SciYAngle)

    # ############################
    # # De-rotate to sky orientation

    # ax = axes[2]
    # ax.imshow(imrot, extent=extent, norm=norm, cmap='magma')

    # ax.set_title("De-Rotated (log scale)")

    # ax.set_xlabel('RA offset (arcsec)')
    # ax.set_ylabel('Dec offset (arcsec)')
    # plotAxes(ax, position=(0.95,0.35), label1='E', label2='N')

    # fig.suptitle(f"Fomalhaut ({siaf_ap.AperName})", fontsize=14)
    # fig.tight_layout()


    # hdul_disk_model_sci[0].header



    # #%% Save image to FITS file
    # hdu_diff = fits.PrimaryHDU(imdisk)

    # copy_keys = [
    #     'PIXELSCL', 'DISTANCE', 
    #     'INSTRUME', 'FILTER', 'PUPIL', 'CORONMSK',
    #     'APERNAME', 'MODULE', 'CHANNEL',
    #     'DET_NAME', 'DET_X', 'DET_Y', 'DET_V2', 'DET_V3'
    # ]

    # hdr = hdu_diff.header
    # for head_temp in (inst.psf_coeff_header, hdul_out[0].header):
    #     for key in copy_keys:
    #         try:
    #             hdr[key] = (head_temp[key], head_temp.comments[key])
    #         except (AttributeError, KeyError):
    #             pass

    # hdr['PIXELSCL'] = inst.pixelscale

    # name = star_A_params['name']

    # outfile = f'HD141569_models/{name}_{inst.aperturename}_{inst.filter}.fits'.replace(' ','')
    # hdu_diff.writeto(outfile, overwrite=True)


    # print('end')



#%% Simulation parameters

star_A_Q = False  # Must be False in the disk modeling framework (we don't want to simulate starlight residuals)
star_B_Q = True   # Optional, depends on the field of view of the final image
star_C_Q = True   # Optional, depends on the field of view of the final image
disk_Q = True     # Must be True in the disk modeling framework
psf_sub = False   # Must be False in the disk modeling framework (we don't want to simulate starlight residuals)
export_Q = False

display_all_Q = True

# Mask information
mask_id = '1140'
filt = f'F{mask_id}C'
mask = f'FQPM{mask_id}'
pupil = 'MASKFQPM'

# Set desired PSF size and oversampling
# MIRI 4QPM: 24" x24" at 0.11 pixels, so 219x219 pixels
# MIRISim synthetic datasets: 224 x 288
fov_pix = 100 #256
osamp = 2


# Observations structure and parameters
if mask_id == '1065':
    pos_ang_list = [107.73513043, 117.36721388]            #deg, list of telescope V3 axis PA for each observation
    base_offset_list =[(0,0), (0,0)]                       #arcsec, list of nominal pointing offsets for each observation ((BaseX, BaseY) columns in .pointing file)
    dith_offsets_list = [[(0,0)], [(0,0)]]                 #arcsec, list of nominal dither offsets for each observation ((DithX, DithY) columns in .pointing file)
    point_error_list = [(-0.20, -1.17), (-0.12, -1.10)]   #pix, list of measured pointing errors for each observation (ErrX, ErrY)

elif mask_id == '1140':
    pos_ang_list = [107.7002475 , 117.34017049]            #deg
    base_offset_list =[(0,0), (0,0)]                       #arcsec
    dith_offsets_list = [[(0,0)], [(0,0)]]                 #arcsec
    point_error_list = [(0.12, 0.05), (0.07, -0.02)]      #pix

elif mask_id == '1550':
    pos_ang_list = [107.65929307, 117.31215657]            #deg
    base_offset_list =[(0,0), (0,0)]                       #arcsec
    dith_offsets_list = [[(0,0)], [(0,0)]]                 #arcsec
    point_error_list = [(0.26, 0.22), (0.14, 0.01)]       #pix

n_obs = len(pos_ang_list)


# Information necessary to create pysynphot spectrum of star, even if star_A_Q = False
star_A_params = {
    'name': 'HD 141569 A', 
    'sptype': 'A2V', 
    'Teff': 10000, 'log_g': 4.28, 'metallicity': -0.5, # Merin et al. 2004
    'dist': 111.6,
    'flux': 64.02, 'flux_units': 'mJy', 'bp_ref': miri_filter('F1065C'),
    'RA_obj'  :  +237.49054012422786,     # RA (decimal deg) of source
    'Dec_obj' :  -3.921290953058801,      # Dec (decimal deg) of source
}


star_B_params = {
    'name': 'HD 141569 B', 
    'sptype': 'M5V', 
    'Teff': 3000, 'log_g': 4.28, 'metallicity': -0.5, # Merin et al. 2004
    'dist': 111.6,
    'flux': 34.22, 'flux_units': 'mJy', 'bp_ref': miri_filter('F1065C'),
    'RA_obj'  :  237.48896543762223,     # RA (decimal deg) of source
    'Dec_obj' :  -3.919897397019599,      # Dec (decimal deg) of source
}


star_C_params = {
    'name': 'HD 141569 C', 
    'sptype': 'M5V', 
    'Teff': 3000, 'log_g': 4.28, 'metallicity': -0.5, # Merin et al. 2004
    'dist': 111.6,
    'flux': 45.49, 'flux_units': 'mJy', 'bp_ref': miri_filter('F1065C'),
    'RA_obj'  :  237.4886562041795,     # RA (decimal deg) of source
    'Dec_obj' :  -3.919700403624408,      # Dec (decimal deg) of source
}

if filt == 'F1550C':
    star_A_params['flux'] = 30.71  
    star_B_params['flux'] = 18.49 
    star_C_params['flux'] = 23.64 
    

disk_params = {
    'file': "/Users/echoquet/Documents/Research/Astro/JWST_Programs/Cycle-1_ERS-1386_Hinkley/Disk_Work/2021-10_Synthetic_Datasets/1_Disk_Modeling/MIRI_Model_Oversampled/HD141569_Model_Pantin_F1065C.fits",
    # 'file': "./radmc_model/images/image_MIRI_FQPM_{}_{}.fits".format(target,10.575),
    'pixscale': 0.027491, 
    'wavelength': 10.65,
    'units': 'Jy/pixel',
    'dist' : 116,
    'cen_star' : True,
}

#%% Create the PSF structure


# Initiate instrument class with selected filters, pupil mask, and image mask
inst = webbpsf_ext.MIRI_ext(filter=filt, pupil_mask=pupil, image_mask=mask)

# Set desired PSF size and oversampling
inst.fov_pix = fov_pix
inst.oversample = osamp
pixscale = inst.pixelscale


# Calculate PSF coefficients, or import them if already computed.
# Can take a while if need to be calculated
inst.gen_psf_coeff()

# Calculate position-dependent PSFs due to FQPM
# Equivalent to generating a giant library to interpolate over
inst.gen_wfemask_coeff()



### Calculate the grid of PSFs for extended objects convolution
t0 = time()
hdul_psfs = generate_grid_psf(inst)
t1 = time()
print('\nPSF Grid Calculation time: {} s'.format(t1-t0))
print('Number of PSFs: {}'.format(len(hdul_psfs)))
print('PSF shape: {}'.format(hdul_psfs[0].data.shape))




#%% Observation setup
'''
Configuring observation settings

Observations consist of nested visit, mosaic tiles, exposures, and dithers. 
In this section, we configure a pointing class that houses information for a single 
observation defined in the APT .pointing file. The primary information includes a 
pointing reference SIAF aperturne name, RA and Dec of the ref aperture, Base X/Y offset
relative to the ref aperture position, and Dith X/Y offsets. From this information, 
along with the V2/V3 position angle, we can determine the orientation and location 
of objects on the detector focal plane.

Note: The reference aperture is not necessarily the same as the observed aperture. 
For instance, you may observe simultaneously with four of NIRCam's SWA detectors, 
so the reference aperture would be the entire SWA channel, while the observed apertures 
are A1, A2, A3, and A4.
'''

# Observed and reference apertures
ap_obs = inst.aperturename
ap_ref = ap_obs   # f'MIRIM_MASK{mask_id}'
ra_ref = star_A_params['RA_obj']
dec_ref = star_A_params['Dec_obj']

siaf_obs = inst.siaf[ap_obs]
ny_pix, nx_pix = (siaf_obs.YSciSize, siaf_obs.XSciSize)
shape_new = (ny_pix * osamp, nx_pix * osamp)

# book keeping arrays:
obs_image_list = np.zeros((n_obs, ny_pix, nx_pix))
obs_image_sub_list = np.zeros((n_obs, ny_pix, nx_pix))
obs_image_derot_list = np.zeros((n_obs, ny_pix, nx_pix))

for obs in range(n_obs):
    print('###### Generating Observation {}/{} ######'.format(obs+1, n_obs))
    
    
    print('--- Setting the pointing parameters:')
    # For each observation, define the telescope pointing. 
    # This accounts for telescope V3 axis angle, nominal offset, nominal dither offsets, and measured pointing error.
    
    # NOTE: the pointing errors are implemented as an (X,Y) dither offset. 
    # This is done rather crudely in the pixel to arcsec conversion, compared to 
    # using the SIAF system in the rest of the code.
    # TODO: improve on the implementation of the poiting error.

    pos_ang = pos_ang_list[obs]
    base_offset = base_offset_list[obs]
    point_error = point_error_list[obs]
    dith_offsets_mod = [(dith[0] + point_error[0]*pixscale, dith[1] + point_error[1]*pixscale)  
                        for dith in dith_offsets_list[obs]]
 
    
 
    # Telescope pointing information
    tel_point = jwst_point(ap_obs, ap_ref,ra_ref, dec_ref, 
                           pos_ang=pos_ang, base_offset=base_offset, dith_offsets=dith_offsets_mod,
                           base_std=0, dith_std=0)
    
    # Get sci position of center in units of detector pixels
    # Elodie: gives the position of the mask center, in pixels 
    # siaf_ap = tel_point.siaf_ap_obs
    # # x_cen, y_cen = siaf_ap.reference_point('sci')
    
    # # Elodie: gives the full frame image size in pixel, inc. oversampling (432x432 with osamp=2)
    # ny_pix, nx_pix = (siaf_ap.YSciSize, siaf_ap.XSciSize)
    # shape_new = (ny_pix * osamp, nx_pix * osamp)
    
    print('     Reference aperture: {}'.format(tel_point.siaf_ap_ref.AperName))
    print('     Telescope orientation: {:.3f} deg'.format(pos_ang))
    print('     Nominal RA, Dec = ({:.6f}, {:.6f})'.format(tel_point.ra_ref, tel_point.dec_ref))     
    print('     Scene nominal offet: ({:.3f}, {:.3f}) arcsec'.format(base_offset[0], base_offset[1]))   
    print('     Relative offsets for each dither position (incl. pointing errors)')
    for i, offset in enumerate(tel_point.position_offsets_act):
        print('  Position {}: ({:.4f}, {:.4f}) arcsec'.format(i, offset[0],offset[1]))
        
        
    
    print('--- Creating the instrument image with stars and disk:')
    # Generate the stars PSFs
    obs_image_over = np.zeros(shape_new)
    if star_A_Q:
        print('     Creating Star A image')
        psf_star_A = generate_star_psf(star_A_params, tel_point, inst, shape_new)
        obs_image_over += psf_star_A
    if star_B_Q:
        print('     Creating Star B image')
        psf_star_B = generate_star_psf(star_B_params, tel_point, inst, shape_new)
        obs_image_over += psf_star_B
    if star_C_Q:
        print('     Creating Star C image')
        psf_star_C = generate_star_psf(star_C_params, tel_point, inst, shape_new)
        obs_image_over += psf_star_C

    
    # Display the simulated stars
    if display_all_Q and (star_A_Q or star_B_Q or star_C_Q):
        fig, ax = plt.subplots(1,1)
        fig.suptitle('Oversampled image stars '+filt)
        ax.imshow(obs_image_over, vmin=-0.5,vmax=1) 
        fig.tight_layout()
        plt.show()
    
    
    # Generate the disk image
    if disk_Q:
        print('     Creating Disk image')
        disk_image = add_disk_into_model(disk_params, hdul_psfs, tel_point, inst, shape_new, star_params=star_A_params)
        obs_image_over += disk_image
    
    # Display the total image
    if display_all_Q :
        fig, ax = plt.subplots(1,1)
        fig.suptitle('Oversampled full image '+filt)
        ax.imshow(obs_image_over, vmin=0,vmax=20) 
        fig.tight_layout()
        plt.show()


    # Rebin science data to detector pixels
    obs_image = image_manip.frebin(obs_image_over, scale=1/osamp)
    print('Detector sampled final image shape: {}'.format(obs_image.shape)) #216x216

    # Subtract a reference PSF from the science data
    if psf_sub:
        print('--- Subtracting the central star:')
        coord_vals = tel_point.position_offsets_act[0]
        im_psf = quick_ref_psf(coord_vals, inst, obs_image.shape, star_params=star_A_params)
        im_ref = image_manip.frebin(im_psf, scale=1/osamp)
        obs_image_sub = obs_image - im_ref
    else:
        obs_image_sub = obs_image

    rotate_to_idl = -1 * tel_point.pos_ang
    obs_image_derot = image_manip.rotate_offset(obs_image_sub, rotate_to_idl, reshape=False, cval=np.nan)
    
    obs_image_list[obs] = obs_image
    obs_image_sub_list[obs] = obs_image_sub
    obs_image_derot_list[obs] = obs_image_derot

#%% Derotating and Combining the two rolls
print('Generation of each observation complete!\n')
if display_all_Q:
    plmax = obs_image_sub_list.max()/10
    fig, ax = plt.subplots(3, n_obs)
    fig.suptitle('Roll images '+filt)
    for obs in range(n_obs):
        ax[0, obs].imshow(obs_image_list[obs], vmin=0,vmax=plmax) 
        ax[1, obs].imshow(obs_image_sub_list[obs], vmin=0,vmax=plmax) 
        ax[2, obs].imshow(obs_image_derot_list[obs], vmin=0,vmax=plmax) 
    fig.tight_layout()
    plt.show()


print('###### Combining the dataset')
# Finally getting the final combined image to compare with the real MIRI image
cropsize = 101
model_image_full = np.nanmean(obs_image_derot_list, axis=0)
model_image = resize(model_image_full, [cropsize,cropsize]) #, cent=np.round(star_center_sci).astype(int)) 

# sma = [0.397, 1.775, 3.29] 
vmax = 700
fig7, ax7 = plt.subplots(1,1,figsize=(8,6), dpi=130)
xsize_asec = cropsize * siaf_obs.XSciScale
ysize_asec = cropsize * siaf_obs.YSciScale
extent = [-1*xsize_asec/2, xsize_asec/2, -1*ysize_asec/2, ysize_asec/2]
im = ax7.imshow(model_image, norm=LogNorm(vmin=vmax/500, vmax=vmax))#, extent=extent)
# im = ax7.imshow(combined_image, vmin=vmin_lin, vmax=vmax)
# ax7.set_xlabel('RA offset (arcsec)')
# ax7.set_ylabel('Dec offset (arcsec)')
# plotAxes(ax7, position=(0.95,0.35), label1='E', label2='N')
ax7.set_title('COMBINED HD141569 MODEL '+filt)
plt.tight_layout()
cbar = fig7.colorbar(im, ax=ax7)
cbar.ax.set_title('Units TBD$')
plt.show()

