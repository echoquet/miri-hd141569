#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sat Oct 23 15:16:26 2021: Creation

@author: echoquet
"""
import os
from glob import glob
import numpy as np
# from skimage.transform import rotate
import scipy.ndimage as ndimage
from scipy.optimize import minimize
from copy import deepcopy
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams['image.origin'] = 'lower'
plt.rcParams["image.cmap"] = 'gist_heat'#'hot'#'copper'

#%% FUNCTIONS
def create_mask(rIn, dims, cent=None, polar_system=False, rOut=float('inf')):
    """ Creates a boolean frame with the pixels outside rIn at True.
    
    Parameters
    ----------
    rIn : the inner radius of the mask in pixels.
    dims : the shape of the output array (pix, pix).
    cent : (optional) the coordinates of the mask. Default is the frame center.
    system : (optional) 'cartesian' or 'polar' system for the center coordinates. Default is cartesian.
    rOut : (optional) the outer radius of the mask in pixels
    
    Returns
    -------
    mask : 2D Array of booleans, True outside rIn, False inside.
        
    """
    if len(dims) != 2:
        raise TypeError('dims should be tuple of 2 elements')
    im_center = (np.array(dims)-1) / 2 #minus 1 because of difference between length and indexation
    
    if cent is None:
        center = im_center
    else:
        if polar_system:
            phi = cent[1] *np.pi/180
            center = [cent[0] * np.cos(phi), -cent[0] * np.sin(phi)] # + im_center
        else:
            center = cent 
            
    y, x = np.indices(dims)
    rads = np.sqrt((y-center[0])**2+(x-center[1])**2)
    mask = (rOut > rads) & (rads >= rIn)
    return mask


def create_elliptical_annulus(rIn, rOut, inc, pa, shape, cent=None):
    """ Creates a boolean frame with the pixels within rIn and rOut at True.
    
    Parameters
    ----------
    rIn : the inner radius of the annulus in pixels.
    rOut : the outer radius of the annulus in pixels.
    inc: inclination in degrees to compute the minor axis.
    pa: orientation in degrees of the ellipse.
    shape : the shape of the output array (pix, pix).
    cent : coordinates of the center  (y0, x0)
        
    Returns
    -------
    mask : 2D Array of booleans, True within rIn and rOut, False outside.
        
    """
    if len(shape) != 2:
        raise TypeError('Shape should be tuple of 2 elements')
        
    if cent is None:
        cent = (np.array(shape)-1) / 2
        
    y, x = np.indices(shape)
    dx = (x-cent[1])
    dy = (y-cent[0])
    
    pa_rad = pa*np.pi/180
    dx_rot = (dx*np.cos(pa_rad) + dy*np.sin(pa_rad))/np.cos(inc*np.pi/180)
    dy_rot = -dx*np.sin(pa_rad) + dy*np.cos(pa_rad)
    
    
    rads = np.sqrt(dx_rot**2 + dy_rot**2)
    mask = (rOut > rads) & (rads >= rIn)
    return mask


def create_fqpm_mask(dims, center, width, angle):
    fqpm_mask_tmp = np.zeros(dims)
    fqpm_mask_tmp[center[0] - width//2 : center[0] + width//2, :] =1
    fqpm_mask_tmp[:, center[1] - width//2 : center[1] + width//2] =1
    fqpm_mask = frame_rotate_interp(fqpm_mask_tmp, angle, center=center)
    return fqpm_mask


### DEPRECATED
# def frame_rotate(array, angle, rot_center=None, interp_order=4, border_mode='constant'):
#     """ Rotates a frame or 2D array.
    
#     Parameters
#     ----------
#     array : Input image, 2d array.
#     angle : Rotation angle.
#     rot_center : Coordinates X,Y  of the point with respect to which the rotation will be 
#                 performed. By default the rotation is done with respect to the center 
#                 of the frame; central pixel if frame has odd size.
#     interp_order: Interpolation order for the rotation. See skimage rotate function.
#     border_mode : Pixel extrapolation method for handling the borders. 
#                 See skimage rotate function.
        
#     Returns
#     -------
#     array_out : Resulting frame.
        
#     """
#     if array.ndim != 2:
#         raise TypeError('Input array is not a frame or 2d array')

#     min_val = np.nanmin(array)
#     im_temp = array - min_val
#     max_val = np.nanmax(im_temp)
#     im_temp /= max_val

#     array_out = rotate(im_temp, angle, order=interp_order, center=rot_center, 
#                         cval=np.nan, mode=border_mode)

#     array_out *= max_val
#     array_out += min_val
#     array_out = np.nan_to_num(array_out)
             
#     return array_out



def frame_rotate_interp(array, angle, center=None, mode='constant', cval=0, order=3):
    ''' Rotates a frame or 2D array.
        
        Parameters
        ----------
        array : Input image, 2d array.
        angle : Rotation angle (deg).
        center : Coordinates X,Y  of the point with respect to which the rotation will be 
                    performed. By default the rotation is done with respect to the center 
                    of the frame; central pixel if frame has odd size.
        interp_order: Interpolation order for the rotation. See skimage rotate function.
        border_mode : Pixel extrapolation method for handling the borders. 
                    See skimage rotate function.
            
        Returns
        -------
        rotated_array : Resulting frame.

    '''
    dtype = array.dtype
    dims  = array.shape
    angle_rad = -np.deg2rad(angle)

    if center is None:
        center = (np.array(dims)-1) / 2 # The minus 1 is because of python indexation at 0
    
    x, y = np.meshgrid(np.arange(dims[1], dtype=dtype), np.arange(dims[0], dtype=dtype))

    xp = (x-center[1])*np.cos(angle_rad) + (y-center[0])*np.sin(angle_rad) + center[1]
    yp = -(x-center[1])*np.sin(angle_rad) + (y-center[0])*np.cos(angle_rad) + center[0]

    rotated_array = ndimage.map_coordinates(array, [yp, xp], mode=mode, cval=cval, order=order)
    
    return rotated_array


def shift_interp(array, shift_value, mode='constant', cval=0, order=3):
    ''' Shifts a frame or 2D array. From Vigan imutil toolbox
        
        Parameters
        ----------
        array : Input image, 2d array.
        shift_value : array/list/tuple of two elements with the offsets.
    
            
        Returns
        -------
        shifted : Resulting frame.

    '''
    Ndim  = array.ndim
    dims  = array.shape
    dtype = array.dtype.kind

    if (Ndim == 1):
        pass
    elif (Ndim == 2):
        x, y = np.meshgrid(np.arange(dims[1], dtype=dtype), np.arange(dims[0], dtype=dtype))

        x -= shift_value[0]
        y -= shift_value[1]

        shifted = ndimage.map_coordinates(array, [y, x], mode=mode, cval=cval, order=order)

    return shifted


def reg_criterion(parameters, image, template, mask):    
    dx, dy, nu = parameters
    return np.sum((nu * shift_interp(image, [dx, dy]) - template)[mask]**2)



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



def import_data_cube_from_files(file_list, scaling_list=None):
    tmp = fits.getdata(file_list[0])
    data_cube = np.zeros(np.shape(file_list) + np.shape(tmp))
    if scaling_list is None:
        scaling_list = np.ones(len(file_list))
    for i, file in enumerate(file_list):
        data_cube[i] = fits.getdata(file) * scaling_list[i]
    return data_cube


def import_keyword_from_files(file_list, keyword, extension=0):
    keyword_values = np.zeros(len(file_list))
    for i, file in enumerate(file_list):
        header = fits.getheader(file, extension)
        keyword_values[i] = header[keyword]
    return keyword_values


def median_filter(image, box_half_size, threshold):
    dims = np.shape(image)
    nShhifts = (box_half_size * 2 + 1)**2
    image_shifts = np.empty((nShhifts, dims[0], dims[1]))
    k = 0
    for i in range(-box_half_size, box_half_size+1):
        for j in range(-box_half_size, box_half_size+1):
            image_shifts[k,:,:] = np.roll(image, (i,j), axis=(0,1))
            k+=1
    medians = np.median(image_shifts, axis=0)
    stddevs = np.std(image_shifts, axis=0)
    bad_pix_flags = np.abs(image - medians) > threshold * stddevs
    print('Number of bad pixels: {}'.format(np.sum(bad_pix_flags)))
    
    image_filtered = deepcopy(image)
    image_filtered[bad_pix_flags] = medians[bad_pix_flags]
    return image_filtered


def median_filter_cube(cube, box_half_size, threshold, iter_max=10000, verbose=True):
    dims = np.shape(cube)
    ndims = len(dims)
    nShifts = (box_half_size * 2 + 1)**2
    cube_filtered = deepcopy(cube)
    
    nb_bad_pix = 1
    it = 0
    while (nb_bad_pix > 0) and (it < iter_max):
        cube_shifts = np.empty((nShifts,)+ dims)
        k = 0
        for i in range(-box_half_size, box_half_size+1):
            for j in range(-box_half_size, box_half_size+1):
                cube_shifts[k] = np.roll(cube_filtered, (i,j), axis=(ndims-2, ndims-1))
                k+=1
        medians = np.median(cube_shifts, axis=0)
        stddevs = np.std(cube_shifts, axis=0)
        bad_pix_flags = np.abs(cube_filtered - medians) > threshold * stddevs
        nb_bad_pix = np.sum(bad_pix_flags)
        if verbose:
            print('Number of bad pixels: {}'.format(nb_bad_pix))
        
        cube_filtered[bad_pix_flags] = medians[bad_pix_flags]
        it += 1
        
    if verbose:
        print('Number of iterations: {}'.format(it))
    return cube_filtered

#%%
print('##### IDENTIFY DATA FILES ### ')

# Parameters to locate / select the datasets:
root = '/Users/echoquet/Documents/Research/Astro/JWST_Programs/Cycle-1_ERS-1386_Hinkley/Disk_Work/2021-10_Synthetic_Datasets/2_Raw_Synthetic_Data'

version = 'v0'
filt = 'F1550C'

inspec_raw = False
inspec_cal1 = False


# Locate the datasets
print('Root folder: \n{}'.format(root))
print('Filter name: {}'.format(filt))

# Paths:
path_raw = os.path.join(root, 'MIRI_RAW_Data_Kim', filt)
path_cal1 = os.path.join(root, 'MIRI_CAL1', filt)
path_cal2 = os.path.join(root, 'MIRI_CAL2', filt)


basename_sci = 'HD141569_'+version+'*MIRIMAGE_'+filt+'exp1'
basename_ref = 'SGD*MIRIMAGE_'+filt+'exp1'
print('Basename SCI: {}'.format(basename_sci))
print('Basename REF: {}\n'.format(basename_ref))

raw_sci_files = glob(os.path.join(path_raw, basename_sci+'.fits'))
cal1_sci_files = glob(os.path.join(path_cal1, basename_sci+'_rateints.fits'))
cal2_sci_files = glob(os.path.join(path_cal2, basename_sci+'_calints.fits'))
print('Number of RAW SCI files: {}'.format(len(raw_sci_files)))
print('Number of CAL1 SCI files (rateints): {}'.format(len(cal1_sci_files)))
print('Number of CAL2 SCI files (calints): {}'.format(len(cal2_sci_files)))

raw_ref_files = glob(os.path.join(path_raw, basename_ref+'.fits'))
cal1_ref_files = glob(os.path.join(path_cal1, basename_ref+'_rateints.fits'))
cal2_ref_files = glob(os.path.join(path_cal2, basename_ref+'_calints.fits'))
print('Number of RAW REF files: {}'.format(len(raw_ref_files)))
print('Number of CAL1 REF files (rateints): {}'.format(len(cal1_ref_files)))
print('Number of CAL2 REF files (calints): {}'.format(len(cal2_ref_files)))

#%% RAW DATA INSPECTION
if inspec_raw:
    print('##### RAW DATA INSPECTION #####')
    
    # SCIENCE DATASETS RAMPS
    # index -1 gives the end of the ramp, with all the accumulated counts
    raw_sci_cube = import_data_cube_from_files(raw_sci_files) 
    PA_V3_sci = import_keyword_from_files(raw_sci_files, 'PA_V3', extension=1)
    NGROUPS_sci = import_keyword_from_files(raw_sci_files, 'NGROUPS', extension=0)
    NINTS_sci = import_keyword_from_files(raw_sci_files, 'NINTS', extension=0)
    TINT_sci = import_keyword_from_files(raw_sci_files, 'EFFINTTM', extension=0)
    TEXP_sci = import_keyword_from_files(raw_sci_files, 'EFFEXPTM', extension=0)
    for i, file in enumerate(raw_sci_files):
        print('SCI Exposure {}:\n  NGROUPS = {:n}\n  NINTS = {:n}\n  PA_V3 = {} deg'.format(i, NGROUPS_sci[i], NINTS_sci[i], PA_V3_sci[i]))
    dim = (np.shape(raw_sci_cube))[-2:]
    print('Image dimensions: {}'.format(dim))
    
    
    
    mask_cent = ~create_mask(15, dim, cent=[110,120])
    max_val = np.max(raw_sci_cube[0,0,-1,:,:]*mask_cent)
    median_val = np.median(raw_sci_cube[0,0,-1,:,:])
    print('Median value: ', median_val)
    print('Max value: ', max_val)
    # fig0, ax0 = plt.subplots(1,1,figsize=(6,6), dpi=130)
    # plt.imshow(raw_sci_cube[0,0,-1,:,:]*mask_cent, vmin=median_val, vmax=max_val)
    # plt.show()
    
    
    int_index = 0
    vmin = 9000 #median_val*0.8
    vmax = 17000 #max_val*0.7
    fig1, ax1 = plt.subplots(len(raw_sci_files),1,figsize=(8,6), dpi=130)
    fig1.suptitle('RAW HD141569  '+filt)
    images = []
    for i in range(len(raw_sci_files)):
        images.append(ax1[i].imshow(raw_sci_cube[i,int_index,-1,:,:], vmin=vmin, vmax=vmax))
        ax1[i].set_title('ORIENT {}: {}deg'.format(i, PA_V3_sci[i]))
    plt.tight_layout()
    cbar = fig1.colorbar(images[0], ax=ax1)
    cbar.ax.set_title('counts')
    plt.show()
    
    
    # REFERENCE STAR DATASET RAMPS:
    raw_ref_cube = import_data_cube_from_files(raw_ref_files) 
    PA_V3_ref = import_keyword_from_files(raw_ref_files, 'PA_V3', extension=1)
    NGROUPS_ref = import_keyword_from_files(raw_ref_files, 'NGROUPS', extension=0)
    NINTS_ref = import_keyword_from_files(raw_ref_files, 'NINTS', extension=0)
    TINT_ref = import_keyword_from_files(raw_ref_files, 'EFFINTTM', extension=0)
    TEXP_ref = import_keyword_from_files(raw_ref_files, 'EFFEXPTM', extension=0)
    for i, file in enumerate(raw_ref_files):
        print('REF Exposure {}:\n  NGROUPS = {:n}\n  NINTS = {:n}\n  PA_V3 = {} deg'.format(i, NGROUPS_ref[i], NINTS_ref[i], PA_V3_ref[i]))
    
    
    int_index = 0
    ncol = 2
    nrow = int(np.ceil(len(raw_ref_files)/2))
    fig3, ax3 = plt.subplots(nrow, ncol,figsize=(8,6), dpi=130)
    ax3[-1, -1].axis('off')
    fig3.suptitle('RAW REFSTAR  '+filt)
    images = []
    for i in range(len(raw_ref_files)):
        irow = i%nrow
        icol = i//nrow
        images.append(ax3[irow, icol].imshow(raw_ref_cube[i,int_index,-1,:,:], vmin=vmin, vmax=vmax))
        ax3[irow, icol].set_title('ORIENT {}: {}deg'.format(i, PA_V3_ref[i]))
    plt.tight_layout(h_pad=0)
    cbar = fig3.colorbar(images[-1], ax=ax3)
    cbar.ax.set_title('counts')
    plt.show()
    
    
    vmin = 11500 #median_val*0.8
    vmax = 13000 #max_val*0.7
    fig1, ax1 = plt.subplots(1,1,figsize=(7,5), dpi=130)
    image=ax1.imshow(raw_ref_cube[0,int_index,-1,:,:], norm=LogNorm(vmin=vmin, vmax=vmax))
    # ax1.set_title('HD141569 {} - LOG'.format(filt, PA_V3_sci[0]))
    # cbar = fig1.colorbar(image,ax=ax1)
    # cbar.ax.set_title('mJy.arcsec$^{-2}$')
    plt.show()

#%% CAL 1 DATA INSPECTION
if inspec_cal1:
    print('##### CAL 1 DATA INSPECTION #####')
    
    # SCIENCE DATASETS COUNTRATES
    cal1_sci_cube = import_data_cube_from_files(cal1_sci_files) 
    PA_V3_sci = import_keyword_from_files(cal1_sci_files, 'PA_V3', extension=1)
    dim = (np.shape(cal1_sci_cube))[-2:]
    print('Image dimensions: {}'.format(dim))
    
    
    mask_cent = ~create_mask(15, dim, cent=[110,120])
    max_val = np.max(cal1_sci_cube[0,0,:,:]*mask_cent)
    median_val = np.median(cal1_sci_cube[0,0,:,:])
    print('Median value: ', median_val)
    print('Max value: ', max_val)
    # fig0, ax0 = plt.subplots(1,1,figsize=(6,6), dpi=130)
    # plt.imshow(cal1_sci_cube[0,0,:,:]*mask_cent, vmin=median_val, vmax=max_val)
    # plt.show()
    
    
    int_index = 0
    vmin = 0 #median_val*0.8
    vmax = 10 #max_val*0.7
    fig1, ax1 = plt.subplots(len(cal1_sci_cube),1,figsize=(8,6), dpi=130)
    fig1.suptitle('CAL 1 HD141569  '+filt)
    images = []
    for i in range(len(raw_sci_files)):
        images.append(ax1[i].imshow(cal1_sci_cube[i,int_index,:,:], vmin=vmin, vmax=vmax))
        ax1[i].set_title('ORIENT {}: {}deg'.format(i, PA_V3_sci[i]))
    plt.tight_layout()
    cbar = fig1.colorbar(images[0], ax=ax1)
    cbar.ax.set_title('count/s')
    plt.show()
    
    
    # REFERENCE STAR DATASET RAMPS:
    cal1_ref_cube = import_data_cube_from_files(cal1_ref_files) 
    PA_V3_ref = import_keyword_from_files(cal1_ref_files, 'PA_V3', extension=1)
    
    
    int_index = 0
    ncol = 2
    nrow = int(np.ceil(len(cal1_ref_files)/2))
    fig3, ax3 = plt.subplots(nrow, ncol,figsize=(8,6), dpi=130)
    ax3[-1, -1].axis('off')
    fig3.suptitle('CAL 1 REFSTAR  '+filt)
    images = []
    for i in range(len(cal1_ref_files)):
        irow = i%nrow
        icol = i//nrow
        images.append(ax3[irow, icol].imshow(cal1_ref_cube[i,int_index,:,:], vmin=vmin, vmax=vmax))
        ax3[irow, icol].set_title('ORIENT {}: {}deg'.format(i, PA_V3_ref[i]))
    plt.tight_layout(h_pad=0)
    cbar = fig3.colorbar(images[-1], ax=ax3)
    cbar.ax.set_title('count/s')
    plt.show()


#%% CAL 2 DATA INSPECTION
print('##### CAL 2 DATA INSPECTION #####')
if filt == 'F1065C':
    vmax = 2 #mJy/arcsec^2
elif filt == 'F1140C':
    vmax = 10
else:
    vmax=3
vmin_lin = 0
vmin_log = vmax/100

# SCIENCE DATASETS COUNTRATES
PA_V3_sci = import_keyword_from_files(cal2_sci_files, 'PA_V3', extension=1)
phot_MJySr_sci = import_keyword_from_files(cal2_sci_files, 'PHOTMJSR', extension=1)
phot_uJyA2_sci = import_keyword_from_files(cal2_sci_files, 'PHOTUJA2', extension=1)
scaling_values = phot_uJyA2_sci/(1000*phot_MJySr_sci)
cal2_sci_cube = import_data_cube_from_files(cal2_sci_files, scaling_list=scaling_values)
dims = (np.shape(cal2_sci_cube))[-2:]
n_sci_files = len(cal2_sci_files)
n_sci_int = (np.shape(cal2_sci_cube))[1]
print('Image dimensions: {}'.format(dims))

mask_cent = ~create_mask(15, dims, cent=[110,120])
max_val = np.max(cal2_sci_cube[0,0,:,:]*mask_cent)
median_val = np.median(cal2_sci_cube[0,0,:,:])
print('Median value: ', median_val)
print('Max value: ', max_val)
# fig0, ax0 = plt.subplots(1,1,figsize=(6,6), dpi=130)
# plt.imshow(cal2_sci_cube[0,0,:,:]*mask_cent, vmin=median_val, vmax=max_val)
# plt.show()


fig1, ax1 = plt.subplots(n_sci_files,n_sci_int,figsize=(2*n_sci_int,2*n_sci_files), dpi=130)
fig1.suptitle('CAL 2 HD141569  '+filt)
images = []
for i in range(n_sci_files):
    for j in range(n_sci_int):
        images.append(ax1[i,j].imshow(cal2_sci_cube[i,j,:,:], vmin=vmin_lin, vmax=vmax))
        ax1[i,j].set_title('ORIENT {}: {}deg'.format(i, PA_V3_sci[i]))
plt.tight_layout()
cbar = fig1.colorbar(images[0], ax=ax1)
cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()




# REFERENCE STAR DATASET RAMPS:
PA_V3_ref = import_keyword_from_files(cal2_ref_files, 'PA_V3', extension=1)
phot_MJySr_ref = import_keyword_from_files(cal2_ref_files, 'PHOTMJSR', extension=1)
phot_uJyA2_ref = import_keyword_from_files(cal2_ref_files, 'PHOTUJA2', extension=1)
scaling_values_ref = phot_uJyA2_ref/(1000*phot_MJySr_ref)
cal2_ref_cube = import_data_cube_from_files(cal2_ref_files, scaling_list=scaling_values_ref)
n_ref_files = len(cal2_ref_cube)
n_ref_int = (np.shape(cal2_ref_cube))[1]

int_index = 0
ncol = 2
nrow = int(np.ceil(len(cal2_ref_files)/2))
fig3, ax3 = plt.subplots(nrow, ncol,figsize=(8,6), dpi=130)
ax3[-1, -1].axis('off')
fig3.suptitle('CAL 2 REFSTAR  '+filt)
images = []
for i in range(len(cal2_ref_files)):
    irow = i%nrow
    icol = i//nrow
    images.append(ax3[irow, icol].imshow(cal2_ref_cube[i,int_index,:,:], vmin=vmin_lin, vmax=vmax))
    ax3[irow, icol].set_title('ORIENT {}: {}deg'.format(i, PA_V3_ref[i]))
plt.tight_layout(h_pad=0)
cbar = fig3.colorbar(images[-1], ax=ax3)
cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()


#%% DATA COMPBINATION
debug = False
display_all = False

# Parameters for Median-filter
median_filt_thresh = 3 #sigma
median_filt_box_size = 2
median_filt_iter_max = 20

avg_ref_ints = False

# Parameters for the background mask
bck_cropsize = 131
# crop_center = [113, 120] if filt=='F1065C' else [110, 119]      # TMP for synthetic dataset! use header instead
if filt == 'F1065C':
    crop_center = [113, 120]
elif filt == 'F1140C':
    crop_center = [110, 118]
else:
    crop_center=[109, 119]
bck_mask_Rin_sci = 50
bck_mask_Rin_ref = 65
# bck_mask_Rout = 105

# PSF subtraction
subtract_method = 'classical-Ref-Averaged' # 'classical-Ref-Averaged'  'No-Subtraction'
if filt == 'F1065C':
    sci_ref_th_ratio = 64.02/469.45
elif filt == 'F1140C':
    sci_ref_th_ratio = 56.07/410.48
else:
    sci_ref_th_ratio = 30.71/223.75

# Parameters for 4QPM mask and data combination
# TODO: use CDBS calibration mask instead!
fqpm_center = crop_center     # TMP for synthetic dataset! use header instead
fqpm_width = 3 #pix
fqpm_angle = 10 #deg

PA_V3_sci = [35, 49] if filt=='F1065C' else [130, 144] # TMP for synthetic datasets!

cropsize = 101

saveCombinedImageQ = True
overWriteQ = True
path_output = os.path.join(root, 'MIRI_COMBINED_Elodie', filt)
filename_output = basename_sci + '_combined.fits'

''' Reduction notes:
    To achieve the subtraction of the PSF, both SCI and REF must be background-subtracted.
    Background subtrarction is best achieved if estimated in the area close to the system,
    so it is better to crop the data first. 
    
'''


### Cleaning the bad pixels from the SCI and REF datasets
print('--- Cleaning the data from bad pixels ---')
# NOTE: consider using the DQ maps too
cal2_sci_cube_filt = median_filter_cube(cal2_sci_cube, median_filt_box_size, median_filt_thresh, 
                                        iter_max=median_filt_iter_max, verbose=False)
cal2_ref_cube_filt = median_filter_cube(cal2_ref_cube, median_filt_box_size, median_filt_thresh, 
                                        iter_max=median_filt_iter_max, verbose=False)

if avg_ref_ints:
    cal2_ref_cube_filt_list = np.mean(cal2_ref_cube_filt, axis=1)
else:
    cal2_ref_cube_filt_list = np.reshape(cal2_ref_cube_filt, (n_ref_files*n_ref_int,)+dims)
n_ref_frames = len(cal2_ref_cube_filt_list)



### Register the frames
# TODO: Register the SCI frames with companion
# TODO: Register the REF frames?
if debug:
    template = cal2_ref_cube_filt_list[0]
    reg_mask = ~create_mask(40, dims, cent=crop_center)  #np.ones(dims, dtype=bool)
    reg_sol_list = np.empty((n_ref_frames, 4))
    for i in range(n_ref_frames):
        reg_init = np.array([0, 0, 1]) + 0.1 * np.random.randn(3)
        reg_sol = minimize(reg_criterion, reg_init, args=(cal2_ref_cube_filt_list[i], template, reg_mask), tol=1e-5)
        dx, dy, nu = reg_sol.x
        reg_crit = reg_criterion(reg_sol.x, cal2_ref_cube_filt_list[i], template, reg_mask)
        reg_sol_list[i] = np.append(reg_sol.x, [reg_crit]) 
        print('Success!') if reg_sol.success else print(reg_sol.message)
        message = '     dx = {:.3f}"    dy = {:.3f}"    nu = {:.2f}    crit = {:.2f}'
        print(message.format(dx*0.11, dy*0.11, nu, reg_crit))
    
    
    fig13, ax13 = plt.subplots(1,n_ref_frames,figsize=(2*n_ref_frames,2), dpi=130)
    images = []
    for i in range(n_ref_frames):
        dx, dy, nu , crit_val = reg_sol_list[i]
        diff = (nu * shift_interp(cal2_ref_cube_filt_list[i], [dx, dy]) - template) * reg_mask
        images.append(ax13[i].imshow(diff, vmin=-vmax/30, vmax=vmax/30))
        ax13[i].set_title('crit: {:.2f}'.format(crit_val))
    plt.tight_layout()
    cbar = fig13.colorbar(images[0], ax=ax13)
    plt.show()


### Crop and subtract the background
# Note: here we assume the background is spatially invariant.
# If this is not the case, use the mean across the N integrations and M rolls.
# If it is *verry* spatially variant, use an elliptical mask and mean across the N integration per roll.
print('--- Cropping & Subtracting the background level ---')
bck_cropsizes = np.array((bck_cropsize, bck_cropsize))
cal2_sci_cube_crop = resize(cal2_sci_cube_filt, bck_cropsizes, cent=crop_center)
cal2_ref_cube_crop = resize(cal2_ref_cube_filt_list, bck_cropsizes, cent=crop_center)

# Optimize the masks
bck_mask_sci = create_mask(bck_mask_Rin_sci, bck_cropsizes)
bck_mask_ref = create_mask(bck_mask_Rin_ref, bck_cropsizes)

if display_all:
    fig10, ax10 = plt.subplots(1,1,figsize=(8,6), dpi=130)
    ax10.imshow(bck_mask_sci*np.mean(cal2_sci_cube_crop,axis=(0,1)), norm=LogNorm(vmin=0.02, vmax=0.5))
    
    fig11, ax11 = plt.subplots(1,1,figsize=(8,6), dpi=130)
    ax11.imshow(bck_mask_ref*np.mean(cal2_ref_cube_crop,axis=0), norm=LogNorm(vmin=0.02, vmax=0.5))

# Estimate and subtract the background levels
bck_level_sci = np.median(cal2_sci_cube_crop[:, :, bck_mask_sci])
bck_level_ref = np.median(cal2_ref_cube_crop[:, bck_mask_ref])
print('Background_level SCI = {:.2e} mJy.arcsec^-2'.format(bck_level_sci))
print('Background_level REF = {:.2e} mJy.arcsec^-2'.format(bck_level_ref))

cal2_sci_cube_bck_sub = cal2_sci_cube_crop - np.tile(bck_level_sci, (n_sci_files, n_sci_int, 1, 1))
cal2_ref_cube_bck_sub = cal2_ref_cube_crop - np.tile(bck_level_ref, (n_ref_frames, 1, 1))

fig4, ax4 = plt.subplots(n_sci_files,n_sci_int,figsize=(2*n_sci_int,2*n_sci_files), dpi=130)
fig4.suptitle('Background subtracted Integrations HD141569  '+filt)
images = []
for i in range(n_sci_files):
    for j in range(n_sci_int):
        # images.append(ax4[i,j].imshow(cal2_sci_cube_bck_sub[i,j,:,:], vmin=vmin_lin, vmax=vmax))
        images.append(ax4[i,j].imshow(cal2_sci_cube_bck_sub[i,j,:,:], norm=LogNorm(vmin=vmax/100, vmax=vmax/3)))
        ax4[i,j].set_title('ORIENT {}: {}deg'.format(i, PA_V3_sci[i]))
plt.tight_layout()
cbar = fig4.colorbar(images[0], ax=ax4)
cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()

fig3, ax3 = plt.subplots(1,n_ref_frames,figsize=(2*n_ref_frames,2), dpi=130)
fig3.suptitle('Background subtracted Integrations Ref Star  '+filt)
images = []
for i in range(n_ref_frames):
   images.append(ax3[i].imshow(cal2_ref_cube_bck_sub[i,:,:], norm=LogNorm(vmin=vmax/100, vmax=vmax/3)))
plt.tight_layout()
cbar = fig3.colorbar(images[0], ax=ax3)
cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()




### Classical subtraction of the SCI PSF
if subtract_method == 'No-Subtraction':
    cal2_sci_cube_psf_sub = cal2_sci_cube_bck_sub
elif subtract_method =='classical-Ref-Averaged':
    ref_average = sci_ref_th_ratio * np.mean(cal2_ref_cube_bck_sub, axis=0)
    cal2_sci_cube_psf_sub = cal2_sci_cube_bck_sub - np.tile(ref_average, (n_sci_files, n_sci_int, 1, 1))



### Mean-combine integration within a roll (minimizes rotations)
print('--- Combining integrations within rolls ---')
combined_roll_images = np.mean(cal2_sci_cube_psf_sub, axis=1)


# Creates a 4QPM mask for the combination
fqpm_mask = create_fqpm_mask(bck_cropsizes, bck_cropsizes//2, fqpm_width, fqpm_angle)
fqpm_mask_roll = np.tile(fqpm_mask, (n_sci_files,1,1))

fig8, ax8 = plt.subplots(1,2,figsize=(8,6), dpi=130)
for i in range(n_sci_files):
    # ax8[i].imshow((1-fqpm_mask)*combined_roll_images[i], vmin=vmin_lin, vmax=vmax)
    ax8[i].imshow((1-fqpm_mask)*combined_roll_images[i], norm=LogNorm(vmin=vmax/100, vmax=vmax))

# vmin = 0 #median_val*0.8
# vmax = 2 #3 #max_val*0.7
fig5, ax5 = plt.subplots(1,n_sci_files,figsize=(8,4), dpi=130)
fig5.suptitle('ROLLS HD141569  '+filt)
images = []
for i in range(n_sci_files):
    # images.append(ax5[i].imshow(combined_roll_images[i,:,:], vmin=vmin_lin, vmax=vmax))
    images.append(ax5[i].imshow(combined_roll_images[i,:,:], norm=LogNorm(vmin=vmax/100, vmax=vmax)))
    ax5[i].set_title('ORIENT {}: {}deg'.format(i, PA_V3_sci[i]))
plt.tight_layout()
cbar = fig5.colorbar(images[0], ax=ax5)
cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()


### Derotate each roll
print('--- Derotate and combine rolls ---')
derotated_images = np.empty(np.shape(combined_roll_images))
fqpm_mask_roll_derotated = np.empty(np.shape(fqpm_mask_roll))
for i in range(n_sci_files):
    derotated_images[i] = frame_rotate_interp(combined_roll_images[i], PA_V3_sci[i])
    fqpm_mask_roll_derotated[i] = frame_rotate_interp(fqpm_mask_roll[i], PA_V3_sci[i])
nanPoses = (fqpm_mask_roll_derotated > 0.5)
derotated_images[nanPoses] = np.nan


### Combine the two rolls and crop it
combined_image_full = np.nanmean(derotated_images, axis=0)
combined_image = resize(combined_image_full, [cropsize,cropsize]) 


print('Total flux in image: {:.3f} mJy'.format(np.nansum(combined_image)*0.11*0.11))

# vmin = 0.1 #median_val*0.8
# vmax = 10 #3 #max_val*0.7
fig7, ax7 = plt.subplots(1,1,figsize=(8,6), dpi=130)
im = ax7.imshow(combined_image, norm=LogNorm(vmin=vmax/500, vmax=vmax))#,origin='upper')
# im = ax7.imshow(combined_image, vmin=0, vmax=vmax)#,origin='upper')
ax7.set_title('COMBINED HD141569  '+filt)
plt.tight_layout()
cbar = fig7.colorbar(im, ax=ax7)
cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()


if saveCombinedImageQ:
    print('--- Saving the reduced images ---')
    hdu = fits.PrimaryHDU(data=None, header = fits.getheader(cal2_sci_files[0]))
    #Put the data back in the initial untits to match the unit in the header
    
    hdu2 = fits.ImageHDU(combined_image,fits.getheader(cal2_sci_files[0], 1))
    hdu2.header['BUNIT']= 'mJy/arcsec2'
    hdul = fits.HDUList([hdu,hdu2])
    hdul.writeto(os.path.join(path_output, filename_output), overwrite=overWriteQ)

