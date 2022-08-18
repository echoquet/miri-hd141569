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


def create_saber_mask(dims, center, width, angle):
    saber_mask_tmp = np.zeros(dims)
    saber_mask_tmp[center[0] - width//2 : center[0] + width//2, :] =1
    # saber_mask_tmp[:, center[1] - width//2 : center[1] + width//2] =1
    saber_mask = np.round(frame_rotate_interp(saber_mask_tmp, angle, center=center)).astype(int)
    return saber_mask


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
    # keyword_values = np.zeros(len(file_list))
    keyword_values = []
    for i, file in enumerate(file_list):
        header = fits.getheader(file, extension)
        # keyword_values[i] = header[keyword]
        keyword_values = np.append(keyword_values, header[keyword])
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


def display_grid_of_images_from_cube(cube, vmax, suptitle='', imtitle_array=None, logNorm=True, dpi=130):
    dims = np.shape(cube)
    ndims = len(dims)
    if ndims == 4:
        nrow, ncol = dims[0:2]
        cube2 = cube
    elif ndims == 3:
        nrow = dims[0]
        ncol = 1
        cube2 = np.reshape(cube, (dims[0], 1, dims[1], dims[2]))
    else:
        raise TypeError('Cube should be 3D or 4D')
        
    fig, ax = plt.subplots(nrow, ncol, figsize=(2*ncol, 2*nrow), dpi=130)
    fig.suptitle(suptitle)
    images = []
    for i in range(nrow):
        for j in range(ncol):
            if logNorm:
                images.append(ax[i,j].imshow(cube2[i,j,:,:], norm=LogNorm(vmin=vmax/100, vmax=vmax)))
            else:
                images.append(ax[i,j].imshow(cube2[i,j,:,:], vmin=0, vmax=vmax))
            if imtitle_array is not None:
                ax[i,j].set_title(imtitle_array[i,j])
    plt.tight_layout()
    cbar = fig.colorbar(images[0], ax=ax)
    cbar.ax.set_title('mJy.arcsec$^{-2}$')
    plt.show()

#%%
print('##### IDENTIFY DATA FILES ### ')

# Parameters to locate / select the datasets:
# root = '/Users/echoquet/Documents/Research/Astro/JWST_Programs/Cycle-1_ERS-1386_Hinkley/Disk_Work/2021-10_Synthetic_Datasets/2_Raw_Synthetic_Data'
# root = '/Users/echoquet/Documents/Research/Astro/JWST_Programs/Cycle-1_ERS-1386_Hinkley/Disk_Work/2021-10_Synthetic_Datasets/4_Real_JWST_Data/MIRI_Commissioning/jw01037'
base_root = '/Users/echoquet/Documents/Research/Astro/JWST_Programs/Cycle-1_ERS-1386_Hinkley/Disk_Work/2021-10_Synthetic_Datasets'
root_folder = '4_Real_JWST_Data/MIRI_ERS/MIRI_Data/MIRI_CAL2'
save_folder = '4_Real_JWST_Data/MIRI_ERS/MIRI_Data/MIRI_PROCESSED'
root = os.path.join(base_root, root_folder)
path_output = os.path.join(base_root, save_folder)


# inspec_raw = True
inspec_cal1 = False
data_type = '*_calints.fits'

selected_data_files = np.array(glob(os.path.join(root, data_type)))
targname_all_list = import_keyword_from_files(selected_data_files, 'TARGNAME', extension=0)
filt_all_list = import_keyword_from_files(selected_data_files, 'FILTER', extension=0)
print('Data type: {}'.format(data_type))
print('Number of files: {}\n'.format(len(selected_data_files)))
print('All Targets: {}'.format(np.unique(targname_all_list)))
print('All Filters: {}\n'.format(np.unique(filt_all_list)))


filt = 'F1065C'
targname_sci = 'HD 141569'
targname_ref = 'HD 140986'

print('SCI Target name: {}'.format(targname_sci))
print('REF Target name: {}'.format(targname_ref))
print('Filter: {}'.format(filt))

selected_filt_indices = (filt_all_list == filt)
selected_sci_indices = (targname_all_list == targname_sci)
selected_ref_indices = (targname_all_list == targname_ref)
selected_sci_files = selected_data_files[selected_filt_indices * selected_sci_indices]
selected_ref_files = selected_data_files[selected_filt_indices * selected_ref_indices]




# Locate the datasets
# print('Root folder:\n {}'.format(root))
# print('SCI folder: {}'.format(folder_sci))
# print('REF folder: {}'.format(folder_ref))
# print('Filter name: {}'.format(filt))

# Paths:
# path_raw = os.path.join(root, 'MIRI_RAW_Data_Kim', filt)
# path_cal1 = os.path.join(root, 'MIRI_CAL1', filt)
# path_cal2 = os.path.join(root, 'MIRI_CAL2', filt)
# path_cal2_sci = os.path.join(root, folder_sci)
# path_cal2_ref = os.path.join(root, folder_ref)


# basename_sci = 'HD141569_'+version+'*MIRIMAGE_'+filt+'exp1'
# basename_ref = 'SGD*MIRIMAGE_'+filt+'exp1'
# print('Basename SCI: {}'.format(basename_sci))
# print('Basename REF: {}\n'.format(basename_ref))

# raw_sci_files = glob(os.path.join(path_raw, basename_sci+'.fits'))
# cal1_sci_files = glob(os.path.join(path_cal1, basename_sci+'_rateints.fits'))
# cal2_sci_files = glob(os.path.join(path_cal2_sci, '*_calints.fits'))
# print('Number of RAW SCI files: {}'.format(len(raw_sci_files)))
# print('Number of CAL1 SCI files (rateints): {}'.format(len(cal1_sci_files)))
print('Number of selected SCI files: {}'.format(len(selected_sci_files)))

# raw_ref_files = glob(os.path.join(path_raw, basename_ref+'.fits'))
# cal1_ref_files = glob(os.path.join(path_cal1, basename_ref+'_rateints.fits'))
# cal2_ref_files = glob(os.path.join(path_cal2_ref, '*_calints.fits'))
# print('Number of RAW REF files: {}'.format(len(raw_ref_files)))
# print('Number of CAL1 REF files (rateints): {}'.format(len(cal1_ref_files)))
print('Number of selected REF files: {}'.format(len(selected_ref_files)))

#%% RAW DATA INSPECTION
if data_type == '*_uncal.fits':
    print('##### RAW DATA INSPECTION #####')
    raw_sci_files = selected_sci_files
    raw_ref_files = selected_ref_files
    
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
if data_type == '*_calints.fits':
    cal2_sci_files = selected_sci_files
    cal2_ref_files = selected_ref_files

#number of extensions: 
#len(fits.open(cal2_sci_files[0]))
# targname_sci_list = np.unique(import_keyword_from_files(cal2_sci_files, 'TARGNAME', extension=0))
# filt_sci_list = import_keyword_from_files(cal2_sci_files, 'FILTER', extension=0)
# targname_ref_list = np.unique(import_keyword_from_files(cal2_ref_files, 'TARGNAME', extension=0))
# filt_ref_list = import_keyword_from_files(cal2_ref_files, 'FILTER', extension=0)

# if len(targname_sci_list) > 1 or len(targname_ref_list) > 1:
#     raise DatasetError('Mix of several targets')
# else:
#     targname_sci = targname_sci_list[0]
#     targname_ref = targname_ref_list[0]

# if len(np.unique(np.concatenate((filt_sci_list, filt_ref_list)))) > 1:
#     raise DatasetError('Mix of several filters')
# else:
#     filt = filt_sci_list[0]

# print('SCI TARGET NAME: {}'.format(targname_sci))
# print('REFERENCE NAME: {}'.format(targname_ref))
# print('FILTER: {}'.format(filt))
# print('\n')


if filt == 'F1065C':
    vmax = 10 #2 #mJy/arcsec^2
elif filt == 'F1140C':
    vmax = 10
else:
    vmax= 20
vmin_lin = 0
vmin_log = vmax/100

# SCIENCE DATASETS COUNTRATES
print('##### SCI TARGET: #####')
PA_V3_sci = import_keyword_from_files(cal2_sci_files, 'PA_V3', extension=1)
phot_MJySr_sci = import_keyword_from_files(cal2_sci_files, 'PHOTMJSR', extension=1)
phot_uJyA2_sci = import_keyword_from_files(cal2_sci_files, 'PHOTUJA2', extension=1)
scaling_values = phot_uJyA2_sci/(1000*phot_MJySr_sci)
cal2_sci_cube = import_data_cube_from_files(cal2_sci_files, scaling_list=scaling_values)
dims = (np.shape(cal2_sci_cube))[-2:]
n_sci_files = len(cal2_sci_files)
n_sci_int = (np.shape(cal2_sci_cube))[1]
print('Number of exposures: {}'.format(n_sci_files))
print('Number of integrations: {}'.format(n_sci_int))
print('Number of rolls: {}'.format(len(np.unique(PA_V3_sci))))
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




# REFERENCE STAR DATASET COUNTRATES:
print('##### REF TARGET: #####')
PA_V3_ref = import_keyword_from_files(cal2_ref_files, 'PA_V3', extension=1)
phot_MJySr_ref = import_keyword_from_files(cal2_ref_files, 'PHOTMJSR', extension=1)
phot_uJyA2_ref = import_keyword_from_files(cal2_ref_files, 'PHOTUJA2', extension=1)
scaling_values_ref = phot_uJyA2_ref/(1000*phot_MJySr_ref)
cal2_ref_cube = import_data_cube_from_files(cal2_ref_files, scaling_list=scaling_values_ref)
n_ref_files = len(cal2_ref_cube)
n_ref_int = (np.shape(cal2_ref_cube))[1]
print('Number of exposures: {}'.format(n_ref_files))
print('Number of integrations: {}'.format(n_ref_int))
print('Number of rolls: {}'.format(len(np.unique(PA_V3_ref))))

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
display_all = True
verbose = False
commissioning_dataQ = False

# Values from Aarynn C. / Dean H. after commissioning
if filt == 'F1065C':
    fqpm_center = [111.89, 120.81]
elif filt == 'F1140C':
    fqpm_center = [112.20, 119.99]
else:
    fqpm_center=[113.33, 119.84]

## Parameters for Median-filter
median_filt_thresh = 3 #sigma
median_filt_box_size = 2
median_filt_iter_max = 20


## Parameters for the background mask. 
# Croping roughly about the FQPM center before finer registration steps
bck_cropsize = 201
crop_center = (np.round(fqpm_center)).astype(int)
bck_mask_Rin_sci = 50
bck_mask_Rin_ref = 65
bck_saber_glow_mask = True
bck_saber_glow_width = 9 #pix
bck_saber_glow_angle = 5 #deg
# bck_mask_Rout = 105



## Parameters for the PSF subtraction
subtract_method = 'classical-Ref-Averaged' # 'classical-Ref-Averaged'  'No-Subtraction'
if filt == 'F1065C':
    sci_ref_th_ratio = 64.42/1257.45  #64.02/469.45
elif filt == 'F1140C':
    sci_ref_th_ratio = 56.72/1109.31   #56.07/410.48
else:
    sci_ref_th_ratio = 30.65/595.52   #30.71/223.75

if commissioning_dataQ:
    sci_ref_th_ratio = 10**((4.531-4.655)/2.5) # from W3 mag

# Parameters for 4QPM mask and data combination
# TODO: use CDBS calibration mask instead!
# fqpm_center = crop_center     # TMP for synthetic dataset! use header instead
fqpm_width = 3 #pix
fqpm_angle = 5 #deg

# PA_V3_sci = [35, 49] if filt=='F1065C' else [130, 144] # TMP for synthetic datasets!

cropsize = 101

# Parameters for exporting the outputs
export_tmp_filesQ = True
saveCombinedImageQ = True
overWriteQ = True
basename_sci = 'HD141569_'+filt+'_v1'
filename_output = basename_sci + '_combined.fits'
if saveCombinedImageQ:
    os.makedirs(path_output, exist_ok=overWriteQ)



''' Reduction notes:
    To achieve the subtraction of the PSF, both SCI and REF must be background-subtracted.
    Background subtrarction is best achieved if estimated in the area close to the system,
    so it is better to crop the data first. 
    
'''


### Cleaning the bad pixels from the SCI and REF datasets
print('--- Cleaning the data from bad pixels ---')
# NOTE: consider using the DQ maps too
cal2_sci_cube_filt = median_filter_cube(cal2_sci_cube, median_filt_box_size, median_filt_thresh, 
                                        iter_max=median_filt_iter_max, verbose=verbose)
cal2_ref_cube_filt = median_filter_cube(cal2_ref_cube, median_filt_box_size, median_filt_thresh, 
                                        iter_max=median_filt_iter_max, verbose=verbose)

# if avg_ref_ints:
#     cal2_ref_cube_filt_list = np.mean(cal2_ref_cube_filt, axis=1)
# else:
#     cal2_ref_cube_filt_list = np.reshape(cal2_ref_cube_filt, (n_ref_files*n_ref_int,)+dims)
# n_ref_frames = len(cal2_ref_cube_filt_list)
# print('Number of reference frames: {}'.format(n_ref_frames))



### Register the frames
# TODO: Register the SCI frames with companion
# TODO: Register the REF frames?
# TODO: update to using the 4D cal2_ref_cube_filt cube again instead of cal2_ref_cube_filt_list
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
    
    #display_grid_of_images_from_cube(cal2_ref_cube_filt, vmax/30)
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
cal2_ref_cube_crop = resize(cal2_ref_cube_filt, bck_cropsizes, cent=crop_center)

# Optimize the masks
binary_coords = bck_cropsize/2 + np.array([-90, 20])
bck_mask_sci = create_mask(bck_mask_Rin_sci, bck_cropsizes) 
bck_mask_sci *= create_mask(bck_mask_Rin_sci, bck_cropsizes, cent=binary_coords)
bck_mask_ref = create_mask(bck_mask_Rin_ref, bck_cropsizes)
if bck_saber_glow_mask:
    bck_saber_mask = create_saber_mask(bck_cropsizes, bck_cropsizes//2, bck_saber_glow_width, bck_saber_glow_angle)
    bck_mask_sci = bck_mask_sci * ~(bck_saber_mask.astype(bool))
    bck_mask_ref = bck_mask_ref * ~(bck_saber_mask.astype(bool))
# bck_saber_glow_width = 5 #pix
# bck_saber_glow_angle = 10 #deg

if display_all:
    fig10, ax10 = plt.subplots(1,1,figsize=(8,6), dpi=130)
    ax10.imshow(bck_mask_sci*np.mean(cal2_sci_cube_crop,axis=(0,1)), norm=LogNorm(vmin=0.02, vmax=0.5))
    # ax10.imshow(bck_mask_sci, vmin=0, vmax=1)
    
    fig11, ax11 = plt.subplots(1,1,figsize=(8,6), dpi=130)
    ax11.imshow(bck_mask_ref*np.mean(cal2_ref_cube_crop,axis=(0,1)), norm=LogNorm(vmin=0.02, vmax=0.5))
    
    print('Number of NaNs in SCI: {}'.format(np.count_nonzero(np.isnan(cal2_sci_cube_crop[:, :, bck_mask_sci]))))
    print('Number of NaNs in REF: {}'.format(np.count_nonzero(np.isnan(cal2_ref_cube_crop[:, :, bck_mask_sci]))))
    
# Estimate and subtract the background levels
bck_level_sci = np.nanmedian(cal2_sci_cube_crop[:, :, bck_mask_sci])
bck_level_ref = np.nanmedian(cal2_ref_cube_crop[:, :, bck_mask_ref])
print('Background_level SCI = {:.2e} mJy.arcsec^-2'.format(bck_level_sci))
print('Background_level REF = {:.2e} mJy.arcsec^-2'.format(bck_level_ref))

cal2_sci_cube_bck_sub = cal2_sci_cube_crop - np.tile(bck_level_sci, (n_sci_files, n_sci_int, 1, 1))
cal2_ref_cube_bck_sub = cal2_ref_cube_crop - np.tile(bck_level_ref, (n_ref_files, n_ref_int, 1, 1))

display_grid_of_images_from_cube(cal2_sci_cube_bck_sub, vmax/3, #logNorm=False,
                                 suptitle='Background subtracted Integrations HD141569  '+filt)
display_grid_of_images_from_cube(cal2_ref_cube_bck_sub, vmax/3, #logNorm=False,
                                 suptitle='Background subtracted Integrations Ref Star  '+filt)


if export_tmp_filesQ:
    path_tmp_folder = '4_Real_JWST_Data/MIRI_ERS/MIRI_Data/MIRI_Background_subtracted'
    path_tmp_output = os.path.join(base_root, path_tmp_folder)
    
    for i, obs in enumerate(cal2_sci_cube_bck_sub):
        cal2_filename = os.path.basename(cal2_sci_files[i])
        filename_tmp_output = cal2_filename[:-5] + '_bckgrd-sub' + cal2_filename[-5:] 
        hdu = fits.PrimaryHDU(data=None, header = fits.getheader(cal2_sci_files[i]))
        hdu2 = fits.ImageHDU(obs,fits.getheader(cal2_sci_files[i], 1))
        hdu2.header['BUNIT']= 'mJy/arcsec2'
        hdul = fits.HDUList([hdu,hdu2])
        hdul.writeto(os.path.join(path_tmp_output, filename_tmp_output), overwrite=overWriteQ)
    
    for i, obs in enumerate(cal2_ref_cube_bck_sub):
        cal2_filename = os.path.basename(cal2_ref_files[i])
        filename_tmp_output = cal2_filename[:-5] + '_bckgrd-sub' + cal2_filename[-5:] 
        hdu = fits.PrimaryHDU(data=None, header = fits.getheader(cal2_ref_files[i]))
        hdu2 = fits.ImageHDU(obs,fits.getheader(cal2_ref_files[i], 1))
        hdu2.header['BUNIT']= 'mJy/arcsec2'
        hdul = fits.HDUList([hdu,hdu2])
        hdul.writeto(os.path.join(path_tmp_output, filename_tmp_output), overwrite=overWriteQ)
    


### Classical subtraction of the SCI PSF
if subtract_method == 'No-Subtraction':
    cal2_sci_cube_psf_sub = cal2_sci_cube_bck_sub
elif subtract_method =='classical-Ref-Averaged':
    ref_average = sci_ref_th_ratio * np.mean(cal2_ref_cube_bck_sub, axis=(0,1))
    cal2_sci_cube_psf_sub = cal2_sci_cube_bck_sub - np.tile(ref_average, (n_sci_files, n_sci_int, 1, 1))



### Mean-combine integration within a roll (minimizes rotations)
print('--- Combining integrations within rolls ---')
combined_roll_images = np.mean(cal2_sci_cube_psf_sub, axis=1)

# Patch to remove nan values
nan_poses = np.argwhere(np.isnan(combined_roll_images))
for i, inds in enumerate(nan_poses):
    combined_roll_images[inds[0], inds[1], inds[2]] = 0

# Creates a 4QPM mask for the combination
fqpm_mask = create_fqpm_mask(bck_cropsizes, bck_cropsizes//2, fqpm_width, fqpm_angle)
fqpm_mask_roll = np.tile(fqpm_mask, (n_sci_files,1,1))

fig8, ax8 = plt.subplots(1,n_sci_files,figsize=(n_sci_files*3,3), dpi=130)
for i in range(n_sci_files):
    ax8[i].imshow((1-fqpm_mask)*combined_roll_images[i], vmin=vmin_lin, vmax=vmax)
    # ax8[i].imshow((1-fqpm_mask)*combined_roll_images[i], norm=LogNorm(vmin=vmax/100, vmax=vmax))

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
    derotated_images[i] = frame_rotate_interp(combined_roll_images[i], -PA_V3_sci[i])
    fqpm_mask_roll_derotated[i] = frame_rotate_interp(fqpm_mask_roll[i], -PA_V3_sci[i])
nanPoses = (fqpm_mask_roll_derotated > 0.5)
derotated_images[nanPoses] = np.nan

fig6, ax6 = plt.subplots(1,n_sci_files,figsize=(8,4), dpi=130)
fig6.suptitle('ROLLS HD141569  '+filt)
images = []
for i in range(n_sci_files):
    # images.append(ax5[i].imshow(combined_roll_images[i,:,:], vmin=vmin_lin, vmax=vmax))
    images.append(ax6[i].imshow(fqpm_mask_roll_derotated[i,:,:], norm=LogNorm(vmin=vmax/100, vmax=vmax)))
    ax6[i].set_title('ORIENT {}: {}deg'.format(i, PA_V3_sci[i]))
plt.tight_layout()
cbar = fig6.colorbar(images[0], ax=ax6)
cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()



### Combine the two rolls and crop it
combined_image_full = np.nanmean(derotated_images, axis=0)
combined_image = resize(combined_image_full, [cropsize,cropsize]) 


print('Total flux in image: {:.3f} mJy'.format(np.nansum(combined_image)*0.11*0.11))

# vmin = 0.1 #median_val*0.8
# vmax = 10 #3 #max_val*0.7
fig7, ax7 = plt.subplots(1,1,figsize=(8,6), dpi=130)
# im = ax7.imshow(combined_image, norm=LogNorm(vmin=vmax/100, vmax=vmax))
im = ax7.imshow(combined_image, vmin=vmin_lin, vmax=vmax)
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

