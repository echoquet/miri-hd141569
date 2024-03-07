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

import imtoolbox as tb

from scipy.optimize import minimize
from copy import deepcopy
from astropy.io import fits
import pandas as pd
from photutils.centroids import centroid_sources, centroid_com, centroid_quadratic,centroid_2dg, centroid_1dg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.rcParams['image.origin'] = 'lower'
plt.rcParams["image.cmap"] = 'gist_heat'#'hot'#'copper'

#%% FUNCTIONS

def reg_criterion(parameters, image, template, mask):    
    dx, dy, nu = parameters
    return np.sum((nu * tb.shift_interp(image, [dx, dy]) - template)[mask]**2)


def import_data_cube_from_files(file_list, scaling_list=None, ext=None, extname=None):
    tmp = fits.getdata(file_list[0], ext=None, extname=None)
    data_cube = np.zeros(np.shape(file_list) + np.shape(tmp))
    if scaling_list is None:
        scaling_list = np.ones(len(file_list))
    for i, file in enumerate(file_list):
        data_cube[i] = fits.getdata(file, ext=ext, extname=extname) * scaling_list[i]
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


def clean_bad_pixel_list(cube, bad_pixel_coords, method='zeros'):
    # dims = np.shape(cube)

    # for i, inds in enumerate(nan_poses):
    #     cal2_sci_cube_bck_sub[inds[0], inds[1], inds[2], inds[3]] = 0

    cube_clean = deepcopy(cube)
    for coords in bad_pixel_coords:
        cube_clean[tuple(coords)] = 0
    
    return cube_clean




def display_grid_of_images_from_cube(cube, vmax, suptitle='', imtitle_array=None, logNorm=True, 
                                     vmin=None, dpi=130, imsize=3, colorbar = False):
    dims = np.shape(cube)
    ndims = len(dims)
    if ndims == 4:
        nrow, ncol = dims[0:2]
    elif ndims == 3:
        nrow = 1
        ncol = dims[0]
    else:
        raise TypeError('Cube should be 3D or 4D')
    
    if vmin is None:
        vmin = vmax/1000 if logNorm else 0
    
    fig, ax = plt.subplots(nrow, ncol, figsize=(imsize*ncol, imsize*nrow), dpi=dpi)
    fig.suptitle(suptitle)
    images = []
    if ndims == 4:
        for i in range(nrow):
            for j in range(ncol):
                if logNorm:
                    images.append(ax[i,j].imshow(cube[i,j,:,:], norm=LogNorm(vmin=vmin, vmax=vmax)))
                else:
                    images.append(ax[i,j].imshow(cube[i,j,:,:], vmin=vmin, vmax=vmax))
                if imtitle_array is not None:
                    ax[i,j].set_title(imtitle_array[i,j])
    elif ndims == 3:
        for i in range(ncol):
            if logNorm:
                images.append(ax[i].imshow(cube[i,:,:], norm=LogNorm(vmin=vmin, vmax=vmax)))
            else:
                images.append(ax[i].imshow(cube[i,:,:], vmin=vmin, vmax=vmax))
            if imtitle_array is not None:
                ax[i].set_title(imtitle_array[i])
                
    plt.tight_layout()
    if colorbar:
        cbar = fig.colorbar(images[0], ax=ax)
        cbar.ax.set_title('mJy.arcsec$^{-2}$')
    plt.show()


#%% GLOBAL CONSTANTS

pixsize = 0.11
base_root = '/Users/echoquet/Documents/Research/JWST_Programs/Cycle-1_ERS-1386_Hinkley/Disk_Work/2021-10_Synthetic_Datasets'

targname_sci = 'HD 141569'
targname_ref = 'HD 140986'
targname_bck = ''
filt = 'F1065C'

#%% TARGET ACQ FILES
print('\n##### PROCESSING THE TARGET ACQUISITION FILES ### ')
##### CUSTOM Parameters:

target_acq_folder = '4_Real_JWST_Data/MIRI_ERS/MIRI_Data/MIRI_TA_CAL'
path_target_acq_data = os.path.join(base_root, target_acq_folder)

## Parameter for Displaying the images:
vmax_ta= 10

# Parameters for frame selection:
discard_ta_filename = np.array(['jw01386020001_02101_00001_mirimage_cal.fits',
                                'jw01386023001_02101_00001_mirimage_cal.fits'])

## Parameters for bad pixel cleaning:
median_filt_thresh = 3 #sigma
median_filt_box_size = 2
median_filt_iter_max = 200


## Parameters for background subtraction:
rIn_mask_A = 19
rIn_mask_BC = 7

## Parameters for centroiding 
rough_ta_xpos_ABC = np.array([[179, 196.,197.5],[135.5, 152, 154], 
                          [136, 163, 166],
                          [177,194, 195], [137,155,159],
                          [138, 165, 169 ],
                          [177, 194, 196], [133,151,152],
                          [177,205,209], [134,162,165]
                          ]) - 1
rough_ta_ypos_ABC = np.array([[168, 101., 89.2],[125, 58, 47], 
                          [126, 62, 50],
                          [169,103, 91], [128, 61, 50],
                          [128, 65, 54 ],
                          [164, 98, 87],[125,59,48],
                          [164,102, 91], [126,63,52]
                          ]) - 1
rough_ta_xpos_REF = np.array([179, 136, 177, 138, 177, 134]) - 1
rough_ta_ypos_REF = np.array([170, 125, 170, 127, 166, 125]) - 1

centroid_box_size_ta = 9
centroid_method_ta = centroid_2dg #centroid_sources, centroid_com, centroid_quadratic,centroid_2dg, centroid_1dg

## Parameters for saving the data
export_centroid_coordinates = False
export_combined_ta = False
overWriteTAQ = False
combined_ta_folder = '4_Real_JWST_Data/MIRI_ERS/MIRI_Data/MIRI_TA_combined'


#############################################################################################
##### Find the TA frames ####
all_TA_files = np.array(glob(os.path.join(path_target_acq_data, '*_cal.fits')))
all_TA_files.sort()
# all_TA_PA_V3 = import_keyword_from_files(all_TA_files, 'PA_V3', extension=1)
all_TA_ROLL_REF = import_keyword_from_files(all_TA_files, 'ROLL_REF', extension=1)
all_TA_V3I_YANG = import_keyword_from_files(all_TA_files, 'V3I_YANG', extension=1)
all_TA_PA_V3 = all_TA_ROLL_REF + all_TA_V3I_YANG
targname_all_TA_list = import_keyword_from_files(all_TA_files, 'TARGNAME', extension=0)
filt_all_TA_list = import_keyword_from_files(all_TA_files, 'FILTER', extension=0)


# Note: there is only one Filter for the TA. 
# Use the filename or Obs ID to select files by coronagraphic filter
selected_TA_sci_indices = (targname_all_TA_list == targname_sci)
selected_TA_ref_indices = (targname_all_TA_list == targname_ref)
selected_TA_sci_files = all_TA_files[selected_TA_sci_indices]
selected_TA_ref_files = all_TA_files[selected_TA_ref_indices]
selected_TA_sci_PA_V3 = all_TA_PA_V3[selected_TA_sci_indices]

print('Number of selected SCI-TA files: {}'.format(len(selected_TA_sci_files)))
print('Number of selected REF-TA files: {}'.format(len(selected_TA_ref_files)))
print('Filter Name: {}'.format(np.unique(filt_all_TA_list)))


##### Import the TA data (REF and SCI):
# Note: the TA cal is not flux calibrated. 
# The files are in DN/s and there is no photometric keyword.
target_acq_ref_cube = import_data_cube_from_files(selected_TA_ref_files)
display_grid_of_images_from_cube(target_acq_ref_cube, vmax_ta*10, vmin=-vmax_ta*5, logNorm=False, 
                                  suptitle='Target Acq REF STAR')
n_ref_ta = len(target_acq_ref_cube)

target_acq_sci_cube = import_data_cube_from_files(selected_TA_sci_files)
display_grid_of_images_from_cube(target_acq_sci_cube, vmax_ta, vmin=-vmax_ta/2, logNorm=False, 
                                  suptitle='Target Acq HD 141569')

selected_TA_sci_filenames = np.array([os.path.basename(file) for file in selected_TA_sci_files])
n_sci_ta = len(target_acq_sci_cube)
dims_ta = np.shape(target_acq_sci_cube)[1:]



##### Discarding frames with sources on the 4QPM edges (SCI):
print('\nDiscarding {} SCI-TA files: \n{}'.format(len(discard_ta_filename), discard_ta_filename))
discard_ta_flag = np.isin(selected_TA_sci_filenames, discard_ta_filename)
n_sci_ta_cent = n_sci_ta - len(discard_ta_filename)
target_acq_sci_cube_selec = target_acq_sci_cube[~discard_ta_flag]
selected_TA_sci_PA_V3_cent = selected_TA_sci_PA_V3[~discard_ta_flag]


##### Clean the data from bad pixels with median filter (REF and SCI):
print('\nClean bad pixels (REF and SCI TAs)')
target_acq_ref_cube_clean = median_filter_cube(target_acq_ref_cube, median_filt_box_size, median_filt_thresh, 
                                               iter_max=median_filt_iter_max, verbose=False)
target_acq_sci_cube_clean = median_filter_cube(target_acq_sci_cube_selec, median_filt_box_size, median_filt_thresh, 
                                               iter_max=median_filt_iter_max, verbose=False)



#### Estimate the background frame from the median image (REF and SCI):
print('\nBAckground subtraction (REF and SCI TAs)')
target_acq_ref_cube_masked = deepcopy(target_acq_ref_cube_clean)
for i in range(n_ref_ta):
    mask_ta_ref_stars = np.ones(dims_ta, dtype=bool) 
    mask_ta_ref_stars *= tb.create_mask(rIn_mask_BC, dims_ta, cent=[rough_ta_ypos_REF[i], rough_ta_xpos_REF[i]])
    target_acq_ref_cube_masked[i, ~mask_ta_ref_stars] = np.nan
target_acq_ref_median_bckg = np.nanmedian(target_acq_ref_cube_masked, axis=0)

fig6, ax6 = plt.subplots(1,1,figsize=(8,6), dpi=130)
im = ax6.imshow(target_acq_ref_median_bckg, vmin=-vmax_ta*5, vmax=vmax_ta*10)
plt.tight_layout()
ax6.set_title('Bacground frame REF TA')
cbar = fig6.colorbar(im, ax=ax6)
cbar.ax.set_title('DN/s$')
plt.show()

target_acq_sci_cube_masked = deepcopy(target_acq_sci_cube_clean)
for i in range(n_sci_ta_cent):
    mask_ta_sci_stars = np.ones(dims_ta, dtype=bool) 
    mask_ta_sci_stars *= tb.create_mask(rIn_mask_A, dims_ta, cent=[rough_ta_ypos_ABC[i,0], rough_ta_xpos_ABC[i,0]])
    mask_ta_sci_stars *= tb.create_mask(rIn_mask_BC, dims_ta, cent=[rough_ta_ypos_ABC[i,1], rough_ta_xpos_ABC[i,1]])
    mask_ta_sci_stars *= tb.create_mask(rIn_mask_BC, dims_ta, cent=[rough_ta_ypos_ABC[i,2], rough_ta_xpos_ABC[i,2]])
    target_acq_sci_cube_masked[i, ~mask_ta_sci_stars] = np.nan
target_acq_sci_median_bckg = np.nanmedian(target_acq_sci_cube_masked, axis=0)

fig7, ax7 = plt.subplots(1,1,figsize=(8,6), dpi=130)
im = ax7.imshow(target_acq_sci_median_bckg, vmin=-vmax_ta/2, vmax=vmax_ta)
ax7.set_title('Bacground frame SCI TA')
plt.tight_layout()
cbar = fig7.colorbar(im, ax=ax7)
cbar.ax.set_title('DN/s$')
plt.show()


target_acq_ref_cube_clean = target_acq_ref_cube - np.tile(target_acq_ref_median_bckg,  (n_ref_ta, 1, 1))
target_acq_sci_cube_clean = target_acq_sci_cube_selec - np.tile(target_acq_sci_median_bckg,  (n_sci_ta_cent, 1, 1))
display_grid_of_images_from_cube(target_acq_ref_cube_clean, vmax_ta*10, vmin=-vmax_ta*5, logNorm=False, 
                                  suptitle='Background sub Target Acq REF STAR')
display_grid_of_images_from_cube(target_acq_sci_cube_clean, vmax_ta, vmin=-vmax_ta/2, logNorm=False, 
                                  suptitle='Background sub Target Acq SCI STAR')


##### Finding the centroid of the three stars
print('\nFinding the centroid of the three stars:')
fine_ta_xpos_ABC = np.empty((n_sci_ta_cent, 3))
fine_ta_ypos_ABC = np.empty((n_sci_ta_cent, 3))
for i, image_ta in enumerate(target_acq_sci_cube_clean):
    x, y = centroid_sources(image_ta, 
                            rough_ta_xpos_ABC[i], rough_ta_ypos_ABC[i], 
                            box_size=centroid_box_size_ta, centroid_func=centroid_method_ta)
    fine_ta_xpos_ABC[i,:] = x
    fine_ta_ypos_ABC[i,:] = y

if export_centroid_coordinates:
    fine_ta_coords_ABC = np.concatenate((fine_ta_xpos_ABC,fine_ta_ypos_ABC), axis=1)
    TA_sci_filenames = selected_TA_sci_filenames[~discard_ta_flag]
    data_ta = np.concatenate((TA_sci_filenames.reshape(len(TA_sci_filenames),1),fine_ta_coords_ABC),axis=1)
    labels_table = ['Filename', 'X coord Star A','X coord Star B','X coord Star C', 'Y coord Star A','Y coord Star B','Y coord Star C']
    df = pd.DataFrame(data=data_ta, columns=labels_table)
    df.to_csv(os.path.join(path_target_acq_data, 'Centroid_Position_StarsABC.csv'))

vmax_ta_list = [400,25,25]
zoom_size = 15
fig2, ax2 = plt.subplots(3, n_sci_ta_cent, figsize=(4*n_sci_ta_cent, 4*3), dpi=130)
for i in range(n_sci_ta_cent):
    for k in range(3):
        ax2[k,i].imshow(target_acq_sci_cube_clean[i,:,:], interpolation='nearest', 
                        vmax=vmax_ta_list[k], vmin=-vmax_ta_list[k]/2)
        ax2[k,i].scatter(fine_ta_xpos_ABC[i,k], fine_ta_ypos_ABC[i,k], marker='x', color='blue',s=100)    
        ax2[k,i].set_xlim(fine_ta_xpos_ABC[i,k]+ np.array([-1, 1])*zoom_size)
        ax2[k,i].set_ylim(fine_ta_ypos_ABC[i,k]+ np.array([-1, 1])*zoom_size)
plt.tight_layout()
plt.show()

labelStars_ta = ['A star','B star', 'C star']
fig1, ax1 = plt.subplots(1,3,figsize=(4*3, 4), dpi=130)
for i in range(3):
    ax1[i].scatter(fine_ta_xpos_ABC[:,i], fine_ta_ypos_ABC[:,i], marker='x') 
    ax1[i].set_title('{} '.format(labelStars_ta[i]))
    ax1[i].set_xlim(fine_ta_xpos_ABC[:,i].mean()+ np.array([-1, 1])*100)
    ax1[i].set_ylim(fine_ta_ypos_ABC[:,i].mean()+ np.array([-1, 1])*100)
    ax1[i].set_xlabel('X position (pix)')
    ax1[i].set_ylabel('Y position (pix)')
plt.tight_layout()
plt.show()

ref_star_index = 1
fine_ta_xpos_ABC_from_ref = fine_ta_xpos_ABC - np.tile(np.reshape(fine_ta_xpos_ABC[:,ref_star_index],(len(fine_ta_xpos_ABC),1)),(1,3))
fine_ta_ypos_ABC_from_ref = fine_ta_ypos_ABC - np.tile(np.reshape(fine_ta_ypos_ABC[:,ref_star_index],(len(fine_ta_ypos_ABC),1)),(1,3))
xy_coords_A_from_ref = np.array([fine_ta_xpos_ABC_from_ref[:,0], fine_ta_ypos_ABC_from_ref[:,0]])
xy_coords_B_from_ref = np.array([fine_ta_xpos_ABC_from_ref[:,1], fine_ta_ypos_ABC_from_ref[:,1]])
xy_coords_C_from_ref = np.array([fine_ta_xpos_ABC_from_ref[:,2], fine_ta_ypos_ABC_from_ref[:,2]])

# Get the Separation and PA of A&C from B, corrected from the telescope orient
sep_A_from_ref = np.sqrt(np.sum(xy_coords_A_from_ref**2, axis=0)) * pixsize
pa_A_from_ref = np.arctan2(-xy_coords_A_from_ref[0], xy_coords_A_from_ref[1])*180/np.pi + selected_TA_sci_PA_V3_cent
sep_B_from_ref = np.sqrt(np.sum(xy_coords_B_from_ref**2, axis=0)) * pixsize
pa_B_from_ref = np.arctan2(-xy_coords_B_from_ref[0], xy_coords_B_from_ref[1])*180/np.pi + selected_TA_sci_PA_V3_cent
sep_C_from_ref = np.sqrt(np.sum(xy_coords_C_from_ref**2, axis=0)) * pixsize
pa_C_from_ref = np.arctan2(-xy_coords_C_from_ref[0], xy_coords_C_from_ref[1])*180/np.pi + selected_TA_sci_PA_V3_cent

# Convert these to dRA and dDEC
# From Gaia, A should be dRA = 5.67   dDec = 5.02
# From Gaia, C should be dRA = 1.11   dDec = 0.71
ra_A_from_ref = sep_A_from_ref *np.sin(pa_A_from_ref*np.pi/180)
dec_A_from_ref = sep_A_from_ref *np.cos(pa_A_from_ref*np.pi/180)
ra_B_from_ref = sep_B_from_ref *np.sin(pa_B_from_ref*np.pi/180)
dec_B_from_ref = sep_B_from_ref *np.cos(pa_B_from_ref*np.pi/180)
ra_C_from_ref = sep_C_from_ref *np.sin(pa_C_from_ref*np.pi/180)
dec_C_from_ref = sep_C_from_ref *np.cos(pa_C_from_ref*np.pi/180)

# Define the best (median) offset and uncertainty (std-dev)
best_sep_A_from_ref = np.array([np.median(sep_A_from_ref), np.std(sep_A_from_ref)])
best_pa_A_from_ref = np.array([np.median(pa_A_from_ref), np.std(pa_A_from_ref)])
best_sep_B_from_ref = np.array([np.median(sep_B_from_ref), np.std(sep_B_from_ref)])
best_pa_B_from_ref = np.array([np.median(pa_B_from_ref), np.std(pa_B_from_ref)])
best_sep_C_from_ref = np.array([np.median(sep_C_from_ref), np.std(sep_C_from_ref)])
best_pa_C_from_ref = np.array([np.median(pa_C_from_ref), np.std(pa_C_from_ref)])

best_ra_A_from_ref = np.array([np.median(ra_A_from_ref), np.std(ra_A_from_ref)])
best_dec_A_from_ref = np.array([np.median(dec_A_from_ref), np.std(dec_A_from_ref)])
best_ra_B_from_ref = np.array([np.median(ra_B_from_ref), np.std(ra_B_from_ref)])
best_dec_B_from_ref = np.array([np.median(dec_B_from_ref), np.std(dec_B_from_ref)])
best_ra_C_from_ref = np.array([np.median(ra_C_from_ref), np.std(ra_C_from_ref)])
best_dec_C_from_ref = np.array([np.median(dec_C_from_ref), np.std(dec_C_from_ref)])

print('Best coordinates of A from ref:')
print('    Sep = {:.2f}" ± {:.2f}"'.format(best_sep_A_from_ref[0], best_sep_A_from_ref[1]))
print('    PA = {:.2f}º ± {:.2f}º'.format(best_pa_A_from_ref[0], best_pa_A_from_ref[1]))
print('    dRA = {:.2f}" ± {:.2f}"'.format(best_ra_A_from_ref[0], best_ra_A_from_ref[1]))
print('    dDEC = {:.2f}" ± {:.2f}"'.format(best_dec_A_from_ref[0], best_dec_A_from_ref[1]))

print('Best coordinates of B from ref:')
print('    Sep = {:.2f}" ± {:.2f}"'.format(best_sep_B_from_ref[0], best_sep_B_from_ref[1]))
print('    PA = {:.2f}º ± {:.2f}º'.format(best_pa_B_from_ref[0], best_pa_B_from_ref[1]))
print('    dRA = {:.2f}" ± {:.2f}"'.format(best_ra_B_from_ref[0], best_ra_B_from_ref[1]))
print('    dDEC = {:.2f}" ± {:.2f}"'.format(best_dec_B_from_ref[0], best_dec_B_from_ref[1]))

print('Best coordinates of C from ref:')
print('    Sep = {:.2f}" ± {:.2f}"'.format(best_sep_C_from_ref[0], best_sep_C_from_ref[1]))
print('    PA = {:.2f}º ± {:.2f}º'.format(best_pa_C_from_ref[0], best_pa_C_from_ref[1]))
print('    dRA = {:.2f}" ± {:.2f}"'.format(best_ra_C_from_ref[0], best_ra_C_from_ref[1]))
print('    dDEC = {:.2f}" ± {:.2f}"'.format(best_dec_C_from_ref[0], best_dec_C_from_ref[1]))


fig1, ax1 = plt.subplots(1,2,figsize=(4*2, 4), dpi=130)
ax1[0].scatter(ra_A_from_ref, dec_A_from_ref, marker='x') 
ax1[1].scatter(ra_C_from_ref, dec_C_from_ref, marker='x') 
ax1[0].scatter(best_ra_A_from_ref, best_dec_A_from_ref, marker='x', color='red') 
ax1[1].scatter(best_ra_C_from_ref, best_dec_C_from_ref, marker='x', color='red') 
ax1[0].set_title('A star from B')
ax1[1].set_title('C star from B')
ax1[0].set_xlim(ra_A_from_ref.mean()+ np.array([-1, 1])*pixsize)
ax1[0].set_ylim(dec_A_from_ref.mean()+ np.array([-1, 1])*pixsize)
ax1[1].set_xlim(ra_C_from_ref.mean()+ np.array([-1, 1])*pixsize)
ax1[1].set_ylim(dec_C_from_ref.mean()+ np.array([-1, 1])*pixsize)
for i in range(2):
    ax1[i].set_xlabel('dRA from B (arcsec)')
    ax1[i].set_ylabel('dDEC from B (arcsec)')
plt.tight_layout()
plt.show()


#TODO: cross corelate also the reference TA frames, combine them, and send to Karl


###### Registering the images using the B coordinates:
print('\nRegister, derotate, and combine the SCI TA frames:')
central_star_index = 1
center_ta_images = (np.array(target_acq_sci_cube_clean.shape[1:])-1)//2
target_acq_sci_cube_centered_derotated = np.empty(np.shape(target_acq_sci_cube_clean))
for i, ta_image in enumerate (target_acq_sci_cube_clean):
    shfit_values = np.array([fine_ta_xpos_ABC[i,central_star_index] - center_ta_images[1], fine_ta_ypos_ABC[i,central_star_index] - center_ta_images[0]])
    ta_image_centered = tb.shift_interp(ta_image, -shfit_values)
    ta_image_derotated = tb.frame_rotate_interp(ta_image_centered, -selected_TA_sci_PA_V3_cent[i], center=center_ta_images)
    target_acq_sci_cube_centered_derotated[i] = ta_image_derotated

# display_grid_of_images_from_cube(target_acq_sci_cube_centered_derotated, vmax_ta, logNorm=False, 
#                                   suptitle='Target Acq HD 141569')

target_acq_sci_cube_combined = np.median(target_acq_sci_cube_centered_derotated, axis=0)

vmax_ta2 = 50
fig7, ax7 = plt.subplots(1,1,figsize=(8,6), dpi=130)
# im = ax7.imshow(target_acq_sci_cube_combined, norm=LogNorm(vmin=vmax_ta2/1000, vmax=vmax_ta2))
im = ax7.imshow(target_acq_sci_cube_combined, vmin=0, vmax=vmax_ta2)
ax7.set_title('COMBINED HD141569 TA frames')
plt.tight_layout()
cbar = fig7.colorbar(im, ax=ax7)
cbar.ax.set_title('DN/s$')
plt.show()

vmax_ta3 = 500
fig71, ax71 = plt.subplots(1,1,figsize=(8,6), dpi=130)
# im = ax7.imshow(target_acq_sci_cube_combined, norm=LogNorm(vmin=vmax_ta2/1000, vmax=vmax_ta2))
im = ax71.imshow(target_acq_sci_cube_combined, vmin=0, vmax=vmax_ta3)
ax71.set_title('COMBINED HD141569 TA frames')
ax71.set_xlim([82, 103])
ax71.set_ylim([55, 76])
plt.tight_layout()
cbar = fig71.colorbar(im, ax=ax71)
cbar.ax.set_title('DN/s$')
plt.show()



coords_B_round = [int(np.round(center_ta_images[0]-best_dec_A_from_ref[0]/pixsize)),
                  int(np.round(center_ta_images[1]+best_ra_A_from_ref[0]/pixsize))]
fig8, ax8 = plt.subplots(1,1,figsize=(4, 4), dpi=130)
# ax8.plot(target_acq_sci_cube_combined[center_ta_images[0]]) 
ax8.plot(target_acq_sci_cube_combined[:, center_ta_images[1]], marker='x') 
# ax8.plot(target_acq_sci_cube_combined[center_ta_images[0]-10:center_ta_images[0]+10, center_ta_images[1]], marker='x') 
# ax8.plot(14*target_acq_sci_cube_combined[coords_B_round[0] - 10:coords_B_round[0]+ 10, coords_B_round[1]], marker='x') 
ax8.set_xlim(center_ta_images[0]+ np.array([-1, 1])*10)
ax8.set_xlabel('y coordinate (pix)')
ax8.set_ylabel('Flux (DN/s)')
ax8.set_title('Cut along the y axis at the x position of A star')
# ax8.set_yscale('log')


##### Export the combined TA frames
if export_combined_ta:    
    print('\n--- Exporting the combined TA image ---')
    hdu = fits.PrimaryHDU(data=None, header = fits.getheader(selected_TA_sci_files[0]))
    hdu2 = fits.ImageHDU(target_acq_sci_cube_combined,fits.getheader(selected_TA_sci_files[0], 1))
    hdul = fits.HDUList([hdu, hdu2])
    
    path_combined_ta = os.path.join(base_root, combined_ta_folder)
    combined_ta_filename = 'HD141569_FND_TA_combined.fits'
    hdul.writeto(os.path.join(path_combined_ta, combined_ta_filename), overwrite=overWriteTAQ)
    



#%% Identify 4QPM data
print('##### IDENTIFY CORONAGRAPHIC DATA FILES ### ')

# Parameters to locate / select the datasets:
# root = '/Users/echoquet/Documents/Research/Astro/JWST_Programs/Cycle-1_ERS-1386_Hinkley/Disk_Work/2021-10_Synthetic_Datasets/2_Raw_Synthetic_Data'
# root = '/Users/echoquet/Documents/Research/Astro/JWST_Programs/Cycle-1_ERS-1386_Hinkley/Disk_Work/2021-10_Synthetic_Datasets/4_Real_JWST_Data/MIRI_Commissioning/jw01037'

cal2_folder = '4_Real_JWST_Data/MIRI_ERS/MIRI_Data/MIRI_CAL2'
path_cal2_data = os.path.join(base_root, cal2_folder)


# inspec_raw = True
inspec_cal1 = False
data_type = '*_calints.fits'

selected_data_files = np.array(glob(os.path.join(path_cal2_data, data_type)))
selected_data_files.sort()
targname_all_list = import_keyword_from_files(selected_data_files, 'TARGNAME', extension=0)
filt_all_list = import_keyword_from_files(selected_data_files, 'FILTER', extension=0)
nints_all_list = import_keyword_from_files(selected_data_files, 'NINTS', extension=0)
ngroups_all_list = import_keyword_from_files(selected_data_files, 'NGROUPS', extension=0)

print('Data type: {}'.format(data_type))
print('Number of files: {}\n'.format(len(selected_data_files)))
print('All Targets: {}'.format(np.unique(targname_all_list)))
print('All Filters: {}\n'.format(np.unique(filt_all_list)))


print('SCI Target name: {}'.format(targname_sci))
print('REF Target name: {}'.format(targname_ref))
print('Background Field Target name: {}'.format(targname_bck))
print('Filter: {}'.format(filt))

selected_filt_indices = (filt_all_list == filt)
selected_sci_indices = (targname_all_list == targname_sci)
selected_ref_indices = (targname_all_list == targname_ref)
selected_bck_indices = (targname_all_list == targname_bck)
selected_sci_files = selected_data_files[selected_filt_indices * selected_sci_indices]
selected_ref_files = selected_data_files[selected_filt_indices * selected_ref_indices]

nints_selected_sci = np.unique(nints_all_list[selected_filt_indices * selected_sci_indices])[0]
ngroups_selected_sci = np.unique(ngroups_all_list[selected_filt_indices * selected_sci_indices])[0]
selected_exp_sci_indices = (nints_all_list == nints_selected_sci) * (ngroups_all_list == ngroups_selected_sci)
selected_bck_sci_files = selected_data_files[selected_filt_indices * selected_bck_indices * selected_exp_sci_indices]

nints_selected_ref = np.unique(nints_all_list[selected_filt_indices * selected_ref_indices])[0]
ngroups_selected_ref = np.unique(ngroups_all_list[selected_filt_indices * selected_ref_indices])[0]
selected_exp_ref_indices = (nints_all_list == nints_selected_ref) * (ngroups_all_list == ngroups_selected_ref)
selected_bck_ref_files = selected_data_files[selected_filt_indices * selected_bck_indices * selected_exp_ref_indices]

print('Number of selected SCI files: {}'.format(len(selected_sci_files)))
print('Number of selected REF files: {}'.format(len(selected_ref_files)))
print('Number of selected SCI Background files: {}'.format(len(selected_bck_sci_files)))
print('Number of selected REF Background files: {}'.format(len(selected_bck_ref_files)))





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
    
    
    
    mask_cent = ~tb.create_mask(15, dim, cent=[110,120])
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
    cal1_sci_files = selected_sci_files
    cal1_ref_files = selected_ref_files
    
    # SCIENCE DATASETS COUNTRATES
    cal1_sci_cube = import_data_cube_from_files(cal1_sci_files) 
    PA_V3_sci = import_keyword_from_files(cal1_sci_files, 'PA_V3', extension=1)
    dim = (np.shape(cal1_sci_cube))[-2:]
    print('Image dimensions: {}'.format(dim))
    
    
    mask_cent = ~tb.create_mask(15, dim, cent=[110,120])
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
    nrow = int(np.ceil(len(cal1_ref_cube)/2))
    fig3, ax3 = plt.subplots(nrow, ncol,figsize=(8,6), dpi=130)
    ax3[-1, -1].axis('off')
    fig3.suptitle('CAL 1 REFSTAR  '+filt)
    images = []
    for i in range(len(cal1_ref_cube)):
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
    cal2_bck_sci_files = selected_bck_sci_files
    cal2_bck_ref_files = selected_bck_ref_files

#number of extensions: 
#len(fits.open(cal2_sci_files[0]))
# SCI
# ERR
# DQ
# INT_TIMES
# VAR_POISSON
# VAR_RNOISE
# VAR_FLAT
# ASDF


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
# PA_V3_sci = import_keyword_from_files(cal2_sci_files, 'PA_V3', extension=1)
ROLL_REF_sci = import_keyword_from_files(cal2_sci_files, 'ROLL_REF', extension=1)
V3I_YANG_sci = import_keyword_from_files(cal2_sci_files, 'V3I_YANG', extension=1)
PA_V3_sci = ROLL_REF_sci + V3I_YANG_sci
phot_MJySr_sci = import_keyword_from_files(cal2_sci_files, 'PHOTMJSR', extension=1)
phot_uJyA2_sci = import_keyword_from_files(cal2_sci_files, 'PHOTUJA2', extension=1)
scaling_values = phot_uJyA2_sci/(1000*phot_MJySr_sci)

cal2_sci_cube = import_data_cube_from_files(cal2_sci_files, scaling_list=scaling_values)
cal2_sci_cube_err = import_data_cube_from_files(cal2_sci_files, scaling_list=scaling_values, extname='ERR')
dims = (np.shape(cal2_sci_cube))[-2:]
n_sci_files = len(cal2_sci_files)
n_sci_int_all = (np.shape(cal2_sci_cube))[1]
print('Number of exposures: {}'.format(n_sci_files))
print('Number of integrations: {}'.format(n_sci_int_all))
print('Number of rolls: {}'.format(len(np.unique(PA_V3_sci))))
print('Rolls" {}'.format(np.unique(PA_V3_sci)))
print('Roll angles" {}'.format(np.diff(np.unique(PA_V3_sci))))
print('Image dimensions: {}'.format(dims))

display_grid_of_images_from_cube(cal2_sci_cube, vmax, logNorm=False, 
                                 suptitle='CAL 2 HD141569  '+filt)



# REFERENCE STAR DATASET COUNTRATES:
print('##### REF TARGET: #####')
# PA_V3_ref = import_keyword_from_files(cal2_ref_files, 'PA_V3', extension=1)
ROLL_REF_ref = import_keyword_from_files(cal2_ref_files, 'ROLL_REF', extension=1)
V3I_YANG_ref = import_keyword_from_files(cal2_ref_files, 'V3I_YANG', extension=1)
PA_V3_ref = ROLL_REF_ref + V3I_YANG_ref
phot_MJySr_ref = import_keyword_from_files(cal2_ref_files, 'PHOTMJSR', extension=1)
phot_uJyA2_ref = import_keyword_from_files(cal2_ref_files, 'PHOTUJA2', extension=1)
scaling_values_ref = phot_uJyA2_ref/(1000*phot_MJySr_ref)

cal2_ref_cube = import_data_cube_from_files(cal2_ref_files, scaling_list=scaling_values_ref)
cal2_ref_cube_err = import_data_cube_from_files(cal2_ref_files, scaling_list=scaling_values_ref, extname='ERR')
n_ref_files = len(cal2_ref_cube)
n_ref_int_all = (np.shape(cal2_ref_cube))[1]
print('Number of exposures: {}'.format(n_ref_files))
print('Number of integrations: {}'.format(n_ref_int_all))
# print('Number of rolls: {}'.format(len(np.unique(PA_V3_ref))))

display_grid_of_images_from_cube(cal2_ref_cube, vmax, logNorm=False, 
                                 suptitle='CAL 2 REFSTAR  '+filt)



# BACKGROUND FIELD  DATASET COUNTRATES:
print('##### BACKGROUND FIELD SCI : #####')
phot_MJySr_bck_sci = import_keyword_from_files(cal2_bck_sci_files, 'PHOTMJSR', extension=1)
phot_uJyA2_bck_sci = import_keyword_from_files(cal2_bck_sci_files, 'PHOTUJA2', extension=1)
scaling_values_bck_sci = phot_uJyA2_bck_sci/(1000*phot_MJySr_bck_sci)

cal2_bck_sci_cube = import_data_cube_from_files(cal2_bck_sci_files, scaling_list=scaling_values_bck_sci)
cal2_bck_sci_cube_err = import_data_cube_from_files(cal2_bck_sci_files, scaling_list=scaling_values_bck_sci, extname='ERR')
n_bck_sci_files = len(cal2_bck_sci_cube)
n_bck_sci_int_all = (np.shape(cal2_bck_sci_cube))[1]
print('Number of exposures: {}'.format(n_bck_sci_files))
print('Number of integrations: {}'.format(n_bck_sci_int_all))

display_grid_of_images_from_cube(cal2_bck_sci_cube, vmax, logNorm=False, 
                                 suptitle='CAL 2 BACKGROUND SCI  '+filt)


print('##### BACKGROUND FIELD REF : #####')
phot_MJySr_bck_ref = import_keyword_from_files(cal2_bck_ref_files, 'PHOTMJSR', extension=1)
phot_uJyA2_bck_ref = import_keyword_from_files(cal2_bck_ref_files, 'PHOTUJA2', extension=1)
scaling_values_bck_ref = phot_uJyA2_bck_ref/(1000*phot_MJySr_bck_ref)

cal2_bck_ref_cube = import_data_cube_from_files(cal2_bck_ref_files, scaling_list=scaling_values_bck_ref)
cal2_bck_ref_cube_err = import_data_cube_from_files(cal2_bck_ref_files, scaling_list=scaling_values_bck_ref, extname='ERR')
n_bck_ref_files = len(cal2_bck_ref_cube)
n_bck_ref_int_all = (np.shape(cal2_bck_ref_cube))[1]
print('Number of exposures: {}'.format(n_bck_ref_files))
print('Number of integrations: {}'.format(n_bck_ref_int_all))

display_grid_of_images_from_cube(cal2_bck_ref_cube, vmax, logNorm=False, 
                                 suptitle='CAL 2 BACKGROUND REF  '+filt)




#%% DATA COMPBINATION

#***** Defining a set of custom parameters *****
display_all = True
verbose = False

discard_first_ints = False

## Parameters for Median-filtering the bad pixels:
median_filt_thresh = 3 #sigma
median_filt_box_size = 2
median_filt_iter_max = 20


## 4QPM mask centers:
# Will eventually be in the SIAF
if filt == 'F1065C':
    fqpm_center = np.array([113.115523, 121.1844268]) - 1 #Values from J. Aguilar from commissioning. 1-indexed.
    # fqpm_center = np.array([111.89, 120.81]) #Values from Aarynn C. / Dean H. after commissioning.
elif filt == 'F1140C':
    fqpm_center = np.array([113.2361251, 120.7486661]) - 1  #Values from J. Aguilar from commissioning. 1-indexed.
    # fqpm_center = np.array([112.20, 119.99]) #Values from Aarynn C. / Dean H. after commissioning.
elif filt == 'F1550C':
    fqpm_center = np.array([114.2883897, 120.745885]) - 1   #Values from J. Aguilar from commissioning. 1-indexed.
    # fqpm_center = np.array([113.33, 119.84]) #Values from Aarynn C. / Dean H. after commissioning.


## Parameters for background subtraction: 
bck_subtraction_method = 'bck_frames_median_scaled'   # 'uniform'  'bck_frames_median' 'bck_frames_cube' 'bck_frames_combo_from_Max'
bck_mask_size = 201
bck_mask_Rin_sci = 50
bck_mask_Rin_ref = 65
# bck_mask_Rout = 105
mask_glowstick = True
glowstick_width = 9 #pix
glowstick_angle = 5 #deg


## Parameters for star-centering
rough_xpos_BC = np.array([[138.1, 139.1], [149.1,152.1]]) - 1
rough_ypos_BC = np.array([[45.9, 34.0], [49.9,38.2]]) - 1
# rough_xpos_BC = [[138.1, 139.1], [149.1,152.1]]
# rough_ypos_BC = [[45.9, 34.0], [49.9,38.2]]
centroid_box_size = 7
centroid_method = centroid_2dg



## Parameters for the PSF subtraction
psf_subtraction_method = 'classical-Ref-Averaged' # 'classical-Ref-Averaged'  'No-Subtraction'
if filt == 'F1065C':
    sci_ref_th_ratio = 64.42/1257.45  #64.02/469.45
elif filt == 'F1140C':
    sci_ref_th_ratio = 56.72/1109.31   #56.07/410.48
else:
    sci_ref_th_ratio = 30.65/595.52   #30.71/223.75


# Parameters for 4QPM mask and data combination
# TODO: use CDBS calibration mask instead!
# fqpm_center = crop_center     # TMP for synthetic dataset! use header instead
# crop_center = (np.round(fqpm_center)).astype(int)
mask_4qpm_Q = False
fqpm_width = 3 #pix
fqpm_angle = 5 #deg

cropsize = 131

# Parameters for exporting the outputs
export_tmp_filesQ = False
saveCombinedImageQ = False
overWriteQ = False
save_folder = '4_Real_JWST_Data/MIRI_ERS/MIRI_Data/MIRI_PROCESSED'

basename_sci = 'HD141569_'+filt+'_v7'
filename_output = basename_sci + '_combined.fits'


''' Reduction notes:
    To achieve the subtraction of the PSF, both SCI and REF must be background-subtracted.
    Background subtrarction is best achieved if estimated in the area close to the system,
    so it is better to crop the data first. 
    
'''



#****** Discarding the first integration of each observations  ******
if discard_first_ints: print('--- Discarding the first integrations ---')
init_int = 1 if discard_first_ints else 0
n_sci_int = n_sci_int_all - 1 if discard_first_ints else n_sci_int_all
n_ref_int = n_ref_int_all - 1 if discard_first_ints else n_ref_int_all
n_bck_sci_int = n_bck_sci_int_all - 1 if discard_first_ints else n_bck_sci_int_all
n_bck_ref_int = n_bck_ref_int_all - 1 if discard_first_ints else n_bck_ref_int_all


#****** Cleaning the bad pixels  ******
print('\n--- Cleaning the data from bad pixels ---')
# NOTE: consider using the DQ maps too

cal2_sci_cube_clean1 = median_filter_cube(cal2_sci_cube[:, init_int:], median_filt_box_size, median_filt_thresh, 
                                        iter_max=median_filt_iter_max, verbose=verbose)
cal2_ref_cube_clean1 = median_filter_cube(cal2_ref_cube[:, init_int:], median_filt_box_size, median_filt_thresh, 
                                        iter_max=median_filt_iter_max, verbose=verbose)

cal2_bck_sci_cube_clean1 = median_filter_cube(cal2_bck_sci_cube[:, init_int:], median_filt_box_size, median_filt_thresh, 
                                        iter_max=median_filt_iter_max, verbose=verbose)
cal2_bck_ref_cube_clean1 = median_filter_cube(cal2_bck_ref_cube[:, init_int:], median_filt_box_size, median_filt_thresh, 
                                        iter_max=median_filt_iter_max, verbose=verbose)

# Patch to remove nan values (as other bad pixels)
# TODO: replace this by a more general use of the Data Quality map.
# Current version puts the pixels to zero. add the option to put median value of neighbors
cal2_sci_cube_clean = clean_bad_pixel_list(cal2_sci_cube_clean1, np.argwhere(np.isnan(cal2_sci_cube_clean1)))
cal2_ref_cube_clean = clean_bad_pixel_list(cal2_ref_cube_clean1, np.argwhere(np.isnan(cal2_ref_cube_clean1)))
cal2_bck_sci_cube_clean = clean_bad_pixel_list(cal2_bck_sci_cube_clean1, np.argwhere(np.isnan(cal2_bck_sci_cube_clean1)))
cal2_bck_ref_cube_clean = clean_bad_pixel_list(cal2_bck_ref_cube_clean1, np.argwhere(np.isnan(cal2_bck_ref_cube_clean1)))


#****** Background subtraction  ******
print('\n--- Subtracting the background level ---')
print('    Subtraction method: {}'.format(bck_subtraction_method))

if bck_subtraction_method == 'uniform':  
    companion_stars_coords = fqpm_center + np.array([-90, 20])
    bck_mask_sizes = np.array((bck_mask_size, bck_mask_size))
    bck_mask_box = ~tb.create_box_mask(dims, np.round(fqpm_center).astype(int), bck_mask_sizes)
    bck_mask_sci = tb.create_mask(bck_mask_Rin_sci, dims, cent=fqpm_center) * bck_mask_box
    bck_mask_sci *= tb.create_mask(bck_mask_Rin_sci, dims, cent=companion_stars_coords)
    bck_mask_ref = tb.create_mask(bck_mask_Rin_ref, dims, cent=fqpm_center) * bck_mask_box
    if mask_glowstick:
        glowstick_mask = tb.create_glowstick_mask(dims, np.round(fqpm_center).astype(int), glowstick_width, glowstick_angle)
        bck_mask_sci = bck_mask_sci * glowstick_mask
        bck_mask_ref = bck_mask_ref * glowstick_mask
    
    if display_all:
        fig10, ax10 = plt.subplots(1,1,figsize=(8,6), dpi=130)
        ax10.imshow(bck_mask_sci*np.mean(cal2_sci_cube_clean,axis=(0,1)), norm=LogNorm(vmin=0.02, vmax=0.5))
        # ax10.imshow(bck_mask_sci, vmin=0, vmax=1)
        
        fig11, ax11 = plt.subplots(1,1,figsize=(8,6), dpi=130)
        ax11.imshow(bck_mask_ref*np.mean(cal2_ref_cube_clean,axis=(0,1)), norm=LogNorm(vmin=0.02, vmax=0.5))
        
        print('Number of NaNs in SCI: {}'.format(np.count_nonzero(np.isnan(cal2_sci_cube_clean[:, :, bck_mask_sci]))))
        print('Number of NaNs in REF: {}'.format(np.count_nonzero(np.isnan(cal2_ref_cube_clean[:, :, bck_mask_sci]))))
        
    # Estimate and subtract the background levels
    bck_level_sci = np.nanmedian(cal2_sci_cube_clean[:, :, bck_mask_sci])
    bck_level_ref = np.nanmedian(cal2_ref_cube_clean[:, :, bck_mask_ref])
    
    bck_tile_sci = np.tile(bck_level_sci, (n_sci_files, n_sci_int, dims[0], dims[1]))
    bck_tile_ref = np.tile(bck_level_ref, (n_ref_files, n_ref_int, dims[0], dims[1]))

elif bck_subtraction_method == 'bck_frames_median':
    bck_level_sci = np.median(cal2_bck_sci_cube_clean, axis=(0,1))
    bck_level_ref = np.median(cal2_bck_ref_cube_clean, axis=(0,1))
    
    bck_tile_sci = np.tile(bck_level_sci, (n_sci_files, n_sci_int, 1, 1))
    bck_tile_ref = np.tile(bck_level_ref, (n_ref_files, n_ref_int, 1, 1))

elif bck_subtraction_method == 'bck_frames_cube':
    bck_level_sci = np.median(cal2_bck_sci_cube_clean, axis=(0))
    bck_level_ref = np.median(cal2_bck_ref_cube_clean, axis=(0))
    
    bck_tile_sci = np.tile(bck_level_sci, (n_sci_files, 1, 1, 1))
    bck_tile_ref = np.tile(bck_level_ref, (n_ref_files, 1, 1, 1))

elif bck_subtraction_method == 'bck_frames_median_scaled':
    bck_level_sci = np.median(cal2_bck_sci_cube_clean, axis=(0,1))
    bck_level_ref = np.median(cal2_bck_ref_cube_clean, axis=(0,1))
    
    bck_tile_sci = np.tile(bck_level_sci, (n_sci_files, n_sci_int, 1, 1))
    bck_tile_ref = np.tile(bck_level_ref, (n_ref_files, n_ref_int, 1, 1))
    
    bck_scale_coef_sci = np.nansum((cal2_sci_cube_clean * bck_tile_sci)[:,:,180:215,30:100], axis=(2,3))/np.nansum((bck_tile_sci * bck_tile_sci)[:,:,180:215,30:100], axis=(2,3))
    bck_scale_coef_ref = np.nansum((cal2_ref_cube_clean * bck_tile_ref)[:,:,180:215,30:100], axis=(2,3))/np.nansum((bck_tile_ref * bck_tile_ref)[:,:,180:215,30:100], axis=(2,3))
    
    bck_tile_sci *= np.moveaxis(np.tile(bck_scale_coef_sci, (dims[0], dims[1], 1, 1)), [0,1], [-2,-1])
    bck_tile_ref *= np.moveaxis(np.tile(bck_scale_coef_ref, (dims[0], dims[1], 1, 1)), [0,1], [-2,-1])

    
elif bck_subtraction_method == 'bck_frames_cube_scaled':
    bck_level_sci = np.median(cal2_bck_sci_cube_clean, axis=(0))
    bck_level_ref = np.median(cal2_bck_ref_cube_clean, axis=(0))
    
    bck_tile_sci = np.tile(bck_level_sci, (n_sci_files, 1, 1, 1))
    bck_tile_ref = np.tile(bck_level_ref, (n_ref_files, 1, 1, 1))
    
    bck_scale_coef_sci = np.nansum((cal2_sci_cube_clean * bck_tile_sci)[:,:,180:215,30:100], axis=(2,3))/np.nansum((bck_tile_sci * bck_tile_sci)[:,:,180:215,30:100], axis=(2,3))
    bck_scale_coef_ref = np.nansum((cal2_ref_cube_clean * bck_tile_ref)[:,:,180:215,30:100], axis=(2,3))/np.nansum((bck_tile_ref * bck_tile_ref)[:,:,180:215,30:100], axis=(2,3))
    
    bck_tile_sci *= np.moveaxis(np.tile(bck_scale_coef_sci, (dims[0], dims[1], 1, 1)), [0,1], [-2,-1])
    bck_tile_ref *= np.moveaxis(np.tile(bck_scale_coef_ref, (dims[0], dims[1], 1, 1)), [0,1], [-2,-1])



print('Background_level SCI = {:.2e} mJy.arcsec^-2'.format(np.nanmean(bck_tile_sci)))
print('Background_level REF = {:.2e} mJy.arcsec^-2'.format(np.nanmean(bck_tile_ref)))

cal2_sci_cube_bck_sub = cal2_sci_cube_clean - bck_tile_sci
cal2_ref_cube_bck_sub = cal2_ref_cube_clean - bck_tile_ref

cal2_sci_stamps = cal2_sci_cube_bck_sub[:,:,180:215,30:100]*1000.
cal2_ref_stamps = cal2_ref_cube_bck_sub[:,:,180:215,30:100]*1000.
print('Background-subtracted dark zone SCI = {:.2f}+/-{:.2f} muJy.arcsec^-2'.format(np.nanmean(cal2_sci_stamps),np.nanstd(cal2_sci_stamps)))
print('Background-subtracted dark zone REF = {:.2f}+/-{:.2f} muJy.arcsec^-2'.format(np.nanmean(cal2_ref_stamps),np.nanstd(cal2_ref_stamps)))


display_grid_of_images_from_cube(cal2_sci_cube_bck_sub, vmax/3, #logNorm=False,
                                 suptitle='Background subtracted Integrations HD141569  '+filt)
display_grid_of_images_from_cube(cal2_ref_cube_bck_sub, vmax/3, #logNorm=False,
                                 suptitle='Background subtracted Integrations Ref Star  '+filt)


if export_tmp_filesQ:
    print('    Exporting the background subtracted data:')
    path_tmp_folder = '4_Real_JWST_Data/MIRI_ERS/MIRI_Data/MIRI_Background_subtracted'
    path_tmp_output = os.path.join(base_root, path_tmp_folder, bck_subtraction_method)
    print(path_tmp_output)
    
    for i, obs in enumerate(cal2_sci_cube_bck_sub):
        cal2_filename = os.path.basename(cal2_sci_files[i])
        filename_tmp_output = cal2_filename[:-5] + '_bckgrd-sub' + cal2_filename[-5:] 
        hdu = fits.PrimaryHDU(data=None, header = fits.getheader(cal2_sci_files[i]))
        hdu2 = fits.ImageHDU(obs,fits.getheader(cal2_sci_files[i], 1))
        hdu2.header['BUNIT']= 'mJy/arcsec2'
        hdul = fits.HDUList([hdu,hdu2])
        hdul.writeto(os.path.join(path_tmp_output, filename_tmp_output), overwrite=overWriteQ)
        
        #Combined integrations 
        filename_tmp_output_combined = cal2_filename[:-5] + '_bckgrd-sub_combined' + cal2_filename[-5:] 
        hdu3 = fits.ImageHDU(np.nanmean(obs, axis=0),fits.getheader(cal2_sci_files[i], 1))
        hdu3.header['BUNIT']= 'mJy/arcsec2'
        hdulc = fits.HDUList([hdu,hdu3])
        hdulc.writeto(os.path.join(path_tmp_output, filename_tmp_output_combined), overwrite=overWriteQ)
    
    for i, obs in enumerate(cal2_ref_cube_bck_sub):
        cal2_filename = os.path.basename(cal2_ref_files[i])
        filename_tmp_output = cal2_filename[:-5] + '_bckgrd-sub' + cal2_filename[-5:] 
        hdu = fits.PrimaryHDU(data=None, header = fits.getheader(cal2_ref_files[i]))
        hdu2 = fits.ImageHDU(obs,fits.getheader(cal2_ref_files[i], 1))
        hdu2.header['BUNIT']= 'mJy/arcsec2'
        hdul = fits.HDUList([hdu,hdu2])
        hdul.writeto(os.path.join(path_tmp_output, filename_tmp_output), overwrite=overWriteQ)
        
        #Combined integrations 
        filename_tmp_output_combined = cal2_filename[:-5] + '_bckgrd-sub_combined' + cal2_filename[-5:] 
        hdu3 = fits.ImageHDU(np.nanmean(obs, axis=0),fits.getheader(cal2_ref_files[i], 1))
        hdu3.header['BUNIT']= 'mJy/arcsec2'
        hdulc = fits.HDUList([hdu,hdu3])
        hdulc.writeto(os.path.join(path_tmp_output, filename_tmp_output_combined), overwrite=overWriteQ)
    



#****** Frame Registration  ******
print('\n--- Registring the SCI frames ---')

# Companion star fine centers
print('    FindinD the fine center coordinates of B (and C):')
fine_xpos_BC = np.empty((n_sci_files, n_sci_int, 2))
fine_ypos_BC = np.empty((n_sci_files, n_sci_int, 2))
for i in range(n_sci_files):
    for j in range(n_sci_int):
        x, y = centroid_sources(cal2_sci_cube_bck_sub[i,j], 
                                rough_xpos_BC[i], rough_ypos_BC[i], 
                                box_size=centroid_box_size, centroid_func=centroid_method)
        fine_xpos_BC[i,j,:] = x
        fine_ypos_BC[i,j,:] = y

    if export_centroid_coordinates:
        filename = os.path.basename(cal2_sci_files[i])
        fine_sci_coords_BC = np.concatenate((fine_xpos_BC[i],fine_ypos_BC[i]), axis=1)
        labels_table = ['X coord Star B','X coord Star C', 'Y coord Star B','Y coord Star C']
        df = pd.DataFrame(data=fine_sci_coords_BC, columns=labels_table)
        df.to_csv(os.path.join(path_cal2_data, 'Centroid_Position_StarsBC_{}.csv'.format(filename)))



if display_all:
    labelStars = ['B star', 'C star']
    fig1, ax1 = plt.subplots(2,n_sci_files,figsize=(4*n_sci_files, 4*2), dpi=130)
    for i in range(n_sci_files):
        for j in range(2):
            ax1[i,j].scatter(fine_xpos_BC[i,:,j], fine_ypos_BC[i,:,j], marker='x') 
            ax1[i,j].set_title('{} - Roll {}'.format(labelStars[j], i))
            ax1[i,j].set_xlim(fine_xpos_BC[i,:,j].mean()+ np.array([-1, 1])*0.01)
            ax1[i,j].set_ylim(fine_ypos_BC[i,:,j].mean()+ np.array([-1, 1])*0.01)
    plt.tight_layout()
    plt.show()
    
    fig2, ax2 = plt.subplots(n_sci_files, n_sci_int, figsize=(4*n_sci_int, 4*n_sci_files), dpi=130)
    for i in range(n_sci_files):
        for j in range(n_sci_int):
            ax2[i,j].imshow(cal2_sci_cube_bck_sub[i,j,:,:], vmin=vmin_lin, vmax=vmax/3,interpolation='nearest')
            for k in range(2):
                ax2[i,j].scatter(fine_xpos_BC[i,j,k], fine_ypos_BC[i,j,k], marker='x', color='blue',s=100)    
            ax2[i,j].set_xlim(fine_xpos_BC[i,j].mean()+ np.array([-1, 1])*15)
            ax2[i,j].set_ylim(fine_ypos_BC[i,j].mean()+ np.array([-1, 1])*15)
    plt.tight_layout()
    plt.show()

print('    -- Pointing stability from B / X-axis: std < {:.5f} pix'.format(np.max(np.std(fine_xpos_BC[:,:,0], axis=1))))
print('    -- Pointing stability from B / Y-axis: std < {:.5f} pix'.format(np.max(np.std(fine_ypos_BC[:,:,0], axis=1))))


# Patch to remove nan values before interpolating the data
# nan_poses = np.argwhere(np.isnan(cal2_sci_cube_bck_sub))
# for i, inds in enumerate(nan_poses):
#     cal2_sci_cube_bck_sub[inds[0], inds[1], inds[2], inds[3]] = 0
    
print('    Register the SCI frames:')
# Shifting the SCI images in order to place the star center at the center of the FOV
im_center = (np.array(dims)-1)//2   #fqpm_center #(np.array(dims)-1)//2
pointing_error_list = np.empty((n_sci_files, n_sci_int, 2))
cal2_sci_cube_centered = np.empty(np.shape(cal2_sci_cube_bck_sub))
for i in range(n_sci_files):
    for j in range(n_sci_int):
        estimated_xpos_A = fine_xpos_BC[i,j,0] - best_sep_A_from_ref[0] * np.sin((best_pa_A_from_ref[0] - PA_V3_sci[i])*np.pi/180)/pixsize 
        estimated_ypos_A = fine_ypos_BC[i,j,0] + best_sep_A_from_ref[0] * np.cos((best_pa_A_from_ref[0] - PA_V3_sci[i])*np.pi/180)/pixsize 
        pointing_error_list[i, j] = np.array([estimated_xpos_A - fqpm_center[1], estimated_ypos_A - fqpm_center[0]])
        message = '    -- Roll {}, int {}, star A : x = {:.3f}  y = {:.3f}    /   Error from 4qpm center: x_err = {:.3f}  y_err = {:.3f}'
        print(message.format(i, j, estimated_xpos_A, estimated_ypos_A, estimated_xpos_A-fqpm_center[1], estimated_ypos_A-fqpm_center[0]))
        
        shift_values = np.array([estimated_xpos_A - im_center[1], estimated_ypos_A - im_center[0]])
        cal2_sci_cube_centered[i,j] = tb.shift_interp(cal2_sci_cube_bck_sub[i,j], -shift_values)
        
pointing_error_means = np.mean(pointing_error_list, axis=1)
print('    -- Mean Pointing Error of A / X-axis: ROll 1: {:.3f} - ROll 2: {:.3f} pix'.format(pointing_error_means[0,0], pointing_error_means[1,0]))
print('    -- Mean Pointing Error of A / Y-axis: ROll 1: {:.3f} - ROll 2: {:.3f} pix'.format(pointing_error_means[0,1], pointing_error_means[1,1]))


display_grid_of_images_from_cube(cal2_sci_cube_centered,  vmax/3,# logNorm=False,
                                 suptitle='Centered Integrations HD141569  '+filt)


print('    Register the REF frame on the same reference pixel')
# Shifting the reference star at the center of the FOV, like the SCI frames
# here we assume that the REF images are well centered on the fqpm_center, with no error.
# this can be refined with cross-correlation, or retrieving the dither offsets
# TODO: Register the REF frames, because of small grid dither
cal2_ref_cube_centered = np.empty(np.shape(cal2_ref_cube_bck_sub))
for i in range(n_ref_files):
    for j in range(n_ref_int):
        shift_values = np.array([fqpm_center[1] - im_center[1], fqpm_center[0] - im_center[0]])
        cal2_ref_cube_centered[i,j] = tb.shift_interp(cal2_ref_cube_bck_sub[i,j], -shift_values)
        


star_center_sci = im_center
star_center_ref = im_center




### Classical subtraction of the SCI PSF
# TODO: implement subtraction with the separate REF dither frames (lower noise but better subtraction?)
if psf_subtraction_method == 'No-Subtraction':
    cal2_sci_cube_psf_sub = cal2_sci_cube_centered
elif psf_subtraction_method =='classical-Ref-Averaged':
    ref_average = sci_ref_th_ratio * np.mean(cal2_ref_cube_centered, axis=(0,1))
    cal2_sci_cube_psf_sub = cal2_sci_cube_centered - np.tile(ref_average, (n_sci_files, n_sci_int, 1, 1))

if display_all and (psf_subtraction_method =='classical-Ref-Averaged'):
    vmax = 35 #3 #max_val*0.7
    fig9, ax9 = plt.subplots(1,2,figsize=(10,6), dpi=130)
    im = ax9[0].imshow(cal2_sci_cube_centered[0,0], norm=LogNorm(vmin=vmax/500, vmax=vmax))
    im = ax9[1].imshow(ref_average, norm=LogNorm(vmin=vmax/500, vmax=vmax))
    # im = ax7.imshow(combined_image, vmin=vmin_lin, vmax=vmax)
    ax9[0].set_title('HD141569  '+filt)
    ax9[1].set_title('Reference scaled  '+filt)
    plt.tight_layout()
    cbar = fig9.colorbar(im, ax=ax9)
    cbar.ax.set_title('mJy.arcsec$^{-2}$')
    plt.show()



### Mean-combine integration within a roll (minimizes rotations)
print('--- Combining integrations within rolls ---')
combined_roll_images = np.mean(cal2_sci_cube_psf_sub, axis=1)

# Patch to remove nan values
# nan_poses = np.argwhere(np.isnan(combined_roll_images))
# for i, inds in enumerate(nan_poses):
#     combined_roll_images[inds[0], inds[1], inds[2]] = 0

display_grid_of_images_from_cube(combined_roll_images, vmax, #logNorm=False,
                                 suptitle='ROLLS HD141569  '+filt,imsize=4)



### Derotate each roll
print('--- Derotate and combine rolls ---')
derotated_images = np.empty(np.shape(combined_roll_images))
for i in range(n_sci_files):
    derotated_images[i] = tb.frame_rotate_interp(combined_roll_images[i], -PA_V3_sci[i], center=star_center_sci)


#TODO : Make the 4QPM mask centered on float values.
if mask_4qpm_Q:
    fqpm_masks = np.empty((n_sci_files, dims[0], dims[1]), dtype=bool)
    for i, roll_angle in enumerate(PA_V3_sci):
        fqpm_masks[i] = tb.create_fqpm_mask(dims, im_center, fqpm_width, fqpm_angle-roll_angle)
    derotated_images[~fqpm_masks] = np.nan

display_grid_of_images_from_cube(derotated_images, vmax, #logNorm=False,
                                 suptitle='Rerotated ROLLS HD141569  '+filt,imsize=4)



### Combine the two rolls and crop it
combined_image_full = np.nanmean(derotated_images, axis=0)
combined_image = tb.resize(combined_image_full, [cropsize,cropsize], cent=np.round(star_center_sci).astype(int)) 


print('Total flux in image: {:.3f} mJy'.format(np.nansum(combined_image)*0.11*0.11))

# vmin = 0.1 #median_val*0.8
vmax = 35 #3 #max_val*0.7
fig7, ax7 = plt.subplots(1,1,figsize=(8,6), dpi=130)
im = ax7.imshow(combined_image, norm=LogNorm(vmin=vmax/500, vmax=vmax))
# im = ax7.imshow(combined_image, vmin=vmin_lin, vmax=vmax)
ax7.set_title('COMBINED HD141569  '+filt)
plt.tight_layout()
cbar = fig7.colorbar(im, ax=ax7)
cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()


if saveCombinedImageQ:
    print('--- Saving the reduced images ---')
    path_output = os.path.join(base_root, save_folder)
    os.makedirs(path_output, exist_ok=overWriteQ)
    
    hdu = fits.PrimaryHDU(data=None, header = fits.getheader(cal2_sci_files[0]))
    #Put the data back in the initial untits to match the unit in the header
    
    hdu2 = fits.ImageHDU(combined_image,fits.getheader(cal2_sci_files[0], 1))
    hdu2.header['BUNIT']= 'mJy/arcsec2'
    hdul = fits.HDUList([hdu,hdu2])
    
    hdul.writeto(os.path.join(path_output, filename_output), overwrite=overWriteQ)




#%% Analyze images

save_folder = '4_Real_JWST_Data/MIRI_ERS/MIRI_Data/MIRI_PROCESSED'

filt_list = ['F1065C', 'F1140C', 'F1550C']
basename_sci = 'HD141569_'+filt+'_v4'
filename_output = basename_sci + '_combined.fits'


path_output = os.path.join(base_root, save_folder)

combined_image_F1065C = fits.getdata(os.path.join(path_output, 'HD141569_F1065C_v4_combined.fits'))
combined_image_F1140C = fits.getdata(os.path.join(path_output, 'HD141569_F1140C_v4_combined.fits'))
combined_image_F1550C = fits.getdata(os.path.join(path_output, 'HD141569_F1550C_v4_combined.fits'))


vmax = 15 #3 #max_val*0.7
fig12, ax12 = plt.subplots(1,3,figsize=(6,18), dpi=130)
im1 = ax12[0].imshow(combined_image_F1065C, norm=LogNorm(vmin=vmax/500, vmax=vmax))#, cmap='bone')
im2 = ax12[1].imshow(combined_image_F1140C, norm=LogNorm(vmin=vmax/500, vmax=vmax))#, cmap='pink')
im3 = ax12[2].imshow(combined_image_F1550C, norm=LogNorm(vmin=vmax/500, vmax=vmax))#, cmap='gist_heat')
ax12[0].set_title('F1065C')
ax12[1].set_title('F1140C')
ax12[2].set_title('F1550C')
plt.tight_layout()
# cbar = fig12.colorbar(im, ax=ax12[0])
# cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()



vmax = 15 #3 #max_val*0.7
fig7, ax7 = plt.subplots(1,1,figsize=(8,6), dpi=130)
im = ax7.imshow(combined_image_F1140C - combined_image_F1065C, norm=LogNorm(vmin=vmax/500, vmax=vmax))
# im = ax7.imshow(combined_image_F1140C - combined_image_F1065C, vmin=vmin_lin, vmax=vmax)
ax7.set_title('F1140C - F1065C  ')
plt.tight_layout()
cbar = fig7.colorbar(im, ax=ax7)
cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()


vmax = 15 #3 #max_val*0.7
fig7, ax7 = plt.subplots(1,1,figsize=(8,6), dpi=130)
im = ax7.imshow(combined_image_F1550C - combined_image_F1140C, norm=LogNorm(vmin=vmax/500, vmax=vmax))
# im = ax7.imshow(combined_image_F1140C - combined_image_F1065C, vmin=vmin_lin, vmax=vmax)
ax7.set_title('F1550C - F1140C  ')
plt.tight_layout()
cbar = fig7.colorbar(im, ax=ax7)
cbar.ax.set_title('mJy.arcsec$^{-2}$')
plt.show()
