#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:32:04 2024

@author: echoquet
"""

import numpy as np
import scipy.ndimage as ndimage



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


def create_box_mask(dims, center, box_sizes):
    square_mask = np.ones(dims)
    
    y1i, y1f = resize_list_limits(dims[0], box_sizes[0], cent1=center[0])
    x1i, x1f = resize_list_limits(dims[1], box_sizes[1], cent1=center[1])
    
    square_mask[y1i:y1f, x1i:x1f] *= 0 
    
    return square_mask.astype(bool)


def create_fqpm_mask(dims, center, width, angle):
    fqpm_mask_tmp = np.ones(dims)
    fqpm_mask_tmp[center[0] - width//2 : center[0] + width//2, :] = 0
    fqpm_mask_tmp[:, center[1] - width//2 : center[1] + width//2] = 0
    fqpm_mask = np.round(frame_rotate_interp(fqpm_mask_tmp, angle, center=center, cval=1))
    return fqpm_mask.astype(bool)


def create_glowstick_mask(dims, center, width, angle):
    glowstick_mask_tmp = np.ones(dims)
    glowstick_mask_tmp[center[0] - width//2 : center[0] + width//2, :] = 0
    # glowstick_mask_tmp[:, center[1] - width//2 : center[1] + width//2] =1
    glowstick_mask = np.round(frame_rotate_interp(glowstick_mask_tmp, angle, center=center, cval=1))
    return glowstick_mask.astype(bool)



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



def info(array, label='', show_mean=True, show_med=False, show_min=False, show_max=False):
    print(label)
    if show_mean: print('    Mean    : {}'.format(np.mean(array))) 
    if show_med: print('    Median  : {}'.format(np.median(array))) 
    if show_min: print('    Min     : {}'.format(np.min(array)))
    if show_max: print('    Max     : {}'.format(np.max(array)))
    print('    Std dev : {}'.format(np.std(array)))
    print('    Max-Min : {}'.format(np.max(array) - np.min(array)))
