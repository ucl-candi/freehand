
import os

import h5py
import numpy as np
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from calib import read_calib_matrices


RESAMPLE_FACTOR = 4
FILENAME_CALIB = "calib_matrix.csv"
FLAG_SCANFILE = False


## T{image->tool} = T{image_mm -> tool} * T{image_pix -> image_mm} * T{resampled_image_pix -> image_pix}
tform_calib = read_calib_matrices(filename_calib=FILENAME_CALIB, resample_factor=RESAMPLE_FACTOR)


## using the frame file
FILENAME_FRAMES = os.path.join(os.path.expanduser("~"), "workspace", 'frames_res{}'.format(RESAMPLE_FACTOR)+".h5")
idx_sub = 15
idx_scan = 5
fh5_frames = h5py.File(FILENAME_FRAMES,'r')  
num_frames = fh5_frames['num_frames'][()]
frame_size = fh5_frames['frame_size'][()]
# sub_folders = fh5_frames['sub_folders']

# plot images
for ii in range(0,num_frames[idx_sub,idx_scan],25):
    print(ii)
    frame = fh5_frames['/sub{:03d}_scan{:02d}_frame{:04d}'.format(idx_sub,idx_scan,ii)]
    tform = fh5_frames['/sub{:03d}_scan{:02d}_tform{:04d}'.format(idx_sub,idx_scan,ii)]
    print(tform[()])
    plt.figure()
    plt.imshow(frame,cmap="gray")
    plt.show()

# plot transformations in world (w): T{image->world} = T{tool->world} * T{image->tool} 
cx0, cy0 = np.meshgrid(
    range(frame_size[0]),  # x
    range(frame_size[1]),  # y
    indexing='xy')  # z
cz0 = np.zeros_like(cx0)

ax = plt.figure().add_subplot(projection='3d')
for ii in range(0,num_frames[idx_sub,idx_scan],5):

    frame = fh5_frames['/sub{:03d}_scan{:02d}_frame{:04d}'.format(idx_sub,idx_scan,ii)]
    tform = fh5_frames['/sub{:03d}_scan{:02d}_tform{:04d}'.format(idx_sub,idx_scan,ii)]

    tform_i2w = tform @ tform_calib
    pix_intensities = np.repeat(frame[()].transpose()[...,None]/255,3,2)

    ps_w = tform_i2w @ np.vstack((cx0.flatten(),cy0.flatten(),cz0.flatten(),np.ones(cx0.size)))
    cx, cy, cz = [ps_w[ii,...].reshape(frame_size[1],frame_size[0]) for ii in range(3)]
    ax.plot_surface(cx,cy,cz, facecolors=pix_intensities, linewidth=0, edgecolors=None)

plt.show()



if not FLAG_SCANFILE: 
    raise SystemExit(0)

## using the scans file
FILENAME_SCANS = os.path.join(os.path.expanduser("~"), "workspace", 'scans_res{}'.format(RESAMPLE_FACTOR)+".h5")
# read data from a scan from one subject
idx_sub = 15
idx_scan = 5
fh5_scans = h5py.File(FILENAME_SCANS,'r')  
frames = fh5_scans['/sub{:03d}_frames{:02d}'.format(idx_sub,idx_scan)]
tforms = fh5_scans['/sub{:03d}_tforms{:02d}'.format(idx_sub,idx_scan)]

# plot images
for ii in range(0,frames.shape[0],20):
    print(ii)
    print(tforms[ii,...])
    plt.figure()
    plt.imshow(frames[ii,...],cmap="gray")
plt.show()

# plot transformation # in world (w): T{image->world} = T{tool->world} * T{image->tool} 
# frame_size = (frames.shape[1],frames.shape[2])  # (x,y)
cx0, cy0 = np.meshgrid(
    range(frames.shape[1]),  # x
    range(frames.shape[2]),  # y
    indexing='xy')  # z
cz0 = np.zeros_like(cx0)

ax = plt.figure().add_subplot(projection='3d')
for ii in range(0,frames.shape[0],20):

    tform_i2w = tforms[ii,...] @ tform_calib
    pix_intensities = np.repeat(frames[ii,...].transpose()[...,None]/255,3,2)

    ps_w = tform_i2w @ np.vstack((cx0.flatten(),cy0.flatten(),cz0.flatten(),np.ones(cx0.size)))
    cx, cy, cz = [ps_w[ii,...].reshape(frames.shape[2],frames.shape[1]) for ii in range(3)]
    ax.plot_surface(cx,cy,cz, facecolors=pix_intensities, linewidth=0, edgecolors=None)

plt.show()