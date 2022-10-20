
import os

import h5py
import numpy as np

from utils import read_frame_transform, read_scan_crop_indices_file


FLAG_SCANFILE = False

DIR_RAW = os.path.join(os.path.expanduser("~"), "workspace/forearm_US")
NUM_SCANS = 12
RESAMPLE_FACTOR = 4
PATH_SAVE = os.path.join(os.path.expanduser("~"), "workspace/")
DELAY_TFORM = 4  # delayed tform from temporal calibration

fh5_frames = h5py.File(os.path.join(PATH_SAVE,'frames_res{}'.format(RESAMPLE_FACTOR)+".h5"),'a')
if FLAG_SCANFILE:
    fh5_scans = h5py.File(os.path.join(PATH_SAVE,'scans_res{}'.format(RESAMPLE_FACTOR)+".h5"),'a')

folders_subject = [f for f in os.listdir(DIR_RAW) if os.path.isdir(os.path.join(DIR_RAW, f))]
num_frames = np.zeros((len(folders_subject),NUM_SCANS),dtype=np.uint16)
for i_sub, folder in enumerate(folders_subject):

    fn_xls = [f for f in os.listdir(os.path.join(DIR_RAW, folder)) if f.endswith(".xlsx")]
    if len(fn_xls) != 1: 
        raise('Should contain 1 xlsx file in folder "{}"'.format(folder))
    scan_crop_idx = read_scan_crop_indices_file(os.path.join(DIR_RAW, folder, fn_xls[0]), NUM_SCANS)  # TBA: checks for 1) item name/order and if the 12 scans are complete

    fn_mha = [f for f in os.listdir(os.path.join(DIR_RAW, folder)) if f.endswith(".mha")]
    fn_mha.sort(reverse=False)  # ordered in acquisition time
    if len(fn_mha) != NUM_SCANS: raise('Should contain 12 mha files in folder "{}"'.format(folder))    

    for i_scan, fn in enumerate(fn_mha):

        frames, tforms, _ = read_frame_transform(
            filename = os.path.join(DIR_RAW, folder, fn),
            scan_crop_indices = scan_crop_idx[i_scan][1:3],
            resample_factor = RESAMPLE_FACTOR,
            delay_tform = DELAY_TFORM
        )
        tforms_inv = np.linalg.inv(tforms)  # pre-compute the iverse

        num_frames[i_sub,i_scan] = frames.shape[0]
        for i_frame in range(frames.shape[0]):
            fh5_frames.create_dataset('/sub%03d_scan%02d_frame%04d' % (i_sub,i_scan,i_frame), frames.shape[1:3], dtype=frames.dtype, data=frames[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_tform%04d' % (i_sub,i_scan,i_frame), tforms.shape[1:3], dtype=tforms.dtype, data=tforms[i_frame,...])
            fh5_frames.create_dataset('/sub%03d_scan%02d_tform_inv%04d' % (i_sub,i_scan,i_frame), tforms.shape[1:3], dtype=tforms.dtype, data=tforms_inv[i_frame,...])
        if FLAG_SCANFILE:
            fh5_scans.create_dataset('/sub%03d_frames%02d' % (i_sub,i_scan), frames.shape, dtype=frames.dtype, data=frames)
            fh5_scans.create_dataset('/sub%03d_tforms%02d' % (i_sub,i_scan), tforms.shape, dtype=tforms.dtype, data=tforms)
            fh5_scans.create_dataset('/sub%03d_tforms_inv%02d' % (i_sub,i_scan), tforms.shape, dtype=tforms.dtype, data=tforms_inv)

fh5_frames.create_dataset('num_frames', num_frames.shape, dtype=num_frames.dtype, data=num_frames)
fh5_frames.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
fh5_frames.create_dataset('frame_size', 2, data=frames.shape[1:3])
fh5_frames.flush()
fh5_frames.close()

if FLAG_SCANFILE:
    fh5_scans.create_dataset('num_frames', num_frames.shape, dtype=num_frames.dtype, data=num_frames)
    fh5_scans.create_dataset('sub_folders', len(folders_subject), data=folders_subject)
    fh5_scans.create_dataset('frame_size', 2, data=frames.shape[1:3])
    fh5_scans.flush()
    fh5_scans.close()

print('Saved at: %s' % os.path.join(PATH_SAVE,'frames_res{}'.format(RESAMPLE_FACTOR)+".h5"))
if FLAG_SCANFILE:
    print('Saved at: %s' % os.path.join(PATH_SAVE,'scans_res{}'.format(RESAMPLE_FACTOR)+".h5"))
