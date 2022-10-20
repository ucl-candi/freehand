
import numpy as np
import SimpleITK as sitk
import cv2
import openpyxl


def read_frame_transform(filename, scan_crop_indices, resample_factor=1, delay_tform=0):

    get_transform = lambda ii, image : list(map(float,image.GetMetaData('Seq_Frame{:04d}_ProbeToTrackerTransform'.format(ii)).split(' ')))
    get_transform_status = lambda ii, image : image.GetMetaData('Seq_Frame{:04d}_ProbeToTrackerTransformStatus'.format(ii))=="OK"

    image = sitk.ReadImage(filename)  # imageIO="MetaImageIO"    
    frames = sitk.GetArrayFromImage(image)  # [no_frames,h,w] - nb. size = image.GetSize()  # [w,h,no_frames]

    if (scan_crop_indices[1]+delay_tform) > frames.shape[0]:
        scan_crop_indices[1] = frames.shape[0]
        print("WARNING: scan_crop_indices has been reduced due to delayed transform.")

    frames = frames[scan_crop_indices[0]:scan_crop_indices[1],...] 
    tforms = [get_transform(ii,image) for ii in range(scan_crop_indices[0]+delay_tform,scan_crop_indices[1]+delay_tform)]
    tforms = np.stack([np.array(t,dtype=np.float32).reshape(4,4) for t in tforms],axis=0)

    tform_status = [get_transform_status(ii,image) for ii in range(scan_crop_indices[0]+delay_tform,scan_crop_indices[1]+delay_tform)]
    if not all(tform_status):
        frames = frames[tform_status,:,:]
        tforms = tforms[tform_status,:,:]
    
    if resample_factor != 1:
        frames = np.stack(
            [frame_resize(frames[ii,...], resample_factor) for ii in range(frames.shape[0])], 
            axis=0
            )  # resample on 2D frames
    
    return frames, tforms, tform_status


def frame_resize(image, resample_factor):
    # frame_resize = lambda im : cv2.resize(im, None, fx=1/RESAMPLE_FACTOR, fy=1/RESAMPLE_FACTOR, interpolation = cv2.INTER_LINEAR)
    return cv2.resize(
        image, 
        dsize=None, 
        fx=1/resample_factor, 
        fy=1/resample_factor, 
        interpolation = cv2.INTER_LINEAR
        )


def read_scan_crop_indices_file(filename, num_scans):
    fid_xls = openpyxl.load_workbook(filename).active
    return list(fid_xls.iter_rows(values_only=True))[1:num_scans+1]
