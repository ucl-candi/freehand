
import csv

import numpy as np

def read_calib_matrices(filename_calib, resample_factor):
    # T{image->tool} = T{image_mm -> tool} * T{image_pix -> image_mm} * T{resampled_image_pix -> image_pix}
    tform_calib = np.empty((8,4), np.float32)
    with open(filename_calib) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')    
        for ii, row in enumerate(csv_reader):
            tform_calib[ii,:] = (list(map(float,row)))
    return tform_calib[4:8,:] @ tform_calib[0:4,:] @ np.array([[resample_factor,0,0,0], [0,resample_factor,0,0], [0,0,1,0], [0,0,0,1]], np.float32)