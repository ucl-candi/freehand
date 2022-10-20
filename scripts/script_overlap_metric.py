
import os

from freehand.fileio import read_json_points
from freehand.metric import frame_volume_overlap

# load ground-truth and predicted point sets
foldername = os.path.join(os.path.expanduser('~/'),'Scratch/overlap/testing_val_results')
ps_true = read_json_points(os.path.join(foldername,'y_actual_overlap_LH_Para_S_0000.json'))
ps_pred = read_json_points(os.path.join(foldername,'y_predicted_overlap_LH_Para_S_0000.json'))

DSC = frame_volume_overlap(ps_true[...,:100], ps_pred[...,:50])
