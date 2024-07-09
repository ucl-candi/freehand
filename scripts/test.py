
import os
from matplotlib import pyplot as plt
import torch
from torchvision.models import efficientnet_b1
import sys
sys.path.append(os.getcwd())

from freehand.loader import SSFrameDataset
from freehand.network import build_model
from data.calib import read_calib_matrices
from freehand.transform import LabelTransform, TransformAccumulation, PredictionTransform
from freehand.utils import *


os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = 'cpu'

RESAMPLE_FACTOR = 4
FILENAME_CALIB = "data/calib_matrix.csv"
FILENAME_FRAMES = os.path.join(os.getcwd(), "data/Freehand_US_data", 'frames_res{}'.format(RESAMPLE_FACTOR)+".h5")

## algorithm parameters
PRED_TYPE = "parameter"  # {"transform", "parameter", "point"}
LABEL_TYPE = "point"  # {"point", "parameter"}
NUM_SAMPLES = 10
SAMPLE_RANGE = 10
NUM_PRED = 9
LEARNING_RATE = 1e-4

saved_results = 'seq_len' + str(NUM_SAMPLES) + '__' + 'lr' + str(LEARNING_RATE)\
        + '__pred_type_'+str(PRED_TYPE) + '__label_type_'+str(LABEL_TYPE) 
SAVE_PATH = os.path.join('results', saved_results)
if not os.path.exists(os.path.join(os.getcwd(),SAVE_PATH,'plotting')):
    os.makedirs(os.path.join(os.getcwd(),SAVE_PATH,'plotting'))
FILENAME_TEST = "fold_04.json"
FILENAME_WEIGHTS = "best_validation_dist_model"


## create the validation/test set loader
dset_test = SSFrameDataset.read_json(os.path.join(SAVE_PATH,FILENAME_TEST))
# TODO compare with train parameters before changing to all frames
if NUM_SAMPLES != dset_test.num_samples:
    raise("Inconsistent num_samples.")
if SAMPLE_RANGE != dset_test.sample_range:
    raise("Inconsistent sample_range.")
dset_test = SSFrameDataset.read_json(os.path.join(SAVE_PATH,FILENAME_TEST), num_samples=-1)

data_pairs = pair_samples(NUM_SAMPLES, NUM_PRED)
frame_points = reference_image_points(dset_test.frame_size,2).to(device)
tform_calib = torch.tensor(read_calib_matrices(
    filename_calib=FILENAME_CALIB, 
    resample_factor=RESAMPLE_FACTOR
    ), device=device)


transform_prediction = PredictionTransform(
    PRED_TYPE, 
    'transform', 
    num_pairs=data_pairs.shape[0], 
    image_points=frame_points, 
    tform_image_to_tool=tform_calib
    )
accumulate_prediction = TransformAccumulation(
    image_points=frame_points, 
    tform_image_to_tool=tform_calib
    )

pred_dim = type_dim(PRED_TYPE, frame_points.shape[1], data_pairs.shape[0])
label_dim = type_dim(LABEL_TYPE, frame_points.shape[1], data_pairs.shape[0])


## load the model
model = build_model(
    efficientnet_b1, 
    in_frames = NUM_SAMPLES, 
    out_dim = pred_dim
    ).to(device)
model.load_state_dict(torch.load(os.path.join(SAVE_PATH,'saved_model',FILENAME_WEIGHTS), map_location=torch.device(device)))
model.train(False)


## inference
for i_scan in range(len(dset_test)):
        
    SCAN_INDEX = i_scan  # plot one scan
    PAIR_INDEX = 0  # which prediction to use
    START_FRAME_INDEX = 0  # starting frame - the reference 

    frames, tforms, tforms_inv = dset_test[SCAN_INDEX]
    frames, tforms, tforms_inv = (torch.tensor(t).to(device) for t in [frames,tforms,tforms_inv])
    
    predictions_allpts = torch.zeros((frames.shape[0],3,frame_points.shape[-1]))

    data_pairs_all = data_pairs_cal_label(frames.shape[0])
    transform_label = LabelTransform(
            "point",
            pairs=data_pairs_all,
            image_points=frame_points,
            tform_image_to_tool=tform_calib
            )
    labels_allpts = torch.squeeze(transform_label(tforms[None,...], tforms_inv[None,...]))
    
    
    idx_f0 = START_FRAME_INDEX # this is the reference starting frame for network prediction 
    idx_p0 = idx_f0 + data_pairs[PAIR_INDEX][0] # this is the reference frame for transformaing others to
    idx_p1 = idx_f0 + data_pairs[PAIR_INDEX][1]
    interval_pred = data_pairs[PAIR_INDEX][1] - data_pairs[PAIR_INDEX][0]


    tform_1to0 = torch.eye(4) 
    predictions_allpts[idx_f0+1] = labels_allpts[0]
    while 1:
        frames_test = frames[idx_f0:idx_f0+NUM_SAMPLES,...]
        frames_test = frames_test/255
        outputs_test = model(frames_test.unsqueeze(0)) 

        tform_2to1 =  transform_prediction(outputs_test)[0,PAIR_INDEX]
        preds_val, tform_1to0 = accumulate_prediction(tform_1to0, tform_2to1)
        predictions_allpts[idx_f0+1] = preds_val.cpu()
        
        idx_f0 += interval_pred
        idx_p1 += interval_pred 
        if (idx_f0+NUM_SAMPLES) > frames.shape[0]:
            break

    if NUM_SAMPLES > 2:
        predictions_allpts[idx_f0:,...] = predictions_allpts[idx_f0-1].expand(predictions_allpts[idx_f0:,...].shape[0],-1,-1)

    # plot trajactory
    scan_plot_gt_pred(labels_allpts.numpy(),predictions_allpts.detach().numpy(),SAVE_PATH +'/'+'plotting'+'/' + str(i_scan),color = 'g',width = 4, scatter = 8, legend_size=50, legend = 'GT')

