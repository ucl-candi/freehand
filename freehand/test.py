
import os

from matplotlib import pyplot as plt
import torch
from torchvision.models import efficientnet_b1

from loader import SSFrameDataset
from network import build_model
from data.calib import read_calib_matrices
from transform import LabelTransform, TransformAccumulation, ImageTransform
from utils import pair_samples, reference_image_points, type_dim


os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = 'cpu'

RESAMPLE_FACTOR = 4
FILENAME_CALIB = "data/calib_matrix.csv"
FILENAME_FRAMES = os.path.join(os.path.expanduser("~"), "workspace", 'frames_res{}'.format(RESAMPLE_FACTOR)+".h5")

## algorithm parameters
PRED_TYPE = "transform"  # {"transform", "parameter", "point"}
LABEL_TYPE = "point"  # {"point", "parameter"}
NUM_SAMPLES = 5
SAMPLE_RANGE = 5
NUM_PRED = 3
SAVE_PATH = "results"
FILENAME_VAL = "fold_00.json"
FILENAME_WEIGHTS = "model_epoch00100000"


## create the validation/test set loader
dset_val = SSFrameDataset.read_json(os.path.join(SAVE_PATH,FILENAME_VAL))
# TODO compare with train parameters before changing to all frames
if NUM_SAMPLES != dset_val.num_samples:
    raise("Inconsistent num_samples.")
if SAMPLE_RANGE != dset_val.sample_range:
    raise("Inconsistent sample_range.")
dset_val = SSFrameDataset(filename_h5=dset_val.filename, num_samples=-1)

data_pairs = pair_samples(NUM_SAMPLES, NUM_PRED)
#
tform_calib = torch.tensor(read_calib_matrices(
    filename_calib=FILENAME_CALIB, 
    resample_factor=RESAMPLE_FACTOR
    ), device=device)

# using GT: transform pixel_points from current image to starting (reference) image 0
pixel_points = reference_image_points(dset_val.frame_size,dset_val.frame_size).to(device)
transform_label = LabelTransform(
    label_type = "point",  # for plotting  
    pairs=torch.tensor([0,1])[None,], 
    image_points=pixel_points, 
    in_image_coords=True,  # for plotting  
    tform_image_to_tool=tform_calib
    )

# using prediction: transform frame_points from current image to starting (reference) image 0
frame_points = reference_image_points(dset_val.frame_size,2).to(device)
accumulate_prediction = TransformAccumulation(
    image_points=frame_points, 
    tform_image_to_tool=tform_calib
    )

pred_dim = type_dim(PRED_TYPE, frame_points.shape[1], data_pairs.shape[0])
label_dim = type_dim(LABEL_TYPE, frame_points.shape[1], data_pairs.shape[0])

transform_image = ImageTransform(mean=32, std=32)


## load the model
model = build_model(
    efficientnet_b1, 
    in_frames = NUM_SAMPLES, 
    out_dim = pred_dim
    ).to(device)
model.load_state_dict(torch.load(os.path.join(SAVE_PATH,FILENAME_WEIGHTS), map_location=torch.device(device)))
model.train(False)


## inference
SCAN_INDEX = 5  # plot one scan
PAIR_INDEX = 8  # which prediction to use
START_FRAME_INDEX = 10  # starting frame - the reference 

frames, tforms, tforms_inv = dset_val[SCAN_INDEX]
frames, tforms, tforms_inv = (torch.tensor(t).to(device) for t in [frames,tforms,tforms_inv])
idx_f0 = START_FRAME_INDEX # this is the reference starting frame for network prediction 
idx_p0 = idx_f0 + data_pairs[PAIR_INDEX][0] # this is the reference frame for transformaing others to
idx_p1 = idx_f0 + data_pairs[PAIR_INDEX][1]
interval_pred = data_pairs[PAIR_INDEX][1] - data_pairs[PAIR_INDEX][0]

# plot the reference
px, py, pz = [pixel_points[ii,].reshape(dset_val.frame_size[0],dset_val.frame_size[1]).cpu() for ii in range(3)]
pix_intensities = (frames[idx_p0,...,None].float()/255).cpu().expand(-1,-1,3).numpy()
fx, fy, fz = [frame_points[ii,].reshape(2,2).cpu() for ii in range(3)]

ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(px,py,pz, facecolors=pix_intensities, linewidth=0, edgecolors=None, antialiased=True)
ax.plot_surface(fx,fy,fz, edgecolor='y', linewidth=2, alpha=0.2, antialiased=True)


tform_1to0 = torch.eye(4) 
while 1:
    # prediction -> points in image coords
    frames_val = frames[idx_f0:idx_f0+NUM_SAMPLES,...]
    frames_val = transform_image(frames_val)    
    outputs_val = model(frames_val.unsqueeze(0)) 
    tform_2to1 =  outputs_val.reshape(data_pairs.shape[0],3,4)[PAIR_INDEX,:,:].cpu()
    preds_val, tform_1to0 = accumulate_prediction(tform_1to0, tform_2to1)
    fx, fy, fz = [preds_val[ii,].reshape(2,2).cpu().detach().numpy() for ii in range(3)]
    ax.plot_surface(fx,fy,fz, edgecolor='y', linewidth=2, alpha=0.1, antialiased=True)
    # label -> points in image coords, wrt. the "unchanged" reference starting frame, idx_p0
    tforms_val, tforms_inv_val = (t[[idx_p0,idx_p1],...] for t in [tforms, tforms_inv])  
    label = transform_label(tforms_val.unsqueeze(0), tforms_inv_val.unsqueeze(0))
    px, py, pz = [label[:,:,ii,:].reshape(dset_val.frame_size[0],dset_val.frame_size[1]).cpu() for ii in range(3)]
    pix_intensities = (frames[idx_p1,...,None].float()/255).cpu().expand(-1,-1,3).numpy()
    ax.plot_surface(px,py,pz, facecolors=pix_intensities, linewidth=0, edgecolors=None, antialiased=True)
    # update for the next prediction
    idx_f0 += interval_pred
    idx_p1 += interval_pred 
    if (idx_f0+NUM_SAMPLES) > frames.shape[0]:
        break

plt.show()
