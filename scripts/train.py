
import os
import torch
from torchvision.models import efficientnet_b1
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append(os.getcwd())

from freehand.loader import SSFrameDataset
from freehand.network import build_model
from freehand.loss import PointDistance
from data.calib import read_calib_matrices
from freehand.transform import LabelTransform, PredictionTransform, ImageTransform
from freehand.utils import *



os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RESAMPLE_FACTOR = 4
FILENAME_CALIB = "data/calib_matrix.csv"
FILENAME_FRAMES = os.path.join(os.getcwd(), "data/Freehand_US_data", 'frames_res{}'.format(RESAMPLE_FACTOR)+".h5")

## settings
# training
MINIBATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = int(1e6)
# algorithm
PRED_TYPE = "parameter"  # {"transform", "parameter", "point"}
LABEL_TYPE = "point"  # {"point", "parameter"}
NUM_SAMPLES = 10
SAMPLE_RANGE = 10
NUM_PRED = 9

FREQ_INFO = 10
FREQ_SAVE = 100
# saved model path
saved_results = 'seq_len' + str(NUM_SAMPLES) + '__' + 'lr' + str(LEARNING_RATE)\
        + '__pred_type_'+str(PRED_TYPE) + '__label_type_'+str(LABEL_TYPE) 
SAVE_PATH = os.path.join('results', saved_results)
if not os.path.exists(os.path.join(os.getcwd(),SAVE_PATH)):
    os.makedirs(os.path.join(os.getcwd(),SAVE_PATH))
if not os.path.exists(os.path.join(os.getcwd(),SAVE_PATH, 'saved_model')):
    os.makedirs(os.path.join(os.getcwd(),SAVE_PATH, 'saved_model'))
writer = SummaryWriter(os.path.join(os.getcwd(),SAVE_PATH))


dataset_all = SSFrameDataset(
    filename_h5=FILENAME_FRAMES, 
    num_samples=NUM_SAMPLES, 
    sample_range=SAMPLE_RANGE
    )


## setup for cross-validation
dset_folds = dataset_all.partition_by_ratio(
    ratios = [1]*5, 
    randomise=True, 
    subject_level=False
    )
for (idx, ds) in enumerate(dset_folds):
    ds.write_json(os.path.join(SAVE_PATH,"fold_{:02d}.json".format(idx)))  # see test.py for file reading

dset_train = dset_folds[0]+dset_folds[1]+dset_folds[2]+dset_folds[3]
dset_val = dset_folds[4]

train_loader = torch.utils.data.DataLoader(
    dset_train,
    batch_size=MINIBATCH_SIZE, 
    shuffle=True,
    num_workers=8
    )

val_loader = torch.utils.data.DataLoader(
    dset_val,
    batch_size=1, 
    shuffle=False,
    num_workers=1
    )


## loss
tform_calib = torch.tensor(read_calib_matrices(filename_calib=FILENAME_CALIB, resample_factor=RESAMPLE_FACTOR), device=device)
data_pairs = pair_samples(NUM_SAMPLES, NUM_PRED).to(device)
if (PRED_TYPE=="point") or (LABEL_TYPE=="point"):
    image_points = reference_image_points(dset_train.frame_size,2).to(device)
    pred_dim = type_dim(PRED_TYPE, image_points.shape[1], data_pairs.shape[0])
    label_dim = type_dim(LABEL_TYPE, image_points.shape[1], data_pairs.shape[0])
else:
    image_points = None
    pred_dim = type_dim(PRED_TYPE)
    label_dim = type_dim(LABEL_TYPE)

transform_label = LabelTransform(
    LABEL_TYPE, 
    pairs=data_pairs, 
    image_points=image_points, 
    tform_image_to_tool=tform_calib
    )
transform_prediction = PredictionTransform(
    PRED_TYPE, 
    LABEL_TYPE, 
    num_pairs=data_pairs.shape[0], 
    image_points=image_points, 
    tform_image_to_tool=tform_calib
    )

if LABEL_TYPE == "point":
    criterion = torch.nn.MSELoss()
    metrics = PointDistance()
elif LABEL_TYPE == "parameter":
    criterion = torch.nn.L1Loss()
    metrics = torch.nn.L1Loss()


## network
model = build_model(
    efficientnet_b1, 
    in_frames = NUM_SAMPLES, 
    out_dim = pred_dim
    ).to(device)


## train
val_loss_min = 1e10
val_dist_min = 1e10
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
for epoch in range(NUM_EPOCHS):
    train_epoch_loss = 0
    train_epoch_dist = 0
    for step, (frames, tforms, tforms_inv) in enumerate(train_loader):

        frames, tforms, tforms_inv = frames.to(device), tforms.to(device), tforms_inv.to(device)    
        frames = frames/255    
        labels = transform_label(tforms, tforms_inv)

        optimiser.zero_grad()

        outputs = model(frames)
        preds = transform_prediction(outputs)

        loss = criterion(preds, labels)
        dist = metrics(preds, labels)

        loss.backward()
        optimiser.step()
        train_epoch_loss += loss.item()
        train_epoch_dist += dist

    train_epoch_loss /= (step + 1)
    train_epoch_dist /= (step + 1)

    if epoch in range(0, NUM_EPOCHS, FREQ_INFO):
        print('[Epoch %d] train-loss=%.3f, train-dist=%.3f' % (epoch, train_epoch_loss, train_epoch_dist.mean()))
        if dist.shape[0]>1:
            print('%.2f '*dist.shape[0] % tuple(dist))
    

    # validation    
    if epoch in range(0, NUM_EPOCHS, FREQ_SAVE):

        model.train(False)

        running_loss_val = 0
        running_dist_val = 0
        for step, (fr_val, tf_val, tf_val_inv) in enumerate(val_loader):
            fr_val, tf_val, tf_val_inv = fr_val.to(device), tf_val.to(device), tf_val_inv.to(device) 
            fr_val = fr_val/255    
            la_val = transform_label(tf_val, tf_val_inv)
            out_val = model(fr_val)    
            pr_val = transform_prediction(out_val)
            running_loss_val += criterion(pr_val, la_val)
            running_dist_val += metrics(pr_val, la_val)

        running_loss_val /= (step+1)
        running_dist_val /= (step+1)
        print('[Epoch %d] val-loss=%.3f, val-dist=%.3f' % (epoch, running_loss_val, running_dist_val.mean()))
        if running_dist_val.shape[0]>1:
            print('%.2f '*running_dist_val.shape[0] % tuple(running_dist_val)) 
        
        # save model at current epoch
        save_model(model,epoch,NUM_EPOCHS,FREQ_SAVE,SAVE_PATH)
        # save best validation model
        val_loss_min, val_dist_min = save_best_network(epoch, model, running_loss_val, running_dist_val.mean(), val_loss_min, val_dist_min,SAVE_PATH)
        # add to tensorboard
        loss_dists = {'train_epoch_loss': train_epoch_loss, 'train_epoch_dist': train_epoch_dist,'val_epoch_loss':running_loss_val,'val_epoch_dist':running_dist_val}
        add_scalars(writer, epoch, loss_dists)
        
        model.train(True)        
