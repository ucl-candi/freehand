
import torch,os
from matplotlib import pyplot as plt


def pair_samples(num_samples, num_pred):
    """
    :param num_samples:
    :param num_pred: number of the (last) samples, for which the transformations are predicted
        For each "pred" frame, pairs are formed with every one previous frame 
    """
    return torch.tensor([[n0,n1] for n1 in range(num_samples-num_pred,num_samples) for n0 in range(n1)])



def type_dim(label_pred_type, num_points=None, num_pairs=1):
    type_dim_dict = {
        "transform": 12,
        "parameter": 6,
        "point": num_points*3
    }
    return type_dim_dict[label_pred_type] * num_pairs  # num_points=self.image_points.shape[1]), num_pairs=self.pairs.shape[0]


def reference_image_points(image_size, density=2):
    """
    :param image_size: (x, y), used for defining default grid image_points
    :param density: (x, y), point sample density in each of x and y, default n=2
    """
    if isinstance(density,int):
        density=(density,density)

    image_points = torch.cartesian_prod(
        torch.linspace(0,image_size[0]-1,density[0]),
        torch.linspace(0,image_size[1]-1,density[1])
        ).t()  # transpose to 2-by-n
    
    image_points = torch.cat([
        image_points, 
        torch.zeros(1,image_points.shape[1]),
        torch.ones(1,image_points.shape[1])
        ], axis=0)
    
    return image_points

def save_model(model,epoch,NUM_EPOCHS,FREQ_SAVE,SAVE_PATH):
    # save the model at current epoch, and keep the number of models at 4
    if epoch in range(0, NUM_EPOCHS, FREQ_SAVE):

        torch.save(model.state_dict(), os.path.join(os.getcwd(),SAVE_PATH, 'saved_model', 'model_epoch%08d' % epoch))
        print('Model parameters saved.')
        list_dir = os.listdir(os.path.join(os.getcwd(),SAVE_PATH, 'saved_model'))
        saved_models = [i for i in list_dir if i.startswith('model_epoch')]
        if len(saved_models)>4:
            os.remove(os.path.join(os.getcwd(),SAVE_PATH,'saved_model',sorted(saved_models)[0]))

def save_best_network(epoch_label,model,running_loss_val, running_dist_val, val_loss_min, val_dist_min,SAVE_PATH):
    '''
    :param epoch_label: current epoch
    :param running_loss_val: validation loss of this epoch
    :param running_dist_val: validation distance of this epoch
    :param val_loss_min: min of previous validation losses
    :param val_dist_min: min of previous validation distances
    :return:
    '''

    if running_loss_val < val_loss_min:
        val_loss_min = running_loss_val
        file_name = os.path.join(os.getcwd(),SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ best validation loss result - epoch %s: -------------\n' % (str(epoch_label)))
        
        torch.save(model.state_dict(), os.path.join(os.getcwd(),SAVE_PATH, 'saved_model', 'best_validation_loss_model' ))
        print('Best validation loss parameters saved.')
    
    if running_dist_val < val_dist_min:
        val_dist_min = running_dist_val
        file_name = os.path.join(os.getcwd(),SAVE_PATH, 'config.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write('------------ best validation dist result - epoch %s: -------------\n' % (str(epoch_label)))
        
        torch.save(model.state_dict(), os.path.join(os.getcwd(),SAVE_PATH, 'saved_model', 'best_validation_dist_model'))
        print('Best validation dist parameters saved.')
    
    return val_loss_min, val_dist_min

def add_scalars(writer,epoch, loss_dists):
    # loss and average distance in training and val
    train_epoch_loss = loss_dists['train_epoch_loss']
    train_epoch_dist = loss_dists['train_epoch_dist'].mean()
    epoch_loss_val = loss_dists['val_epoch_loss']
    epoch_dist_val = loss_dists['val_epoch_dist'].mean()
    
    writer.add_scalars('loss', {'train_loss': train_epoch_loss},epoch)
    writer.add_scalars('loss', {'val_loss': epoch_loss_val},epoch)
    writer.add_scalars('dist', {'train_dist': train_epoch_dist}, epoch)
    writer.add_scalars('dist', {'val_dist': epoch_dist_val}, epoch)


def scan_plot(gt,axs,color,width = 4, scatter = 8, legend_size=50,legend = None):

    gx_all, gy_all, gz_all = [gt[:, ii, :] for ii in range(3)]
    for i,ax in enumerate(axs):
        ax.scatter(gx_all[...,0], gy_all[...,0], gz_all[...,0],  alpha=0.5, c = color, s=scatter, label=legend)
        ax.scatter(gx_all[...,1], gy_all[...,1], gz_all[...,1],  alpha=0.5,c = color, s=scatter)
        ax.scatter(gx_all[...,2], gy_all[...,2], gz_all[...,2],  alpha=0.5, c = color,s=scatter)
        ax.scatter(gx_all[...,3], gy_all[...,3], gz_all[...,3],  alpha=0.5,c = color, s=scatter)
        # plot the first frame and the last frame
        ax.plot(gt[0,0,0:2], gt[0,1,0:2], gt[0,2,0:2], color, linewidth = width)
        ax.plot(gt[0,0,[1,3]], gt[0,1,[1,3]], gt[0,2,[1,3]], color, linewidth = width) 
        ax.plot(gt[0,0,[3,2]], gt[0,1,[3,2]], gt[0,2,[3,2]], color, linewidth = width) 
        ax.plot(gt[0,0,[2,0]], gt[0,1,[2,0]], gt[0,2,[2,0]], color, linewidth = width)
        ax.plot(gt[-1,0,0:2], gt[-1,1,0:2], gt[-1,2,0:2], color, linewidth = width)
        ax.plot(gt[-1,0,[1,3]], gt[-1,1,[1,3]], gt[-1,2,[1,3]], color, linewidth = width) 
        ax.plot(gt[-1,0,[3,2]], gt[-1,1,[3,2]], gt[-1,2,[3,2]], color, linewidth = width) 
        ax.plot(gt[-1,0,[2,0]], gt[-1,1,[2,0]], gt[-1,2,[2,0]], color, linewidth = width)


        ax.axis('equal')
        ax.grid(False)
        ax.legend(fontsize = legend_size,markerscale = 5,scatterpoints = 5)
        # ax.axis('off')
        ax.set_xlabel('x',fontsize=legend_size)
        ax.set_ylabel('y',fontsize=legend_size)
        ax.set_zlabel('z',fontsize=legend_size)
        plt.rc('xtick', labelsize=legend_size)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=legend_size)    # fontsize of the tick labels
        

def scan_plot_gt_pred(gt,pred,saved_name,color,width = 4, scatter = 8, legend_size=50,legend = None):
    # plot the scan in 3D

    fig = plt.figure(figsize=(35,15))
    axs=[]
    for i in range(1):
        axs.append(fig.add_subplot(1,1,i+1,projection='3d'))
    plt.tight_layout()

    scan_plot(gt,axs,'g',width, scatter, legend_size,legend = 'gt')
    scan_plot(pred,axs,'r',width, scatter, legend_size,legend = 'pred')

    plt.savefig(saved_name +'.png')
    plt.close()

def data_pairs_cal_label(num_frames):
    # obtain the data_pairs to compute the tarnsfomration between frames and the reference (first) frame
    
    return torch.tensor([[0,n0] for n0 in range(num_frames)])