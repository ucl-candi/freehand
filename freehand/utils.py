
import torch


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
