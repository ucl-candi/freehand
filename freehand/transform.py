
import torch

''' Geometric transformation used in transforms:
Use left-multiplication:
    T{image->world} = T{tool->world} * T{image->tool} 
    T{image->tool} is the calibration matrix: T_calib
    pts{tool} = T{image->tool} * pts{image}
    pts{world} = T{image->world} * pts{image}
        where pts{image1} are four (corners) or five (and centre) image points in the image coordinate system.

The ImagePointTransformLoss 
    - a pair of "past" and "pred" images, image0 and image1
    - a pair of ground-truth (GT) and predicted (Pred) transformation
    pts{world} = T{tool0->world} * T_calib * pts{image0}
    pts{world} = T{tool1->world} * T_calib * pts{image1}
    => T{tool0->world} * T_calib * pts{image0} = T{tool1->world} * T_calib * pts{image1}
    => pts{image0} = T_calib^(-1) * T{tool0->world}^(-1) * T{tool1->world} * T_calib * pts{image1}
    => pts{image0} = T_calib^(-1) * T(tool1->tool0) * T_calib * pts{image1}

Denote world-coordinates-independent transformation, 
    T(tool1->tool0) = T{tool0->world}^(-1) * T{tool1->world}, 
    which can be predicted and its GT can be obtained from the tracker.
When image_coords = True (pixel): 
    => loss = ||pts{image0}^{GT} - pts{image0}^{Pred}||
When image_coords = False (mm):
    => loss = ||pts{tool0}^{GT} - pts{tool0}^{Pred}||
        where pts{tool0} = T(tool1->tool0) * T_calib * pts{image1}

Accumulating transformations using TransformAccumulation class:
    e.g. working in tool coordinates, so round-off error does not accumulate on calibration
    Given often-predicted T(tool1->tool0) and then T(tool2->tool1):
        T(tool2->tool0) = T(tool1->tool0) * T(tool2->tool1)
    Similarly then: 
        pts{image0} = T_calib^(-1) * T(tool2->tool0) * T_calib * pts{image2}

'''


class LabelTransform():

    def __init__(
        self, 
        label_type, 
        pairs,
        image_points=None,
        in_image_coords=False,
        tform_image_to_tool=None
        ):        
        """
        :param label_type: {"point", "parameter"}
        :param pairs: data pairs, between which the transformations are constructed, returned from pair_samples
        :param image_points: 2-by-num or 3-by-num in pixel, used for computing loss
        :param in_image_coords: label points coordinates, True (in pixel) or False (in mm), default is False
        :param tform_image_to_tool: transformation from image coordinates to tool coordinates, usually obtained from calibration
        """


        self.label_type = label_type
        self.pairs = pairs
        
        
        if self.label_type=="point":        
            self.image_points = image_points
            self.tform_image_to_tool = tform_image_to_tool
            # pre-compute reference points in tool coordinates
            self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)
            self.in_image_coords = in_image_coords
            if self.in_image_coords:  # pre-compute the inverse
                self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool)
            
            self.call_function = self.to_points
        
        elif self.label_type=="transform":  # not in use
            self.call_function = self.to_transform_t2t   

        elif self.label_type=="parameter":
            self.call_function = self.to_parameters   

        else:
            raise('Unknown label_type!')
    

    def __call__(self, *args, **kwargs):
        return self.call_function(*args, **kwargs)        
    

    def to_points(self, tforms, tforms_inv=None):
        _tforms = self.to_transform_t2t(tforms, tforms_inv)
        if self.in_image_coords:
            _tforms = torch.matmul(self.tform_tool_to_image[None,None,...], _tforms)  # tform_tool1_to_image0
        return torch.matmul(_tforms, self.image_points_in_tool)[:,:,0:3,:]  # [batch,num_pairs,(x,y,z,1),num_image_points]

    
    def to_transform_t2t(self, tforms, tforms_inv):

        if tforms_inv is None:
            tforms_inv = torch.linalg.inv(tforms)
        
        tforms_world_to_tool0 = tforms_inv[:,self.pairs[:,0],:,:]
        tforms_tool1_to_world = tforms[:,self.pairs[:,1],:,:]
        return torch.matmul(tforms_world_to_tool0, tforms_tool1_to_world)  # tform_tool1_to_tool0


    def to_parameters(self, tforms, tforms_inv):
        raise('Not implemented.')  # TODO: SVD
    


class PredictionTransform():

    def __init__(
        self, 
        pred_type, 
        label_type, 
        num_pairs=None,
        image_points=None,
        in_image_coords=False,
        tform_image_to_tool=None
        ):
        
        """
        :param pred_type = {"transform", "parameter", "point"}
        :param label_type = {"point", "parameter"}
        :param pairs: data pairs, between which the transformations are constructed, returned from pair_samples
        """


        self.pred_type = pred_type
        self.label_type = label_type
        self.num_pairs = num_pairs


        if self.pred_type=="point":
            if self.label_type=="point":
                self.call_function = self.point_to_point
            elif self.label_type=="parameter":
                raise('Not implemented.')  #self.call_function = self.point_to_parameter    
            else:
                raise('Unknown label_type!')
            
        else:            
            self.image_points = image_points
            self.tform_image_to_tool = tform_image_to_tool
            self.in_image_coords = in_image_coords
            # pre-compute reference points in tool coordinates
            self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)
            self.in_image_coords = in_image_coords
            if self.in_image_coords:  # pre-compute the inverse
                self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool)

            if self.pred_type=="parameter":
                if self.label_type=="point":
                    self.call_function = self.parameter_to_point
                elif self.label_type=="parameter":
                    self.call_function = self.parameter_to_parameter
                else:
                    raise('Unknown label_type!')

            elif self.pred_type=="transform":
                if self.label_type=="point":
                    self.call_function = self.transform_to_point
                elif self.label_type=="parameter":
                    raise('Not implemented.')  #self.call_function = self.transform_to_parameter
                else:
                    raise('Unknown label_type!')
            
            else:
                raise('Unknown pred_type!')
        

    def __call__(self, outputs):
        preds = outputs.reshape((outputs.shape[0],self.num_pairs,-1))
        return self.call_function(preds)


    def point_to_point(self,pts):
        return pts.reshape(pts.shape[0],self.num_pairs,-1,3)
    
    def transform_to_point(self,_tforms):
        last_rows = torch.cat([
            torch.zeros_like(_tforms[...,0])[...,None,None].expand(-1,-1,1,3),
            torch.ones_like(_tforms[...,0])[...,None,None]
            ], axis=3)
        _tforms = torch.cat((
            _tforms.reshape(-1,self.num_pairs,3,4),
            last_rows
            ), axis=2)
        if self.in_image_coords:
            _tforms = torch.matmul(self.tform_tool_to_image[None,None,...], _tforms)  # tform_tool1_to_image0
        return torch.matmul(_tforms, self.image_points_in_tool)[:,:,0:3,:]  # [batch,num_pairs,(x,y,z,1),num_image_points]

    def parameter_to_parameter(self,params):
        return params
    
    def parameter_to_point(self,params):
        _tforms = self.param_to_transform(params)
        if self.in_image_coords:
            _tforms = torch.matmul(self.tform_tool_to_image[None,None,...], _tforms)  # tform_tool1_to_image0
        return torch.matmul(_tforms, self.image_points_in_tool)[:,:,0:3,:]  # [batch,num_pairs,(x,y,z,1),num_image_points]
    
    @staticmethod
    def param_to_transform(params):
        # params: (batch,ch,6), "ch": num_pairs, "6":rx,ry,rz,tx,ty,tz
        cos_x = torch.cos(params[...,0])
        sin_x = torch.sin(params[...,0])
        cos_y = torch.cos(params[...,1])
        sin_y = torch.sin(params[...,1])
        cos_z = torch.cos(params[...,2])
        sin_z = torch.sin(params[...,2])
        return torch.cat((
            torch.stack([cos_y*cos_z, sin_x*sin_y*cos_z-cos_x*sin_z,  cos_x*sin_y*cos_z-sin_x*sin_z,  params[...,3]], axis=2)[:,:,None,:],
            torch.stack([cos_y*sin_z, sin_x*sin_y*sin_z-cos_x*cos_z,  cos_x*sin_y*sin_z-sin_x*cos_z,  params[...,4]], axis=2)[:,:,None,:],
            torch.stack([-sin_y,      sin_x*cos_y,                    cos_x*cos_y,                    params[...,5]], axis=2)[:,:,None,:],
            torch.cat((torch.zeros_like(params[...,0:3])[...,None,:], torch.ones_like(params[...,0])[...,None,None]), axis=3)
            ), axis=2)



class ImageTransform:
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def __call__(self,images):
        images = (images.to(torch.float32) - self.mean) / self.std
        return images + torch.normal(mean=0, std=torch.ones_like(images)*0.01)


class TransformAccumulation:

    def __init__(
        self, 
        image_points=None,
        in_image_coords=False,  # TODO: accepting image1->image0, rather than tool1->tool0
        tform_image_to_tool=None
        ):
        if in_image_coords:
            raise("not implemented.")
        
        self.image_points = image_points
        self.tform_image_to_tool = tform_image_to_tool
        self.tform_tool_to_image = torch.linalg.inv(self.tform_image_to_tool) # pre-compute the inverse
        # pre-compute reference points in tool coordinates
        self.image_points_in_tool = torch.matmul(self.tform_image_to_tool,self.image_points)

    def __call__(self, tform_tool1_to_tool0, tform_tool2_to_tool1):  
        # safer to input and output explicitly the previous transform: tform_tool1_to_tool0
        # TODO: __call__ should branch if working in image coordinates
        # TODO: for single transformations for now
        last_row = torch.zeros_like(tform_tool2_to_tool1[None,0,:])
        last_row[0,3] = 1.0
        tform_tool2_to_tool1 = torch.cat((tform_tool2_to_tool1,last_row), axis=0)
        tform_tool2_to_tool0 = torch.matmul(tform_tool1_to_tool0, tform_tool2_to_tool1)
        tform_tool2_to_image0 = torch.matmul(self.tform_tool_to_image, tform_tool2_to_tool0)
        return torch.matmul(tform_tool2_to_image0, self.image_points_in_tool)[0:3,:], tform_tool2_to_tool0
