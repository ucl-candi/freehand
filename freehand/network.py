
import torch


def build_model(model_function, in_frames, out_dim):
    """
    :param model_function: classification model
    """
    if model_function.__name__[:12] == "efficientnet":
        model = model_function(weights=None)
        model.features[0][0] = torch.nn.Conv2d(
            in_channels  = in_frames, 
            out_channels = model.features[0][0].out_channels, 
            kernel_size  = model.features[0][0].kernel_size, 
            stride       = model.features[0][0].stride, 
            padding      = model.features[0][0].padding, 
            bias         = model.features[0][0].bias
        )
        model.classifier[1] = torch.nn.Linear(
            in_features   = model.classifier[1].in_features,
            out_features  = out_dim
        )
    elif model_function.__name__[:6] == "resnet":
        model = model_function()
        model.conv1 = torch.nn.Conv2d(
            in_channels  = in_frames, 
            out_channels = model.features[0][0].out_channels, 
            kernel_size  = model.features[0][0].kernel_size, 
            stride       = model.features[0][0].stride, 
            padding      = model.features[0][0].padding, 
            bias         = model.features[0][0].bias
        )
        model.fc = torch.nn.Linear(
            in_features   = model.fc.in_features,
            out_features  = out_dim
        )
    else:
        raise("Unknown model.")
    
    return model