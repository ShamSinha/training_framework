import timm
import torch
from timm.models.layers.conv2d_same import Conv2dSame

from .conv3d_same import Conv3dSame
from .layers import LayerNorm3d, get_ConvNeXtBlock3D


def convert_3d(module):

    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0])
        )
    elif isinstance(module, torch.nn.Dropout2d):
        module_output = torch.nn.Dropout3d(
            p=module.p,
            inplace=module.inplace,
        )
    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
        module_output.weight = torch.nn.Parameter(
            module.weight.unsqueeze(-1).repeat(1, 1, 1, 1, module.kernel_size[0])
        )

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size
            if isinstance(module.kernel_size, int)
            else module.kernel_size[0],
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, timm.models.layers.LayerNorm2d):
        module_output = LayerNorm3d(
            num_channels=module.normalized_shape[0],
            eps=module.eps,
            affine=module.elementwise_affine,
        )
        if module.elementwise_affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
    elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
        module_output = torch.nn.AdaptiveAvgPool3d(
            output_size=module.output_size,
        )
    elif isinstance(module, torch.nn.AdaptiveMaxPool2d):
        module_output = torch.nn.AdaptiveMaxPool3d(
            output_size=module.output_size, return_indices=module.return_indices
        )
    elif isinstance(module, timm.models.convnext.ConvNeXtBlock):
        module_output = get_ConvNeXtBlock3D(module)
    for name, child in module.named_children():
        module_output.add_module(name, convert_3d(child))
    del module

    return module_output
