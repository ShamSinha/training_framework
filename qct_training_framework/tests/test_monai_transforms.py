import torch
import numpy as np
from monai.transforms import Compose
from monai.transforms.croppad.array import CenterSpatialCrop
import numpy as np
import torch
import monai.transforms as Transform

def test_FixZDim():
    transform = FixZDim(keys=["image"], target_z=32)
    input_data = {"image": torch.randn(40, 256, 256)}
    output_data = transform(input_data)
    assert output_data["image"].shape == (32, 256, 256)
    assert isinstance(output_data["image"], torch.Tensor)

def test_ResizeImage():
    transform = ResizeImage(size=(128, 128), keys=["image", "label"], interpolation={"image": "bicubic", "label": "nearest"})
    input_data = {"image": np.random.rand(64, 64), "label": np.random.randint(0, 2, size=(64, 64))}
    output_data = transform(input_data)
    assert output_data["image"].shape == (128, 128)
    assert isinstance(output_data["image"], np.ndarray)
    assert output_data["label"].shape == (128, 128)
    assert isinstance(output_data["label"], np.ndarray)

def test_ExpandDim():
    transform = ExpandDim(keys=["image"])
    input_data = {"image": np.random.rand(64, 64)}
    output_data = transform(input_data)
    assert output_data["image"].shape == (1, 64, 64)
    assert isinstance(output_data["image"], np.ndarray)

def test_FixZDim_with_random_crop():
    transform = FixZDim(keys=["image"], target_z=32, random_center_crop=True)
    input_data = {"image": torch.randn(40, 256, 256)}
    output_data = transform(input_data)
    assert output_data["image"].shape == (32, 256, 256)
    assert isinstance(output_data["image"], torch.Tensor)

def test_Compose():
    transform1 = CenterSpatialCrop(roi_size=[224, 224, 224])
    transform2 = ExpandDim()
    transform = Compose([transform1, transform2])
    input_data = {"image": np.random.rand(256, 256, 256)}
    output_data = transform(input_data)
    assert output_data["image"].shape == (1, 224, 224, 224)
    assert isinstance(output_data["image"], np.ndarray)

def test_random_affine():
    # Test that the transform doesn't change the data when prob = 0
    data = {"image": torch.rand(3, 64, 64)}
    keys = ["image"]
    transform = RandomAffine(keys=keys, prob=0)
    result = transform(data)
    assert torch.all(torch.eq(data["image"], result["image"]))

    # Test that the transform changes the data when prob > 0
    data = {"image": torch.rand(3, 64, 64)}
    keys = ["image"]
    transform = RandomAffine(keys=keys, prob=1)
    result = transform(data)
    assert not torch.all(torch.eq(data["image"], result["image"]))

def test_downscale_3d():
    # Test that the transform downscales the data by the specified factor
    data = {"image": torch.rand(3, 64, 64, 64)}
    keys = ["image"]
    transform = DownScale3D(keys=keys, downscale_factor_z=2, downscale_factor_y=2, downscale_factor_x=2)
    result = transform(data)
    assert result["image"].shape == (3, 32, 32, 32)

def test_load_dcm_data():
    # Test that the transform loads DICOM data into PyTorch Tensors
    dcm_paths = ["path/to/dcm1", "path/to/dcm2", "path/to/dcm3"]
    data = {"dcm": dcm_paths}
    dcm_data = [np.ones((64, 64)), np.ones((64, 64)), np.ones((64, 64))]
    dcmread_mock = lambda path: type('dcm', (), {'pixel_array': dcm_data.pop(0)})()
    keys = ["dcm"]
    transform = LoadDcmData(keys=keys)
    with monkeypatch.context() as m:
        m.setattr("pydicom.dcmread", dcmread_mock)
        result = transform(data)
    assert torch.all(torch.eq(result["dcm"], torch.Tensor(np.array(dcm_data))))

