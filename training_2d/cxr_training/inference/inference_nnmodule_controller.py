import torch.nn as nn
from cxr_training.nnmodule.nnmodule_controller import LitModel
from cxr_training.data.datasets.test_dataset import Test_dataset
import torch
import os
import omegaconf


def load_model(path):
    data = torch.load(path)
    sd = data["state_dict"]
    args = data["args"]
    data["args"]["path"]["checkpoint_path"] = ""
    data["args"]["model"]["load_prt"] = False
    data["args"]["model"]["pretrained_ckpt"] = ""
    args = omegaconf.OmegaConf.create(args)
    model = LitModel(args)

    print(model.load_state_dict(sd, strict=True))
    model = model.eval()
    model = nn.DataParallel(model)
    return model.cuda(), args


def get_dataloader(args):
    dataset = Test_dataset(args)
    batch_size = args.trainer.batch_size
    num_workers = args.trainer.num_workers
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2,
    )
    return test_dataloader


def set_cuda_gpus(gpus):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
