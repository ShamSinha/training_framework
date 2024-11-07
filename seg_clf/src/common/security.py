import os
import tempfile
from pathlib import Path
from typing import Union

import torch
from qure_adhoc.enc.qryptography import Crypter


def torch_load_enc(checkpoint_file: Union[Path, str], **kwargs):
    """
    It will first check whether checkpoint is encrypted or not
    and if it is encrypted then decrypt it and load it.
    Args:
        checkpoint_file: checkpoints file path
    Return:
        loaded checkpoint (in memory)
    """
    kwargs["map_location"] = "cpu"
    if os.path.exists(checkpoint_file) and checkpoint_file.endswith(".enc"):
        crypter = Crypter()
        with tempfile.NamedTemporaryFile() as named_tmp_file:
            crypter.decrypt(checkpoint_file, out_filename=named_tmp_file.name)
            checkpoint = torch.load(named_tmp_file.name, **kwargs)
    elif os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, **kwargs)
    else:
        raise FileNotFoundError(f"{checkpoint_file} not found")
    return checkpoint
