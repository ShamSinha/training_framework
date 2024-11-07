import pyrootutils
import hydra
import torch
import pandas as pd
from tqdm.auto import tqdm
from src.common.nn_modules.nets.slicewise.v3_slicelabels import (
    V3Net,
    modify_model_state_dict,
    logsumexp_attention
)
from torch.utils.data import DataLoader


root = pyrootutils.setup_root(
    search_from="./", indicator=[".git", "pyproject.toml"], pythonpath=True, dotenv=True
)

hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize("../configs/datamodule/", version_base="1.2")
data_module = hydra.compose("inference_bleed_cls.yaml")

datamodule = hydra.utils.instantiate(data_module)
test_ds = datamodule.test_ds

ckpt_path = "/data_nas5/qer/shubham/ich_checkpoints/ich_classification/runs/2023-08-01_04-40-47/checkpoints/epoch=9_step=260_val_loss=0.99.ckpt"
k = torch.load(ckpt_path, map_location="cpu")
state_dict = k["state_dict"]
new_state_dict = modify_model_state_dict(state_dict, "net.backbone", "backbone")
new_state_dict = modify_model_state_dict(
    new_state_dict, "net.classification_head", "classification_head"
)

model = V3Net()

model.load_state_dict(new_state_dict)
model.to("cuda:2")
model.eval()

li = []


activation = torch.nn.Softmax(dim=1)

for i in tqdm(range(len(test_ds))):
    try:
        data = test_ds[i]
        image = data["image"].unsqueeze(0)

        z_size = image.size(2)

        # image (b,3,z,224,224)

        classification_out = []

        dataset = []
        for y in torch.split(image, 1,2):
            dataset.append(y.squeeze())

        dataloader = DataLoader(dataset , batch_size= 64, pin_memory=True)

        for i_batch, batch in enumerate(dataloader):
            sdh_outputs = model(batch.swapaxes(0,1).unsqueeze(0).to("cuda:2"))
            # torch.cuda.empty_cache()      
            # logger.debug(torch.cuda.memory_allocated(device=torch.device("cuda:1")))
            classification_out.append(sdh_outputs["slice_label"].detach())

        classification_out = torch.concat(classification_out)

        # logger.debug(torch.cuda.memory_allocated(device=torch.device("cuda:1")))

        slice_output = torch.split(classification_out, z_size)  # list of (b, z, 2)
        slice_output = torch.stack(slice_output)
        slice_output = torch.swapaxes(slice_output, 1 ,-1)  ## torch.Size([b, 2, z])

        scan_output = logsumexp_attention(slice_output)  # (b,2)


        sdh_model_score = activation(scan_output)
        sdh_score = sdh_model_score.detach().cpu().numpy()   
        li.append(
            {"StudyUID": data["study_uid"], "sdh_model_score_crop": sdh_score[0][1]}
        )
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)

df = pd.DataFrame.from_records(li)
df.to_csv("v3net_crop_new_test_set_v2.csv", index=False)

df_already_saved = pd.read_csv("v3net_crop_new_test_set.csv")

pd.concat([df_already_saved, df]).to_csv("v3net_crop_new_test_set.csv", index=False)
