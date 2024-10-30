import hydra
import logging
from tqdm.auto import tqdm

logging.basicConfig(filename='example_e2e8_v4.log', format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize("hydra_configs/data/", version_base="1.2")
data_config = hydra.compose("det_data_window.yaml")

data_config.dataloader.ts.include = data_config.dataloader.ts.include[4:]
data_config.dataloader.ts.ds_paths = data_config.dataloader.ts.ds_paths[4:]
data_config.dataloader.ts.individual_lung = data_config.dataloader.ts.individual_lung[4:]


from time import time
import multiprocessing as mp
from torch.utils.data import DataLoader


for batch_size in [4,6] : #range(6,12,2) : 
    for num_workers in [14]: 

        data_config.dataloader.ts.dl.batch_size = batch_size
        data_config.dataloader.ts.dl.num_workers = num_workers
        
        data = hydra.utils.instantiate(data_config.dataloader)
        
        train_loader = data.train_dl()
        start = time()
        for epoch in range(1, 3):
            for i, data in tqdm(enumerate(train_loader)):
                pass
        end = time()
        logger.info("Finish with:{} second, num_workers={} , batch_size= {}".format(end - start, num_workers,batch_size))
