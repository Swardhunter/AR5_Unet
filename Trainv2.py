# The code is importing necessary modules and classes for training a UNet model using PyTorch
from Functions import MyCall,UNetLightning,CustomDataset
import torch 
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
import ssl
import par 
torch.set_float32_matmul_precision("medium")
ssl._create_default_https_context = ssl._create_unverified_context
import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
from torch.utils.data import ConcatDataset

logger = TensorBoardLogger("tb_logsv2",name=par.encoder)
dataset = CustomDataset(par.tensorspath)
#dataset2 = CustomDataset('/home/mskenawi/SW/PreProcessed_Tensorsv4')
#dataset3 = CustomDataset('/home/mskenawi/SW/PreProcessed_Tensorsv5')

#datasets_to_concat = [dataset3, dataset2]
#dataset = ConcatDataset(datasets_to_concat)

print (len(dataset))


# # Load the checkpoint
checkpoint = torch.load(par.chkpointpath)
print(par.chkpointpath)

# # Modify the hyperparameters
checkpoint['hyper_parameters']['dataset'] = dataset
#checkpoint['hyper_parameters']['batch_size']= 40

# Save the checkpoint
torch.save(checkpoint, "/home/mskenawi/SW/trial2.ckpt")
model = UNetLightning.load_from_checkpoint("/home/mskenawi/SW/trial2.ckpt")


trainer = Trainer(logger=logger,
        max_epochs=50,
        devices='auto',
        strategy="auto",
        log_every_n_steps=1,
        callbacks=[MyCall(), EarlyStopping(monitor="IoU", patience=10, mode="max")],
    )
    

tuner = Tuner(trainer)

# #     # Run learning rate finder
lr_finder = tuner.lr_find(model)

# #     # Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()



trainer.fit(model)
trainer.test(model)