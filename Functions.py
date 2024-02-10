import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule, Trainer
import torch.nn as nn
from torch.utils.data import DataLoader, random_split,Dataset
import torchmetrics
from pytorch_lightning.callbacks import Callback
import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from osgeo import gdal, ogr,gdal_array
import os
import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
# Import ListedColormap



def D_Augm(patches_x, patches_y):
    augmented_x = []
    augmented_y = []
    for i in range(patches_x.shape[0]):
        # append original patch
        augmented_x.append(patches_x[i])
        augmented_y.append(patches_y[i])

        # rotate by 45 degrees
        x_rot45 = torch.rot90(patches_x[i], k=1, dims=(1, 2))
        y_rot45 = torch.rot90(patches_y[i], k=1, dims=(0, 1))
        augmented_x.append(x_rot45)
        augmented_y.append(y_rot45)

        # rotate by 90 degrees
        x_rot90 = torch.rot90(patches_x[i], k=2, dims=(1, 2))
        y_rot90 = torch.rot90(patches_y[i], k=2, dims=(0, 1))
        augmented_x.append(x_rot90)
        augmented_y.append(y_rot90)

        # rotate by 135 degrees
        x_rot135 = torch.rot90(patches_x[i], k=3, dims=(1, 2))
        y_rot135 = torch.rot90(patches_y[i], k=3, dims=(0, 1))
        augmented_x.append(x_rot135)
        augmented_y.append(y_rot135)

        # flip horizontally
        x_flipped_h = torch.flip(patches_x[i], dims=(2,))
        y_flipped_h = torch.flip(patches_y[i], dims=(1,))
        augmented_x.append(x_flipped_h)
        augmented_y.append(y_flipped_h)

        # flip vertically
        x_flipped_v = torch.flip(patches_x[i], dims=(1,))
        y_flipped_v = torch.flip(patches_y[i], dims=(0,))
        augmented_x.append(x_flipped_v)
        augmented_y.append(y_flipped_v)

    augmented_x = torch.stack(augmented_x)
    augmented_y = torch.stack(augmented_y)
    return augmented_x, augmented_y


class MyCall(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train")

    def on_train_end(self, trainer, pl_module):
        print("Training Done")


class UNetLightning(LightningModule):
    def __init__(self, num_classes, learning_rate, dataset, batch_size, endcoder):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = endcoder
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset = dataset
        self.crit = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.train_F1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.IoU = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        # Split the dataset into training and validation sets
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)  # 70% for training
        validate_size = int(0.15 * total_size)  # 15% for validation
        test_size = total_size - train_size - validate_size  # Remaining for testing
        self.train_dataset, self.validate_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, validate_size, test_size]
        )
        # Model Structure
        self.model = smp.Unet(
            encoder_name=f"{self.encoder}",  # Use VGG16 with batch normalization
            encoder_weights="imagenet",  # Pre-trained on ImageNet
            classes=num_classes,
            activation="softmax2d",  # No activation in final layer
        )
        # Freeze the weights of the encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
        # Loaders

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=12
        )

    def val_dataloader(self):
        return DataLoader(
            self.validate_dataset, batch_size=self.batch_size, num_workers=12
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=12)

    # Common Step
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.float() / 255.0  # Normalize the data by dividing by 255
        y = y.long()
        y_hat = self.forward(x)
        loss = self.crit(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, y_hat, y = self._common_step(batch, batch_idx)
        preds = y_hat
        t_acc = self.train_acc(preds, y)
        t_f1 = self.train_F1(preds, y)
        t_IoU = self.IoU(preds, y)
        # Logging
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": t_acc,
                "train_f1_score": t_f1,
                "IoU": t_IoU,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": loss, "x": x, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {"valid_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {"test_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        x = x / 255.0
        x = x.float()
        return self(batch)



def PreProcessing_RGB(filename):
    InputImage = filename
    # Get the base name and remove extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    # Define output directory
    output_dir = rf"/home/mskenawi/SW/PreProcessed_Tensorsv3"
    Tensor_name = f"{base_name}_Tensor"
    inVectorPath = rf"/home/mskenawi/Basisdata_0000_Norge_25833_FKB-AR5_FGDB.gdb/"
    from matplotlib.colors import ListedColormap  # Import ListedColormap

    filename = InputImage
    # Get the base name and remove extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    tensor_output_dir = os.path.join(output_dir, Tensor_name)
    # Check if the output directory for tensors already exists, and if it does, skip processing
    if os.path.exists(tensor_output_dir):
        print(f"Tensor folder already exists for {Tensor_name}, skipping processing.")
        return
    # Define output directory

    # define file paths
    inRasterPath = InputImage
    # inVectorPath = r'D:\mskenawi\one drive\OneDrive - NTNU\PhD\LUHP\WetLand\CNN\FKB_CLIP.shp'
    data = gdal.Open(inRasterPath)
    geo_transform = data.GetGeoTransform()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    mb_v = ogr.Open(inVectorPath)
    mb_l = mb_v.GetLayer()
    pixel_width = geo_transform[1]
    mb_l.SetSpatialFilterRect(x_min, y_min, x_max, y_max)
    target_ds = gdal.GetDriverByName("MEM").Create("", x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, -pixel_width))
    gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=arealtype"])
    target_ds.ReadAsArray()
    print(np.unique(target_ds.ReadAsArray()))
    Yar = target_ds.ReadAsArray().astype(np.int32)
    Yar[np.isin(Yar, [11, 12])] = 0
    Yar[np.isin(Yar, [21, 22, 23])] = 4
    Yar[Yar == 30] = 1
    Yar[np.isin(Yar, [50, 70])] = 5
    Yar[Yar == 60] = 3
    Yar[np.isin(Yar, [81, 82])] = 2
    print(np.unique(Yar))
    array = data.ReadAsArray()
    array = data.ReadAsArray()
    tensor_x = torch.Tensor(data.ReadAsArray())
    tensor_y = torch.Tensor(Yar)
    tensor_y = tensor_y.to(torch.uint8)
    # tensor_y = torch.rot90(tensor_y, k=3)
    ph = (((array.shape[1] // 512)) * 512) - array.shape[1]
    pw = (((array.shape[2] // 512)) * 512) - array.shape[2]
    tensor_x = torch.nn.functional.pad(tensor_x, (0, pw, 0, ph), mode="constant")
    tensor_y = torch.nn.functional.pad(tensor_y, (0, pw, 0, ph), mode="constant")
    print(tensor_x.shape)
    print(tensor_y.shape)
    patches_x = tensor_x.unfold(1, 512, 256)
    patches_x = patches_x.unfold(2, 512, 256)
    patches_x = torch.reshape(
        patches_x, (3, patches_x.size(1) * patches_x.size(2), 512, 512)
    )
    patches_x = patches_x.permute(1, 0, 2, 3)
    patches_y = tensor_y.unfold(0, 512, 256)
    patches_y = patches_y.unfold(1, 512, 256)
    patches_y = torch.reshape(
        patches_y, (patches_y.size(1) * patches_y.size(0), 512, 512)
    )
    # Step 1: Remove patches if any of the conditions is met
    # Create a boolean mask to identify patches that meet the conditions
    #for idx in range(patches_x.shape[0] - 1, -1, -1):
    #    if torch.unique(patches_y[idx]).numel() == 1 or (patches_x[idx, 0] == 0).all() :
    #        patches_x = torch.cat((patches_x[:idx], patches_x[idx + 1:]))
    #        patches_y = torch.cat((patches_y[:idx], patches_y[idx + 1:]))
    #mask = torch.zeros(len(patches_y), dtype = bool)
    #for idx in range(len(mask)):
    #    if (torch.unique(patches_y[idx]).numel()) == 1 :
    #        mask[idx] = True
    #patches_y = patches_y[~mask]
    #patches_x = patches_x[~mask]
    mask =  (patches_x[:, 0] == 0).any(dim=1).any(dim=1)
    patches_x = patches_x[~mask]
    patches_y = patches_y[~mask]
    mask = (patches_y==99).any(dim=1).any(dim=1)
    patches_x = patches_x[~mask]
    patches_y = patches_y[~mask]
    if patches_x.numel() == 0:
        print(f"No valid patches in image {InputImage}, skipping...")
        return
    patches_x, patches_y = D_Augm(patches_x, patches_y)
    print(rf"Writing Tensors{base_name}")
    os.makedirs(os.path.join(output_dir, Tensor_name), exist_ok=True)
    njobs = math.ceil(len(patches_x) / 4)
    Parallel(n_jobs=njobs, prefer="threads")(
        delayed(write_tensor)(patches_x, patches_y, output_dir, base_name, i)
        for i in range(len(patches_x))
    )
    return
from joblib import Parallel,delayed
import math

def write_tensor(patches_x, patches_y, output_dir, base_name, i):
    # Save each tensor pair as a separate file
    tensor_name = f"{base_name}_{i}.pt"  # unique name for each tensor pair
    tensor_path = os.path.join(output_dir, rf"{base_name}_Tensor", tensor_name)
    torch.save(
        {"tensor_x": patches_x[i].clone(), "tensor_y": patches_y[i].clone()},
        tensor_path,
    )


class CustomDataset(Dataset):
    def __init__(self, data_root):
        self.data_paths = glob.glob(f"{data_root}/**/*.pt", recursive=True)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        data = torch.load(data_path)

        x = data['tensor_x'].float()  # data loaded from file is already a tensor
        # Scale the data to the range [0, 1]
        y = data['tensor_y'].long()  # data loaded from file is already a tensor

        return x,y



def Plotting_Dataset(dataset):
    # Randomly select 16 indices from the dataset
    num_samples_to_display = 16
    selected_indices = np.random.choice(len(dataset), num_samples_to_display, replace=False)

    # Define the color map for the labels
    label_to_color = {
        0: [0.5, 0.5, 0.5],    # Grey
        1: [0, 1, 0],          # Green
        2: [0, 0, 1],          # Blue
        3: [1, 0, 0],          # Red
        4: [1, 0.5, 0],        # Orange
        5: [1, 1, 0],          # Yellow
    }

    # Plot the patches overlaid with colors
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    displayed_count = 0

    for idx in selected_indices:
        data_item = dataset[idx]
        if data_item is not None:
            x_patch = data_item[0]  # First layer of x
            y_labels = data_item[1]

            overlay = np.zeros((*x_patch.shape, 3), dtype=np.float32)

            for row in range(y_labels.shape[0]):
                for col in range(y_labels.shape[1]):
                    y_label = y_labels[row, col]
                    y_color = label_to_color[y_label.item()]
                    overlay[row, col] = y_color

            ax = axes[displayed_count // 4, displayed_count % 4]
            ax.imshow(x_patch, cmap='gray')  # Plot the grayscale patch
            ax.imshow(overlay, alpha=0.5)  # Overlay with colors
            ax.axis('off')

            displayed_count += 1

        if displayed_count >= num_samples_to_display:
            break

    plt.tight_layout()
    plt.show()
    return


def Prediction(Image,model_chkpt,opdir):
    base_name = os.path.splitext(os.path.basename(Image))[0]
    model = UNetLightning.load_from_checkpoint(rf"{model_chkpt}")
        # Check if output directory exists
    if not os.path.exists(opdir):
        os.makedirs(opdir)
# Weight Tensor
    shape = (512, 512)
    center_x, center_y = shape[0] // 2, shape[1] // 2
    weight_tensor = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            weight_value = 0.2 + 0.8 * (1 - distance / (np.sqrt(center_x**2 + center_y**2)))
            weight_tensor[i, j] = weight_value
    weight = torch.Tensor(weight_tensor)
#Open The image 
    INP = gdal.Open(Image)
    gt = INP.GetGeoTransform()
    pr = INP.GetProjection()
    sf = INP.GetSpatialRef()
    array = INP.ReadAsArray()
    pixelrs = gt[1]
    # Calculate the required padding to avoid missing pixels
    ph = ((array.shape[0] // 512) + 1) * 512 - array.shape[0]
    pw = ((array.shape[1] // 512) + 1) * 512 - array.shape[1]
    # Add padding to the original image tensor
    tensor_x = torch.tensor(array)
    tensor_x = torch.nn.functional.pad(tensor_x, (0, ph, 0, pw), mode='constant')
    # Split the aerial image into patches of size 512x512
    patch_size = 512
    stride = patch_size
    # use unfold to split the tensor into patches of size [n, 512, 512]
    patches_x = tensor_x.unfold(1, 512,512)
    patches_x = patches_x.unfold(2,512,512)
    npx ,npy = patches_x.size(1),patches_x.size(2)
    # reshape the patches tensor to have shape [n, num_patches_h, num_patches_w, 512, 512]
    patches_x = torch.reshape(patches_x,(3,patches_x.size(1)*patches_x.size(2),512,512))
    patches_x = patches_x.permute(1, 0, 2, 3)
    C, H, W = tensor_x.shape
    #Rotation
    patches_x = patches_x/255.0
    patches_x = patches_x.float()
    # Rotate patches by 90 degrees
    rotated_patches_90 = torch.rot90(patches_x, k=1, dims=(2, 3))
    # Rotate patches by 180 degrees
    rotated_patches_180 = torch.rot90(patches_x, k=2, dims=(2, 3))
    # Rotate patches by 270 degrees
    rotated_patches_270 = torch.rot90(patches_x, k=3, dims=(2, 3))
    all_patches = torch.cat((patches_x, rotated_patches_90, rotated_patches_180, rotated_patches_270), dim=0)

    # Create a DataLoader
    dl_all = DataLoader(all_patches, batch_size=80, shuffle=False, num_workers=12)
    trainer = Trainer(devices='1')
    # Predict
    predictions_all = trainer.predict(model, dl_all)
    y_all = torch.cat(predictions_all)

    # Split the results back into four tensors
    y = y_all[:len(patches_x)]
    y_90_rot = y_all[len(patches_x):2*len(patches_x)]
    y_180_rot = y_all[2*len(patches_x):3*len(patches_x)]
    y_270_rot = y_all[3*len(patches_x):]

    # Rotate the predictions back
    y_90 = torch.rot90(y_90_rot, k=-1, dims=(2, 3))
    y_180 = torch.rot90(y_180_rot, k=-2, dims=(2, 3))
    y_270 = torch.rot90(y_270_rot, k=-3, dims=(2, 3))
#Fold
    Y1_avg = (y+y_90+y_180+y_270)/4.0
    #Apply weight then fold 
    Y1_avg = Y1_avg*weight
    #Fold the Y1 
    y_label = Y1_avg.unsqueeze(0)
    B, C, H, W = 1,6,H,W
    y_label = y_label.permute(0,2,1,3,4) 
    y_label = y_label.contiguous().view(B,C, -1, 512*512)
    y_label = y_label.permute(0,1,3,2)

    y_label = y_label.contiguous().view(C, 1*512*512, -1)

    # # y_label = y_label.squeeze(0)
    # # # print("Start Folding")
    output = F.fold(y_label, output_size=(H, W),kernel_size=512, stride=512)
    output = output.squeeze(1)
    output_Y1 = output[:,:-pw,:-ph]
    #Now 256 
    patch_size = 512
    stride = 256
    # use unfold to split the tensor into patches of size [n, 512, 512]
    patches_x = tensor_x.unfold(1, 512,256)
    patches_x = patches_x.unfold(2,512,256)
    npx ,npy = patches_x.size(1),patches_x.size(2)
    # reshape the patches tensor to have shape [n, num_patches_h, num_patches_w, 512, 512]
    patches_x = torch.reshape(patches_x,(3,patches_x.size(1)*patches_x.size(2),512,512))
    patches_x = patches_x.permute(1, 0, 2, 3)
    C, H, W = tensor_x.shape
    patches_x = patches_x/255.0
    patches_x = patches_x.float()
    # Rotate patches by 90 degrees
    rotated_patches_90 = torch.rot90(patches_x, k=1, dims=(2, 3))
    # Rotate patches by 180 degrees
    rotated_patches_180 = torch.rot90(patches_x, k=2, dims=(2, 3))
    # Rotate patches by 270 degrees
    rotated_patches_270 = torch.rot90(patches_x, k=3, dims=(2, 3))
    all_patches = torch.cat((patches_x, rotated_patches_90, rotated_patches_180, rotated_patches_270), dim=0)

    # Create a DataLoader
    dl_all = DataLoader(all_patches, batch_size=80, shuffle=False, num_workers=12)

    # Predict
    predictions_all = trainer.predict(model, dl_all)
    y_all = torch.cat(predictions_all)

    # Split the results back into four tensors
    y = y_all[:len(patches_x)]
    y_90_rot = y_all[len(patches_x):2*len(patches_x)]
    y_180_rot = y_all[2*len(patches_x):3*len(patches_x)]
    y_270_rot = y_all[3*len(patches_x):]

    # Rotate the predictions back
    y_90 = torch.rot90(y_90_rot, k=-1, dims=(2, 3))
    y_180 = torch.rot90(y_180_rot, k=-2, dims=(2, 3))
    y_270 = torch.rot90(y_270_rot, k=-3, dims=(2, 3))
    
    Y2_avg = (y+y_90+y_180+y_270)/4.0
    #Apply weight then fold 
    Y2_avg = Y2_avg*weight
    #Fold the Y1 
    y_label = Y2_avg.unsqueeze(0)
    B, C, H, W = 1,6,H,W
    y_label = y_label.permute(0,2,1,3,4) 
    y_label = y_label.contiguous().view(B,C, -1, 512*512)
    y_label = y_label.permute(0,1,3,2)
    y_label = y_label.contiguous().view(C, 1*512*512, -1)
    output = F.fold(y_label, output_size=(H, W),kernel_size=512, stride=256)
    output = output.squeeze(1)
    output_Y2= output[:,:-pw,:-ph]
    #Get the Mean of the two weights
    Y_F = torch.max(output_Y1, output_Y2)
    Y_F = Y_F.argmax(dim=0)
    Y_F_NP = Y_F.detach().numpy()
    Y_F_NP[Y_F_NP == 3] = 5
    # cmap = ListedColormap([ 'green', 'blue', 'red','orange', 'yellow', 'pink'])
    # # Plot the tensor and its corresponding label as a transparent overlay
    # fig, ax = plt.subplots()
    # ax.imshow(array[0], cmap='gray')
    # ax.imshow(Y_F_NP, cmap=cmap, alpha=0.4)
    # plt.show() 
    options = ["COMPRESS=LZW"]
    Y = gdal_array.OpenArray(Y_F_NP)
    gdal_array.CopyDatasetInfo(INP,Y)
    gdal.Translate(rf'{opdir}/{base_name}_OP.tif',Y,creationOptions=options)
    Y = None
    return 


def predict_and_rotate(model, data_loader, device):
    trainer = Trainer(devices=[device])
    predictions = trainer.predict(model, data_loader)
    
    # Split the results back into four tensors
    y = predictions[:len(data_loader.dataset) // 4]
    y_90_rot = predictions[len(data_loader.dataset) // 4:2 * len(data_loader.dataset) // 4]
    y_180_rot = predictions[2 * len(data_loader.dataset) // 4:3 * len(data_loader.dataset) // 4]
    y_270_rot = predictions[3 * len(data_loader.dataset) // 4:]
    
    # Rotate the predictions back
    y_90 = torch.rot90(y_90_rot, k=-1, dims=(2, 3))
    y_180 = torch.rot90(y_180_rot, k=-2, dims=(2, 3))
    y_270 = torch.rot90(y_270_rot, k=-3, dims=(2, 3))
    
    return (y, y_90, y_180, y_270)


def PreProcessing_RGB_Raster(filename,label):
    InputImage = filename
    # Get the base name and remove extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    # Define output directory
    output_dir = rf"/home/mskenawi/PreProcessed_Tensors/{base_name}"
    Tensor_name = f"{base_name}_Tensor"
    filename = InputImage
    # Get the base name and remove extension
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    tensor_output_dir = os.path.join(output_dir, Tensor_name)
    # Check if the output directory for tensors already exists, and if it does, skip processing
    if os.path.exists(tensor_output_dir):
        print(f"Tensor folder already exists for {Tensor_name}, skipping processing.")
        return
    # Define output directory

    # define file paths
    inRasterPath = InputImage
    # inVectorPath = r'D:\mskenawi\one drive\OneDrive - NTNU\PhD\LUHP\WetLand\CNN\FKB_CLIP.shp'
    data = gdal.Open(inRasterPath)
    geo_transform = data.GetGeoTransform()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    y_l = gdal.Open(label)
    pixel_width = geo_transform[1]
    vrt_options = gdal.BuildVRTOptions(outputBounds=(x_min, y_min, x_max, y_max))
    vrt_ds = gdal.BuildVRT(rf"{base_name}.vrt", y_l, options=vrt_options)
    Yar = vrt_ds.ReadAsArray()
    print(np.unique(Yar))    
    array = data.ReadAsArray()
    
#     plt.imshow(array[0], cmap='viridis')
# # Overlay 'y' array with transparency (alpha)
#     plt.imshow(Yar, cmap='jet', alpha=0.4)
# # Show the plot
#     plt.title("Overlay of X and Y Arrays")
#     plt.show()
    
    tensor_x = torch.Tensor(array)
    tensor_y = torch.Tensor(Yar)
    tensor_y = tensor_y.to(torch.uint8)
    ph = (((array.shape[1] // 512)) * 512) - array.shape[1]
    pw = (((array.shape[2] // 512)) * 512) - array.shape[2]
    tensor_x = torch.nn.functional.pad(tensor_x, (0, pw, 0, ph), mode="constant")
    tensor_y = torch.nn.functional.pad(tensor_y, (0, pw, 0, ph), mode="constant")
    print(tensor_x.shape)
    print(tensor_y.shape)
    patches_x = tensor_x.unfold(1, 512, 256)
    patches_x = patches_x.unfold(2, 512, 256)
    patches_x = torch.reshape(
        patches_x, (3, patches_x.size(1) * patches_x.size(2), 512, 512)
    )
    patches_x = patches_x.permute(1, 0, 2, 3)
    patches_y = tensor_y.unfold(0, 512, 256)
    patches_y = patches_y.unfold(1, 512, 256)
    patches_y = torch.reshape(
        patches_y, (patches_y.size(1) * patches_y.size(0), 512, 512)
    )
    
    # Step 1: Remove patches if any of the conditions is met
    # Create a boolean mask to identify patches that meet the conditions
    #for idx in range(patches_x.shape[0] - 1, -1, -1):
    #    if torch.unique(patches_y[idx]).numel() == 1 or (patches_x[idx, 0] == 0).all() :
    #        patches_x = torch.cat((patches_x[:idx], patches_x[idx + 1:]))
    #        patches_y = torch.cat((patches_y[:idx], patches_y[idx + 1:]))
    #mask = torch.zeros(len(patches_y), dtype = bool)
    #for idx in range(len(mask)):
    #    if (torch.unique(patches_y[idx]).numel()) == 1 :
    #        mask[idx] = True
    #patches_y = patches_y[~mask]
    #patches_x = patches_x[~mask]
    mask =  (patches_x[:, 0] == 0).any(dim=1).any(dim=1)
    patches_x = patches_x[~mask]
    patches_y = patches_y[~mask]
    mask = ((patches_y == 99) | (patches_y == 15)).any(dim=1).any(dim=1)
    patches_x = patches_x[~mask]
    patches_y = patches_y[~mask]
    print(torch.unique(patches_y))
    if patches_x.numel() == 0:
        print(f"No valid patches in image {InputImage}, skipping...")
        return
    patches_x, patches_y = D_Augm(patches_x, patches_y)
    print(rf"Writing Tensors{base_name}")
    os.makedirs(os.path.join(output_dir, Tensor_name), exist_ok=True)
    njobs = math.ceil(len(patches_x) / 4)
    Parallel(n_jobs=njobs, prefer="threads")(
        delayed(write_tensor)(patches_x, patches_y, output_dir, base_name, i)
        for i in range(len(patches_x))
    )
    return


class CustomDataset2(Dataset):
    def __init__(self, data_root):
        self.data_paths = glob.glob(f"{data_root}/**/*.pt", recursive=True)
        self.data = []

        # Load all tensors into memory
        for data_path in self.data_paths:
            data = torch.load(data_path)
            x = data['tensor_x'].float()  # data loaded from file is already a tensor
            y = data['tensor_y'].long()  # data loaded from file is already a tensor
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
