import torch
import pandas as pd
import cv2
import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import imgaug as ia

from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split
from skimage.morphology import label
from torch import nn
from skimage.morphology import label
from torch.utils.data import Dataset, DataLoader
from unet_models import UNetResNet


### CONFIG

path_to_dataset = "/home/timur/Disk/Airbus_Ship_Detection_Challenge_DATA/"
csv_name = "train_ship_segmentations.csv"
batch_size = 8
num_epoch = 10
n_epoch = 10
img_size = (256, 256)
###

path_to_img_train = os.path.join(path_to_dataset, "train")
path_to_img_test = os.path.join(path_to_dataset, "test")
path_to_img_subm = os.path.join(path_to_dataset, "sample_submission.csv")
path_to_csv = os.path.join(path_to_dataset, csv_name)
data_csv = pd.read_csv(path_to_csv)
submission_csv = pd.read_csv(path_to_img_subm)
data_csv = data_csv[pd.notna(data_csv.EncodedPixels)]
img_list = data_csv.ImageId.unique()
train_data_img, val_data_img = train_test_split(img_list, 
                                                test_size = 0.2, 
                                                shuffle = True, 
                                                random_state = 123)

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

print("size train data : {}\nsize validation data : {}".format(len(train_data_img), len(val_data_img)))


sometimes = lambda aug, prt: iaa.Sometimes(prt, aug)

seq = iaa.Sequential(sometimes(iaa.Affine(
                        scale=(0.8, 1.2), # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                        rotate=(-45, 45) # rotate by -45 to +45 degrees
                    ), 0.3),
                    sometimes(iaa.CropAndPad(
                        percent=(-0.05, 0.1),
                        pad_mode=ia.ALL,
                        pad_cval=(0, 255)
                    ), 0.5),
                     iaa.Fliplr(0.5),
                     iaa.Flipud(0.5))

class Airbus_Dataset(Dataset):
    def __init__(self, data_img_name, data_csv, path_to_img, img_size, aug=False):
        self.len = len(data_img_name)
        self.path_to_img = path_to_img
        self.data_csv = data_csv
        self.data_img_name = data_img_name
        self.img_size = img_size
        self.aug = aug
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, inx):
        img_name = self.data_img_name[inx]
        rle = self.data_csv[self.data_csv.ImageId == img_name]
        img = cv2.imread(os.path.join(self.path_to_img, img_name))
        mask = masks_as_image(rle.EncodedPixels)
        
        if self.aug:
            _aug = seq._to_deterministic()
            img =  _aug.augment_image(img.astype(np.uint8))
            mask = _aug.augment_image(mask)
    
        img = cv2.resize(img, img_size).transpose(2, 0, 1).astype(np.float32) / 255.0
        mask = cv2.resize(mask, img_size).astype(np.int64)
        
        return img, mask
    

train_dataset = Airbus_Dataset(train_data_img, data_csv, path_to_img_train, img_size, aug=True)
val_dataset = Airbus_Dataset(val_data_img, data_csv, path_to_img_train, img_size)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

model = UNetResNet(encoder_depth=152, num_classes=2, pretrained=True).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


def iou_loss_calc(predict, target):
    predict = torch.argmax(predict, dim=1)
    inter = predict * target
    union = predict + target - inter
    return torch.sum(inter).float() / torch.sum(union).float()


pred = 0
def train(model, dataloader, optimizer, criterion):
    global pred
    model.train()
    running_loss = 0.0
    iou_loss = 0.0
    len_ = len(dataloader)
    for step, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        pred = (outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        iou_loss += iou_loss_calc(outputs, labels)
        if not step % 1:
            loss_ = running_loss / (step + 1)
            iou_ = iou_loss / (step + 1)
            print(f"step {step}/{len_}, loss {loss_}, iou {iou_}", end = "\r")
    return running_loss / len(dataloader), iou_loss / len(dataloader)


def eval(model, dataloader, criterion):
    running_loss = 0.0
    iou_loss = 0.0
    model.eval()
    for step, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        iou_loss += iou_loss_calc(outputs, labels)
    return running_loss / len(dataloader), iou_loss / len(dataloader)


for epoch in range(n_epoch):
    loss_train, iou_train = train(model, train_dataloader, optimizer, criterion)
    loss_val, iou_val = eval(model, val_dataloader, criterion)
    print(f"epoch {epoch}, loss train {loss_train}, iou train {iou_train}, loss val {loss_val}, iou val {iou_val}")
    torch.save(model.state_dict(), f"./unet16_{epoch}.w")
    

model.eval()
pred_list = []
for i in tqdm.tqdm(range(len(submission_csv))):
    image_name = submission_csv.iloc[i].ImageId
    image = cv2.imread(os.path.join(path_to_img_test, image_name))
    image = cv2.resize(image, img_size).transpose(2, 0, 1).astype(np.float32) / 255.0
    image = image[np.newaxis, ...]
    image = torch.from_numpy(image).cuda()
    
    output = model.forward(image)
    output = torch.argmax(output, dim=1).cpu().data.numpy()
    multi_rle = multi_rle_encode(output.squeeze())
    if len(multi_rle) > 0:
        for _rle in multi_rle:
            pred_list.append([image_name, _rle])
    else:
        pred_list.append([image_name, None])

_dataframe = pd.DataFrame(pred_list, columns = ["ImageId", "EncodedPixels"], index=False)
_dataframe.to_csv("submission.csv")