import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import transforms

from efficientunet import *
from dataset import *
from config import *
from utils import *
from metric import iou_score

cfg = Config()

train_transform = A.Compose([
    A.Resize(512, 512),
    # A.OneOf([
    #     A.GaussNoise(p=0.2),
    #     A.MultiplicativeNoise(p=0.2),
    #     ], p=0.3),

    A.Normalize(mean=(0.485), std=(0.229)),
    transforms.ToTensorV2(),
    ])

val_transform = A.Compose([
    A.Resize(512, 512),
    # A.OneOf([
    #     A.GaussNoise(p=0.2),
    #     A.MultiplicativeNoise(p=0.2),
    #     ], p=0.3),

    A.Normalize(mean=(0.485), std=(0.229)),
    transforms.ToTensorV2(),
])

# Set Dataset
train_dataset = Dataset(imgs_dir=TRAIN_IMGS_DIR, mask_dir=TRAIN_LABELS_DIR, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
val_dataset = Dataset(imgs_dir=VAL_IMGS_DIR, mask_dir=VAL_LABELS_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

train_data_num = len(train_dataset)
val_data_num = len(val_dataset)

train_batch_num = int(np.ceil(train_data_num / cfg.BATCH_SIZE)) # np.ceil 반올림
val_batch_num = int(np.ceil(val_data_num / cfg.BATCH_SIZE))

# Network
# net = UNet().to(device)
net = get_socar_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True).to(device)

# Loss Function
loss_fn = nn.CrossEntropyLoss().to(device)

# Optimizer
optim = torch.optim.Adam(params=net.parameters(), lr=cfg.LEARNING_RATE)


# Training
start_epoch = 0

num_epochs = cfg.NUM_EPOCHS

for epoch in range(start_epoch+1, num_epochs+1):
    net.train()  # Train Mode
    train_loss_arr = []
    
    for batch_idx, data in enumerate(train_loader, 1):
        # Forward Propagation
        img = data['img'].to(device)
        label = data['label'].to(device)
        
        label = label // 255
        
        output = net(img)

        # Backward Propagation
        optim.zero_grad()
        
        loss = loss_fn(output, label)

        loss.backward()
        
        optim.step()
        
        # Calc Loss Function
        train_loss_arr.append(loss.item())

        print_form = '[Train] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
        print(print_form.format(epoch, num_epochs, batch_idx, train_batch_num, train_loss_arr[-1]))

    train_loss_avg = np.mean(train_loss_arr)
    
    # Validation (No Back Propagation)
    with torch.no_grad():
        net.eval()  # Evaluation Mode
        val_loss_arr, val_iou_arr = [], []
        
        for batch_idx, data in enumerate(val_loader, 1):
            # Forward Propagation
            img = data['img'].to(device)
            label = data['label'].to(device)
            
            label = label // 255

            output = net(img)
            output_t = torch.argmax(output, dim=1).float()
            
            # Calc Loss Function
            loss = loss_fn(output, label)
            iou = iou_score(output_t, label)

            val_loss_arr.append(loss.item())
            val_iou_arr.append(iou.item())
            
            print_form = '[Validation] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f} | IoU: {:.4f}'
            print(print_form.format(epoch, num_epochs, batch_idx, val_batch_num, val_loss_arr[-1], iou))
            
    val_loss_avg = np.mean(val_loss_arr)
    val_iou_avg =  np.mean(val_iou_arr)
    # val_writer.add_scalar(tag='loss', scalar_value=val_loss_avg, global_step=epoch)
    
    print_form = '[Epoch {:0>4d}] Training Avg Loss: {:.4f} | Validation Avg Loss: {:.4f} | Validation Avg IoU: {:.4f}'
    print(print_form.format(epoch, train_loss_avg, val_loss_avg, val_iou_avg))
    
    save_net(ckpt_dir=CKPT_DIR, net=net, optim=optim, epoch=epoch)
    