import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import transforms
from torch.utils.data import DataLoader

from efficientunet import *
from dataset import *
from utils import *
from config import *
from metric import iou_score

# Evaluation
# 1. Delete RandomFlip
# 2. shuffle=False
# 3. Tensorboard 사용 X
# 4. Train X (Epoch 존재하지 않음)

cfg = Config()

transform = A.Compose([
    A.Resize(512, 512),
    # A.OneOf([
    #     A.GaussNoise(p=0.2),
    #     A.MultiplicativeNoise(p=0.2),
    #     ], p=0.3),

    A.Normalize(mean=(0.485), std=(0.229)),
    transforms.ToTensorV2(),
    ])

test_dataset = Dataset(imgs_dir=TEST_IMGS_DIR, mask_dir=TEST_LABELS_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

test_data_num = len(test_dataset)
test_batch_num = int(np.ceil(test_data_num / cfg.BATCH_SIZE)) # np.ceil 반올림

# Network
net = get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True).to(device)

# Loss Function
loss_fn = nn.CrossEntropyLoss().to(device)

# Optimizer
optim = torch.optim.Adam(params=net.parameters(), lr=cfg.LEARNING_RATE)

start_epoch = 0

# Load Checkpoint File
if os.listdir(CKPT_DIR):
    net, optim, _ = load_net(ckpt_dir=CKPT_DIR, net=net, optim=optim)

# Evaluation
with torch.no_grad():
    net.eval()  # Evaluation Mode
    loss_arr, iou_arr = [], []

    for batch_idx, data in enumerate(test_loader, 1):
        # Forward Propagation
        img = data['img'].to(device)
        label = data['label'].to(device)

        label = label // 255

        output = net(img)
        output_t = torch.argmax(output, dim=1).float()

        # Calc Loss Function
        loss = loss_fn(output, label)
        iou = iou_score(output_t, label)

        loss_arr.append(loss.item())
        iou_arr.append(iou.item())
        
        print_form = '[Test] | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f} | IoU: {:.4f}'
        print(print_form.format(batch_idx, test_batch_num, loss_arr[-1], iou))

        img = to_numpy(denormalization(img, mean=0.5, std=0.5))
        # 이미지 캐스팅
        img = np.clip(img, 0, 1) 

        label = to_numpy(label)
        output_t = to_numpy(classify_class(output_t))
        
        for j in range(label.shape[0]):
            crt_id = int(test_batch_num * (batch_idx - 1) + j)
            
            plt.imsave(os.path.join(RESULTS_DIR, f'img_{crt_id:04}.png'), img[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(RESULTS_DIR, f'label_{crt_id:04}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(RESULTS_DIR, f'output_{crt_id:04}.png'), output_t[j].squeeze(), cmap='gray')
            
print_form = '[Result] | Avg Loss: {:0.4f} | Avg IoU: {:0.4f}'
print(print_form.format(np.mean(loss_arr), np.mean(iou_arr)))