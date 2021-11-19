import numpy as np
import matplotlib.pyplot as plt

import torch
import copy
import time

from efficientunet import *
from dataset import *
from config import *
from utils import *
from metric import iou_score

def train_model(dataloaders, batch_num, net, criterion, optim, ckpt_dir, wandb, num_epoch):
    wandb.watch(net, criterion, log='all', log_freq=10)

    since = time.time()
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_iou = 0

    for epoch in range(1, num_epoch+1):
        net.train()  # Train Mode
        train_loss_arr = []
        
        for batch_idx, data in enumerate(dataloaders['train'], 1):
            # Forward Propagation
            img = data['img'].to(device)
            label = data['label'].to(device)
            
            label = label // 255
            
            output = net(img)

            # Backward Propagation
            optim.zero_grad()
            
            loss = criterion(output, label)

            loss.backward()
            
            optim.step()
            
            # Calc Loss Function
            train_loss_arr.append(loss.item())

            print_form = '[Train] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'
            print(print_form.format(epoch, num_epoch, batch_idx, batch_num['train'], train_loss_arr[-1]))

        train_loss_avg = np.mean(train_loss_arr)

        # Validation (No Back Propagation)
        with torch.no_grad():
            net.eval()  # Evaluation Mode
            val_loss_arr, val_iou_arr = [], []
            
            for batch_idx, data in enumerate(dataloaders['val'], 1):
                # Forward Propagation
                img = data['img'].to(device)
                label = data['label'].to(device)
                
                label = label // 255

                output = net(img)
                output_t = torch.argmax(output, dim=1).float()
                
                # Calc Loss Function
                loss = criterion(output, label)
                iou = iou_score(output_t, label)

                val_loss_arr.append(loss.item())
                val_iou_arr.append(iou.item())
                
                print_form = '[Validation] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f} | IoU: {:.4f}'
                print(print_form.format(epoch, num_epoch, batch_idx, batch_num['val'], val_loss_arr[-1], iou))


        val_loss_avg = np.mean(val_loss_arr)
        val_iou_avg =  np.mean(val_iou_arr)
        # val_writer.add_scalar(tag='loss', scalar_value=val_loss_avg, global_step=epoch)
        
        if val_iou_avg < best_iou:
            best_iou = val_iou_avg
            best_model_wts = copy.deepcopy(net.state_dict())    

        wandb.log({'train_epoch_loss': train_loss_avg , 'val_epoch_loss': val_loss_avg, 'val_epoch_iou': val_iou_avg}, step=epoch)

        print_form = '[Epoch {:0>4d}] Training Avg Loss: {:.4f} | Validation Avg Loss: {:.4f} | Validation Avg IoU: {:.4f}'
        print(print_form.format(epoch, train_loss_avg, val_loss_avg, val_iou_avg))
        
        save_net(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val IoU: {:4f}'.format(best_iou))

    net.load_state_dict(best_model_wts)
    save_net(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch, is_best=True, best_iou=best_iou)


def eval_model(test_loader, test_batch_num, net, criterion, optim, ckpt_dir, result_dir, wandb):
    # Load Checkpoint File
    if os.listdir(ckpt_dir):
        net, optim, _ = load_net(ckpt_dir=ckpt_dir, net=net, optim=optim)

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
            loss = criterion(output, label)
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
                
                plt.imsave(os.path.join(result_dir, f'img_{crt_id:04}.png'), img[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, f'label_{crt_id:04}.png'), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, f'output_{crt_id:04}.png'), output_t[j].squeeze(), cmap='gray')
    
    eval_loss_avg = np.mean(loss_arr)
    eval_iou_avg  = np.mean(iou_arr)
    print_form = '[Result] | Avg Loss: {:0.4f} | Avg IoU: {:0.4f}'
    wandb.log({'eval_loss': eval_loss_avg , 'eval_iou': eval_iou_avg}, commit=False)
    print(print_form.format(eval_loss_avg, eval_iou_avg))