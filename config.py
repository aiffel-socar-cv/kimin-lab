import os
import torch


sweep_config = {
    'method': 'grid',
    'name':'grid-socar_sweep_v1',
    'metric' : {
        'name': 'val_epoch_iou',
        'goal': 'maximize'   
        },
    'parameters' : {
        'epochs': {
            'value' : 50},
        'batch_size': {
            'value' : 15},
        'optimizer': { 
            'value': 'adabelief'}, # ['adam', 'adabelief']},
        'model': { 
             'values': ['ssl-b0','ssl-b1']}, # 'values': ['ssl-b1', 'imagenet-b1', 'ssl-b0', 'imagenet-b0']},
        'loss': { 
             'values': ['focal', 'cross_entropy']}, # 'values': ['ssl-b1', 'imagenet-b1', 'ssl-b0', 'imagenet-b0']},
        'seed':{
            'value': 0},
        'learning_rate': {
            'value': 1e-3}, #'values': [1e-3, 1e-4]},
    }
}


def mkdir(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

ROOT_DIR = os.path.join('/home/pung/repo/', 'kimin-lab')
DATA_DIR = os.path.join(ROOT_DIR, 'accida_segmentation/dent_mask_only' )
CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoints_dir', 'checkpoints_')
RESULTS_DIR = os.path.join(ROOT_DIR, 'test_results_dir', 'test_results_')

IMGS_DIR = os.path.join(DATA_DIR, 'imgs')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')

TRAIN_IMGS_DIR = os.path.join(IMGS_DIR, 'train')
VAL_IMGS_DIR = os.path.join(IMGS_DIR, 'val')
TEST_IMGS_DIR = os.path.join(IMGS_DIR, 'test')

TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, 'train')
VAL_LABELS_DIR = os.path.join(LABELS_DIR, 'val')
TEST_LABELS_DIR = os.path.join(LABELS_DIR, 'test')

mkdir(
    CKPT_DIR, RESULTS_DIR, IMGS_DIR, LABELS_DIR, TRAIN_IMGS_DIR, VAL_IMGS_DIR, TEST_IMGS_DIR,
    TRAIN_LABELS_DIR, VAL_LABELS_DIR, TEST_LABELS_DIR, 
    )

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# Hyper parameters
class Config:
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 11
    NUM_EPOCHS = 100
