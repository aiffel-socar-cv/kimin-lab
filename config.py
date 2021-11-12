import os
import torch

def mkdir(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

ROOT_DIR = os.path.dirname(os.path.abspath('__file__'))
DATA_DIR = os.path.join(ROOT_DIR, 'accida_segmentation')
CKPT_DIR = os.path.join(ROOT_DIR, 'checkpoints')

IMGS_DIR = os.path.join(DATA_DIR, 'imgs')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')

TRAIN_IMGS_DIR = os.path.join(IMGS_DIR, 'train')
VAL_IMGS_DIR = os.path.join(IMGS_DIR, 'val')
TEST_IMGS_DIR = os.path.join(IMGS_DIR, 'test')

TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, 'train')
VAL_LABELS_DIR = os.path.join(LABELS_DIR, 'val')
TEST_LABELS_DIR = os.path.join(LABELS_DIR, 'test')

mkdir(
    CKPT_DIR, IMGS_DIR, LABELS_DIR, TRAIN_IMGS_DIR, VAL_IMGS_DIR, TEST_IMGS_DIR,
    TRAIN_LABELS_DIR, VAL_LABELS_DIR, TEST_LABELS_DIR,
    )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
class Config:
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 11
    NUM_EPOCHS = 100