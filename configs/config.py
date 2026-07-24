import os
import torch

class Config:
    # Seed for reproducibility
    SEED = 42
    
    # Dataset Configs
    CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    NUM_CLASSES = len(CLASSES)
    CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
    IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}
    
    # Image Preprocessing & Augmentation
    IMAGE_SIZE = (224, 224)
    USE_CLAHE = True
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    
    # Normalization (ImageNet Standard)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    
    # Training Hyperparameters (Teacher)
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    USE_AMP = True  # Automatic Mixed Precision
    
    # Knowledge Distillation Hyperparameters
    KD_EPOCHS = 25
    KD_LEARNING_RATE = 1e-3
    KD_TEMPERATURE = 4.0        # Softmax temperature (higher = softer distributions)
    KD_ALPHA = 0.3              # Weight: hard label CE loss
    KD_BETA = 0.5               # Weight: soft label KD loss
    KD_GAMMA = 0.2              # Weight: attention transfer loss
    STUDENT_MODEL = 'mobilenetv3'  # Options: 'mobilenetv3', 'efficientnet_b0'
    
    # Hardware Config
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoint & Directory Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'OCT2017')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    RESULT_DIR = os.path.join(OUTPUT_DIR, 'results')
    VISUALS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')
    
    @classmethod
    def setup_directories(cls):
        for path in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.CHECKPOINT_DIR, 
                     cls.LOG_DIR, cls.RESULT_DIR, cls.VISUALS_DIR]:
            os.makedirs(path, exist_ok=True)
