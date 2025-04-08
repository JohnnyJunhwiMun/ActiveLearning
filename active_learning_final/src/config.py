import os

# Get the absolute path of the workspace
WORKSPACE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data related settings
DATA_DIR = r'C:\Users\johnn\Desktop\python_prac\active_learning\MP_mixed'
UNLABELED_DIR = r'C:\Users\johnn\Desktop\python_prac\active_learning\MP_mixed\Unlabeled'
TEST_DIR = r'C:\Users\johnn\Desktop\python_prac\active_learning\MP_testset'
INITIAL_MODEL_PATH = r'C:\Users\johnn\Desktop\python_prac\active_learning\initial_model.h5'
BEST_MODEL_PATH = r'C:\Users\johnn\Desktop\python_prac\active_learning\best_model.h5'

# Model related settings
INPUT_DIM = 1662
NUM_CLASSES = 3
ACTIONS = ['Like', 'Fire works', 'Heart']

# Training related settings
LEARNING_RATE = 0.001
NUM_EPOCHS = 1500
BATCH_SIZE = 32
SEQUENCE_LENGTH = 30
MIN_LOSS_THRESHOLD = 0.001  # If loss is lower than this value, training stops

# Active Learning related settings
UNCERTAINTY_SAMPLING_SIZE = 10
STOPPING_CRITERIA = 0.9  # F1 score threshold

# Visualization related settings
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
LANDMARK_COLORS = {
    'pose': (0, 255, 0),    # 녹색
    'face': (255, 0, 0),    # 파란색
    'hand': (0, 0, 255)     # 빨간색
} 