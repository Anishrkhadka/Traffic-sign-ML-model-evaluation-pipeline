import os
root = os.path.dirname(os.path.abspath(__file__))

BATCH_STEPS = 8
EPOCH_START = 0

TOTAL_EPOCHS = 500
MIN_EPOCHS = 4
DATASET_NAME = 'GTSRB'
ROOT_DIR = 'dataset/GTSRB/Final_Training/Images/'
PKL_DIR = 'dataset/GTSRB/'
WEIGHTS_DIR = 'Weights/'
NUM_CLASSES = 43
IMG_SIZE = 48

LOG_DIR = 'tensorLog'
MODEL_INDEX = 0

LR = 1e-4
MIN_LR = 1e-4
MAX_LR = 1e-5

MAX_BATCH_SIZE = 64
MIN_BATCH_SIZE = 8
BATCH_CYCLE_STEPS = 8


# cycle learning rate
STEP_SIZE = 8
CLR_METHOD_OPTION = ['triangular', 'triangular2', 'exp_range']
CLR_METHOD = CLR_METHOD_OPTION[2]
LEARNING_RATE_OPTION = ['cycle', 'poly', 'fixed']
CURRENT_LR_METHOD = LEARNING_RATE_OPTION[1]

METHOD_OPTION = ['triangular', 'triangular2', 'exp_range']
BATCH_RATE_OPTION = ['cycle', 'poly']
CURRENT_BATCHSIZE_METHOD = BATCH_RATE_OPTION[1]
BATCH_CLR_METHOD = METHOD_OPTION[2]



#Configuration for detection i.e traffic_sign_detection.py
PATH_TO_TENSORFLOW_RESEARCH = f'ExternalLib/tensorflow/models/research/'
PATH_TO_TENSORFLOW_OBJECT_DETECTION_FOLDER = f'{PATH_TO_TENSORFLOW_RESEARCH}/object_detection/'
PATH_TO_YOLO_FOLDER = f'ExternalLib/darkflow/'

MODEL_NAME_LIST = [
    'ssd_mobilenet_v1', 'faster_rcnn_resnet_101', 'faster_rcnn_inception_v2',
    'rfcn_resnet101', 'ssd_inception_v2', 'ssd_mobilenet_v2_synth'
]

MODEL_NAME_FOR_DETECTION = MODEL_NAME_LIST[5]
MODEL_PATH = f'ExternalLib/models/{MODEL_NAME_FOR_DETECTION}'
PATH_TO_CKPT = f'{MODEL_PATH}/inference_graph/frozen_inference_graph.pb'

#--  Current model version -- #
AUTOGAN = 'version_8'
AUTOENCODER = 'version_4'  
ANOMALYCLASSIFICATION = 'version_2'
RECOGNITION = 'version_3' #best v3
RECONSTRUCT = 'version_3'
PRETRAIN = 'pre_trained_version_1'
ANOMALY_DETECTION = 'version_12'
ANOMALY_DETECTION_VAE = 'version_18' # best:v17
TO_META = 'version_5' #best : v5
TO_FOURIER = 'version_2'
ANOMALY_DETECTION_WITH_SCORE = 'version_2'
#ANOMALY_DETECTION_VAE good = v7
