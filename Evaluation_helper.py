# Evaluation pipeline helper.py
import numpy as np
from Attack_Augmentation import salt_pepper, gaussian_noise, create_box, poission, speckle
from UtilityManager import loadDatasetFromPklFormat, saveDatasetInPklFormat, class_cat, preprocess_img
from tqdm import tqdm
from ModelManager import load_model_from_json, loadWeights
import LogManager
import UtilityManager
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix

noise_attacks = [salt_pepper, gaussian_noise, poission, speckle]
noise_attacks_name = ['salt_pepper', 'gaussian_noise', 'poission', 'speckle']
# noise_attacks = [salt_pepper, gaussian_noise, create_box, poission, speckle]

ROOT = '/home/anish/Documents/CARAMEL/Project/traffic_signs_v2/'
Y_PATH = f'{ROOT}/dataset/Synthetic/Y_test_synth.pkl'
X_PATH = f'{ROOT}/dataset/Synthetic/X_test_synth.pkl'

# x_test = loadDatasetFromPklFormat(X_PATH)

y_test = loadDatasetFromPklFormat('dataset/evaluation_dataset_Y.pkl')
x_test = loadDatasetFromPklFormat('dataset/evaluation_dataset_X.pkl')

labelNames = open(f"signName.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

mask = UtilityManager.loadDatasetFromPklFormat(
    f'{ROOT}/dataset/GTSRB/mask.pkl', False)

anomaly_model = load_model_from_json(
    f'{ROOT}/Models/anomaly_detection_vae_version_18_for_coral.json')
# anomaly_model.summary()
anomaly_model = loadWeights(
    anomaly_model,
    f'{ROOT}/Weights/anomaly_detection_vae_synth_version_18_for_coral.h5',
    False)

traffic_sign_recognition_model = load_model_from_json(
    f'{ROOT}/Models/traffic_signs_recognition_v5_for_coral.json')
traffic_sign_recognition_model = loadWeights(
    traffic_sign_recognition_model,
    f'{ROOT}/Weights/traffic_signs_recognition_v5_for_coral_synth.h5', False)

anomaly_reconstruction_model = load_model_from_json(
    f'{ROOT}Models/traffic_to_meta_traffic_version_5_for_coral.json')
anomaly_reconstruction_model = loadWeights(
    anomaly_reconstruction_model,
    f'{ROOT}/Weights/traffic_to_meta_traffic_version_5_for_coral.h5', False)

# run once
# class_index = get_class_index()
# saveDatasetInPklFormat('Class_index.pkl', class_index)
# for ts_class in range(1):
# ts_class = 0
# print(f'{ts_class}/43')
# traffic_sign_type = ts_class
# print(f' Traffic Sign Label:{labelNames[traffic_sign_type]}')
# class_index = loadDatasetFromPklFormat('Class_index.pkl', False)

# x = class_index[traffic_sign_type]


# 400 image from each classy
def get_class_index():
    all_class_index = []
    for x in range(43):
        in_class = []
        for i in range(len(y_test)):
            if y_test[i] == x and len(in_class) < 400:
                in_class.append(i)
        all_class_index.append(in_class)

    return all_class_index


def apply_noise_attack(InTestImage, InNoise, InIntensity, InNoiseArea):
    # traffic_signs = x_test[InClassIndex]
    # traffic_signs = InTestImage
    noise_signs = InTestImage.copy()
    # LogManager.displayLog('Applying Noise Attack ...')
    attack_area = 0
    for x in range(InTestImage.shape[0]):
        image, attack_area = InNoise(noise_signs[x], InIntensity, InNoiseArea)
    out = [
        preprocess_img(x, IsRescaleImage=False, IsCenteralCrop=False)
        for x in noise_signs
    ]
    return np.array(InTestImage), np.array(out), attack_area


import cv2, glob
from skimage import io
from skimage.transform import resize
maskPath = glob.glob('dataset/pattern_attack/*.png')
maskPath.sort()
masklist = [resize(io.imread(x), (48, 48)) for x in maskPath]


def apply_pattern_attack(InTestImage, InMaskIndex, InSize=1):
    pattern = np.array(masklist[InMaskIndex] * 255, dtype=np.uint8)[:, :, 3]
    pattern = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

    patten_attack = InTestImage.copy()

    for x in range(InTestImage.shape[0]):
        image = patten_attack[x]
        # convert to 255 if value image is in range of 0-1
        if image.dtype == np.float64 or image.dtype == np.float32:
            image = np.array(image * 255, dtype=np.uint8)
        mask_out = cv2.subtract(image, pattern)
        patten_attack[x] = np.clip(np.array(mask_out / 255, dtype=np.float64),
                                   0, 1)

    return InTestImage, patten_attack, maskPath[InMaskIndex].split('.')[-2]


def MaskMeanSquareError(yTrue, yPred):
    return np.sqrt(np.sqrt(np.mean(np.square(yTrue - yPred))))


def anomaly_detection_per_class(InNormal, InAbnormal, InTotalSignPerClass=300, InFileName=None):
    total_signs_per_class = InTotalSignPerClass
    label_names = ['Normal', 'Anomaly']
    LogManager.displayLog(f'Label, precission, recall, f1-score, confuse matrix')
    for i in range(43):
        normal = InNormal[i * total_signs_per_class:total_signs_per_class *
                          (1 + i)]
        abnormal = InAbnormal[i * total_signs_per_class:total_signs_per_class *
                              (1 + i)]
        out = anomaly_detection(normal, abnormal, f'{InFileName}_{labelNames[i]}.png')
        print(
            f"{labelNames[i]}, {out['precision']}, {out['recall']},  {out['f1-score']}")
        # print(f'{confusion_mat}')
        # LogManager.displayLog(
        #     confusion_matrix(final_dataset_y,
        #                      np.array(errors >= threshold),
        #                      labels=label_names, normalize=True), 'white')

        # UtilityManager.get_confusion_matrix(final_dataset_y, np.array(errors >= threshold),
        #                                     InFileName)


def anomaly_detection(InNormalSigns, InAbnormalSigns, InFileName=None):
    labelNames = ['Normal', 'Anomaly']

    finalDataset = np.concatenate([InNormalSigns, InAbnormalSigns])
    y_ = [0] * len(InNormalSigns) + [1] * len(InAbnormalSigns)

    predictions = anomaly_model.predict(np.array(finalDataset),
                                        batch_size=1024,
                                        verbose=False)

    mse = 0
    errors = []
    i = 0
    x = []

    for batchLoop in range(len(finalDataset)):
        # image, pred = finalDataset[i], predictions[i]
        image, pred = UtilityManager.apply_mask(
            finalDataset[i], 0,
            mask), UtilityManager.apply_mask(predictions[i], 0, mask)
        mse = MaskMeanSquareError(image, pred)
        errors.append(mse)
        x.append(i)
        i += 1

    threshold = np.array(0.3093111931900183)
    out = classification_report(np.array(y_),
                                np.array(errors >= threshold),
                                target_names=labelNames,
                                output_dict=True)


    # LogManager.displayLog((classification_report(np.array(y_),
    #                                              np.array(errors >= threshold),
    #                                              target_names=labelNames,
    #                                              output_dict=True)))

    UtilityManager.get_confusion_matrix(y_, np.array(errors >= threshold),
                                        InFileName)
    # LogManager.displayLog(confusion_matrix(y_, np.array(errors >= threshold)),
    #                       'white')
    return out['Anomaly']


def recoganise_traffic_signs(X_test, Y_test, InFilename=None):
    # label = [f'Not {labelNames}', f'{labelNames}']
    # Y_test = np.array([1] * len(X_test))

    predictions = traffic_sign_recognition_model.predict(np.array(X_test),
                                                         batch_size=1024)

    # LogManager.displayLog((classification_report(Y_test.argmax(axis=-1),
    #                                              predictions.argmax(axis=-1),
    #                                              target_names=labelNames)),
    #                       'grey')
    out = classification_report(Y_test.argmax(axis=-1),
                                                 predictions.argmax(axis=-1),
                                                 target_names=labelNames, output_dict=True)
    for i in range(43):
        # print(out[labelNames[i]])
        results=out[labelNames[i]]
        print(
            f"{labelNames[i]}, {results['precision']}, {results['recall']},  {results['f1-score']}")
    print(f"Model accuracy:{out['accuracy']}")
    UtilityManager.get_confusion_matrix(Y_test.argmax(axis=-1),
                                        predictions.argmax(axis=-1),InFilename)
    # print(confusion_matrix(Y_test.argmax(axis=-1), predictions.argmax(axis=-1)))


def anomaly_traffic_sign_reconstruction(In_Anomaly_Crop_Traffic_Signs,
                                        In_Name=None):

    predictions = anomaly_reconstruction_model.predict(
        np.array(In_Anomaly_Crop_Traffic_Signs), batch_size=1024)

    return predictions


def plot_figure(InImages, InFileName):
    plt.figure(figsize=(10, 10))
    for i in range(InImages.shape[0]):
        plt.subplot(10, 10, i + 1)
        image = InImages[i, :, :, :]
        plt.imshow((image * 255).astype(np.uint8))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(InFileName)
    plt.close()
