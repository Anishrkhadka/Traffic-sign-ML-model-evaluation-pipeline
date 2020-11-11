import cv2
import glob
import numpy as np
from random import shuffle
import ConfigManager as config
from UtilityManager import preprocess_img, loadDatasetFromPklFormat, read_image
from keras.preprocessing.image import ImageDataGenerator
root = '/home/anish/Documents/CARAMEL/Project/traffic_signs_v2/'

datagen = ImageDataGenerator(rotation_range=180,
                             zoom_range=0.2,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode="nearest",
                             channel_shift_range=0)


# load mask and resize to config.IMG_SIZE
def read_all_mask_image(root):
    root = f'{root}/dataset/attack_pattern_png/'
    maskImagePaths = glob.glob(f'{root}/*.png')
    maskImageList = []
    for imagePath in maskImagePaths:
        mask = read_image(imagePath)
        mask = cv2.resize(mask, (config.IMG_SIZE, config.IMG_SIZE),
                          cv2.INTER_AREA)
        maskImageList.append(mask)
    return maskImageList


# load all the mask at the import --
maskList = read_all_mask_image(root)
datagen.fit(maskList)


def get_box(InImage, InBoxScale, InUP=0, InDown=0, InLeft=0, InRight=0):
    row, col, ch = InImage.shape
    #
    drawingBox_minX = int(col * 0.15) + InBoxScale
    drawingBox_minY = int(row * 0.15) + InBoxScale
    drawingBox_maxX = int(col * 0.95) - InBoxScale
    drawingBox_maxY = int(row * 0.95) - InBoxScale

    h = drawingBox_maxY - drawingBox_minY
    w = drawingBox_maxX - drawingBox_minX
    area = h * w

    if InUP:
        drawingBox_minY += InUP
        drawingBox_maxY += InUP
    elif InDown:
        drawingBox_minY -= InDown
        drawingBox_maxY -= InDown

    if InRight:
        drawingBox_minX += InRight
        drawingBox_maxX += InRight
    elif InLeft:
        drawingBox_minY -= InLeft
        drawingBox_maxY -= InLeft

    return drawingBox_minX, drawingBox_maxX, drawingBox_minY, drawingBox_maxY, h, w, area


def create_box(InImage,
               InScale=1,
               InBoxScale=0,
               InUp=0,
               InDown=0,
               InLeft=0,
               InRight=0):
    # width, height = InImage.shape[:2]

    minX, maxX, minY, maxY, h, w, area = get_box(InImage, InBoxScale, InUp,
                                                 InDown, InLeft, InRight)

    return np.clip(
        cv2.rectangle(InImage, (minX, minY), (maxX, maxY), (0, 0, 0), -1), 0,
        1), area,


def gaussian_noise(InImage,
                   InNoseScale=0.01,
                   InBoxScale=0,
                   InUp=0,
                   InDown=0,
                   InLeft=0,
                   InRight=0):
    minX, maxX, minY, maxY, h, w, area = get_box(InImage, InBoxScale)
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (h, w, 3))
    gauss = gauss.reshape(h, w, 3)
    InImage[minY:maxY, minX:maxX] += gauss * InNoseScale
    return np.clip(InImage, 0, 1), area


def salt_pepper(InImage,
                InScale=0.004,
                InBoxScale=0,
                InUp=0,
                InDown=0,
                InLeft=0,
                InRight=0):
    # row, col, ch = image.shape
    s_vs_p = 0.5
    amount = InScale

    minX, maxX, minY, maxY, h, w, area = get_box(InImage, InBoxScale)
    cropBox = InImage[minY:maxY, minX:maxX]
    out = np.copy(cropBox)

    # Salt mode
    num_salt = np.ceil(amount * cropBox.size * s_vs_p)
    coords = [
        np.random.randint(0, i - 1, int(num_salt)) for i in cropBox.shape
    ]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * cropBox.size * (1. - s_vs_p))
    coords = [
        np.random.randint(0, i - 1, int(num_pepper)) for i in cropBox.shape
    ]
    out[coords] = 0

    InImage[minY:maxY, minX:maxX] = out

    return np.clip(InImage, 0, 1), area


def poission(InImage,
             InNoiseScale=0.004,
             InBoxScale=0,
             InUp=0,
             InDown=0,
             InLeft=0,
             InRight=0):
    minX, maxX, minY, maxY, h, w, area = get_box(InImage, InBoxScale)

    cropBox = InImage[minY:maxY, minX:maxX]

    vals = len(np.unique(cropBox))
    vals = 2**np.ceil(np.log2(vals))
    noisy = np.random.poisson(cropBox * vals) / float(vals)
    InImage[minY:maxY, minX:maxX] = noisy * InNoiseScale

    return np.clip(InImage, 0, 1), area


def speckle(InImage,
            InNoiseScale=0.004,
            InBoxScale=0,
            InUp=0,
            InDown=0,
            InLeft=0,
            InRight=0):
    minX, maxX, minY, maxY, h, w, area = get_box(InImage, InBoxScale)

    # row, col, ch = InImage.shape
    gauss = np.random.randn(h, w, 3)
    gauss = gauss.reshape(h, w, 3)
    InImage[minY:maxY,
            minX:maxX] += InImage[minY:maxY, minX:maxX] * gauss * InNoiseScale
    return np.clip(InImage, 0, 1), area


def apply_pattern_batch(InTrafficSign, InMaskList):
    out = []
    for trafficSign, mask in zip(InTrafficSign, InMaskList):
        # mask = InMaskList[np.random.randint(0, len(InMaskList))]
        # convert mask to 1 channel to 3 channel
        mask = np.squeeze(mask)
        mask = cv2.cvtColor(mask[:, :, 3], cv2.COLOR_GRAY2BGR)
        # convert to 255 if value image is in range of 0-1
        if trafficSign.dtype == np.float64 or trafficSign.dtype == np.float32:
            trafficSign = np.array(trafficSign * 255, dtype=np.uint8)
        mask_out = cv2.subtract(trafficSign, mask)
        out.append(np.clip(np.array(mask_out / 255, dtype=np.float64), 0, 1))
    return out


def get_mask_from_img_gen(InMaskList, InBatchSize=32):
    if InBatchSize < 32:
        for x_batch in datagen.flow(np.array(InMaskList),
                                    batch_size=InBatchSize):
            return np.array(x_batch, dtype=np.uint8)
    else:
        out = []
        for x_batch in datagen.flow(np.array(InMaskList), batch_size=1):
            out.append(x_batch)
            if len(out) > InBatchSize:
                return np.array(out, dtype=np.uint8)


list_of_attack = [salt_pepper, gaussian_noise, create_box, poission, speckle]


def get_noise_attack(InImageBatch, InBatchSize=32):

    for i in range(len(InImageBatch)):
        shuffle(list_of_attack)
        attackIndex = np.random.randint(0, 4)
        noise_intensity = np.random.randint(1, 5)
        box_size = np.random.randint(5, 10)
        attack = list_of_attack[attackIndex]

        InImageBatch[i], _ = attack(InImageBatch[i], noise_intensity,
                                    int(10 / box_size))
    return InImageBatch


def get_noise_attack_test(InImageBatch, InBatchSize=32):

    for i in range(len(InImageBatch)):
        #  shuffle(list_of_attack)
        attackIndex = np.random.randint(0, 4)
        noise_intensity = np.random.randint(1, 5)
        #maxBoxSize = np.random.randint(10, 15)
        box_size = np.random.randint(1, 8)
        attack = list_of_attack[attackIndex]
        up, down, left, right = np.random.randint(0, 3), np.random.randint(
            0, 3), np.random.randint(0, 3), np.random.randint(0, 3)

        InImageBatch[i], _ = attack(InImageBatch[i], noise_intensity,
                                    int(8 / box_size), up, down, left, right)
    return InImageBatch


def get_graffiti_attack(InImageBatch, InBatchSize=32):
    maskImageBatch = get_mask_from_img_gen(maskList, len(InImageBatch))
    attackedBatch = apply_pattern_batch(InImageBatch, maskImageBatch)
    return np.array(attackedBatch)


attack_list = [get_noise_attack, get_graffiti_attack]


def get_attack_augmentated_data(InImageBatch, InBatchSize=32):

    noise_out = get_noise_attack(InImageBatch[:len(InImageBatch) // 2],
                                 len(InImageBatch) // 2)
    grafity_out = get_graffiti_attack(InImageBatch[len(InImageBatch) // 2:],
                                      len(InImageBatch) // 2)
    out = [
        preprocess_img(x, IsRescaleImage=False, IsCenteralCrop=False)
        for x in noise_out
    ]
    out += [
        preprocess_img(x, IsRescaleImage=False, IsCenteralCrop=False)
        for x in grafity_out
    ]
    return np.array(out)


# from tqdm import tqdm
# X_test = loadDatasetFromPklFormat(
#     f'{root}/dataset/GTSRB/BalanceDB/X_balance.pkl')
# for i in tqdm(range(2)):
#     # maskImageBatch = get_mask_from_img_gen(maskList, 32)
#     imageBatch = X_test[np.random.randint(0, X_test.shape[0], size=32)]
#     # attackeBatch = get_noise_attack_test(imageBatch, 32)
#     attackeBatch = get_attack_augmentated_data(imageBatch,32)

#     for j in range(len(attackeBatch)):
#         cv2.imwrite(
#             f'{root}/outputs/attack_dataset_test/{i}_{j}.png',
#             np.array(attackeBatch[j] * 255, dtype=np.uint8)[:, :, ::-1])
