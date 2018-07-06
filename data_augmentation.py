import numpy as np

import cv2 as cv
from scipy.misc import imread
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]


def random_crop(img, area_ratio, hw_vari):
    """
    area_ratio:the proportion of the crop image to the original
    hw_vari: The proportion of the disturbance to the ratio of the original height to width
    """
    h, w = img.shape[:2]
    hw_delta = np.random.uniform(-hw_vari, hw_vari)
    hw_mult = 1 + hw_delta

    w_crop = int(round(w * np.sqrt(area_ratio * hw_mult)))
    if w_crop > w:
        w_crop = w
    h_crop = int(round(h * np.sqrt(area_ratio / hw_mult)))
    if h_crop > h:
        h_crop = h
    # generate the upper left coordinate randomly
    x0 = np.random.randint(0, w - w_crop + 1)
    y0 = np.random.randint(0, h - h_crop + 1)

    return crop_image(img, x0, y0, w_crop, h_crop)


def rotate_image(img, angle, crop):
    """
    rotate an image
    angle: angle against clockwise rotation
    crop: bool, indicate if the black edge should be removed
    """
    h, w = img.shape[:2]
    angle %= 360
    M_rotate = cv.getRotationMatrix2D((w / 2, h / 2), angle, 1)

    img_rotated = cv.warpAffine(img, M_rotate, (w, h))
    if crop:
        angle_crop = angle % 180  # period=180
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180.0 # transform angle to radian
        hw_ratio = float(h) / float(w) # calculation of the ratio of height to width
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        r = hw_ratio if h > w else 1 / hw_ratio
        denominator = r * tan_theta + 1 # calculate denominator
        crop_mult = numerator / denominator

        w_crop = int(round(crop_mult * w))
        h_crop = int(round(crop_mult * h))
        x0 = int((w - w_crop) / 2)
        y0 = int((h - h_crop) / 2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated


def random_rotate(img, angle_vari, p_crop):
    """
    rotate an image with a random angle
    """
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True
    return rotate_image(img, angle, crop)


def hsv_transform(img, hue_delta, sat_mult, val_mult):
    """
    deploy hsv transfomation
    hue_delta: hue change ratio
    sat_mult:  saturation change ratio
    val_mult: value change ratio
    """
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float)
    img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
    img_hsv[:, :, 1] *= sat_mult
    img_hsv[:, :, 2] *= val_mult
    img_hsv[img_hsv > 255] = 255
    return cv.cvtColor(np.round(img_hsv).astype(np.uint8), cv.COLOR_HSV2BGR)


def random_hsv_transform(img, hue_vari, sat_vari, val_vari):
    """
    random hsv transformation using the given range of hsv
    """
    hue_delta = np.random.randint(-hue_vari, hue_vari)
    sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
    val_mult = 1 + np.random.uniform(-val_vari, val_vari)
    return hsv_transform(img, hue_delta, sat_mult, val_mult)


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv.LUT(img, gamma_table)


def random_gama_transform(img, gamma_vari):

    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)

