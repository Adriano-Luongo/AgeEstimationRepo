import csv
import random
import cv2
import sys
import os
import numpy as np


def enclosing_square(rect):
    def _to_wh(s, l, ss, ll, width_is_long):
        if width_is_long:
            return l, s, ll, ss
        else:
            return s, l, ss, ll

    def _to_long_short(rect):
        x, y, w, h = rect
        if w > h:
            l, s, ll, ss = x, y, w, h
            width_is_long = True
        else:
            s, l, ss, ll = x, y, w, h
            width_is_long = False
        return s, l, ss, ll, width_is_long

    s, l, ss, ll, width_is_long = _to_long_short(rect)

    hdiff = (ll - ss) // 2
    s -= hdiff
    ss = ll

    return _to_wh(s, l, ss, ll, width_is_long)


def add_margin(roi, qty):
    return (
        roi[0] - qty,
        roi[1] - qty,
        roi[2] + 2 * qty,
        roi[3] + 2 * qty)


def cut(frame, roi):
    pA = (int(roi[0]), int(roi[1]))
    pB = (int(roi[0] + roi[2] - 1), int(roi[1] + roi[3] - 1))  # pB will be an internal point
    W, H = frame.shape[1], frame.shape[0]
    A0 = pA[0] if pA[0] >= 0 else 0
    A1 = pA[1] if pA[1] >= 0 else 0
    data = frame[A1:pB[1], A0:pB[0]]
    if pB[0] < W and pB[1] < H and pA[0] >= 0 and pA[1] >= 0:
        return data
    w, h = int(roi[2]), int(roi[3])
    img = np.zeros((h, w, frame.shape[2]), dtype=np.uint8)
    offX = int(-roi[0]) if roi[0] < 0 else 0
    offY = int(-roi[1]) if roi[1] < 0 else 0
    np.copyto(img[offY:offY + data.shape[0], offX:offX + data.shape[1]], data)
    return img


############### CUTOUT ################################
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.15, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser


############################################
def findRelevantFace(objs, W, H):
    mindistcenter = None
    minobj = None
    for o in objs:
        cx = o['roi'][0] + (o['roi'][2] / 2)
        cy = o['roi'][1] + (o['roi'][3] / 2)
        distcenter = (cx - (W / 2)) ** 2 + (cy - (H / 2)) ** 2
        if mindistcenter is None or distcenter < mindistcenter:
            mindistcenter = distcenter
            minobj = o
    return minobj


def top_left(f):
    return (f['roi'][0], f['roi'][1])


def bottom_right(f):
    return (f['roi'][0] + f['roi'][2], f['roi'][1] + f['roi'][3])


