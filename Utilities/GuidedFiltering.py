# Guided filtering operator, DO NOT EDIT!!!
import numpy as np

def box_filter(imgSrc, r):
    """
    Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
    :param imgSrc: np.array, image
    :param r: int, radius
    :return: imDst: np.array. result of calculation
    """
    if imgSrc.ndim == 2:
        h, w = imgSrc.shape[:2]
        imDst = np.zeros(imgSrc.shape[:2])

        # cumulative sum over h axis
        imCum = np.cumsum(imgSrc, axis=0)
        # difference over h axis
        imDst[0: r + 1] = imCum[r: 2 * r + 1]
        imDst[r + 1: h - r] = imCum[2 * r + 1: h] - imCum[0: h - 2 * r - 1]
        imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

        # cumulative sum over Nets axis
        imCum = np.cumsum(imDst, axis=1)

        # difference over Nets axis
        imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
        imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
        imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r]) - \
                             imCum[:, w - 2 * r - 1: w - r - 1]
    else:
        h, w = imgSrc.shape[:2]
        imDst = np.zeros(imgSrc.shape)

        # cumulative sum over h axis
        imCum = np.cumsum(imgSrc, axis=0)
        # difference over h axis
        imDst[0: r + 1] = imCum[r: 2 * r + 1]
        imDst[r + 1: h - r, :] = imCum[2 * r + 1: h, :] - imCum[0: h - 2 * r - 1, :]
        imDst[h - r: h, :] = np.tile(imCum[h - 1, :], [r, 1, 1]) - imCum[h - 2 * r - 1: h - r - 1, :]

        # cumulative sum over Nets axis
        imCum = np.cumsum(imDst, axis=1)

        # difference over Nets axis
        imDst[:, 0: r + 1] = imCum[:, r: 2 * r + 1]
        imDst[:, r + 1: w - r] = imCum[:, 2 * r + 1: w] - imCum[:, 0: w - 2 * r - 1]
        imDst[:, w - r: w] = np.tile(np.expand_dims(imCum[:, w - 1], axis=1), [1, r, 1]) - \
                             imCum[:, w - 2 * r - 1: w - r - 1]
    return imDst


def guided_filter(I, p, r, eps=0.1):
    """
    Guided Filter
    :param I: np.array, guided image
    :param p: np.array, input image
    :param r: int, radius
    :param eps: float
    :return: np.array, filter result
    """
    h, w = I.shape[:2]
    if I.ndim == 2:
        N = box_filter(np.ones((h, w)), r)
    else:
        N = box_filter(np.ones((h, w, 1)), r)
    mean_I = box_filter(I, r) / N
    mean_p = box_filter(p, r) / N
    mean_Ip = box_filter(I * p, r) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = box_filter(I * I, r) / N
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)

    if I.ndim == 2:
        b = mean_p - a * mean_I
        mean_a = box_filter(a, r) / N
        mean_b = box_filter(b, r) / N
        q = mean_a * I + mean_b
    else:
        b = mean_p - np.expand_dims(np.sum((a * mean_I), 2), 2)
        mean_a = box_filter(a, r) / N
        mean_b = box_filter(b, r) / N
        q = np.expand_dims(np.sum(mean_a * I, 2), 2) + mean_b
    return q


