import cv2
import csv
import os

import matplotlib.pyplot as plt
import torch
import numpy
import numpy as np
import torch.nn as nn


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    img = img.astype(np.float64)
    return img


def mse(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate MSE (Mean Squared Error).

    Ref: https://en.wikipedia.org/wiki/Mean_squared_error

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the MSE calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: mse result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    return np.mean((img1 - img2)**2)


def lfd(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate LFD (Log Frequency Distance).

    Ref:
    Focal Frequency Loss for Image Reconstruction and Synthesis. In ICCV 2021.
    <https://arxiv.org/pdf/2012.12821.pdf>

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the LFD calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: lfd result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    img1 = img1.transpose(2, 0, 1)
    img2 = img2.transpose(2, 0, 1)
    freq1 = np.fft.fft2(img1)
    freq2 = np.fft.fft2(img2)
    return np.log(np.mean((freq1.real - freq2.real)**2 + (freq1.imag - freq2.imag)**2) + 1.0)


def psnr(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, crop_border=0, input_order='HWC'):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def log_values_to_csv(values, csv_file_path):
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(values)


def metrics_calculate_and_log(batch_tensor1, batch_tensor2, metrics, log_file_path, name, epoch):
    # batch_tensor1 is the tgt, batch_tensor2 is the pred.
    # batch_tensor1 and batch tensor2 are torch.Tensor, shape is N C H W !
    # the keywords of the metrics are as followed: [ssim, psnr, mse, fid, floss]
    # parameter log_file_path is open the csv file and save path !
    # parameter log_name_flag is to control whether to save the metrics name in the csv files !

    assert len(batch_tensor1.shape) == len(batch_tensor2.shape) == 4, "the tensor shape should be same !(N C H W) "
    assert log_file_path is not None, "the file path should not be None !"
    assert isinstance(metrics, (str, tuple, list)), "the metrics should be str, tuple or list !"

    # to detect the keywords.
    if isinstance(metrics, str):
        metrics = [metrics]
    for metric in metrics:
        if metric not in ["ssim", "psnr", "mse", "lfd", "floss"]:
            raise ValueError("The keywords of metrics only can include ssim, psnr, mse, lfd, floss !")

    csv_save_path = log_file_path + name

    pred_img_path = os.path.join(log_file_path, 'pred')
    target_img_path = os.path.join(log_file_path, 'target')

    if not os.path.exists(pred_img_path):
        os.makedirs(pred_img_path)
    if not os.path.exists(target_img_path):
        os.makedirs(target_img_path)

    # turn the input tensors to numpy array!
    for i in range(batch_tensor1.shape[0]):
        temp1 = batch_tensor1[i, ...].detach().cpu().numpy().squeeze(0)
        temp2 = batch_tensor2[i, ...].detach().cpu().numpy().squeeze(0)

        # map the data from [0, 1] to [0, 255]
        temp1 = (temp1 - numpy.min(temp1)) / (numpy.max(temp1) - numpy.min(temp1))
        temp2 = (temp2 - numpy.min(temp2)) / (numpy.max(temp2) - numpy.min(temp2))
        img1 = numpy.asarray(temp1 * 255, dtype=int)
        img2 = numpy.asarray(temp2 * 255, dtype=int)

        # calculate 1 mini-batch value parameters.
        metric_values = []
        for metric in metrics:
            if metric == "ssim":
                s = ssim(img1, img2, 0, 'HWC')
                metric_values.append(s)
                print("ssim: %f" % s)
            elif metric == "psnr":
                p = psnr(img1, img2, 0, "HWC")
                metric_values.append(p)
                print("psnr: %f" % p)
            elif metric == "mse":
                m = mse(img1, img2, 0, "HWC")
                metric_values.append(m)
                print("mse: %f" % m)
            elif metric == "lfd":
                l = lfd(img1, img2, 0, "HWC")
                metric_values.append(l)
                print("lfd: %f" % l)
            # elif metric == "floss":
            #     metric_values.append(mse(img1, img2, 0, "HWC"))

        cv2.imwrite(os.path.join(target_img_path, 'batch{}_{}.jpg'.format(epoch, i)), img1)
        cv2.imwrite(os.path.join(pred_img_path, 'batch{}_{}.jpg'.format(epoch, i)), img2)

        # log 1 mini-batch metric values !
        log_values_to_csv(metric_values, csv_save_path)


def metrics_calculate(batch_tensor1, batch_tensor2, metrics):
    assert len(batch_tensor1.shape) == len(batch_tensor2.shape) == 4, "the tensor shape should be same !(N C H W) "
    assert isinstance(metrics, (str, tuple, list)), "the metrics should be str, tuple or list !"

    # to detect the keywords.
    if isinstance(metrics, str):
        metrics = [metrics]
    for metric in metrics:
        if metric not in ["ssim", "psnr", "mse", "lfd", "floss"]:
            raise ValueError("The keywords of metrics only can include ssim, psnr, mse, lfd, floss !")

    batch_num = batch_tensor1.shape[0]

    # turn the input tensors to numpy array!
    ssim_bar, psnr_bar, mse_bar, lfd_bar = 0, 0, 0, 0
    for i in range(batch_tensor1.shape[0]):
        temp1 = batch_tensor1[i, ...].detach().cpu().numpy().squeeze(0)
        temp2 = batch_tensor2[i, ...].detach().cpu().numpy().squeeze(0)

        # map the data from [0, 1] to [0, 255]
        temp1 = (temp1 - numpy.min(temp1)) / (numpy.max(temp1) - numpy.min(temp1))
        temp2 = (temp2 - numpy.min(temp2)) / (numpy.max(temp2) - numpy.min(temp2))
        img1 = numpy.asarray(temp1 * 255, dtype=int)
        img2 = numpy.asarray(temp2 * 255, dtype=int)

        # calculate 1 mini-batch value parameters.
        for metric in metrics:
            if metric == "ssim":
                s = ssim(img1, img2, 0, 'HWC')
                ssim_bar += s
            elif metric == "psnr":
                p = psnr(img1, img2, 0, "HWC")
                psnr_bar += p
            elif metric == "mse":
                m = mse(img1, img2, 0, "HWC")
                mse_bar += m
            elif metric == "lfd":
                l = lfd(img1, img2, 0, "HWC")
                lfd_bar += l

    return ssim_bar/batch_num, psnr_bar/batch_num, mse_bar/batch_num, lfd_bar/batch_num


def print_and_write_log(message, log_file=None):
    """Print message and write to a log file.

    Args:
        message (str): The message to print out and log.
        log_file (str, optional): Path to the log file. Default: None.
    """
    print(message)
    if log_file is not None:
        with open(log_file, 'a+') as f:
            f.write('%s\n' % message)


def print_and_log(message, log_file_path):
    print(message)
    log_values_to_csv(message, log_file_path)

