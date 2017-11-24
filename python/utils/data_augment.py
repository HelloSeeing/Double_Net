import sys
import numpy as np
import math
import cv2
from PIL import Image

class Center_Expand:
  def __init__(self):
    pass

  def get_bound_color(self, img, bound_width=5):
    if img.shape[0] < 2 * bound_width or img.shape[1] < 2 * bound_width:
      return np.mean(img, axis=(0, 1))
    center = img[bound_width:-bound_width, bound_width:-bound_width]
    bound_count = img.shape[0] * img.shape[1] - center.shape[0] * center.shape[1]
    bound_color = (np.sum(img, axis=(0, 1)) - np.sum(center, axis=(0, 1))) / bound_count
    return bound_color

  def apply(self, image, target_size):
    if image.shape == target_size:
      return image

    img_height = image.shape[0]
    img_width = image.shape[1]
    try:
      img_depth = image.shape[2]
    except IndexError:
      img_depth = None
    target_height = target_size[0]
    target_width = target_size[1]
    height_scale = 1.0 * img_height / target_height
    width_scale = 1.0 * img_width / target_width
    scale = max(height_scale, width_scale)

    ideal_height = int(img_height / scale / 2) * 2
    ideal_width = int(img_width / scale / 2) * 2

    # image.shape = (h0, w0)
    # ideal_shape = (h, w)
    # resized_img = cv2.resize(image)
    # resized_img.shape = (w, h)
    resized_img = cv2.resize(image, (ideal_width, ideal_height))


    bound_color = self.get_bound_color(image)

    expanded = np.array([bound_color] * target_height * target_width)
    if img_depth != 1:
      expanded = np.reshape(expanded, (target_height, target_width, img_depth))
    else:
      expanded = np.reshape(expanded, (target_height, target_width))

    height_gap = int((target_height - ideal_height) / 2)
    width_gap = int((target_width - ideal_width) / 2)

    expanded[height_gap:target_height-height_gap, width_gap:target_width-width_gap] = resized_img.copy()
    expanded = np.uint8(expanded)

    return expanded

class Color_Balance:
  def __init__(self):
    pass

  def _apply_mask(self, matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

  def _apply_threshold(self, matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = self._apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = self._apply_mask(matrix, high_mask, high_value)

    return matrix
  def apply(self, img, percent=1):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
      assert len(channel.shape) == 2
      # find the low and high precentile values (based on the input percentile)
      height, width = channel.shape
      vec_size = width * height
      flat = channel.reshape(vec_size)

      assert len(flat.shape) == 1

      flat = np.sort(flat)

      n_cols = flat.shape[0]

      low_val  = flat[int(math.floor(n_cols * half_percent))]
      high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]

      #print "Lowval: ", low_val
      #print "Highval: ", high_val

      # saturate below the low percentile and above the high percentile
      thresholded = self._apply_threshold(channel, low_val, high_val)
      # scale the channel
      normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
      out_channels.append(normalized)

    return cv2.merge(out_channels)
