# 5.将图像透明化并重叠在一起
import matplotlib as plt
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as T
import mmcv
import cv2
def save_mask(pred, img):
    image = mpimg.imread(img)
    # image.flags.writeable = True
    Image.fromarray(np.uint8(image))
    
    pred = Image.fromarray(np.uint8(pred))

    image[:,:,:][pred[:,:,:]>0] = 255
    cv2.imwrite('test.png', image)




# def show_result(img,
#                 result,
#                 n_classes,
#                 idx,
#                 palette=None,
#                 win_name='',
#                 show=False,
#                 wait_time=0,
#                 out_file=None,
#                 opacity=0.5,):
#     img = img.squeeze(0).permute(1, 2, 0).numpy()
#     img = img.copy() # (h, w, 3)
#     seg = result[0]  # seg.shape=(h, w). The value in the seg represents the index of the palette.
#     if palette is None:
#         palette = np.random.randint(
#             0, 255, size = (len(n_classes), 3)
#         )
#     else:
#         palette = palette
#     palette = np.array(palette)
#     assert palette.shape[0] == len(n_classes)
#     assert palette.shape[1] == 3
#     assert len(palette.shape) == 2
#     assert 0 < opacity <= 1.0
#     color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # (h, w, 3). Drawing board.
#     for label, color in enumerate(palette):
#         color_seg[seg == label, :] = color  # seg.shape=(h, w). The value in the seg represents the index of the palette.
#         # convert to BGR
#     color_seg = color_seg[..., ::-1]
 
#     img = img * (1 - opacity) + color_seg * opacity
#     img = img.astype(np.uint8)
#     # if out_file specified, do not show image in window
    
#     if out_file is not None:
#         show = False
#     if show:
#         mmcv.imshow(img, win_name, wait_time)
#     if out_file is not None:
#         mmcv.imwrite(img, out_file + f'{idx}' + 'result.jpg')