# 第3章语义分割（PSPNet）
# 注意标注图像为调色板形式(索引彩色图像)。

# 包装的import
import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torchvision

class Compose(object):
    """按顺序执行在参数transform中存储的变形的类
              同时变换目标图像和标注图像。
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):

        width = img.size[0]  # img.size=[宽][高]
        height = img.size[1]  # img.size=[宽][高]

        # 随机设定放大倍率
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)  # img.size=[宽][高]
        scaled_h = int(height * scale)  # img.size=[宽][高]

        # 图像的大小调整
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)

        # 标注大小
        anno_class_img = anno_class_img.resize(
            (scaled_w, scaled_h), Image.NEAREST)

        # 把图像调整到原来的大小
        #求剪切位置
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h-height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop(
                (left, top, left+width, top+height))

        else:
            # 比input_size短的边padding
            # p_palette = anno_class_img.copy().getpalette()

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            pad_width = width-scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width))

            pad_height = height-scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0, 0, 0))
            img.paste(img_original, (pad_width_left, pad_height_top))

            anno_class_img = Image.new(
                anno_class_img.mode, (width, height), (0))
            anno_class_img.paste(anno_class_img_original,
                                 (pad_width_left, pad_height_top))
            # anno_class_img.putpalette(p_palette)

        return img, anno_class_img


class RandomRotation(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):

        #决定旋转角度
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1]))

        #旋转
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img


class RandomMirror(object):
    """以50%的概率左右反转的类别"""

    def __call__(self, img, anno_class_img):
        if np.random.randint(2):
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        return img, anno_class_img


class Resize(object):
    """将参数input_size的大小变形的类"""

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):

        # width = img.size[0]  # img.size=[宽][高]
        # height = img.size[1]  # img.size=[宽][高]

        img = img.resize((self.input_size, self.input_size),
                         Image.BICUBIC)
        anno_class_img = anno_class_img.resize(
            (self.input_size, self.input_size), Image.NEAREST)

        return img, anno_class_img


class Normalize_Tensor(object):
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):

        #将PIL图像变成Tensor。最大规格为1
        img = transforms.functional.to_tensor(img)

        # 颜色信息的标准化
        img = transforms.functional.normalize(
            img, self.color_mean, self.color_std)

        #将标注图像转换为Numpy
        anno_class_img = np.array(anno_class_img)  # [高][宽]
        a = np.unique(anno_class_img)
        # 因为‘ambigious’中存储了255，所以作为0的背景。
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0

        # 将标注图像添加到Tensor
        anno_class_img = torch.from_numpy(anno_class_img)
        B = np.unique(anno_class_img)

        return img, anno_class_img
