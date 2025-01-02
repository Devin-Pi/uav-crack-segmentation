import lightning as l
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
# from model.unet import UNet
from model.segnet import SegNet
from model.ussa import UNet
from dataload.dataload import VOCDataset, make_datapath_list, DataTransform
from dataload.data_augmentation import Compose, Scale, RandomMirror, RandomRotation, Resize, Normalize_Tensor

# matplotlib.use('TkAgg')

def data_transform(input_size, img):
    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)
    data_transform = Compose([
                Resize(input_size),  #调整图像尺寸(input_size)
                Normalize_Tensor(color_mean, color_std)  #颜色信息的正规化和张量化
            ])
    img, _ = data_transform(img, img)
    return img

model = UNet(c_in=3,c_out=2)
# model = SegNet(None)
#model = model.load_from_checkpoint("/unet-semantic/check_point/epoch=199-val_loss=0.00-train_miou=0.00-v1.ckpt",
#                                   map_location='cpu')
model = model.load_from_checkpoint("check_point/epoch=279-val_loss=0.00-train_miou=0.00.ckpt", map_location='cpu')
# model.cuda()
model.eval()
print('网络设置完毕 ：成功载入了训练完毕的权重。')


#img_path = "/unet-semantic/images/processing/flight1_image_results/img/ppm_690_737579624.jpg" # you can change the path whatever you want

img_path = "/unet-semantic/TestImage/crack1_1.jpg"
# img_path = "data/crack_seg/JPEGImages/0777.jpg"
# 1.显示原有图像

img_original = Image.open(img_path)  # [高度][宽度][颜色RGB]
img_width, img_height = img_original.size
print(img_width)
print(img_height)
# plt.imshow(img_original)
# plt.savefig("img_original.png")
img_original.save('img_original.png')
# plt.show()

# 2. 用UNET进行推论
img = data_transform(input_size=256, img=img_original) # data augmentation actually resize->256
x = img.unsqueeze(0)  # 小批量化：torch.Size([1, 3, 256, 256])
outputs = model(x)
y = outputs

# 3. 从uNet的输出结果求取最大分类，并转换为颜色调色板格式，将图像尺寸恢复为原有尺寸
y = y[0].detach().cpu().numpy()# y：torch.Size([1, 21, 475, 475])
# a = np.unique(y)
y = np.argmax(y, axis=0)
anno_class_img = Image.fromarray(np.uint8(y), mode="P")
anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
print(anno_class_img.size)
# plt.imshow(anno_class_img)
# plt.savefig("anno_class_img.png")
anno_class_img.save('anno_class_img_3--.png')
# plt.show()

# 4.将图像透明化并重叠在一起
trans_img = Image.new('RGBA', anno_class_img.size, (0, 0, 0, 0))
anno_class_img = anno_class_img.convert('RGBA')

for x in range(img_width):
    for y in range(img_height):
        # 获取推测结果的图像的像素数据
        pixel = anno_class_img.getpixel((x, y))
        r, g, b, a = pixel

        # 如果是(0, 0, 0)的背景，直接透明化
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            continue
        else:
            # 将除此之外的颜色写入准备好的图像中
            trans_img.putpixel((x, y), (r, g, b, 200))
            # 150指定的是透明度大小

result = Image.alpha_composite(img_original.convert('RGBA'), trans_img)
# plt.imshow(result)
# plt.savefig("result.png")
result.save('result_3--.png')
# plt.show()
