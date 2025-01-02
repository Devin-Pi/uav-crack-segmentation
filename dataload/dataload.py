# 导入软件包
import os.path as osp
from PIL import Image
import torch.utils.data as data
import numpy as np
from dataload.data_augmentation import Compose, Scale, RandomMirror, RandomRotation, Resize, Normalize_Tensor
import os
import torch
import torchvision.utils as vutils
def make_datapath_list(rootpath):
    """
   创建用于学习、验证的图像数据和标注数据的文件路径列表变量

    Parameters
    ----------
    rootpath : str
        指向数据文件夹的路径

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        保存了指向数据的路径列表变量
    """

    #创建指向图像文件和标注数据的路径的模板
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'SegmentationClass', '%s.png')

    #训练和验证，分别获取相应的文件 ID（文件名）
    train_id_names = osp.join(rootpath + 'ImageSets/Segmentation/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Segmentation/val.txt')
    test_id_names = osp.join(rootpath + "ImageSets/Segmentation/test.txt")

    #创建指向训练数据的图像文件和标注文件的路径列表变量
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  #删除空格和换行符
        img_path = (imgpath_template % file_id)  #图像的路径
        anno_path = (annopath_template % file_id)  #标注数据的路径
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

   #创建指向验证数据的图像文件和标注文件的路径列表变量
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip() #删除空格和换行符
        img_path = (imgpath_template % file_id) #图像的路径
        anno_path = (annopath_template % file_id) #标注数据的路径
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    test_img_list = list()
    test_anno_list = list()
    
    for line in open(test_id_names):
        file_id = line.strip() #删除空格和换行符
        img_path = (imgpath_template % file_id) #图像的路径
        anno_path = (annopath_template % file_id) #标注数据的路径
        test_img_list.append(img_path)
        test_anno_list.append(anno_path)
        
    return train_img_list, train_anno_list, val_img_list, val_anno_list, test_img_list, test_anno_list


class DataTransform():
    """
    图像和标注的预处理类。训练和验证时分别采取不同的处理方法
    将图像的尺寸调整为input_size x input_size
    训练时进行数据增强处理


    Attributes
    ----------
    input_size : int
        指定调整图像尺寸的大小
    color_mean : (R, G, B)
        指定每个颜色通道的平均值
    color_std : (R, G, B)
        指定每个颜色通道的标准差
    """ 

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]), #图像的放大
                RandomRotation(angle=[-10, 10]), #旋转
                RandomMirror(), #随机镜像
                Resize(input_size), #调整尺寸(input_size)
                Normalize_Tensor(color_mean, color_std)#颜色信息的正规化和张量化
            ]),
            'val': Compose([
                Resize(input_size),  #调整图像尺寸(input_size)
                Normalize_Tensor(color_mean, color_std)  #颜色信息的正规化和张量化
            ]),
            'test': Compose([
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            指定预处理的执行模式
        """
        return self.data_transform[phase](img, anno_class_img)


class VOCDataset(data.Dataset):       

    def __init__(self, img_list, anno_list, phase, transform, n_classes, input_shape):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.n_classes = n_classes
        self.input_shape = input_shape
    def __len__(self):
        '''返回图像的张数'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        获取经过预处理的图像的张量形式的数据和标注
        '''
        img, pngs = self.pull_item(index)
        return img, pngs

    def pull_item(self, index):
        '''获取图像的张量形式的数据和标注'''

        #  1.读入图像数据
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)   #[高度][宽度][颜色RGB]

        # 2.读入标注图像数据
        anno_file_path = self.anno_list[index]
        png = Image.open(anno_file_path)   #[ 高度 ][ 宽度 ]
        
        img_name = os.path.splitext(os.path.basename(image_file_path))[0]
        
       # 3.进行预处理操作
        img, png = self.transform(self.phase, img, png)
        return img, png
