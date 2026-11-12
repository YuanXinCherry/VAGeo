# -*- coding: utf-8 -*-

import os
import sys
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations
from shapely.geometry import Polygon
import torch.nn.functional as F

cv2.setNumThreads(0)
from scipy.stats import laplace


class DatasetNotFoundError(Exception):
    pass


# Define the transformation
class MyAugment:
    def __init__(self) -> None:
        self.transform = albumentations.Compose([
            albumentations.Blur(p=0.01),
            albumentations.MedianBlur(p=0.01),
            albumentations.ToGray(p=0.01),
            albumentations.CLAHE(p=0.01),
            albumentations.RandomBrightnessContrast(p=0.0),
            albumentations.RandomGamma(p=0.0),
            albumentations.ImageCompression(quality_lower=75, p=0.0)])

    def augment_hsv(self, im, hgain=0.5, sgain=0.5, vgain=0.5):
        # HSV color-space augmentation
        if hgain or sgain or vgain:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB, dst=im)  # no return needed

    def __call__(self, img, bbox):
        imgh, imgw, _ = img.shape
        x, y, w, h = (bbox[0] + bbox[2]) / 2 / imgw, (bbox[1] + bbox[3]) / 2 / imgh, (bbox[2] - bbox[0]) / imgw, (
                    bbox[3] - bbox[1]) / imgh
        img = self.transform(image=img)['image']
        # self.augment_hsv(img)
        # Flip up-down
        if random.random() < 0.5:
            img = np.flipud(img)
            y = 1 - y

        # Flip left-right
        if random.random() < 0.5:
            img = np.fliplr(img)
            x = 1 - x
        #
        new_imgh, new_imgw, _ = img.shape
        assert new_imgh == imgh, new_imgw == imgw
        x, y, w, h = x * imgw, y * imgh, w * imgw, h * imgh

        # Crop image
        iscropped = False
        if random.random() < 0.5:
            left, top, right, bottom = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            if left >= new_imgw / 2:
                start_cropped_x = random.randint(0, int(0.15 * new_imgw))
                img = img[:, start_cropped_x:, :]
                left, right = left - start_cropped_x, right - start_cropped_x
            if right <= new_imgw / 2:
                start_cropped_x = random.randint(int(0.85 * new_imgw), new_imgw)
                img = img[:, 0:start_cropped_x, :]
            if top >= new_imgh / 2:
                start_cropped_y = random.randint(0, int(0.15 * new_imgh))
                img = img[start_cropped_y:, :, :]
                top, bottom = top - start_cropped_y, bottom - start_cropped_y
            if bottom <= new_imgh / 2:
                start_cropped_y = random.randint(int(0.85 * new_imgh), new_imgh)
                img = img[0:start_cropped_y, :, :]
            cropped_imgh, cropped_imgw, _ = img.shape
            left, top, right, bottom = left / cropped_imgw, top / cropped_imgh, right / cropped_imgw, bottom / cropped_imgh
            if cropped_imgh != new_imgh or cropped_imgw != new_imgw:
                img = cv2.resize(img, (new_imgh, new_imgw))
            new_cropped_imgh, new_cropped_imgw, _ = img.shape
            left, top, right, bottom = left * new_cropped_imgw, top * new_cropped_imgh, right * new_cropped_imgw, bottom * new_cropped_imgh
            x, y, w, h = (left + right) / 2, (top + bottom) / 2, right - left, bottom - top
            iscropped = True
        # if iscropped:
        #    print((new_imgw, new_imgh))
        #    print((cropped_imgw, cropped_imgh), flush=True)
        #    print('============')
        # print(type(img))
        # draw_bbox = np.array([x-w/2, y-h/2, x+w/2, y+h/2], dtype=int)
        # print(('draw_bbox', iscropped, draw_bbox), flush=True)
        # img_new=draw_rectangle(img, draw_bbox)
        # cv2.imwrite('tmp/'+str(random.randint(0,5000))+"_"+str(iscropped)+".jpg", img_new)

        new_bbox = [(x - w / 2), y - h / 2, x + w / 2, y + h / 2]
        # print(bbox)
        # print(new_bbox)
        # print('---end---')
        return img, np.array(new_bbox, dtype=int)


BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, x_max, y_min, y_max = int(bbox[0]), int(bbox[2]), int(bbox[1]), int(bbox[3])
    print(bbox, flush=True)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=2)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


class RSDataset(Dataset):
    def __init__(self, data_root, data_name='CVOGL', split_name='train', img_size=1024,
                 transform=None, augment=False):
        self.data_root = data_root
        self.data_name = data_name
        self.img_size = img_size
        self.transform = transform
        self.split_name = split_name
        self.augment = augment

        self.myaugment = MyAugment()

        if self.data_name == 'CVOGL_DroneAerial':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(self.data_name, split_name))
            self.data_list = torch.load(data_path)
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 256)  # 52 #32
        elif self.data_name == 'CVOGL_SVI':
            data_dir = os.path.join(data_root, self.data_name)
            data_path = os.path.join(data_dir, '{0}_{1}.pth'.format(self.data_name, split_name))
            self.data_list = torch.load(data_path)
            self.queryimg_dir = os.path.join(data_dir, 'query')
            self.rsimg_dir = os.path.join(data_dir, 'satellite')
            self.rs_wh = 1024
            self.query_featuremap_hw = (256, 512)
        else:
            assert (False)

        self.rs_transform = albumentations.Compose([
            albumentations.RandomSizedBBoxSafeCrop(width=self.rs_wh, height=self.rs_wh, erosion_rate=0.2, p=0.2),
            albumentations.RandomRotate90(p=0.5),
            albumentations.GaussNoise(p=0.5),
            albumentations.HueSaturationValue(p=0.3),
            albumentations.OneOf([
                albumentations.Blur(p=0.4),
                albumentations.MedianBlur(p=0.3),
            ], p=0.5),
            albumentations.OneOf([
                albumentations.RandomBrightnessContrast(p=0.4),
                albumentations.CLAHE(p=0.3),
            ], p=0.5),
            albumentations.ToGray(p=0.2),
            albumentations.RandomGamma(p=0.3), ],
            bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['class_name']))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        _, queryimg_name, rsimg_name, _, click_xy, bbox, _, cls_name = self.data_list[idx]

        ## box format: to x1y1x2y2
        bbox = np.array(bbox, dtype=int)

        queryimg = cv2.imread(os.path.join(self.queryimg_dir, queryimg_name))
        queryimg = cv2.cvtColor(queryimg, cv2.COLOR_BGR2RGB)

        rsimg = cv2.imread(os.path.join(self.rsimg_dir, rsimg_name))
        rsimg = cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)
        if self.augment:
            rs_transformed = self.rs_transform(image=rsimg, bboxes=[list(bbox)], class_name=[cls_name])
            rsimg = rs_transformed['image']
            bbox = rs_transformed['bboxes'][0][0:4]

        # Norm, to tensor
        if self.transform is not None:
            rsimg = self.transform(rsimg.copy())
            queryimg = self.transform(queryimg.copy())

        query_featuremap_hw = self.query_featuremap_hw
        click_hw = (int(click_xy[1]), int(click_xy[0]))

        # mat_clickhw = np.zeros((query_featuremap_hw[0], query_featuremap_hw[1]), dtype=np.float32)
        # click_h = [pow(one-click_hw[0],2) for one in range(query_featuremap_hw[0])]
        # click_w = [pow(one-click_hw[1],2) for one in range(query_featuremap_hw[1])]
        # norm_hw = pow(query_featuremap_hw[0]*query_featuremap_hw[0] + query_featuremap_hw[1]*query_featuremap_hw[1], 0.5)
        # for i in range(query_featuremap_hw[0]):
        #     for j in range(query_featuremap_hw[1]):
        #         tmp_val = 1 - (pow(click_h[i]+click_w[j], 0.5)/norm_hw)
        #         mat_clickhw[i, j] = tmp_val * tmp_val

        # Calculate the Euclidean distance
        query_featuremap_hw = self.query_featuremap_hw
        click_hw = (int(click_xy[1]), int(click_xy[0]))
        Euclidean_distance = np.zeros((query_featuremap_hw[0], query_featuremap_hw[1]), dtype=np.float32)
        mat_clickhw = np.zeros((query_featuremap_hw[0], query_featuremap_hw[1]), dtype=np.float32)
        click_h = [pow(one - click_hw[0], 2) for one in range(query_featuremap_hw[0])]
        click_w = [pow(one - click_hw[1], 2) for one in range(query_featuremap_hw[1])]
        norm_hw = pow(query_featuremap_hw[0] * query_featuremap_hw[0] + query_featuremap_hw[1] * query_featuremap_hw[1],
                      0.5)
        for i in range(query_featuremap_hw[0]):
            for j in range(query_featuremap_hw[1]):
                Euclidean_distance[i][j] = pow(click_h[i] + click_w[j], 0.5)
        if self.data_name == 'CVOGL_SVI':
            # The Laplace distribution
            scale = 25  # 分布的标准差，控制分布的宽度
            mat_clickhw = laplace.pdf(Euclidean_distance, loc=0, scale=scale)
            mat_clickhw = mat_clickhw.astype(np.float32)
        elif self.data_name == 'CVOGL_DroneAerial':
            # Square ring division
            query_featuremap_hw = self.query_featuremap_hw
            click_hw = (int(click_xy[1]), int(click_xy[0]))
            mat_clickhw = np.zeros((query_featuremap_hw[0], query_featuremap_hw[1]), dtype=np.float32)
            mat_clickhw = self.picture_partition(click_hw, mat_clickhw)
        else:
            assert (False)

        return queryimg, rsimg, mat_clickhw, np.array(bbox, dtype=np.float32), idx

    def picture_partition(self, click_hw, mat_clickhw, block=4):
        # block copy
        block_limit = block

        # ablation1
        weights = [0.6, 0.15, 0.15, 0.1]

        '''
        #ablation2
        weights = [0.6,0.2,0.15,0.05]
        '''
        '''
        #ablation3
        weights = [0.4,0.3,0.2,0.1]
        '''
        # Half of the size of the first piece
        first_half = 96
        # Click the point coordinates
        x = int(click_hw[0])  # get the x
        y = int(click_hw[1])  # get the y

        # get the maximum height and width
        maxh, maxw = mat_clickhw.shape

        # print('--------------x:'+str(x)+',y:'+str(y))
        # Distance from click point to top
        top = int(x)
        if top < first_half:
            cre_top = top
            top = 0
        else:
            cre_top = first_half
            top = top - cre_top
        # Distance from click point to bottom
        button = int(maxh - 1 - x)
        if button < first_half:
            cre_button = button
            button = 0
        else:
            cre_button = first_half
            button = button - cre_button
        # Distance from click point to left
        left = int(y)
        if left < first_half:
            cre_left = left
            left = 0
        else:
            cre_left = first_half
            left = left - cre_left
        # Distance from click point to right
        right = int(maxw - 1 - y)
        if right < first_half:
            cre_right = right
            right = 0
        else:
            cre_right = first_half
            right = right - cre_right
        # print('the first block cre_top:'+str(cre_top)+',cre_button:'+str(cre_button)+',cre_left:'+str(cre_left)+',cre_right:'+str(cre_right))
        # print('the first block top:'+str(top)+',button:'+str(button)+',left:'+str(left)+',right:'+str(right))
        block_size = torch.ones((maxh - top - button, maxw - left - right))
        # print('the first block block_size:',block_size.shape)
        # feature_pre only represents shape with all elements set to 1
        block_pre_size = block_size
        # Fill weight
        feature_curr = block_size * weights[block_limit - block]
        # print("Size of unpadded feature_curr for block 1:", feature_curr.size())
        # Expand to image size
        feature_curr = F.pad(feature_curr, (left, right, top, button), "constant", 0)
        # print("Size of padded feature_curr for block 1:", feature_curr.size())
        result = feature_curr

        #  Update remaining blocks
        block -= 1

        while (block > 0):
            # From block 2 to block (block-1)
            if block != 1:
                # Distance from top to upper boundary
                # Increment of cre_top to upper boundary
                if top == 0:
                    cre_top = 0
                else:
                    cre_top = int(top / block)
                    if cre_top == 0 and top % block != 0:
                        cre_top = int(top % block)
                    top = int(top - cre_top)
                # Distance from button to lower boundary
                # Increment of cre_button to lower boundary
                if button == 0:
                    cre_button = 0
                else:
                    cre_button = int(button / block)
                    if cre_button == 0 and button % block != 0:
                        cre_button = button % block
                    button = int(button - cre_button)
                # Distance from left to left boundary
                # Increment of cre_left to left boundary
                if left == 0:
                    cre_left = 0
                else:
                    cre_left = int(left / block)
                    if cre_left == 0 and left % block != 0:
                        cre_left = left % block
                    left = int(left - cre_left)

                # Distance from right to right boundary
                # Increment of cre_right to right boundary
                if right == 0:
                    cre_right = 0
                else:
                    cre_right = int(right / block)
                    if cre_right == 0 and right % block != 0:
                        cre_right = right % block
                    right = right - cre_right
                # button and right are index + 1
                # print(f'Block {5-block} cre_top:'+str(cre_top)+',cre_button:'+str(cre_button)+',cre_left:'+str(cre_left)+',cre_right:'+str(cre_right))
                # print(f'Block {5-block} top:'+str(top)+',button:'+str(button)+',left:'+str(left)+',right:'+str(right))
                # Pad previous block
                block_pre_size = F.pad(block_pre_size, (cre_left, cre_right, cre_top, cre_button), "constant", 0)
                # print('Size of padded block_pre_size:', block_pre_size.shape)
                # Get current block size
                block_size = torch.ones((maxh - top - button, maxw - left - right))
                # print(f'Block {5-block} block_size:',block_size.size())
                # Backup a copy of current block with all 1s, because subtracting previous block will create a ring, preventing next block from forming a ring
                mid_size = block_size
                block_size -= block_pre_size
                # block_pre_size only represents shape with all elements set to 1
                block_pre_size = mid_size
                # Fill weight
                feature_curr = block_size * weights[block_limit - block]
                # print(f'Size of unexpanded feature_curr for block {5-block}:',feature_curr.size())
                # Expand to image size
                feature_curr = F.pad(feature_curr, (left, right, top, button), "constant", 0)
                # print(f'{5-block} feature_curr:',feature_curr.size())
                result = np.add(result, feature_curr)
                block -= 1
                # print('--------------------------------------------------------------------------------')
            else:
                # Pad previous block
                block_pre_size = F.pad(block_pre_size, (left, right, top, button), "constant", 0)
                # print('block_pre_size：',block_pre_size.shape)
                block_size = torch.ones((maxh, maxw))
                # print(f'Final block block_size:',block_size.size())
                block_size -= block_pre_size
                # Fill weight
                feature_curr = block_size * weights[block_limit - block]
                # Accumulate result
                result = np.add(result, feature_curr)
                block -= 1

        return result