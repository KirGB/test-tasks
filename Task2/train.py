import numpy as np
import pandas as pd
from pathlib import Path
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from progressbar import ProgressBar
import cv2
import os
import json

def make_mask_img(segment_df):
	""" Turn the run encoded pixels from train.csv into an image mask
		There are multiple rows per image for different apparel items, this groups them into one mask
	"""
    seg_width = segment_df.at[0, "Width"]
    seg_height = segment_df.at[0, "Height"]
    seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)
    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
        pixel_list = list(map(int, encoded_pixels.split(" ")))
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] - 1
            index_len = pixel_list[i+1] - 1
            if int(class_id.split("_")[0]) < category_num - 1:
                seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])
    seg_img = seg_img.reshape((seg_height, seg_width), order='F')
    return seg_img
def acc_fashion(input, target):
	"""Custom metric without background"""
    target = target.squeeze(1)
    mask = target != category_num - 1
    return (input.argmax(dim=1)==target).float().mean()

# create a folder for the mask images
if  not os.path.isdir('../labels'):
    os.makedirs('../labels')
path = Path("../input/imaterialist-fashion-2019-FGVC6")
path_img = path/'train'
path_lbl = Path("../labels")
category_num = 27 + 1
size = 256
# get categories
with open(path/"label_descriptions.json") as f:
    label_descriptions = json.load(f)
label_names = [x['name'] for x in label_descriptions['categories']]

# prepare data
df = pd.read_csv(path/'train.csv')
fnames = get_image_files(path_img)
pbar = ProgressBar()
for img in pbar(images):
    img_df = df[df.ImageId == img].reset_index()
    img_mask = make_mask_img(img_df)
    img_mask_3_chn = np.dstack((img_mask, img_mask, img_mask))
    cv2.imwrite('../labels/' + os.path.splitext(img)[0] + '_P.png', img_mask_3_chn)

get_y_fn = lambda x: path_lbl/f'{Path(x).stem}_P.png'
bs = 32
codes = list(range(category_num))
wd = 1e-2

# create the databunch
images_df = pd.DataFrame(images)

src = (SegmentationItemList.from_df(images_df, path_img)
       .split_by_rand_pct()
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
       .databunch(bs=bs)
       .normalize(imagenet_stats))
learn = unet_learner(data, models.resnet34, metrics=acc_fashion, wd=wd, model_dir="/kaggle/working/models")
lr=1e-4
learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
learn.save('final_model-stage1')
learn.show_results()