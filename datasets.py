import os
import cv2
import json
import glob
import torch
import shutil
import imgviz
import argparse
import numpy as np
import albumentations as A
import torchvision.transforms as T

from labelme import utils
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CocoDetection
from sklearn.model_selection import train_test_split


np.random.seed(41)
 
# Spare 0 for background. Attention!
classname_to_id = {"非典型增生": 1, "非典型增生术后": 2, "口腔白斑": 3, "口腔白斑术后": 4,
             "口腔红斑": 5, "口腔黏膜溃痛": 6, "口腔苔藓化损伤": 7, "口腔苔藓样损伤": 8, 
             "鳞状细胞癌术后": 9, "糜烂": 10, "疣状黄瘤": 11, "肿物": 12, "CA术后": 13}
 
class Lableme2CoCo:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
 
    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent = 2 for more delicate display.
 
    # Transform .json into COCO dataset.
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in json_path_list:
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance
 
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)
 
    # COCO.image
    def _image(self, obj, path):
        image = {}
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image
 
    # COCO.annotation.
    def _annotation(self, shape):
        # print('shape', shape)
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation
 
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
 
    # COCO format [x1,y1,w,h] aligns with COCO bbox.
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

class CocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file, transforms=None)
        self._load_coco_annotations(ann_file)
        self.c_transforms = transforms

    def _load_coco_annotations(self, ann_file):
        with open(ann_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        self.coco = COCO()
        self.coco.dataset = annotations
        self.coco.createIndex()

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        bboxes = [ann['bbox'] for ann in target]
        labels = [ann['category_id'] for ann in target]

        masks = []
        for ann in target:
            mask = self.coco.annToMask(ann)
            masks.append(mask)
        masks = torch.as_tensor(np.stack(masks, axis=0), 
                                dtype=torch.uint8)
        
        img_np = np.array(img.convert('RGB'))

        # transform with Albumentations format. 
        # Before changing library of transforms, 
        # do remember to adjust this part into suitable structure.
        if self.c_transforms is not None:
            transformed = self.c_transforms(
                image = img_np,
                masks=masks.numpy(),
                bboxes = bboxes,
                category_ids = labels
            )
            
            img = transformed['image']
            masks = transformed['masks']
            bboxes = transformed['bboxes']
            labels = transformed['category_ids']

        target = {
            'image_id': torch.tensor([image_id], dtype = torch.int64),
            'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': masks,
        }      
        return img, target


# Below are selectable transforms.
# Before changing library of transforms, 
# do remember to adjust CocoDataset.__getitem__() into suitable structure.
# Do note that also convert mask code into tensors.

# transform = T.ToTensor()

# transform = T.Compose([
#     T.Resize((256, 256)),
#     T.ToTensor()])

transform = A.Compose([
    A.Resize(256, 256),
        # A.ToRGB(),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ToTensorV2()],
    bbox_params = A.BboxParams(
        format = 'coco',
        # min_visibility=0.1, # if required to filt very small frames.
        label_fields = ['category_ids']))

from pycocotools.coco import COCO
from PIL import Image
import os
import tqdm
import cv2
import imgviz
import numpy as np

def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)

def Get_mask(annotation_dir, img_dir):
    """
    Get the mask from an image.
    Run ONLY ONCE when masks are required.
    masks will be saved in .png format.

    params:
    annotation_dir: annotations directory.
    split: train or test directory.
    """
    annotation_file = os.path.join(annotation_dir)

    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))

    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            for i in range(len(anns) - 1):
                mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
            mask_path = os.path.join(img_dir, img['file_name'].replace('.jpg', '.png'))
            save_colored_mask(mask, mask_path)


def Get_COCO(labelme_path, saved_coco_path):
    print('reading...')
    # mkdir.
    if not os.path.exists("%scoco\\annotations" % saved_coco_path):
        os.makedirs("%scoco\\annotations" % saved_coco_path)

    if not os.path.exists("%scoco\\images\\train" % saved_coco_path):
        os.makedirs("%scoco\\images\\train" % saved_coco_path)

    if not os.path.exists("%scoco\\images\\test" % saved_coco_path):
        os.makedirs("%scoco\\images\\test" % saved_coco_path)

    print(labelme_path + "\*.json")
    json_list_path = glob.glob(labelme_path + "\*.json")
    # json_list_path = glob.glob(labelme_path + "\*.png")
    print('json_list_path: ', len(json_list_path))
    # data split without distinguish between train and test.
    train_path, val_path = train_test_split(json_list_path, test_size=0.2, train_size=0.8)
    print(f"train samples: {len(train_path)}, test samples: {len(val_path)}.")

    # Transform train .json into COCO dataset.
    l2c_train = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    l2c_train.save_coco_json(train_instance, '%scoco\\annotations\\instances_train.json' % saved_coco_path)

    for file in train_path:
        # print("Testing: file："+file)
        img_name = file.replace('json', 'png')
        # print("Testing: img_name：" + img_name)
        temp_img = cv2.imread(img_name)
        # if None, img is .jpg format.
        if  temp_img is None:
            img_name_jpg = img_name.replace('png', 'jpg')
            temp_img = cv2.imread(img_name_jpg)
 
        filenames = img_name.split("\\")[-1]
        cv2.imwrite("E:\\kq\\input\\coco\\images\\train\\{}".format(filenames), temp_img)
        # print(temp_img)

    for file in val_path:
        # shutil.copy(file.replace("json", "jpg"), "%scoco\\images\\test\\" % saved_coco_path)
 
        img_name = file.replace('json', 'png')
        temp_img = cv2.imread(img_name)
        if temp_img is None:
            img_name_jpg = img_name.replace('png', 'jpg')
            temp_img = cv2.imread(img_name_jpg)
        filenames = img_name.split("\\")[-1]
        cv2.imwrite("E:\\kq\\dataset\\coco\\images\\test\\{}".format(filenames), temp_img)
 
    # Transform train .json into COCO dataset.
    l2c_val = Lableme2CoCo()
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco\\annotations\\instances_test.json' % saved_coco_path)
    print("COCO dataset generated successfully!")

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        # Convert to 3d tensor [C, H, W].
        if img.dim() == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        images.append(img)
        targets.append(target)
    return images, targets

# finally we get coco-style datasets.
train_loader = CocoDataset('E:\\kq\\input\\coco\\images\\train', 'E:\\kq\\input\\coco\\annotations\\instances_train.json', transforms = transform)
test_loader = CocoDataset('E:\\kq\\input\\coco\\images\\test', 'E:\\kq\\input\\coco\\annotations\\instances_test.json', transforms = transform)


if __name__ == '__main__':

    labelme_path = "E:\\kq\\input\\test"
    saved_coco_path = "E:\\kq\\input\\test"

    # # run only ONCE when first generates a coco directory.
    # Get_COCO(labelme_path, saved_coco_path)
    # Get_mask('E:\\kq\\input\\coco\\annotations\\instances_test.json', 'E:\\kq\\input\\coco\\images\\test')
    # Get_mask('E:\\kq\\input\\coco\\annotations\\instances_train.json', 'E:\\kq\\input\\coco\\images\\train')

