import os
import json
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
from tqdm import tqdm
from pycocotools import mask as maskUtils
from collections import defaultdict
import glob

def load_dcm_to_array(dcm_path, normalize=True):
    """Load DCM into np.array."""
    ds = pydicom.dcmread(dcm_path)
    img = apply_voi_lut(ds.pixel_array, ds)
    
    if normalize:
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
    
    return img, ds.Rows, ds.Columns

def process_case(case_id, base_dir, output_image_dir):
    """All frames for a sample."""
    case_id_str = f"training_{case_id:03d}"
    image_dir = os.path.join(base_dir, "training_image", case_id_str)
    gtv_dir = os.path.join(base_dir, "training_gtv", case_id_str)
    
    image_files = glob.glob(os.path.join(image_dir, "*.dcm"))
    
    case_data = {
        "images": [],
        "annotations": []
    }
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        frame_id = filename.split('_')[-1].split('.')[0]
        
        gtv_filename = f"{case_id_str}_{frame_id}.dcm"
        gtv_path = os.path.join(gtv_dir, gtv_filename)
        
        if not os.path.exists(gtv_path):
            continue
        
        try:
            image_array, height, width = load_dcm_to_array(img_path)
            mask_array, _, _ = load_dcm_to_array(gtv_path, normalize=False)
            
            mask_binary = (mask_array > 0).astype(np.uint8)
            
            img_filename = f"{case_id_str}_{frame_id}.png"
            img_output_path = os.path.join(output_image_dir, img_filename)
            cv2.imwrite(img_output_path, image_array)
            
            image_id = len(case_data["images"]) + 1
            case_data["images"].append({
                "id": image_id,
                "file_name": img_filename,
                "width": int(width),
                "height": int(height),
                "case_id": case_id_str,
                "frame_id": frame_id
            })
            
            # process the mask with RLE coding.
            rle = maskUtils.encode(np.asfortranarray(mask_binary))
            segmentation = {
                "counts": rle["counts"].decode("utf-8"),
                "size": [int(height), int(width)]
            }
            
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(cnt) for cnt in contours]
            if boxes:
                x, y, w, h = zip(*boxes)
                x0, y0, x1, y1 = min(x), min(y), max(x[i] + w[i] for i in range(len(x))), max(y[i] + h[i] for i in range(len(y)))
                bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
                area = float((x1 - x0) * (y1 - y0))
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
                area = 0.0
            
            annotation_id = len(case_data["annotations"]) + 1
            case_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # 1 for GTV catagory.
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            })
                
        except Exception as e:
            print(f"Error processing {case_id_str} frame {frame_id}: {str(e)}")
    
    return case_data

def _2coco_format(base_dir, output_dir):
    """DICOM 2 COCO format."""

    output_image_dir = os.path.join(output_dir, "images")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    image_cases_dir = os.path.join(base_dir, "training_image")
    case_dirs = [d for d in os.listdir(image_cases_dir) 
                if d.startswith('training_') and os.path.isdir(os.path.join(image_cases_dir, d))]

    case_ids = sorted([int(d.split('_')[-1]) for d in case_dirs])
    print(f"Found {len(case_ids)} cases to process")
    
    coco_data = {
        "info": {
            "description": "Medical Image GTV Segmentation Dataset",
            "version": "1.0",
            "year": 2023,
            "contributor": "Medical AI Team",
            "date_created": "2023-01-01"
        },
        "licenses": [{
            "id": 1,
            "name": "Academic Use Only",
            "url": ""
        }],
        "categories": [{
            "id": 1,
            "name": "GTV",
            "supercategory": "lesion"
        }],
        "images": [],
        "annotations": []
    }
    
    image_id_counter = 1
    annotation_id_counter = 1
    
    for case_id in tqdm(case_ids, desc="Processing cases"):
        case_data = process_case(case_id, base_dir, output_image_dir)
        
        for img in case_data["images"]:
            img["id"] = image_id_counter
            coco_data["images"].append(img)
            image_id_counter += 1
        
        for ann in case_data["annotations"]:
            ann["id"] = annotation_id_counter
            ann["image_id"] = image_id_counter - len(case_data["images"]) + ann["image_id"] - 1
            coco_data["annotations"].append(ann)
            annotation_id_counter += 1
    
    annotations_path = os.path.join(output_dir, "annotations", "instances.json")
    with open(annotations_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nConversion complete! COCO dataset saved to: {output_dir}")
    print(f"- Images: {len(coco_data['images'])}")
    print(f"- Annotations: {len(coco_data['annotations'])}")
    print(f"- Categories: {len(coco_data['categories'])}")

if __name__ == "__main__":
    base_directory = "E:/pneumoniaBME/data/training"
    output_directory = "E:/kq/Pneuinput"
    
    _2coco_format(base_directory, output_directory)
