import os
import json
import shutil
from sklearn.model_selection import train_test_split


def save_data(folders, coco_data, images, folder_name):
    folder_path = folders[folder_name]
    annotations = []

    for img in images:
        img_id = img['id']
        img_file = img['file_name']
        
        shutil.copy(os.path.join(train_folder, img_file), folder_path)
        
        img_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        annotations.extend(img_annotations)
    
    annotation_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": coco_data.get("categories", [])
    }
    
    with open(os.path.join(folders['annotations'], f"{folder_name}_annotations.coco.json"), 'w') as f:
        json.dump(annotation_data, f)

if __name__ == '__main__':
    root_folder = '/kaggle/CoDETR/data/vehicle/'
    os.makedirs(root_folder, exist_ok=True)
    train_folder = '/kaggle/input/track1-traffic-vehicle-detection/daytime/daytime/train'
    annotation_file = '/kaggle/input/track1-traffic-vehicle-detection/daytime/daytime/_annotations.coco.json'

    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']

    train_imgs, test_imgs = train_test_split(images, test_size=0.15, random_state=42)
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=0.15 / 0.85, random_state=42)

    # root_folder = os.path.join(root_folder, 'data', 'vehicle')
    folders = {
        'train': os.path.join(root_folder, 'train'),
        'val': os.path.join(root_folder, 'val'),
        'test': os.path.join(root_folder, 'test'),
        'annotations': os.path.join(root_folder, 'annotations')
    }

    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    save_data(folders, coco_data, train_imgs, 'train')
    save_data(folders, coco_data, val_imgs, 'val')
    save_data(folders, coco_data, test_imgs, 'test')
