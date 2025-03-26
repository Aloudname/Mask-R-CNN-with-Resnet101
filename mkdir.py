import os
import json
import shutil
import random
from PIL import Image, ImageEnhance

def prc_dataset(root_dir, target_dir, threshold=10):
    """
    Processing the data from previous directory and enhancing the type which lacks of data.

    params:
    - root_dir: Previous directory containing numerous sample-dirs.
    - target_dir: root directory to .
    - threshold: minimum number of sample in a type to enhance (10 defaultly).
    """
    class_files = {}

    # Collect all samples from the root dir.
    for instance_name in os.listdir(root_dir):
        instance_path = os.path.join(root_dir, instance_name)
        if not os.path.isdir(instance_path):
            continue

        for file in os.listdir(instance_path):
            if file.lower().endswith('.jpg'):
                jpg_path = os.path.join(instance_path, file)
                json_path = os.path.join(instance_path, os.path.splitext(file)[0] + '.json')

                if not os.path.exists(json_path):
                    continue

                with open(json_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue

                    for shape in data.get('shapes', []):
                        label = shape.get('label', '').strip()
                        if label:
                            if label not in class_files:
                                class_files[label] = []
                            class_files[label].append((jpg_path, json_path))

    # Shuffle and save all samples to the target dir.
    for label, files in class_files.items():
        random.shuffle(files)
        split_idx = int(0.8 * len(files))
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        train_dir = os.path.join(target_dir, label, 'train')
        test_dir = os.path.join(target_dir, label, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Enhance the dataset if less than the threshold.
        if len(train_files) < threshold:
            needed = threshold - len(train_files)
            augmented = 0
            while augmented < needed:
                for jpg_path, json_path in train_files:
                    base_name = os.path.splitext(os.path.basename(jpg_path))[0]
                    img = Image.open(jpg_path)
                    
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.7, 1.3))
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(random.uniform(0.7, 1.3))
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(random.uniform(0.5, 1.5))

                    new_jpg = f"{base_name}_aug{augmented}.jpg"
                    new_json = f"{base_name}_aug{augmented}.json"
                    img.save(os.path.join(train_dir, new_jpg))
                    shutil.copy(json_path, os.path.join(train_dir, new_json))
                    
                    augmented += 1
                    if augmented >= needed:
                        break

        for jpg, json_file in train_files:
            shutil.copy(jpg, train_dir)
            shutil.copy(json_file, train_dir)

        for jpg, json_file in test_files:
            shutil.copy(jpg, test_dir)
            shutil.copy(json_file, test_dir)

def augment_image(image_path, save_dir, json_path, augment_id):
    """
    Completed in main function.
    """
    pass
import os
import shutil

def reorg_dataset(original_dir, new_root_dir):
    """
    Split all types of samples into test and train directory.
    params:
    - original_dir: Previous dir.
    - new_root_dir: New dir to save processed train and test directory.
    """
    new_train_dir = os.path.join(new_root_dir, "train")
    new_test_dir = os.path.join(new_root_dir, "test")
    os.makedirs(new_train_dir, exist_ok=True)
    os.makedirs(new_test_dir, exist_ok=True)

    for class_name in os.listdir(original_dir):
        class_dir = os.path.join(original_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Move the test file to new directory.
        original_train = os.path.join(class_dir, "train")
        if os.path.exists(original_train):

            new_class_train = os.path.join(new_train_dir, class_name)
            os.makedirs(new_class_train, exist_ok=True)
            for file in os.listdir(original_train):
                src = os.path.join(original_train, file)
                dst = os.path.join(new_class_train, file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        # Move the test file to new directory.
        original_test = os.path.join(class_dir, "test")
        if os.path.exists(original_test):
            new_class_test = os.path.join(new_test_dir, class_name)
            os.makedirs(new_class_test, exist_ok=True)
            for file in os.listdir(original_test):
                src = os.path.join(original_test, file)
                dst = os.path.join(new_class_test, file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)


if __name__ == "__main__":

    prc_dataset(
        root_dir='E:\\kq\\已标记',
        target_dir='E:\\kq\\已标记',
        threshold=10
    )

    reorg_dataset(
        original_dir='E:\\kq\\已标记',
        new_root_dir='E:\\kq\\input'
    )