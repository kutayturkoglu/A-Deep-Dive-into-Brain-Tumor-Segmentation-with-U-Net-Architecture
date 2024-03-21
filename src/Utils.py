import matplotlib.pyplot as plt
import cv2
import os 
import numpy as np
import tifffile as tiff
import json
import shutil

def plot_data(image_names, annotations):
    fig , ax = plt.subplots(1,2,figsize = (10,10))
    for i, img_name in enumerate(image_names):
        # We can use cv2 to read the image
        image = cv2.imread(img_name)
        # Get the img file name to be able to find the corresponding data
        img_file_name = os.path.basename(img_name)
        # Get the image id from the data
        img_id = [item for item in annotations["images"] if item["file_name"] == img_file_name][0]["id"]
        # Get the boxes corresponding to the image
        boxes = [item for item in annotations["annotations"] if item["image_id"] == img_id]
        for box in boxes:
            points = np.array(box["segmentation"]).reshape((-1, 1, 2)).astype(np.int32)
            # Draw the bounding box
            cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        ax[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    plt.show()

def create_mask(json_file, mask_output_folder, image_output_folder, original_image_dir):

    with open(json_file) as f:
        data = json.load(f)
    images = data["images"]
    annotations = data["annotations"]

    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)
    
    for img in images:
        mask_np = np.zeros((img["height"], img["width"]), dtype=np.uint8)
        for annotation in annotations:
            if(annotation["image_id"] == img["id"]):
                for segment in annotation["segmentation"]:
                    points = np.array(segment).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(mask_np, [points], 255)
                    mask_path = os.path.join(mask_output_folder, img["file_name"].replace(".png", ".jpg"))
                    tiff.imsave(mask_path, mask_np)
        original_image_path = os.path.join(original_image_dir, img['file_name'])
        new_image_path = os.path.join(image_output_folder, os.path.basename(original_image_path))
        shutil.copy(original_image_path, new_image_path)

def plot_mask(mask_dir, image_dir):
    mask_files = os.listdir(mask_dir)
    image_files = os.listdir(image_dir)
    for i, mask_file in enumerate(mask_files):
        mask_path = os.path.join(mask_dir, mask_file)
        image_path = os.path.join(image_dir, image_files[i])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(mask)
        ax[1].imshow(image)
        plt.show()
        break

def compare_folders(folder1, folder2):
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    files_only_in_folder1 = set(files1) - set(files2)
    for file_name in files_only_in_folder1:
        file_path = os.path.join(folder1, file_name)
        os.remove(file_path)
        print(f"Deleted file: {file_path}")

    files_only_in_folder2 = set(files2) - set(files1)
    for file_name in files_only_in_folder2:
        file_path = os.path.join(folder2, file_name)
        os.remove(file_path)
        print(f"Deleted file: {file_path}")