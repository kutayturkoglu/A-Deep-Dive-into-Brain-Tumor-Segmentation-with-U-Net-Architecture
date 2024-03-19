import matplotlib.pyplot as plt
import cv2
import os 
import numpy as np

def plot_data(image_names, data):
    fig , ax = plt.subplots(1,2,figsize = (10,10))
    for i, img_name in enumerate(image_names):
        # We can use cv2 to read the image
        image = cv2.imread(img_name)
        # Get the img file name to be able to find the corresponding data
        img_file_name = os.path.basename(img_name)
        # Get the image id from the data
        img_id = [item for item in data["images"] if item["file_name"] == img_file_name][0]["id"]
        # Get the boxes corresponding to the image
        boxes = [item for item in data["annotations"] if item["image_id"] == img_id]
        for box in boxes:
            points = np.array(box["segmentation"]).reshape((-1, 1, 2)).astype(np.int32)
            # Draw the bounding box
            cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        ax[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    plt.show()
    