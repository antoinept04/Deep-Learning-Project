
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
from IPython.display import Image, display
import cv2
import random
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
import cv2
# Define interactive parameters
dataset_path = "C:/Intelligence artificielle/dataset_detection/"  #@param ["dataset/MOCS_Small/","dataset/MOCS_Medium/","dataset/MOCS_Big/"] {type:"string"}
model_to_be_trained = "yolo11m"  #@param ["yolo11n"]
nbr_epochs = 60  #@param {type:"integer"}
img_size = 640  #@param {type:"integer"}
batch_size = 8  #@param {type:"integer"}
load_weights = True  #@param {type:"boolean"}
test_conf_level = 0.35  #@param {type:"slider", min:0, max:1, step:0.05}



image_dir = dataset_path+'/dataset_detection/val/images'
label_dir = dataset_path+'/dataset_detection/val/labels'

# Class color map
CLASS_NAMES = {
    0: "Worker", 1: "Static crane", 2: "Hanging head", 3: "Crane",
    4: "Roller", 5: "Bulldozer", 6: "Excavator", 7: "Truck",
    8: "Loader", 9: "Pump truck", 10: "Concrete mixer",
    11: "Pile driving", 12: "Other vehicle"
}

CLASS_COLORS = {
    0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0),
    4: (255, 0, 255), 5: (0, 255, 255), 6: (128, 128, 0), 7: (128, 0, 128),
    8: (0, 128, 128), 9: (100, 100, 255), 10: (255, 100, 100),
    11: (100, 255, 100), 12: (200, 200, 200)
}



def draw_bboxes(image_path, label_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                cls = int(cls)
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                label = CLASS_NAMES.get(cls, str(cls))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


if __name__ == "__main__":

    

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    sample_images = random.sample(image_files, 6)

    

    plt.figure(figsize=(18, 10))
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(image_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))
        vis_img = draw_bboxes(img_path, label_path)

        plt.subplot(2, 3, i + 1)
        plt.imshow(vis_img)
        plt.title(f"{img_name}", fontsize=10)
        plt.axis("off")

    plt.suptitle("YOLOv11 Object Detection Bounding Boxes on MOCS Samples", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


    if load_weights:
        model = YOLO("yolo11m.pt")
    else:
        model = YOLO("yolo11m")
    
    

       
    output_dir = "./runs"      
    run_name   = "my_experience5"  

    
    model.train(
        data     = os.path.join(dataset_path, 'data.yaml'),
        epochs   = nbr_epochs,
        imgsz    = img_size,
        batch    = batch_size,
        project  = output_dir,
        name     = run_name,
        exist_ok = True
    )
    

    best_directory = os.path.join(output_dir, run_name, 'weights', 'best.pt')
    model = YOLO(best_directory)


  


    # === ÉVALUATION & VISUALISATION ===

    results = model.predict(source=img_path, conf=test_conf_level, imgsz=img_size, save = True, show = True)


    results[0].show()

    metrics = model.val(
    data=os.path.join(dataset_path, 'data.yaml'),
    split='test',
    imgsz=img_size,
    conf=test_conf_level
    )

    print("la map50-95 vaut : ",metrics.box.map)


      
    video_path_unity = "C:/Intelligence artificielle/MOCS_Small/MOCS_Small/unity_video.mp4"


    results = model.predict(
        source=video_path_unity,
        conf=test_conf_level,
        imgsz=img_size,
        stream=False,     
        save=True,
        save_txt=False,     
        save_conf=True,    
    )

       
    video_path_chantier = "C:/Intelligence artificielle/dataset_detection/Video_chantier.mp4"


    results = model.predict(
        source=video_path_chantier,
        conf=test_conf_level,
        imgsz=img_size,
        stream=False,      
        save=True,
        save_txt=False,    
        save_conf=True,   
    )

        
    
    video_path_test = "C:/Intelligence artificielle/dataset_detection/test2.mp4"

    results = model.predict(
        source=video_path_test,
        conf=test_conf_level,
        imgsz=img_size,
        stream=False,    
        save=True,
        save_txt=False,   
        save_conf=True,     
    )
    

    video_path1 = "C:/Intelligence artificielle/dataset_detection/example_video.mp4"

    results = model.predict(
    source=video_path1,
    conf=test_conf_level,
    imgsz=img_size,
    stream=False,    
    save=True,
    save_txt=False,    
    save_conf=True,    
)
    

    video_path2 = "C:/Intelligence artificielle/dataset_detection/test_classification.mp4"

    results = model.predict(
    source=video_path2,
    conf=test_conf_level,
    imgsz=img_size,
    stream=False,   
    save=True,
    save_txt=False,   
    save_conf=True,   
)
    