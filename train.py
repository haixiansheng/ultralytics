from ultralytics import YOLO

# Load a model
model = YOLO(r"E:\code\ultralytics\runs\detect\train8\weights\last.pt")  

# Train the model
train_results = model.train(
    data=r"E:/code/ultralytics/ultralytics/cfg/datasets/scanner.yaml",  # path to dataset YAML
    epochs=500,  # number of training epochs
    imgsz=640,  # training image size
    device="cuda:0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=8,
    # save_dir="runs/detect/"
)
 