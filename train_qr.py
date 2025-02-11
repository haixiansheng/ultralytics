from ultralytics import YOLO

# Load a model
model = YOLO(r"E:\code\model_zoo\yolo11n.pt")  

# Train the model
train_results = model.train(
    data=r"E:/code/ultralytics/ultralytics/cfg/datasets/qrcode.yaml",  # path to dataset YAML
    epochs=500,  # number of training epochs
    imgsz=640,  # training image size
    device="cuda:0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=16,
    # save_dir="runs/detect/"
)
 