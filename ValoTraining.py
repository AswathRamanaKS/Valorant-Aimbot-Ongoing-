from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolov8n.yaml")

# Train the model using the path to the configuration file
results = model.train(data="config.yaml", epochs=50)
