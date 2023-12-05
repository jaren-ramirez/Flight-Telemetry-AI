from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
#model = YOLO('yolov8n.pt')
model = YOLO("yolov8n.pt") 
#model.to('cuda')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='panel.yaml', epochs=1000, freeze=5, seed=0, dropout=0.7)

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format='onnx')