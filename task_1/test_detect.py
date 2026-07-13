"""Quick test to see exactly what YOLO detects on the test image."""
from ultralytics import YOLO
import cv2

model = YOLO('yolov8s.pt')
image = cv2.imread('test_images/traffic.png')

if image is None:
    print("ERROR: Could not read test_images/traffic.png")
else:
    print(f"Image size: {image.shape}")
    
    # Run with very low confidence
    results = model(image, conf=0.05)[0]
    
    print(f"\nTotal detections: {len(results.boxes)}")
    print("-" * 50)
    
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"  {class_name:15s}  conf={confidence:.3f}  box=({x1},{y1})-({x2},{y2})")
