from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')
results = model("Image\mobile.jpg", show=True)
annotated_frame = results[0].plot()
cv2.imshow("Yolo detector" , annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
