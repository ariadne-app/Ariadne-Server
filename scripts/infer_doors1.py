from super_gradients.training import models
import cv2

DEVICE = 'cpu'
MODEL_ARCH = "yolo_nas_m"
classes = ['bathroom', 'bathtub', 'door', 'en_suite', 'kitchen', 'window']

best_model = models.get(
    MODEL_ARCH,
    num_classes=len(classes),
    checkpoint_path="../models/average_model.pth"
).to(DEVICE)

image_path="../assets/images/hospital_1_upscaled_sharpened.jpg"
image = cv2.imread(image_path)

# predict
model_result = best_model.predict(image, conf=0.25)

print(model_result.prediction)