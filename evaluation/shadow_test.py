from ultralytics import YOLO
import os

def shadow_test(old_model_path, new_model_path, image_dir):
    old_model = YOLO(old_model_path)
    new_model = YOLO(new_model_path)

    discrepancies = 0

    for img in os.listdir(image_dir)[:50]:
        img_path = os.path.join(image_dir, img)

        old_pred = old_model.predict(img_path, verbose=False)
        new_pred = new_model.predict(img_path, verbose=False)

        if len(old_pred[0].boxes) != len(new_pred[0].boxes):
            discrepancies += 1

    return discrepancies
