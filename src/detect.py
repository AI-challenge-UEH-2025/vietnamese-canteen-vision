from ultralytics import YOLO
import cv2
import os
import torch

class FoodDetector:
    def __init__(self, model_path='models/best.pt', output_dir='data/cropped_items'):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.class_names = [
            'banh mi', 'bap cai luoc', 'bap cai xao', 'bo xao', 'ca chien', 'ca chua', 'ca kho',
            'ca rot', 'canh bau', 'canh bi do', 'canh cai', 'canh chua', 'canh rong bien', 'chuoi',
            'com', 'dau bap', 'dau hu', 'dau que', 'do chua', 'dua hau', 'dua leo', 'ga chien',
            'ga kho', 'kho qua', 'kho tieu', 'kho trung', 'nuoc mam', 'nuoc tuong', 'oi', 'ot',
            'rau', 'rau muong', 'rau ngo', 'suon mieng', 'suon xao', 'thanh long', 'thit chien',
            'thit luoc', 'tom', 'trung chien', 'trung luoc'
        ]

    def detect_and_crop(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img, conf=0.5, device='0' if torch.cuda.is_available() else 'cpu')

        cropped_paths = []
        yolo_classes = []
        for i, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy()
            masks = result.masks.data.cpu().numpy() if result.masks is not None else []
            classes = result.boxes.cls.cpu().numpy()
            for j, (box, cls) in enumerate(zip(boxes, classes)):
                x1, y1, x2, y2 = map(int, box)
                cropped_img = img[y1:y2, x1:x2]
                crop_path = os.path.join(self.output_dir, f'item_{i}_{j}.jpg')
                cv2.imwrite(crop_path, cropped_img)
                cropped_paths.append(crop_path)
                yolo_classes.append(self.class_names[int(cls)])
        return cropped_paths, yolo_classes, results