from src.detect import FoodDetector
from src.classify import FoodClassifier
from src.billing import BillingSystem

def main(image_path):
    class_names = [
        'banh mi', 'bap cai luoc', 'bap cai xao', 'bo xao', 'ca chien', 'ca chua', 'ca kho',
        'ca rot', 'canh bau', 'canh bi do', 'canh cai', 'canh chua', 'canh rong bien', 'chuoi',
        'com', 'dau bap', 'dau hu', 'dau que', 'do chua', 'dua hau', 'dua leo', 'ga chien',
        'ga kho', 'kho qua', 'kho tieu', 'kho trung', 'nuoc mam', 'nuoc tuong', 'oi', 'ot',
        'rau', 'rau muong', 'rau ngo', 'suon mieng', 'suon xao', 'thanh long', 'thit chien',
        'thit luoc', 'tom', 'trung chien', 'trung luoc'
    ]
    detector = FoodDetector()
    classifier = FoodClassifier(class_names=class_names)
    billing = BillingSystem()

    # Detect and crop
    cropped_paths, yolo_classes, results = detector.detect_and_crop(image_path)
    if len(cropped_paths) < 4:
        print("Error: Less than 4 food items detected!")
        return

    # Classify and compare
    food_items = []
    for path, yolo_class in zip(cropped_paths, yolo_classes):
        resnet_class = classifier.classify(path)
        if resnet_class:
            food_items.append(resnet_class)
            print(f"Cropped image: {path}, YOLO: {yolo_class}, ResNet50: {resnet_class}")
            if yolo_class != resnet_class:
                print(f"Discrepancy detected, using ResNet50 class: {resnet_class}")
        else:
            print(f"ResNet50 failed for {path}, using YOLO class: {yolo_class}")
            food_items.append(yolo_class)

    # Calculate bill
    bill_details, total_cost, total_calories = billing.calculate_bill(food_items)

    # Print results
    print("\nDetected Food Items:")
    for detail in bill_details:
        print(f"Item: {detail['item']}, Price: {detail['price']} VND, Calories: {detail['calories']} kcal")
    print(f"\nTotal Cost: {total_cost} VND")
    print(f"Total Calories: {total_calories} kcal")

if __name__ == "__main__":
    image_path = "D:\khay_com\Tom rim me, dau hu nhoi, canh bi dao.png"
    main(image_path)