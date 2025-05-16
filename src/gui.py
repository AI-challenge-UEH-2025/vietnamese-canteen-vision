import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
from detect import FoodDetector
from classify import FoodClassifier
from billing import BillingSystem

class FoodBillingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Canteen Billing System")
        self.class_names = [
            'banh mi', 'bap cai luoc', 'bap cai xao', 'bo xao', 'ca chien', 'ca chua', 'ca kho',
            'ca rot', 'canh bau', 'canh bi do', 'canh cai', 'canh chua', 'canh rong bien', 'chuoi',
            'com', 'dau bap', 'dau hu', 'dau que', 'do chua', 'dua hau', 'dua leo', 'ga chien',
            'ga kho', 'kho qua', 'kho tieu', 'kho trung', 'nuoc mam', 'nuoc tuong', 'oi', 'ot',
            'rau', 'rau muong', 'rau ngo', 'suon mieng', 'suon xao', 'thanh long', 'thit chien',
            'thit luoc', 'tom', 'trung chien', 'trung luoc'
        ]
        self.detector = FoodDetector()
        self.classifier = FoodClassifier(class_names=self.class_names)
        self.billing = BillingSystem()

        self.upload_btn = tk.Button(root, text="Upload Image", command=self.start_upload_thread)
        self.upload_btn.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.result_text = tk.Text(root, height=15, width=60)
        self.result_text.pack(pady=10)

    def start_upload_thread(self):
        """Start a new thread to handle image processing."""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Processing...\n")
        self.upload_btn.config(state='disabled')  # Disable button during processing
        thread = threading.Thread(target=self.upload_image)
        thread.daemon = True  # Thread terminates when main program exits
        thread.start()

    def upload_image(self):
        """Process the image in a separate thread."""
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
            if not file_path or not os.path.exists(file_path):
                self.root.after(0, lambda: messagebox.showerror("Error", "No valid image file selected!"))
                self.root.after(0, self.enable_button)
                return

            img = cv2.imread(file_path)
            if img is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to load image. Ensure the file is a valid image format."))
                self.root.after(0, self.enable_button)
                return

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((400, 400))
            img_tk = ImageTk.PhotoImage(img_pil)

            # Update GUI on the main thread
            self.root.after(0, lambda: self.image_label.configure(image=img_tk))
            self.root.after(0, lambda: setattr(self.image_label, 'image', img_tk))

            cropped_paths, yolo_classes, results = self.detector.detect_and_crop(file_path)
            if len(cropped_paths) < 4:
                self.root.after(0, lambda: self.result_text.delete(1.0, tk.END))
                self.root.after(0, lambda: self.result_text.insert(tk.END, "Error: Less than 4 food items detected!"))
                self.root.after(0, self.enable_button)
                return

            food_items = []
            result_text = "Detected Food Items:\n"
            for path, yolo_class in zip(cropped_paths, yolo_classes):
                resnet_class = self.classifier.classify(path)
                if resnet_class:
                    food_items.append(resnet_class)
                    result_text += f"YOLO: {yolo_class}, ResNet50V2: {resnet_class}\n"
                    if yolo_class != resnet_class:
                        result_text += f"Discrepancy detected, using ResNet50V2 class: {resnet_class}\n"
                else:
                    food_items.append(yolo_class)
                    result_text += f"ResNet50V2 failed, using YOLO class: {yolo_class}\n"

            bill_details, total_cost, total_calories = self.billing.calculate_bill(food_items)
            for detail in bill_details:
                result_text += f"Item: {detail['item']}, Price: {detail['price']} VND, Calories: {detail['calories']} kcal\n"
            result_text += f"\nTotal Cost: {total_cost} VND\nTotal Calories: {total_calories} kcal"

            # Update GUI with results
            self.root.after(0, lambda: self.result_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.result_text.insert(tk.END, result_text))
            self.root.after(0, self.enable_button)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.root.after(0, self.enable_button)

    def enable_button(self):
        """Re-enable the upload button after processing."""
        self.upload_btn.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    app = FoodBillingGUI(root)
    root.mainloop()