import os
import sys

# Xác định đường dẫn gốc khi chạy từ file EXE
if getattr(sys, 'frozen', False):
    # Nếu đang chạy từ EXE (đã đóng gói)
    application_path = os.path.dirname(sys.executable)
else:
    # Nếu đang chạy từ script Python
    application_path = os.path.dirname(os.path.abspath(__file__))

# Điều chỉnh các đường dẫn khi khởi tạo Flask app
static_folder = os.path.join(application_path, 'web_ui', 'static')
template_folder = os.path.join(application_path, 'web_ui', 'templates')

from flask import Flask, render_template, request, jsonify
import os
import base64
import tempfile
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import json

# Import your existing modules
from src.detect import FoodDetector
from src.classify import FoodClassifier
from src.billing import BillingSystem

app = Flask(__name__, 
            static_folder=static_folder,
            template_folder=template_folder)
            
# Increase max content length to 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Configure JSON serialization to handle NumPy types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(CustomJSONEncoder, self).default(obj)

app.json_encoder = CustomJSONEncoder

# Initialize your existing systems
class_names = [
    'banh mi', 'bap cai luoc', 'bap cai xao', 'bo xao', 'ca chien', 'ca chua', 'ca kho',
    'ca rot', 'canh bau', 'canh bi do', 'canh cai', 'canh chua', 'canh rong bien', 'chuoi',
    'com', 'dau bap', 'dau hu', 'dau que', 'do chua', 'dua hau', 'dua leo', 'ga chien',
    'ga kho', 'kho qua', 'kho tieu', 'kho trung', 'nuoc mam', 'nuoc tuong', 'oi', 'ot',
    'rau', 'rau muong', 'rau ngo', 'suon mieng', 'suon xao', 'thanh long', 'thit chien',
    'thit luoc', 'tom', 'trung chien', 'trung luoc'
]

detector = None
classifier = None
billing = None

def initialize_models():
    global detector, classifier, billing
    detector = FoodDetector()
    classifier = FoodClassifier(class_names=class_names)
    billing = BillingSystem()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    # Check if models are initialized
    if detector is None or classifier is None or billing is None:
        initialize_models()
    
    try:
        # Get the image data from the request
        if 'image' not in request.files:
            # Handle base64 encoded image
            if 'imageData' in request.form:
                image_data = request.form['imageData']
                # Remove the data URL prefix if present
                if 'data:image' in image_data:
                    image_data = image_data.split(',')[1]
                
                try:
                    # Convert base64 to image
                    image_bytes = base64.b64decode(image_data)
                    
                    # Create a temporary file to save the image
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                    temp_file.write(image_bytes)
                    temp_file.close()
                    image_path = temp_file.name
                except Exception as e:
                    return jsonify({'error': f'Error processing image data: {str(e)}'}), 400
            else:
                return jsonify({'error': 'No image provided'}), 400
        else:
            # Handle file upload
            image_file = request.files['image']
            if not image_file:
                return jsonify({'error': 'Empty file provided'}), 400
                
            try:
                filename = secure_filename(image_file.filename)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
                image_file.save(temp_file.name)
                temp_file.close()
                image_path = temp_file.name
            except Exception as e:
                return jsonify({'error': f'Error saving uploaded file: {str(e)}'}), 400
        
        # Verify the image can be read
        try:
            img = cv2.imread(image_path)
            if img is None:
                return jsonify({'error': 'Cannot read image file. Please try a different image.'}), 400
                
            # Resize the image if it's too large
            max_dimension = 1920  # Maximum dimension for processing
            height, width = img.shape[:2]
            print(f"Original image dimensions: {width}x{height}")
            if height > max_dimension or width > max_dimension:
                if height > width:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                else:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                img = cv2.resize(img, (new_width, new_height))
                cv2.imwrite(image_path, img)
                print(f"Resized image to {new_width}x{new_height}")
        except Exception as e:
            return jsonify({'error': f'Error reading image: {str(e)}'}), 400
        
        # Process the image using your existing pipeline
        try:
            print("Starting food detection with YOLO...")
            cropped_paths, yolo_classes, results = detector.detect_and_crop(image_path)
            print(f"YOLO detection completed. Found {len(cropped_paths)} items.")
            for i, cls in enumerate(yolo_classes):
                print(f"  Item {i+1}: {cls}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': f'Error detecting food items: {str(e)}'
            }), 500
        
        # Check if any food items were detected
        if len(cropped_paths) == 0:
            return jsonify({
                'error': 'Không phát hiện được món ăn nào trong hình ảnh. Vui lòng thử lại với ảnh khác.',
                'items_found': 0
            }), 400
        
        # Classify each cropped image and populate food_items
        print("Starting classification of cropped images...")
        food_items = []
        detected_items = []
        
        for i, (path, yolo_class) in enumerate(zip(cropped_paths, yolo_classes)):
            try:
                print(f"Processing item {i+1}: {yolo_class} (YOLO class)")
                # Load the cropped image for display
                crop_img = cv2.imread(path)
                if crop_img is None:
                    print(f"  Failed to read cropped image at {path}")
                    continue
                
                # Compress and resize the cropped image to reduce size
                max_crop_size = 300
                height, width = crop_img.shape[:2]
                if height > max_crop_size or width > max_crop_size:
                    if height > width:
                        new_height = max_crop_size
                        new_width = int(width * (max_crop_size / height))
                    else:
                        new_width = max_crop_size
                        new_height = int(height * (max_crop_size / width))
                    crop_img = cv2.resize(crop_img, (new_width, new_height))
                
                # KHÔNG chuyển đổi BGR sang RGB ở đây, giữ nguyên định dạng BGR cho imencode
                # cv2 làm việc với BGR mặc định, khi encode thành JPEG nó sẽ tự chuyển về định dạng đúng
                _, buffer = cv2.imencode('.jpg', crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                crop_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Classify with ResNet50V2
                print(f"  Classifying with ResNet50V2...")
                resnet_class = classifier.classify(path)
                print(f"  ResNet50V2 classification result: {resnet_class}")
                
                if resnet_class:
                    food_class = resnet_class
                    detected_items.append({
                        'id': i,
                        'yolo_class': yolo_class,
                        'resnet_class': resnet_class,
                        'final_class': resnet_class,
                        'image': f'data:image/jpeg;base64,{crop_b64}'
                    })
                    food_items.append(resnet_class)
                else:
                    food_class = yolo_class
                    detected_items.append({
                        'id': i,
                        'yolo_class': yolo_class,
                        'resnet_class': None,
                        'final_class': yolo_class,
                        'image': f'data:image/jpeg;base64,{crop_b64}'
                    })
                    food_items.append(yolo_class)
            except Exception as e:
                print(f"Error processing cropped image {i}: {e}")
                continue
        
        # If we didn't process any food items
        if len(food_items) == 0:
            return jsonify({
                'error': 'Không thể xử lý được món ăn nào. Vui lòng thử lại với ảnh rõ ràng hơn.',
                'items_processed': 0
            }), 400
        
        try:
            # Calculate the bill
            print("Calculating bill for food items:", food_items)
            bill_details, total_cost, total_calories = billing.calculate_bill(food_items)
            print(f"Bill calculation complete: {len(bill_details)} items, {total_cost} VND, {total_calories} kcal")
            
            # Convert NumPy types to Python types
            total_cost = float(total_cost) if hasattr(total_cost, 'item') else total_cost
            total_calories = float(total_calories) if hasattr(total_calories, 'item') else total_calories
            
            # Format the bill details for better display
            formatted_bill = []
            for i, detail in enumerate(bill_details):
                price = float(detail['price']) if hasattr(detail['price'], 'item') else detail['price']
                calories = float(detail['calories']) if hasattr(detail['calories'], 'item') else detail['calories']
                
                print(f"Item {i+1}: {detail['item']}, Price: {price} VND, Calories: {calories} kcal")
                
                item_details = {
                    'id': i,
                    'item': detail['item'],
                    'price': price,
                    'calories': calories,
                    'image': detected_items[i]['image'] if i < len(detected_items) else None
                }
                formatted_bill.append(item_details)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error calculating bill: {str(e)}'}), 500
        
        # Clean up temporary files
        try:
            os.unlink(image_path)
            for path in cropped_paths:
                if os.path.exists(path):
                    os.unlink(path)
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
        
        # Return the results
        result = {
            'success': True,
            'detected_items': detected_items,
            'bill_details': formatted_bill,
            'total_cost': total_cost,
            'total_calories': total_calories,
            'items_count': len(formatted_bill)
        }
        print("Final result:", result)
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Lỗi phân tích: {str(e)}'}), 500

@app.route('/api/food-info', methods=['GET'])
def get_food_info():
    """Return information about available food items"""
    try:
        if billing is None:
            initialize_models()
        
        # Get food info from the menu CSV
        menu_data = billing.menu.to_dict('records')
        
        # Format the data for the frontend
        food_info = []
        for item in menu_data:
            food_info.append({
                'name': item['item'],
                'calories': item['calories'],
                'price': item['price'],
                'category': get_food_category(item['item'])
            })
        
        return jsonify({'success': True, 'food_info': food_info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/update-food-item', methods=['POST'])
def update_food_item():
    """Update a food item classification and recalculate bill"""
    try:
        # Get request data
        data = request.json
        if not data or 'itemIndex' not in data or 'newFoodItem' not in data:
            return jsonify({'error': 'Missing required data'}), 400
            
        item_index = data['itemIndex']
        new_food_item = data['newFoodItem']
        bill_data = data.get('billData', {})
        
        # If we have no billing system initialized, initialize it
        if billing is None:
            initialize_models()
        
        # Get the menu data to find the new food item details
        menu_data = billing.menu.to_dict('records')
        matching_item = next((item for item in menu_data if item['item'] == new_food_item), None)
        
        if not matching_item:
            return jsonify({'error': f'Food item {new_food_item} not found in menu'}), 404
            
        # Calculate new bill with updated item
        # If bill_data was provided, use it to update the specific item
        if bill_data and 'bill_details' in bill_data and item_index < len(bill_data['bill_details']):
            # Save old details for reference
            old_item = bill_data['bill_details'][item_index]['item']
            old_price = bill_data['bill_details'][item_index]['price']
            old_calories = bill_data['bill_details'][item_index]['calories']
            
            # Update the item
            bill_data['bill_details'][item_index]['item'] = new_food_item
            bill_data['bill_details'][item_index]['price'] = matching_item['price']
            bill_data['bill_details'][item_index]['calories'] = matching_item['calories']
            
            # Recalculate totals
            total_cost = sum(item['price'] for item in bill_data['bill_details'])
            total_calories = sum(item['calories'] for item in bill_data['bill_details'])
            
            # Update the totals
            bill_data['total_cost'] = total_cost
            bill_data['total_calories'] = total_calories
            
            return jsonify({
                'success': True, 
                'updated_bill': bill_data,
                'old_item': {
                    'name': old_item,
                    'price': old_price,
                    'calories': old_calories
                },
                'new_item': {
                    'name': new_food_item,
                    'price': matching_item['price'],
                    'calories': matching_item['calories']
                }
            })
        else:
            # Return just the new item details if no bill data was provided
            return jsonify({
                'success': True,
                'new_item': {
                    'name': new_food_item,
                    'price': matching_item['price'],
                    'calories': matching_item['calories']
                }
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error updating food item: {str(e)}'}), 500

def get_food_category(item_name):
    """Simple function to categorize food items"""
    item_name = item_name.lower()
    
    # Categories based on common Vietnamese food types
    if any(x in item_name for x in ['com', 'bun', 'pho', 'banh']):
        return 'Carbohydrates'
    elif any(x in item_name for x in ['thit', 'ga', 'bo', 'ca chien', 'ca kho', 'trung', 'tom', 'suon', 'kho tieu', 'dau hu']):
        return 'Protein'
    elif any(x in item_name for x in ['rau', 'dau', 'bap cai', 'cai', 'dua leo', 'ot', 'ca chua', 'kho qua']):
        return 'Vegetable'
    elif any(x in item_name for x in ['canh', 'ca chua', 'ca rot']):
        return 'Soup'
    else:
        return 'Other'

if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)