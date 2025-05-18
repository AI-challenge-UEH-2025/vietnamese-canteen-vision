import os
import sys
from threading import Timer
import webbrowser

# Thêm đường dẫn hiện tại vào PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import ứng dụng Flask
from app import app, initialize_models

def open_browser():
    """Mở trình duyệt web khi ứng dụng khởi động"""
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Khởi tạo các mô hình
    initialize_models()
    
    # Mở trình duyệt web sau 1.5 giây
    Timer(1.5, open_browser).start()
    
    # Chạy ứng dụng Flask (với debug=False khi đóng gói)
    app.run(debug=False, host='127.0.0.1', port=5000)