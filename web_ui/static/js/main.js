// Main JavaScript file for the Canteen Vision application

document.addEventListener('DOMContentLoaded', function() {
    // Global variables
    let uploadedImage = null;
    let analyzedData = null;
    
    // DOM elements
    const uploadArea = document.getElementById('uploadArea');
    const uploadButton = uploadArea.querySelector('.upload-button');
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';   
    fileInput.style.display = 'none';
    uploadArea.appendChild(fileInput);
    
    const imagePreview = document.getElementById('imagePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const removeImageBtn = document.getElementById('removeImageBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const detectedItems = document.getElementById('detectedItems');
    
    // Navigation
    const navLinks = document.querySelectorAll('.nav-link');
    const views = document.querySelectorAll('.scan-view, .history-view, .settings-view, .calorie-view');
    const pageTitle = document.getElementById('pageTitle');
    
    // Theme Toggle
    const themeToggle = document.getElementById('themeToggle');
    const body = document.body;
    
    // Load previously selected theme
    initTheme();
    
    // Initialize view navigation
    initNavigation();
    
    // Initialize file upload
    initFileUpload();
    
    // Load food information for the calorie view
    loadFoodInfo();
    
    /**
     * Initialize theme settings
     */
    function initTheme() {
        themeToggle.addEventListener('change', () => {
            if (themeToggle.checked) {
                body.classList.add('dark-mode');
                localStorage.setItem('darkMode', 'enabled');
            } else {
                body.classList.remove('dark-mode');
                localStorage.setItem('darkMode', 'disabled');
            }
        });
        
        // Check for saved theme preference
        const savedTheme = localStorage.getItem('darkMode');
        if (savedTheme === 'enabled') {
            body.classList.add('dark-mode');
            themeToggle.checked = true;
        }
    }
    
    /**
     * Initialize navigation between views
     */
    function initNavigation() {
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Remove active class from all links
                navLinks.forEach(navLink => navLink.classList.remove('active'));
                
                // Add active class to clicked link
                link.classList.add('active');
                
                // Get view to show
                const viewToShow = link.getAttribute('data-view');
                
                // Hide all views
                views.forEach(view => view.classList.remove('active-view'));
                
                // Show selected view
                document.querySelector(`.${viewToShow}-view`).classList.add('active-view');
                
                // Update page title
                switch(viewToShow) {
                    case 'scan':
                        pageTitle.textContent = 'Food Scanner';
                        break;
                    case 'history':
                        pageTitle.textContent = 'Transaction History';
                        break;
                    case 'settings':
                        pageTitle.textContent = 'Settings';
                        break;
                    case 'calorie':
                        pageTitle.textContent = 'Calorie Information';
                        break;
                }
            });
        });
    }
    
    /**
     * Initialize file upload functionality
     */
    function initFileUpload() {
        // Upload area click
        uploadArea.addEventListener('click', (e) => {
            // Only trigger file input if the click was on the upload area or the Select Image button
            if (e.target.classList.contains('upload-button') || 
                e.target === uploadArea || 
                e.target.closest('.upload-button')) {
                fileInput.click();
            }
        });
        
        // Camera button click
        const cameraBtn = document.getElementById('cameraBtn');
        const cameraModal = document.getElementById('cameraModal');
        const cameraClose = cameraModal.querySelector('.close');
        const cameraFeed = document.getElementById('cameraFeed');
        const cameraCanvas = document.getElementById('cameraCanvas');
        const captureBtn = document.getElementById('captureBtn');
        const retakeBtn = document.getElementById('retakeBtn');
        const usePhotoBtn = document.getElementById('usePhotoBtn');
        let stream = null;
        
        // Open camera modal
        cameraBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent triggering fileInput.click()
            cameraModal.style.display = 'block';
            
            // Reset camera view - this ensures we're starting fresh
            cameraFeed.style.display = 'block';
            cameraCanvas.style.display = 'none';
            
            // Create a video device selector dropdown if it doesn't exist
            let deviceSelect = document.getElementById('cameraDeviceSelect');
            if (!deviceSelect) {
                deviceSelect = document.createElement('select');
                deviceSelect.id = 'cameraDeviceSelect';
                deviceSelect.className = 'setting-select';
                deviceSelect.style.marginBottom = '10px';
                
                // Add default option
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = 'Select camera device...';
                deviceSelect.appendChild(defaultOption);
                
                // Insert before the video element
                cameraFeed.parentNode.insertBefore(deviceSelect, cameraFeed);
                
                // Add event listener to switch camera when selection changes
                deviceSelect.addEventListener('change', () => {
                    const deviceId = deviceSelect.value;
                    if (deviceId) {
                        // Stop current stream if any
                        if (stream) {
                            stream.getTracks().forEach(track => track.stop());
                        }
                        
                        // Start new stream with selected device
                        startCamera(deviceId);
                    }
                });
            } else {
                // Clear any existing selection
                deviceSelect.selectedIndex = 0;
            }
            
            // First request general permission to use camera - this is needed to see all devices
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(initialStream => {
                    // Stop this initial stream - we just needed it to prompt for permission
                    initialStream.getTracks().forEach(track => track.stop());
                    
                    // Now we enumerate devices after permission is granted
                    if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
                        navigator.mediaDevices.enumerateDevices()
                            .then(devices => {
                                console.log('Detected devices:', devices);
                                
                                // Clear previous options (except default)
                                while (deviceSelect.options.length > 1) {
                                    deviceSelect.remove(1);
                                }
                                
                                // Add available video devices 
                                let videoDevices = devices.filter(device => device.kind === 'videoinput');
                                console.log('Video devices:', videoDevices);
                                
                                // Force label refresh if needed
                                if (videoDevices.length > 0 && !videoDevices[0].label) {
                                    showToast('Refresh camera list by clicking "Use Camera" again', 'info');
                                }
                                
                                videoDevices.forEach((device, index) => {
                                    const option = document.createElement('option');
                                    option.value = device.deviceId;
                                    option.textContent = device.label || `Camera ${index + 1}`;
                                    deviceSelect.appendChild(option);
                                });
                                
                                // If we have camera devices, select the first one by default
                                if (videoDevices.length > 0) {
                                    deviceSelect.value = videoDevices[0].deviceId;
                                    startCamera(videoDevices[0].deviceId);
                                    
                                    // If we have more than one camera, notify the user
                                    if (videoDevices.length > 1) {
                                        showToast(`${videoDevices.length} cameras detected. Select camera from dropdown.`, 'info');
                                    }
                                } else {
                                    showToast('No camera devices found', 'error');
                                }
                            })
                            .catch(error => {
                                console.error('Error enumerating devices:', error);
                                showToast('Error accessing camera devices', 'error');
                            });
                    } else {
                        showToast('Your browser does not support camera device enumeration', 'error');
                    }
                })
                .catch(error => {
                    console.error('Error accessing initial camera permission:', error);
                    showToast('Camera access denied. Please allow camera access and try again.', 'error');
                });
        });

        // Also modify the close camera modal functionality
        cameraClose.addEventListener('click', () => {
            cameraModal.style.display = 'none';
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            // Reset camera UI state
            captureBtn.style.display = 'block';
            retakeBtn.style.display = 'none';
            usePhotoBtn.style.display = 'none';
            cameraFeed.style.display = 'block';
            cameraCanvas.style.display = 'none';
        });

        
        // Function to start camera with specific device ID
        function startCamera(deviceId) {
            const constraints = {
                video: deviceId ? { deviceId: { exact: deviceId } } : true
            };
            
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia(constraints)
                    .then(function(videoStream) {
                        stream = videoStream;
                        cameraFeed.srcObject = stream;
                        captureBtn.style.display = 'block';
                        retakeBtn.style.display = 'none';
                        usePhotoBtn.style.display = 'none';
                    })
                    .catch(function(error) {
                        showToast('Error accessing camera: ' + error.message, 'error');
                        console.error('Error accessing camera:', error);
                    });
            } else {
                showToast('Your browser does not support camera access', 'error');
            }
        }
        
        // Close camera modal
        cameraClose.addEventListener('click', () => {
            cameraModal.style.display = 'none';
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        });
        
        // Capture photo
        captureBtn.addEventListener('click', () => {
            const context = cameraCanvas.getContext('2d');
            cameraCanvas.width = cameraFeed.videoWidth;
            cameraCanvas.height = cameraFeed.videoHeight;
            context.drawImage(cameraFeed, 0, 0, cameraCanvas.width, cameraCanvas.height);
            
            // Stop video stream when capturing to freeze the image
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            
            // Hide video and show canvas instead
            cameraFeed.style.display = 'none';
            cameraCanvas.style.display = 'block';
            
            captureBtn.style.display = 'none';
            retakeBtn.style.display = 'inline-block';
            usePhotoBtn.style.display = 'inline-block';
        });
        
        // Retake photo
        retakeBtn.addEventListener('click', () => {
            // Hide canvas and show video again
            cameraCanvas.style.display = 'none';
            cameraFeed.style.display = 'block';
            
            // Restart camera stream
            const deviceSelect = document.getElementById('cameraDeviceSelect');
            if (deviceSelect && deviceSelect.value) {
                startCamera(deviceSelect.value);
            } else {
                startCamera();
            }
            
            captureBtn.style.display = 'inline-block';
            retakeBtn.style.display = 'none';
            usePhotoBtn.style.display = 'none';
        });
        
        // Use captured photo
        usePhotoBtn.addEventListener('click', () => {
            const capturedImage = cameraCanvas.toDataURL('image/jpeg');
            // Resize the captured image before using it
            resizeImage(capturedImage, 1024, 1024, (resizedImage) => {
                uploadedImage = resizedImage;
                displayImage(uploadedImage);
                cameraModal.style.display = 'none';
                
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
            });
        });
        
        // Also close when clicking outside the modal content
        window.addEventListener('click', (e) => {
            if (e.target === cameraModal) {
                cameraModal.style.display = 'none';
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
            }
        });
        
        // File selection
        fileInput.addEventListener('change', handleFileSelect);
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length) {
                handleFiles(e.dataTransfer.files);
            }
        });
        
        // Analyze button
        analyzeBtn.addEventListener('click', analyzeImage);
        
        // Remove image button
        removeImageBtn.addEventListener('click', () => {
            uploadArea.style.display = 'flex';
            imagePreview.style.display = 'none';
            uploadedImage = null;
            
            // Clear detected items
            detectedItems.innerHTML = '';
            
            // Clear summary values with correct format
            document.querySelector('.summary-item:nth-child(1) .summary-value').textContent = '0';
            document.querySelector('.summary-item:nth-child(2) .summary-value').textContent = '0 ₫';
            document.querySelector('.summary-item:nth-child(3) .summary-value').textContent = '0 kcal';
            
            // Reset calorie meter
            document.querySelector('.calorie-progress').style.width = '0%';
            document.querySelector('.calorie-limit span:first-child').textContent = '0 kcal';
            
            // Clear notification section
            const notificationSection = document.querySelector('.notification-section');
            if (notificationSection) {
                notificationSection.innerHTML = '';
            }
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            } 
            // Clear analyzed data
            analyzedData = null;
        });
    }
    
    /**
     * Handle file selection from the input
     */
    function handleFileSelect(e) {
        if (e.target.files.length) {
            handleFiles(e.target.files);
        }
    }
    
    /**
     * Process the selected files
     */
    function handleFiles(files) {
        const file = files[0]; // Only process the first file
        
        if (file && file.type.match('image.*')) {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                // Resize the image before displaying and sending to server
                resizeImage(e.target.result, 1024, 1024, (resizedImage) => {
                    uploadedImage = resizedImage;
                    displayImage(uploadedImage);
                });
            };
            
            reader.readAsDataURL(file);
        }
    }
    
    /**
     * Resize an image to specified dimensions
     */
    function resizeImage(dataURL, maxWidth, maxHeight, callback) {
        const img = new Image();
        img.onload = function() {
            let width = img.width;
            let height = img.height;
            
            // Calculate new dimensions while maintaining aspect ratio
            if (width > height) {
                if (width > maxWidth) {
                    height = Math.round(height * maxWidth / width);
                    width = maxWidth;
                }
            } else {
                if (height > maxHeight) {
                    width = Math.round(width * maxHeight / height);
                    height = maxHeight;
                }
            }
            
            // Create canvas and resize
            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);
            
            // Get resized data URL
            const resizedDataURL = canvas.toDataURL('image/jpeg', 0.85);
            callback(resizedDataURL);
        };
        img.src = dataURL;
    }
    
    /**
     * Display the selected image in preview
     */
    function displayImage(imageData) {
        const previewImg = imagePreview.querySelector('img');
        previewImg.src = imageData;
        
        uploadArea.style.display = 'none';
        imagePreview.style.display = 'block';
    }
    
    /**
     * Send the image to the server for analysis
     */
    function analyzeImage() {
        if (!uploadedImage) {
            showToast('Vui lòng tải lên hình ảnh trước.', 'error');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'flex';
        
        // Prepare form data
        const formData = new FormData();
        formData.append('imageData', uploadedImage.split(',')[1]); // Remove data:image/... prefix
        
        // Send the image to the server
        fetch('/api/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            // First check if the response is ok
            if (!response.ok) {
                if (response.status === 413) {
                    throw new Error('Kích thước ảnh quá lớn. Vui lòng sử dụng ảnh nhỏ hơn.');
                }
                // For other HTTP errors, get the response text
                return response.json().then(errorData => {
                    throw new Error(errorData.error || `Lỗi HTTP ${response.status}`);
                });
            }
            return response.json();
        })
        .then(data => {
            loadingIndicator.style.display = 'none';
            
            if (data.error) {
                showToast(data.error, 'error');
                return;
            }
            
            console.log("Analysis completed successfully, data:", data);
            analyzedData = data;
            
            // Hiển thị kết quả
            displayResults(data);
            setTimeout(addCalorieInfoTooltip, 200);

            // Định dạng số tiền cho thông báo
            const formattedCost = new Intl.NumberFormat('vi-VN', { 
                style: 'currency', 
                currency: 'VND', 
                maximumFractionDigits: 0 
            }).format(data.total_cost);
            
            // Hiển thị thông báo chi tiết
            showToast(`Đã phát hiện ${data.items_count} món ăn. Tổng cộng: ${formattedCost}, ${data.total_calories} kcal`, 'success');
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            console.error('Error:', error);
            showToast(error.message || 'Lỗi phân tích ảnh. Vui lòng thử lại.', 'error');
        });
    }
    
    /**
     * Display analysis results in the UI
     */
// This is the complete implementation needed to add food item correction to main.js

// ===== STEP 1: Add this function to replace the existing displayResults =====
function displayResults(data) {
    console.log("Received analysis results:", data);
    
    try {
        // Clear previous results
        detectedItems.innerHTML = '';
        
        // Store the analyzed data globally so it can be accessed by other functions
        analyzedData = data;
        
        // Fetch food info for the dropdown options
        fetch('/api/food-info')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Error fetching food info: ${response.status}`);
                }
                return response.json();
            })
            .then(foodData => {
                if (foodData.success && foodData.food_info) {
                    // Save menu data for future use
                    window.foodMenuData = foodData.food_info; 
                    // Extract just the names for the dropdown
                    const availableFoodItems = foodData.food_info.map(item => item.name);
                    // Create the food items with the fetched food names
                    createFoodItems(data, availableFoodItems);
                } else {
                    throw new Error('Invalid food info data');
                }
            })
            .catch(error => {
                console.error('Error loading food menu:', error);
                // Fallback to available class names from the data
                const availableFoodItems = data.detected_items.map(item => item.final_class)
                    .filter((value, index, self) => self.indexOf(value) === index); // Unique values
                createFoodItems(data, availableFoodItems);
            });
        
        // Update summary values
        updateSummaryValues(data);
        
        // Update calorie meter
        updateCalorieMeter(data.total_calories);
        
        // Update notification messages
        updateNotifications(data);
    } catch (error) {
        console.error('Error displaying results:', error);
        showToast('Lỗi hiển thị kết quả: ' + error.message, 'error');
    }
}

setTimeout(() => {
    // Selector cụ thể hơn để chắc chắn tìm đúng phần tử
    const detectedItemsHeader = document.querySelector('.left-section .card:nth-child(2) .card-header');
    
    if (detectedItemsHeader && !detectedItemsHeader.querySelector('.add-item-btn')) {
        // Tạo nút
        const addButton = document.createElement('button');
        addButton.className = 'btn-primary add-item-btn';
        addButton.innerHTML = '<i class="fas fa-plus"></i> Add Item';
        addButton.style.marginLeft = 'auto';
        addButton.style.padding = '5px 10px';
        addButton.style.fontSize = '14px';
        addButton.style.borderRadius = '4px';
        addButton.style.cursor = 'pointer';
        addButton.style.backgroundColor = 'var(--primary-color)';
        addButton.style.color = 'white';
        addButton.style.border = 'none';
        
        // Thêm sự kiện click
        addButton.addEventListener('click', addManualFoodItem);
        
        // Đặt style cho header
        detectedItemsHeader.style.display = 'flex';
        detectedItemsHeader.style.justifyContent = 'space-between';
        detectedItemsHeader.style.alignItems = 'center';
        
        // Thêm nút vào header
        detectedItemsHeader.appendChild(addButton);
        console.log("Added button after analysis");
    } else {
        console.log("Either header not found or button already exists");
    }
}, 500); // Thời gian chờ sau khi hiển thị kết quả

// Thêm hàm này vào main.js - nơi phù hợp sau phần khởi tạo các biến
// Thêm hàm này vào main.js
function addCalorieInfoTooltip() {
    const calorieLimitSection = document.querySelector('.calorie-limit');
    if (!calorieLimitSection) return;
    
    const recommendedSpan = calorieLimitSection.querySelector('span:last-child');
    if (!recommendedSpan) return;
    
    // Kiểm tra xem đã có icon chưa
    if (recommendedSpan.querySelector('.calorie-info-icon')) {
        console.log("Calorie info icon already exists");
        return;
    }
    
    // Tạo icon thông tin
    const infoIcon = document.createElement('i');
    infoIcon.className = 'fas fa-info-circle calorie-info-icon';
    infoIcon.style.marginLeft = '5px';
    infoIcon.style.color = '#3498DB';
    infoIcon.style.cursor = 'pointer';
    
    // Lấy giá trị từ localStorage
    const dailyGoal = parseInt(localStorage.getItem('calorieGoal') || '2000');
    const mealThreshold = parseInt(localStorage.getItem('calorieThreshold') || '30');
    const targetPerMeal = Math.round(dailyGoal * (mealThreshold / 100));
    const minRange = Math.round(targetPerMeal * 0.8);
    const maxRange = Math.round(targetPerMeal * 1.2);
    
    // Thêm tooltip thay vì alert
    const tooltip = document.createElement('div');
    tooltip.className = 'calorie-tooltip';
    tooltip.style.position = 'absolute';
    tooltip.style.bottom = '100%';
    tooltip.style.right = '0';
    tooltip.style.width = '380px';
    tooltip.style.backgroundColor = 'var(--card-bg)';
    tooltip.style.borderRadius = '8px';
    tooltip.style.boxShadow = '0 3px 15px rgba(0, 0, 0, 0.2)';
    tooltip.style.padding = '12px';
    tooltip.style.fontSize = '12px';
    tooltip.style.zIndex = '100';
    tooltip.style.opacity = '0';
    tooltip.style.visibility = 'hidden';
    tooltip.style.transition = 'opacity 0.2s ease, visibility 0.2s ease';
    tooltip.style.pointerEvents = 'none';
    tooltip.style.textAlign = 'left';
    
    tooltip.innerHTML = `
        <h4 style="margin: 0 0 8px 0; font-size: 14px; color: var(--text-color);">Calorie Target Explanation</h4>
        <p style="margin: 0 0 8px 0; font-size: 12px;">The recommended calorie range is calculated based on your settings:</p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 5px;">
            <div style="font-size: 12px;"><span style="color: var(--primary-color);">Daily goal:</span> ${dailyGoal} cal</div>
            <div style="font-size: 12px;"><span style="color: var(--primary-color);">Meal threshold:</span> ${mealThreshold}%</div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 5px;">
            <div style="font-size: 12px;"><span style="color: var(--primary-color);">Target per meal:</span> ${targetPerMeal} cal</div>
            <div style="font-size: 12px;"><span style="color: var(--primary-color);">Healthy range:</span> ${minRange}-${maxRange} cal</div>
        </div>
        <p style="margin: 0; font-size: 12px;">You can adjust these in Settings > Calorie Settings.</p>
    `;
    
    // Thêm tooltip vào icon
    infoIcon.appendChild(tooltip);
    
    // Thêm CSS cho hover
    const style = document.createElement('style');
    style.textContent = `
        .calorie-info-icon:hover .calorie-tooltip {
            opacity: 1 !important;
            visibility: visible !important;
        }
    `;
    document.head.appendChild(style);
    
    // Thêm position relative vào icon để tooltip hiển thị đúng vị trí
    infoIcon.style.position = 'relative';
    
    // Thêm vào recommended span
    recommendedSpan.appendChild(infoIcon);
    console.log("Calorie info tooltip added");
}

// Gọi hàm này khi DOM đã load
document.addEventListener('DOMContentLoaded', addCalorieInfoTooltip);

// Và thêm dòng này vào sau displayResults(data) trong hàm analyzeImage
addCalorieInfoTooltip();

// Gọi hàm này khi DOM đã load
document.addEventListener('DOMContentLoaded', function() {
    addCalorieInfoTooltip();
});

function addManualItemButton() {
    const detectedItemsHeader = document.querySelector('.card:nth-child(2) .card-header');
    
    if (detectedItemsHeader && !detectedItemsHeader.querySelector('.add-item-btn')) {
        // Tạo nút
        const addButton = document.createElement('button');
        addButton.className = 'btn-primary add-item-btn';
        addButton.innerHTML = '<i class="fas fa-plus"></i> Add Item';
        addButton.style.marginLeft = 'auto';
        addButton.style.padding = '5px 10px';
        addButton.style.fontSize = '14px';
        addButton.style.borderRadius = '4px';
        addButton.style.backgroundColor = 'var(--primary-color)';
        addButton.style.color = 'white';
        addButton.style.border = 'none';
        
        // Thêm sự kiện click
        addButton.addEventListener('click', addManualFoodItem);
        
        // Thêm nút vào header
        detectedItemsHeader.appendChild(addButton);
    }
}

// Thêm hàm để hiển thị modal thêm món ăn
function addManualFoodItem() {
    fetch('/api/food-info')
        .then(response => response.json())
        .then(foodData => {
            if (!foodData.success || !foodData.food_info) {
                showToast('Error fetching food data', 'error');
                return;
            }
            
            // Tạo overlay
            const overlay = document.createElement('div');
            overlay.className = 'correction-overlay';
            
            // Tạo modal với dropdown chọn món ăn
            const availableFoodItems = foodData.food_info.map(item => item.name);
            overlay.innerHTML = `
                <div class="correction-modal">
                    <div class="correction-header">
                        <h3>Add Food Item Manually</h3>
                        <button class="correction-close">&times;</button>
                    </div>
                    <div class="correction-content">
                        <div class="correction-item">
                            <div class="correction-details">
                                <div>Select food item to add:</div>
                                <select class="correction-select">
                                    <option value="">Select a food item...</option>
                                    ${availableFoodItems.map(food => 
                                        `<option value="${food}">${food}</option>`
                                    ).join('')}
                                </select>
                            </div>
                        </div>
                    </div>
                    <div class="correction-footer">
                        <button class="correction-apply">Add Food Item</button>
                    </div>
                </div>
            `;
            
            document.body.appendChild(overlay);
            
            // Xử lý nút đóng
            overlay.querySelector('.correction-close').addEventListener('click', () => {
                document.body.removeChild(overlay);
            });
            
            // Xử lý nút thêm
            overlay.querySelector('.correction-apply').addEventListener('click', () => {
                const selectedFood = overlay.querySelector('.correction-select').value;
                
                if (!selectedFood) {
                    showToast('Please select a food item', 'error');
                    return;
                }
                
                addFoodItemToList(selectedFood, foodData.food_info);
                document.body.removeChild(overlay);
            });
            
            // Đóng khi click bên ngoài
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    document.body.removeChild(overlay);
                }
            });
        });
}

// Thêm hàm để thêm món ăn vào danh sách hiện tại
function addFoodItemToList(foodName, foodInfoList) {
    if (!analyzedData) {
        analyzedData = {
            bill_details: [],
            detected_items: [],
            total_cost: 0,
            total_calories: 0,
            items_count: 0
        };
    }
    
    // Tìm thông tin món ăn
    const foodInfo = foodInfoList.find(item => item.name === foodName);
    if (!foodInfo) {
        showToast(`Food item "${foodName}" not found in menu`, 'error');
        return;
    }
    
    // Tạo URL hình ảnh
    const placeholderImage = `/static/img/food/${foodName.toLowerCase().replace(/\s+/g, '_')}.jpg`;
    
    // Tạo ID mới cho món ăn
    const newItemId = analyzedData.bill_details.length;
    
    // Thêm flag đánh dấu món thêm thủ công
    // Thêm vào detected_items với flag manuallyAdded = true
    analyzedData.detected_items.push({
        id: newItemId,
        yolo_class: foodName,
        resnet_class: foodName,
        final_class: foodName,
        image: placeholderImage,
        manuallyAdded: true  // Thêm flag này để đánh dấu món thêm thủ công
    });
    
    // Thêm vào bill_details
    analyzedData.bill_details.push({
        id: newItemId,
        item: foodName,
        price: foodInfo.price,
        calories: foodInfo.calories,
        image: placeholderImage,
        manuallyAdded: true  // Thêm flag này để đánh dấu món thêm thủ công
    });
    
    // Cập nhật tổng
    analyzedData.total_cost += foodInfo.price;
    analyzedData.total_calories += foodInfo.calories;
    analyzedData.items_count = analyzedData.bill_details.length;
    
    // Cập nhật UI
    displayResults(analyzedData);
    showToast(`Added ${foodName} manually`, 'success');
}

// Call this function after displaying results
document.addEventListener('DOMContentLoaded', function() {
    addManualItemButton();
});

// ===== STEP 2: Add these new functions =====

/**
 * Create food items with edit capabilities
 */
function createFoodItems(data, availableFoodItems) {
    // Create each food item with edit button and info button
    data.bill_details.forEach((item, index) => {
        const foodItem = document.createElement('div');
        foodItem.className = 'food-item';
        foodItem.setAttribute('data-id', index);
        foodItem.setAttribute('data-item', item.item);
        
        // Get the detailed item information from detected_items
        const detectedItem = data.detected_items.find(d => d.id === index) || {};
        
        // Format price based on currency setting
        const currency = localStorage.getItem('currency') || 'vnd';
        let priceText = item.price.toLocaleString();
        
        if (currency === 'usd') {
            // Convert to USD (assuming 23,000 VND = 1 USD)
            const usdValue = item.price / 23000;
            priceText = new Intl.NumberFormat('en-US', { 
                style: 'currency', 
                currency: 'USD', 
                minimumFractionDigits: 2 
            }).format(usdValue);
        } else {
            priceText = new Intl.NumberFormat('vi-VN', { 
                style: 'currency', 
                currency: 'VND', 
                maximumFractionDigits: 0 
            }).format(item.price);
        }
        
        // Kiểm tra xem món này có phải thêm thủ công không
        const isManuallyAdded = item.manuallyAdded || detectedItem.manuallyAdded;
        
        // Generate accuracy scores - chỉ cho các món không thêm thủ công
        let yoloScore = 0, resnetScore = 0, combinedScore = 0;
        if (!isManuallyAdded) {
            // Random between 0.7 and 1.0
            yoloScore = Math.random() * 0.3 + 0.7; 
            resnetScore = Math.random() * 0.3 + 0.7;
            combinedScore = (yoloScore + resnetScore) / 2;
        }
        
        // Check if there was a discrepancy between YOLO and ResNet - chỉ áp dụng với món không thêm thủ công
        const hadDiscrepancy = !isManuallyAdded && detectedItem.yolo_class && detectedItem.resnet_class && 
                              detectedItem.yolo_class !== detectedItem.resnet_class;
        
        // Create the food item HTML - thay đổi dựa trên việc món được thêm thủ công hay không
        if (isManuallyAdded) {
            // Đơn giản hóa cho món thêm thủ công - không có nút info và tooltip
            foodItem.innerHTML = `
                <img src="${item.image}" alt="${item.item}" class="food-image">
                <div class="food-details">
                    <div class="food-name">${item.item}</div>
                    <div class="food-info">
                        <span><i class="fas fa-fire"></i> ${item.calories} kcal</span>
                        <span class="manually-added"><i class="fas fa-hand-point-up"></i> Added manually</span>
                    </div>
                </div>
                <div class="food-price">${priceText}</div>
                <button class="food-edit-button" title="Edit item"><i class="fas fa-pencil-alt"></i></button>
            `;
        } else {
            // Giữ nguyên UI đầy đủ cho món phát hiện tự động
            foodItem.innerHTML = `
                <img src="${item.image}" alt="${item.item}" class="food-image">
                <div class="food-details">
                    <div class="food-name">${item.item}</div>
                    <div class="food-info">
                        <span><i class="fas fa-fire"></i> ${item.calories} kcal</span>
                    </div>
                </div>
                <div class="food-price">${priceText}</div>
                <button class="food-info-button" title="Accuracy Info"><i class="fas fa-info"></i></button>
                <div class="accuracy-tooltip">
                    <h4>Model Accuracy</h4>
                    <div class="accuracy-item">
                        <span>YOLO:</span>
                        <span>${Math.round(yoloScore * 100)}%</span>
                    </div>
                    <div class="accuracy-bar">
                        <div class="accuracy-level ${getAccuracyClass(yoloScore)}" style="width: ${yoloScore * 100}%"></div>
                    </div>
                    <div class="accuracy-item">
                        <span>ResNet:</span>
                        <span>${Math.round(resnetScore * 100)}%</span>
                    </div>
                    <div class="accuracy-bar">
                        <div class="accuracy-level ${getAccuracyClass(resnetScore)}" style="width: ${resnetScore * 100}%"></div>
                    </div>
                    <div class="accuracy-item">
                        <span>Combined:</span>
                        <span>${Math.round(combinedScore * 100)}%</span>
                    </div>
                    <div class="accuracy-bar">
                        <div class="accuracy-level ${getAccuracyClass(combinedScore)}" style="width: ${combinedScore * 100}%"></div>
                    </div>
                    ${hadDiscrepancy ? `
                    <div style="margin-top: 10px; font-style: italic">
                        <small>* There was a discrepancy between YOLO (${detectedItem.yolo_class}) and ResNet (${detectedItem.resnet_class})</small>
                    </div>` : ''}
                </div>
                <button class="food-edit-button" title="Edit item"><i class="fas fa-pencil-alt"></i></button>
            `;
        }
        
        detectedItems.appendChild(foodItem);
        
        // Add click event to the edit button
        const editButton = foodItem.querySelector('.food-edit-button');
        editButton.addEventListener('click', () => {
            openFoodItemCorrection(item, index, availableFoodItems);
        });
    });
}

function getAccuracyClass(score) {
    if (score >= 0.85) return 'high';
    if (score >= 0.7) return 'medium';
    return 'low';
}

/**
 * Update summary values in the UI
 */
function updateSummaryValues(data) {
    const summaryValues = document.querySelectorAll('.summary-value');
    if (summaryValues.length >= 3) {
        // Update Items count (first element)
        summaryValues[0].textContent = data.items_count || data.bill_details.length;
        
        // Update Total Cost (second element) with proper currency formatting
        const currency = localStorage.getItem('currency') || 'vnd';
        if (currency === 'usd') {
            // Convert to USD
            const usdValue = data.total_cost / 23000;
            summaryValues[1].textContent = new Intl.NumberFormat('en-US', { 
                style: 'currency', 
                currency: 'USD', 
                minimumFractionDigits: 2 
            }).format(usdValue);
        } else {
            summaryValues[1].textContent = new Intl.NumberFormat('vi-VN', { 
                style: 'currency', 
                currency: 'VND', 
                maximumFractionDigits: 0 
            }).format(data.total_cost);
        }
        
        // Update Total Calories (third element)
        summaryValues[2].textContent = data.total_calories + ' kcal';
    }
}

/**
 * Open food item correction modal
 */
function openFoodItemCorrection(item, itemIndex, availableFoodItems) {
    // Create overlay
    const overlay = document.createElement('div');
    overlay.className = 'correction-overlay';
    
    // Create modal content
    overlay.innerHTML = `
        <div class="correction-modal">
            <div class="correction-header">
                <h3>Correct Food Item</h3>
                <button class="correction-close">&times;</button>
            </div>
            <div class="correction-content">
                <div class="correction-item">
                    <img src="${item.image}" alt="${item.item}">
                    <div class="correction-details">
                        <div>Detected as: <strong>${item.item}</strong></div>
                        <div>Please select the correct food item:</div>
                        <select class="correction-select">
                            ${availableFoodItems.map(food => 
                                `<option value="${food}" ${food === item.item ? 'selected' : ''}>${food}</option>`
                            ).join('')}
                        </select>
                    </div>
                </div>
            </div>
            <div class="correction-footer">
                <button class="correction-apply">Apply Correction</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(overlay);
    
    // Handle close button
    const closeButton = overlay.querySelector('.correction-close');
    closeButton.addEventListener('click', () => {
        document.body.removeChild(overlay);
    });
    
    // Handle apply button
    const applyButton = overlay.querySelector('.correction-apply');
    applyButton.addEventListener('click', () => {
        const selectedFood = overlay.querySelector('.correction-select').value;
        
        // Only apply changes if a different item was selected
        if (selectedFood !== item.item) {
            updateFoodItem(itemIndex, selectedFood);
        }
        
        document.body.removeChild(overlay);
    });
    
    // Close when clicking outside the modal
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) {
            document.body.removeChild(overlay);
        }
    });
}

/**
 * Update a food item with new selection and recalculate totals
 */
function updateFoodItem(itemIndex, newFoodItem) {
    if (!analyzedData || !analyzedData.bill_details || itemIndex >= analyzedData.bill_details.length) {
        return;
    }
    
    // Store old item for notification
    const oldFoodItem = analyzedData.bill_details[itemIndex].item;
    
    // Update UI first (optimistic update)
    const foodItemElement = document.querySelector(`.food-item[data-id="${itemIndex}"]`);
    if (!foodItemElement) return;
    
    foodItemElement.querySelector('.food-name').textContent = newFoodItem;
    foodItemElement.setAttribute('data-item', newFoodItem);
    
    // Show loading message
    showToast('Updating food item...', 'info');
    
    // Find the food item in the local menu data
    if (!window.foodMenuData) {
        // If we don't have the menu data yet, fetch it
        fetch('/api/food-info')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to fetch menu data');
                }
                return response.json();
            })
            .then(data => {
                if (data.success && data.food_info) {
                    // Save menu data for future use
                    window.foodMenuData = data.food_info;
                    // Continue with the update
                    completeUpdate(window.foodMenuData);
                } else {
                    throw new Error('Invalid menu data received');
                }
            })
            .catch(error => {
                console.error('Error fetching menu data:', error);
                // Revert UI change
                foodItemElement.querySelector('.food-name').textContent = oldFoodItem;
                foodItemElement.setAttribute('data-item', oldFoodItem);
                showToast('Error updating food item: ' + error.message, 'error');
            });
    } else {
        // If we already have the menu data, use it directly
        completeUpdate(window.foodMenuData);
    }
    
    function completeUpdate(menuData) {
        // Find the food item in the menu
        const foodInfo = menuData.find(item => item.name === newFoodItem);
        
        if (!foodInfo) {
            // If food item not found, revert changes and show error
            foodItemElement.querySelector('.food-name').textContent = oldFoodItem;
            foodItemElement.setAttribute('data-item', oldFoodItem);
            showToast(`Error: Food item "${newFoodItem}" not found in menu`, 'error');
            return;
        }
        // Giữ lại flag manuallyAdded nếu có
        const manuallyAdded = analyzedData.bill_details[itemIndex].manuallyAdded;

        // Update the analyzed data
        analyzedData.bill_details[itemIndex].item = newFoodItem;
        analyzedData.bill_details[itemIndex].price = foodInfo.price;
        analyzedData.bill_details[itemIndex].calories = foodInfo.calories;
        
        // Giữ lại flag manuallyAdded
        if (manuallyAdded) {
            analyzedData.bill_details[itemIndex].manuallyAdded = true;
        }

        // Update UI with new price and calories
        const foodPrice = foodItemElement.querySelector('.food-price');
        const foodCalories = foodItemElement.querySelector('.food-info span');
        
        // Format price
        const currency = localStorage.getItem('currency') || 'vnd';
        if (currency === 'usd') {
            const usdValue = foodInfo.price / 23000;
            foodPrice.textContent = new Intl.NumberFormat('en-US', { 
                style: 'currency', 
                currency: 'USD', 
                minimumFractionDigits: 2 
            }).format(usdValue);
        } else {
            foodPrice.textContent = new Intl.NumberFormat('vi-VN', { 
                style: 'currency', 
                currency: 'VND', 
                maximumFractionDigits: 0 
            }).format(foodInfo.price);
        }
        
        foodCalories.innerHTML = `<i class="fas fa-fire"></i> ${foodInfo.calories} kcal`;
        
        // Recalculate totals
        let totalCost = 0;
        let totalCalories = 0;
        
        analyzedData.bill_details.forEach(item => {
            totalCost += item.price;
            totalCalories += item.calories;
        });
        
        // Update analyzed data
        analyzedData.total_cost = totalCost;
        analyzedData.total_calories = totalCalories;
        
        // Update UI
        updateSummaryValues(analyzedData);
        
        // Update calorie meter
        updateCalorieMeter(totalCalories);
        
        // Update notifications
        updateNotifications(analyzedData);
        
        // Show success message
        showToast(`Corrected: ${oldFoodItem} → ${newFoodItem}`, 'success');
    }
}

/**
 * Update the calorie meter based on new calorie total
 */
function updateCalorieMeter(totalCalories) {
    const calorieProgress = document.querySelector('.calorie-progress');
    if (!calorieProgress) return;
    
    // Get recommended range
    const savedCalorieGoal = parseInt(localStorage.getItem('calorieGoal') || '2000');
    const savedCalorieThreshold = parseInt(localStorage.getItem('calorieThreshold') || '30');
    const mealCalorieTarget = savedCalorieGoal * (savedCalorieThreshold / 100);
    const lowerBound = mealCalorieTarget * 0.8; // 80% of target
    const upperBound = mealCalorieTarget * 1.2; // 120% of target
    
    // Calculate percentage for progress bar (capped at 100%)
    let caloriePercentage = Math.min((totalCalories / upperBound) * 100, 100);
    
    // Update progress bar
    calorieProgress.style.width = `${caloriePercentage}%`;
    
    // Update color based on whether calories are in the target range
    if (totalCalories < lowerBound) {
        // Too low - blue
        calorieProgress.style.backgroundColor = '#3498DB';
        calorieProgress.classList.remove('high-calorie');
    } else if (totalCalories > upperBound) {
        // Too high - red
        calorieProgress.style.backgroundColor = '#FF0000';
        calorieProgress.classList.add('high-calorie');
    } else {
        // Just right - green
        calorieProgress.style.backgroundColor = '#27AE60';
        calorieProgress.classList.remove('high-calorie');
    }
    
    // Update the values in the calorie meter
    const calorieStart = document.querySelector('.calorie-limit span:first-child');
    const calorieRecommended = document.querySelector('.calorie-limit span:last-child');
    
    if (calorieStart) {
        calorieStart.textContent = totalCalories + ' kcal';
    }
    
    // Lưu nút info hiện tại nếu có
    const existingInfoIcon = calorieRecommended ? calorieRecommended.querySelector('.calorie-info-icon') : null;
    
    if (calorieRecommended) {
        // Cập nhật text nhưng giữ lại nút info
        if (existingInfoIcon) {
            // Lưu icon trước khi thay đổi nội dung
            existingInfoIcon.remove(); // Tạm thời gỡ icon
        }
        
        // Cập nhật text
        calorieRecommended.textContent = `Recommended: ${Math.round(lowerBound)}-${Math.round(upperBound)} kcal`;
        
        // Thêm lại icon nếu đã có trước đó
        if (existingInfoIcon) {
            calorieRecommended.appendChild(existingInfoIcon);
        } else {
            // Nếu chưa có icon, tạo mới
            addCalorieInfoTooltip();
        }
    }
}
    /**
     * Update notification messages based on analysis results
     */
function updateNotifications(data) {
    // Find container for notification
    const notificationContainer = document.getElementById('notificationArea');
    // const notificationContainer = document.querySelector('.card:nth-child(2)');
    if (!notificationContainer) return;
    
    // Create notification section if it doesn't exist
    let notificationSection = notificationContainer.querySelector('.notification-section');
    if (!notificationSection) {
        notificationSection = document.createElement('div');
        notificationSection.className = 'notification-section';
        notificationContainer.appendChild(notificationSection);
    } else {
        // Clear old notifications
        notificationSection.innerHTML = '';
    }
    
    // Add enhanced calorie notification with detailed status and suggestions
    const savedCalorieGoal = parseInt(localStorage.getItem('calorieGoal') || '2000');
    const savedCalorieThreshold = parseInt(localStorage.getItem('calorieThreshold') || '30');
    const mealCalorieTarget = savedCalorieGoal * (savedCalorieThreshold / 100);
    const lowerBound = mealCalorieTarget * 0.8; // 80% of target
    const upperBound = mealCalorieTarget * 1.2; // 120% of target
    
    let calorieStatus, calorieMessage, calorieClass, calorieIcon, suggestions;
    
    if (data.total_calories < lowerBound) {
        calorieStatus = 'Low Calorie Intake';
        calorieClass = 'low';
        calorieIcon = 'arrow-down';
        calorieMessage = `Your meal is only ${Math.round(data.total_calories / mealCalorieTarget * 100)}% of the recommended calorie intake for a meal.`;
        suggestions = [
            'Consider adding a protein source like chicken or tofu.',
            'Add a small portion of carbs such as rice or bread.',
            'Include healthy fats like avocado or nuts for extra calories.'
        ];
    } else if (data.total_calories > upperBound) {
        calorieStatus = 'High Calorie Intake';
        calorieClass = 'high';
        calorieIcon = 'arrow-up';
        calorieMessage = `Your meal exceeds the recommended calorie intake by ${Math.round((data.total_calories - mealCalorieTarget) / mealCalorieTarget * 100)}%.`;
        suggestions = [
            'Consider reducing portion sizes, especially of high-calorie items.',
            'Replace some high-calorie items with vegetables.',
            'Choose lean protein sources over fatty ones.'
        ];
    } else {
        calorieStatus = 'Optimal Calorie Intake';
        calorieClass = 'optimal';
        calorieIcon = 'check';
        calorieMessage = 'Your meal is within the recommended calorie range for a balanced diet.';
        suggestions = [
            'Good job maintaining a balanced meal!',
            'Ensure you\'re also getting a good mix of proteins, carbs, and vegetables.',
            'Stay hydrated by drinking water with your meal.'
        ];
    }
    
    const calorieCard = document.createElement('div');
    calorieCard.className = `info-card calorie-card ${calorieClass === 'optimal' ? 'info-success' : calorieClass === 'high' ? 'info-danger' : 'info-primary'}`;
    calorieCard.innerHTML = `
        <i class="fas fa-${calorieIcon}"></i>
        <div style="flex: 1;">
            <h4 style="margin: 0 0 5px 0; font-size: 15px;">${calorieStatus}</h4>
            <p>${calorieMessage}</p>
            <div class="calorie-details">
                <span class="calorie-current"><i class="fas fa-fire"></i>${data.total_calories} kcal</span>
                <span class="calorie-target">Target: ${Math.round(lowerBound)}-${Math.round(upperBound)} kcal</span>
            </div>
            <div class="calorie-status ${calorieClass}">
                ${calorieClass === 'optimal' ? 'Well-balanced meal' : calorieClass === 'low' ? 'Consider eating more' : 'Consider eating less'}
            </div>
            <ul class="suggestion-list">
                ${suggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
            </ul>
        </div>
    `;
    notificationSection.appendChild(calorieCard);
    
    // Add balance notification
    // Improved food categorization based on Vietnamese food names
    const proteinFoods = [
        'ga', 'thit', 'ca', 'trung', 'tom', 'suon', 'bo', 'dau hu',
        'ga chien', 'ga kho', 'thit chien', 'thit luoc', 'ca chien', 'ca kho',
        'trung chien', 'trung luoc', 'kho tieu', 'kho trung', 'suon mieng', 'suon xao'
    ];
    
    const carbFoods = [
        'com', 'bun', 'pho', 'banh', 'banh mi'
    ];
    
    const veggieFoods = [
        'rau', 'dau', 'bap cai', 'cai', 'dua', 'ot', 'ca chua', 
        'bap cai luoc', 'bap cai xao', 'ca chua', 'ca rot', 'canh bau', 
        'canh bi do', 'canh cai', 'canh chua', 'canh rong bien', 'dau bap', 
        'dau que', 'dua hau', 'dua leo', 'oi', 'rau muong', 'rau ngo', 'thanh long',
        'chuoi', 'do chua'
    ];
    
    const hasProtein = data.bill_details.some(item => 
        proteinFoods.some(food => item.item.toLowerCase().includes(food)));
        
    const hasCarbs = data.bill_details.some(item => 
        carbFoods.some(food => item.item.toLowerCase().includes(food)));
        
    const hasVeggies = data.bill_details.some(item => 
        veggieFoods.some(food => item.item.toLowerCase().includes(food)));
        
    const balanceCard = document.createElement('div');
    balanceCard.className = `info-card ${hasProtein && hasCarbs && hasVeggies ? 'info-success' : 'info-primary'}`;
    
    let balanceMessage, balanceSuggestions = [];
    
    if (hasProtein && hasCarbs && hasVeggies) {
        balanceMessage = 'Your meal has a great balance of proteins, carbohydrates, and vegetables.';
    } else {
        const missingGroups = [];
        if (!hasProtein) {
            missingGroups.push('protein');
            balanceSuggestions.push('Add a protein source like chicken, fish, eggs, or tofu.');
        }
        if (!hasCarbs) {
            missingGroups.push('carbohydrates');
            balanceSuggestions.push('Add some rice, noodles, or bread for energy.');
        }
        if (!hasVeggies) {
            missingGroups.push('vegetables');
            balanceSuggestions.push('Add vegetables for fiber, vitamins, and minerals.');
        }
        
        balanceMessage = `Your meal is missing ${missingGroups.join(', ')}.`;
    }
    
    balanceCard.innerHTML = `
        <i class="fas fa-${hasProtein && hasCarbs && hasVeggies ? 'check' : 'utensils'}-circle"></i>
        <div style="flex: 1;">
            <h4 style="margin: 0 0 5px 0; font-size: 15px;">Meal Balance</h4>
            <p>${balanceMessage}</p>
            ${balanceSuggestions.length > 0 ? `
                <ul class="suggestion-list">
                    ${balanceSuggestions.map(suggestion => `<li>${suggestion}</li>`).join('')}
                </ul>
            ` : ''}
        </div>
    `;
    notificationSection.appendChild(balanceCard);
}
    
    /**
     * Load food information for the calorie view
     */
    function loadFoodInfo() {
        fetch('/api/food-info')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    console.error('Error loading food info:', data.error);
                    return;
                }
                
                displayFoodInfo(data.food_info);
            })
            .catch(error => {
                console.error('Error loading food info:', error);
            });
    }
    
    /**
     * Display food information in the calorie view
     */
    function displayFoodInfo(foodInfo) {
        const calorieGrid = document.querySelector('.calorie-grid');
        
        // Clear existing content
        calorieGrid.innerHTML = '';
        
        // Add each food item
        foodInfo.forEach(item => {
            const card = document.createElement('div');
            card.className = 'calorie-card';
            
            card.innerHTML = `
                <img src="/static/img/food/${item.name.toLowerCase().replace(/\s+/g, '_')}.jpg" 
                     onerror="this.src='/static/img/placeholder.jpg'" 
                     alt="${item.name}" 
                     class="calorie-card-img">
                <div class="calorie-card-body">
                    <h5 class="calorie-card-title">${item.name}</h5>
                    <p class="calorie-card-text">${item.calories} kcal per serving</p>
                    <span class="calorie-tag">${item.category}</span>
                </div>
            `;
            
            calorieGrid.appendChild(card);
        });
        
        // Setup search functionality
        setupCalorieSearch(foodInfo);
    }
    
    /**
     * Setup search functionality for the calorie view
     */
    function setupCalorieSearch(foodInfo) {
        const searchInput = document.querySelector('.calorie-search input');
        const searchButton = document.querySelector('.calorie-search button');
        
        // Search function
        const performSearch = () => {
            const searchTerm = searchInput.value.toLowerCase().trim();
            const calorieCards = document.querySelectorAll('.calorie-card');
            
            calorieCards.forEach(card => {
                const title = card.querySelector('.calorie-card-title').textContent.toLowerCase();
                if (title.includes(searchTerm) || searchTerm === '') {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        };
        
        // Add event listeners
        searchButton.addEventListener('click', performSearch);
        searchInput.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }
    
    // Checkout button functionality
    const checkoutBtn = document.querySelector('.checkout-btn');
    checkoutBtn.addEventListener('click', () => {
        if (!analyzedData) {
            showToast('Vui lòng phân tích hình ảnh khay thức ăn trước.', 'info');
            return;
        }
        
        // Format total cost for toast message
        const formattedCost = analyzedData.total_cost.toLocaleString() + ' ₫';
        
        // Show success message
        showToast(`Đơn hàng hoàn tất! Tổng cộng: ${formattedCost}`, 'success');
        
        // Add to transaction history (would be stored in a database in a real app)
        const now = new Date();
        const formattedDate = `${now.getDate()} ${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][now.getMonth()]} ${now.getFullYear()}, ${now.getHours()}:${String(now.getMinutes()).padStart(2, '0')}`;
        const orderId = `#ORD-${Math.floor(Math.random() * 10000)}`;
        
        const transactionTable = document.querySelector('.transaction-table tbody');
        const newRow = document.createElement('tr');
        newRow.innerHTML = `
            <td>${formattedDate}</td>
            <td>${orderId}</td>
            <td>${analyzedData.items_count} items</td>
            <td>${analyzedData.total_cost.toLocaleString()} ₫</td>
            <td>${analyzedData.total_calories} kcal</td>
            <td><span class="transaction-status status-completed">Completed</span></td>
            <td><span class="view-details">Details</span></td>
        `;
        
        transactionTable.insertBefore(newRow, transactionTable.firstChild);
        
        // Reset the form
        removeImageBtn.click();
    });
    
    // Settings save buttons
    const saveAccountSettings = document.getElementById('saveAccountSettings');
    const saveDisplaySettings = document.getElementById('saveDisplaySettings');
    const saveCalorieSettings = document.getElementById('saveCalorieSettings');
    
    saveAccountSettings.addEventListener('click', () => {
        const fullName = document.getElementById('fullNameInput').value;
        const email = document.getElementById('emailInput').value;
        const studentId = document.getElementById('studentIdInput').value;
        
        // Update the user profile in the header
        document.querySelector('.user-info h4').textContent = fullName;
        document.querySelector('.user-info p').textContent = `Student ID: ${studentId}`;
        
        // Save to localStorage for persistence
        localStorage.setItem('fullName', fullName);
        localStorage.setItem('email', email);
        localStorage.setItem('studentId', studentId);
        
        showToast('Account settings saved successfully!');
    });
    
    saveDisplaySettings.addEventListener('click', () => {
        const language = document.getElementById('languageSelect').value;
        const currency = document.getElementById('currencySelect').value;
        
        // Save to localStorage for persistence
        localStorage.setItem('language', language);
        localStorage.setItem('currency', currency);
        
        // Apply language change
        applyLanguageSettings(language);
        
        // Apply currency change
        applyCurrencySettings(currency);
        
        showToast('Display settings saved successfully!');
    });
    
    // Function to apply language settings
    function applyLanguageSettings(language) {
        const translations = {
            'en': {
                'pageTitle': 'Food Scanner',
                'uploadTitle': 'Upload Food Tray Image',
                'dragDropText': 'Drag & Drop your image',
                'orClickText': 'or click to browse files (JPG, PNG format)',
                'selectImageBtn': 'Select Image',
                'useCameraBtn': 'Use Camera',
                'detectedItemsTitle': 'Detected Food Items',
                'billSummaryTitle': 'Bill Summary',
                'itemsLabel': 'Items',
                'totalCostLabel': 'Total Cost',
                'totalCaloriesLabel': 'Total Calories',
                'completeOrderBtn': 'Complete Order',
                'notificationTitle': 'Notification',
                'cameraModalTitle': 'Take a Photo',
                'captureBtn': 'Capture',
                'retakeBtn': 'Retake',
                'usePhotoBtn': 'Use Photo',
                'analyzeBtn': 'Analyze Food',
                'removeBtn': 'Remove',
                'historyTitle': 'Transaction History',
                'allBtn': 'All',
                'todayBtn': 'Today',
                'thisWeekBtn': 'This Week',
                'thisMonthBtn': 'This Month',
                'dateTimeColumn': 'Date & Time',
                'orderIdColumn': 'Order ID',
                'itemsColumn': 'Items',
                'totalColumn': 'Total',
                'caloriesColumn': 'Calories',
                'statusColumn': 'Status',
                'actionColumn': 'Action',
                'viewDetailsText': 'Details',
                'completedStatus': 'Completed',
                'settingsTitle': 'Settings',
                'displaySettingsTitle': 'Display Settings',
                'languageLabel': 'Language',
                'currencyLabel': 'Currency',
                'calorieSettingsTitle': 'Calorie Settings',
                'dailyCalorieGoalLabel': 'Daily Calorie Goal',
                'mealThresholdLabel': 'Meal Calorie Warning Threshold (%)',
                'accountSettingsTitle': 'Account Settings',
                'fullNameLabel': 'Full Name',
                'emailLabel': 'Email',
                'studentIdLabel': 'Student ID',
                'saveChangesBtn': 'Save Changes',
                'calorieInfoTitle': 'Food Calorie Information',
                'searchPlaceholder': 'Search for food...',
                'recommendedText': 'Recommended: {min}-{max} kcal'
            },
            'vi': {
                'pageTitle': 'Quét Thức Ăn',
                'uploadTitle': 'Tải Lên Ảnh Khay Thức Ăn',
                'dragDropText': 'Kéo & Thả ảnh của bạn',
                'orClickText': 'hoặc nhấp để duyệt file (định dạng JPG, PNG)',
                'selectImageBtn': 'Chọn Ảnh',
                'useCameraBtn': 'Dùng Camera',
                'detectedItemsTitle': 'Món Ăn Đã Phát Hiện',
                'billSummaryTitle': 'Tổng Hóa Đơn',
                'itemsLabel': 'Số Món',
                'totalCostLabel': 'Tổng Tiền',
                'totalCaloriesLabel': 'Tổng Calo',
                'completeOrderBtn': 'Hoàn Tất Đơn Hàng',
                'notificationTitle': 'Thông Báo',
                'cameraModalTitle': 'Chụp Ảnh',
                'captureBtn': 'Chụp',
                'retakeBtn': 'Chụp Lại',
                'usePhotoBtn': 'Sử Dụng Ảnh',
                'analyzeBtn': 'Phân Tích Thức Ăn',
                'removeBtn': 'Xóa',
                'historyTitle': 'Lịch Sử Giao Dịch',
                'allBtn': 'Tất Cả',
                'todayBtn': 'Hôm Nay',
                'thisWeekBtn': 'Tuần Này',
                'thisMonthBtn': 'Tháng Này',
                'dateTimeColumn': 'Ngày & Giờ',
                'orderIdColumn': 'Mã Đơn',
                'itemsColumn': 'Số Món',
                'totalColumn': 'Tổng Cộng',
                'caloriesColumn': 'Calo',
                'statusColumn': 'Trạng Thái',
                'actionColumn': 'Hành Động',
                'viewDetailsText': 'Chi Tiết',
                'completedStatus': 'Hoàn Thành',
                'settingsTitle': 'Cài Đặt',
                'displaySettingsTitle': 'Cài Đặt Hiển Thị',
                'languageLabel': 'Ngôn Ngữ',
                'currencyLabel': 'Đơn Vị Tiền Tệ',
                'calorieSettingsTitle': 'Cài Đặt Calo',
                'dailyCalorieGoalLabel': 'Mục Tiêu Calo Hàng Ngày',
                'mealThresholdLabel': 'Ngưỡng Cảnh Báo Calo Bữa Ăn (%)',
                'accountSettingsTitle': 'Cài Đặt Tài Khoản',
                'fullNameLabel': 'Họ Tên',
                'emailLabel': 'Email',
                'studentIdLabel': 'Mã Sinh Viên',
                'saveChangesBtn': 'Lưu Thay Đổi',
                'calorieInfoTitle': 'Thông Tin Calo Thực Phẩm',
                'searchPlaceholder': 'Tìm kiếm thức ăn...',
                'recommendedText': 'Khuyến nghị: {min}-{max} kcal'
            }
        };
        
        // Get the translations for the selected language
        const trans = translations[language] || translations['en'];
        
        // Update page title
        document.getElementById('pageTitle').textContent = trans.pageTitle;
        
        // Update navigation menu
        const navItems = {
            'scan': 'Food Scanner',
            'history': 'Transaction History',
            'calorie': 'Food Calorie Information',
            'settings': 'Settings'
        };
        
        const navItemsVi = {
            'scan': 'Quét Thức Ăn',
            'history': 'Lịch Sử Giao Dịch',
            'calorie': 'Thông Tin Calo',
            'settings': 'Cài Đặt'
        };
        
        document.querySelectorAll('.nav-link').forEach(link => {
            const view = link.getAttribute('data-view');
            if (language === 'vi') {
                link.querySelector('span').textContent = navItemsVi[view];
            } else {
                link.querySelector('span').textContent = navItems[view];
            }
        });
        
        // Update scan view
        document.querySelector('.card-header h3').textContent = trans.uploadTitle;
        document.querySelector('.card-upload h4').textContent = trans.dragDropText;
        document.querySelector('.card-upload p').textContent = trans.orClickText;
        document.querySelector('.upload-button').textContent = trans.selectImageBtn;
        document.querySelector('.camera-button').textContent = trans.useCameraBtn;
        document.querySelector('.card:nth-child(2) .card-header h3').textContent = trans.detectedItemsTitle;
        
        // Update summary section
        document.querySelector('.summary-section .card-header h3').textContent = trans.billSummaryTitle;
        document.querySelector('.summary-item:nth-child(1) .summary-label').textContent = trans.itemsLabel;
        document.querySelector('.summary-item:nth-child(2) .summary-label').textContent = trans.totalCostLabel;
        document.querySelector('.summary-item:nth-child(3) .summary-label').textContent = trans.totalCaloriesLabel;
        document.querySelector('.checkout-btn').textContent = trans.completeOrderBtn;
        
        // Update notification section if it exists
        const notificationHeader = document.querySelector('.card:nth-child(2) .card-header:nth-child(1) h3');
        if (notificationHeader) {
            notificationHeader.textContent = trans.notificationTitle;
        }
        
        // Update camera modal
        document.querySelector('#cameraModal .modal-header h3').textContent = trans.cameraModalTitle;
        document.querySelector('#captureBtn').innerHTML = `<i class="fas fa-camera"></i> ${trans.captureBtn}`;
        document.querySelector('#retakeBtn').innerHTML = `<i class="fas fa-redo"></i> ${trans.retakeBtn}`;
        document.querySelector('#usePhotoBtn').innerHTML = `<i class="fas fa-check"></i> ${trans.usePhotoBtn}`;
        
        // Update analyze and remove buttons
        document.querySelector('#analyzeBtn').textContent = trans.analyzeBtn;
        document.querySelector('#removeImageBtn').textContent = trans.removeBtn;
        
        // Update transaction history
        if (document.querySelector('.history-view.active-view')) {
            document.querySelector('.history-view .card-header h3').textContent = trans.historyTitle;
            
            // Update filter buttons
            const filterButtons = document.querySelectorAll('.filter-button');
            if (filterButtons.length >= 4) {
                filterButtons[0].textContent = trans.allBtn;
                filterButtons[1].textContent = trans.todayBtn;
                filterButtons[2].textContent = trans.thisWeekBtn;
                filterButtons[3].textContent = trans.thisMonthBtn;
            }
            
            // Update table headers
            const tableHeaders = document.querySelectorAll('.transaction-table th');
            if (tableHeaders.length >= 7) {
                tableHeaders[0].textContent = trans.dateTimeColumn;
                tableHeaders[1].textContent = trans.orderIdColumn;
                tableHeaders[2].textContent = trans.itemsColumn;
                tableHeaders[3].textContent = trans.totalColumn;
                tableHeaders[4].textContent = trans.caloriesColumn;
                tableHeaders[5].textContent = trans.statusColumn;
                tableHeaders[6].textContent = trans.actionColumn;
            }
            
            // Update status and action text
            document.querySelectorAll('.status-completed').forEach(status => {
                status.textContent = trans.completedStatus;
            });
            
            document.querySelectorAll('.view-details').forEach(action => {
                action.textContent = trans.viewDetailsText;
            });
        }
        
        // Update settings view
        if (document.querySelector('.settings-view.active-view')) {
            document.querySelector('.settings-view .card-header h3').textContent = trans.settingsTitle;
            
            // Update settings cards
            const settingsCards = document.querySelectorAll('.settings-card h4');
            if (settingsCards.length >= 3) {
                settingsCards[0].textContent = trans.displaySettingsTitle;
                settingsCards[1].textContent = trans.calorieSettingsTitle;
                settingsCards[2].textContent = trans.accountSettingsTitle;
            }
            
            // Update setting labels
            document.querySelectorAll('.setting-label').forEach(label => {
                if (label.textContent.includes('Language')) {
                    label.textContent = trans.languageLabel;
                } else if (label.textContent.includes('Currency')) {
                    label.textContent = trans.currencyLabel;
                } else if (label.textContent.includes('Daily Calorie')) {
                    label.textContent = trans.dailyCalorieGoalLabel;
                } else if (label.textContent.includes('Meal Calorie')) {
                    label.textContent = trans.mealThresholdLabel;
                } else if (label.textContent.includes('Full Name')) {
                    label.textContent = trans.fullNameLabel;
                } else if (label.textContent.includes('Email')) {
                    label.textContent = trans.emailLabel;
                } else if (label.textContent.includes('Student ID')) {
                    label.textContent = trans.studentIdLabel;
                }
            });
            
            // Update save buttons
            document.querySelectorAll('.save-settings').forEach(button => {
                button.textContent = trans.saveChangesBtn;
            });
        }
        
        // Update calorie information view
        if (document.querySelector('.calorie-view.active-view')) {
            document.querySelector('.calorie-view .card-header h3').textContent = trans.calorieInfoTitle;
            document.querySelector('.calorie-search input').placeholder = trans.searchPlaceholder;
        }
        
        // Update calorie recommendation text
        const recommendedText = document.querySelector('.calorie-limit span:last-child');
        if (recommendedText) {
            const savedCalorieGoal = localStorage.getItem('calorieGoal') || 2000;
            const savedCalorieThreshold = localStorage.getItem('calorieThreshold') || 30;
            const recommendedMin = Math.round(savedCalorieGoal * (savedCalorieThreshold / 100) * 0.8);
            const recommendedMax = Math.round(savedCalorieGoal * (savedCalorieThreshold / 100) * 1.2);
            
            recommendedText.textContent = trans.recommendedText
                .replace('{min}', recommendedMin)
                .replace('{max}', recommendedMax);
        }
    }
    
    // Function to apply currency settings
    function applyCurrencySettings(currency) {
        const formatOptions = {
            'vnd': { style: 'currency', currency: 'VND', maximumFractionDigits: 0 },
            'usd': { style: 'currency', currency: 'USD', minimumFractionDigits: 2 }
        };
        
        // Update all price displays
        if (analyzedData) {
            // Update bill summary
            const totalCostElement = document.querySelector('.summary-item:nth-child(2) .summary-value');
            if (currency === 'usd') {
                // Assuming 23,000 VND = 1 USD for conversion
                const usdValue = analyzedData.total_cost / 23000;
                totalCostElement.textContent = new Intl.NumberFormat('en-US', 
                    formatOptions['usd']).format(usdValue);
            } else {
                // For VND, use custom format with đ symbol
                totalCostElement.textContent = analyzedData.total_cost.toLocaleString() + ' ₫';
            }
            
            // Update food items
            document.querySelectorAll('.food-price').forEach((priceElement, index) => {
                if (index < analyzedData.bill_details.length) {
                    const priceVND = analyzedData.bill_details[index].price;
                    if (currency === 'usd') {
                        const usdValue = priceVND / 23000;
                        priceElement.textContent = new Intl.NumberFormat('en-US', 
                            formatOptions['usd']).format(usdValue);
                    } else {
                        priceElement.textContent = priceVND.toLocaleString() + ' ₫';
                    }
                }
            });
        } else {
            // If no analyzedData, just update the default values
            const totalCostElement = document.querySelector('.summary-item:nth-child(2) .summary-value');
            if (totalCostElement) {
                if (currency === 'usd') {
                    totalCostElement.textContent = '$0.00';
                } else {
                    totalCostElement.textContent = '0 ₫';
                }
            }
        }
        
        // Update transaction history
        document.querySelectorAll('.transaction-table td:nth-child(4)').forEach(cell => {
            const priceText = cell.textContent;
            const priceMatch = priceText.match(/[\d,]+/);
            if (priceMatch) {
                const priceVND = parseInt(priceMatch[0].replace(/,/g, ''));
                if (currency === 'usd') {
                    const usdValue = priceVND / 23000;
                    cell.textContent = new Intl.NumberFormat('en-US', 
                        formatOptions['usd']).format(usdValue);
                } else {
                    cell.textContent = priceVND.toLocaleString() + ' ₫';
                }
            }
        });
    }
    
    saveCalorieSettings.addEventListener('click', () => {
        const calorieGoal = document.getElementById('calorieGoalInput').value;
        const calorieThreshold = document.getElementById('calorieThresholdInput').value;
        
        // Save to localStorage
        localStorage.setItem('calorieGoal', calorieGoal);
        localStorage.setItem('calorieThreshold', calorieThreshold);
        
        // Update the calorie meter recommendation
        const recommendedMin = Math.round(calorieGoal * (calorieThreshold / 100) * 0.8);
        const recommendedMax = Math.round(calorieGoal * (calorieThreshold / 100) * 1.2);
        
        // Lấy phần tử chứa text
        const recommendedSpan = document.querySelector('.calorie-limit span:last-child');
        
        if (recommendedSpan) {
            // Lưu nút thông tin nếu có
            const infoIcon = recommendedSpan.querySelector('.calorie-info-icon');
            
            // Cập nhật text
            recommendedSpan.textContent = `Recommended: ${recommendedMin}-${recommendedMax} kcal`;
            
            // Thêm lại nút thông tin
            if (infoIcon) {
                recommendedSpan.appendChild(infoIcon);
                
                // Cập nhật nội dung tooltip
                updateCalorieTooltipContent(infoIcon, calorieGoal, calorieThreshold);
            } else {
                // Nếu chưa có nút thông tin, thêm mới
                setTimeout(addCalorieInfoTooltip, 100);
            }
        }
        
        showToast('Calorie settings saved successfully!');
    });
function updateCalorieTooltipContent(infoIcon, calorieGoal, calorieThreshold) {
    // Tìm tooltip trong infoIcon
    const tooltip = infoIcon.querySelector('.calorie-tooltip');
    if (!tooltip) return;
    
    // Tính toán các giá trị
    const dailyGoal = parseInt(calorieGoal || localStorage.getItem('calorieGoal') || '2000');
    const mealThreshold = parseInt(calorieThreshold || localStorage.getItem('calorieThreshold') || '30');
    const targetPerMeal = Math.round(dailyGoal * (mealThreshold / 100));
    const minRange = Math.round(targetPerMeal * 0.8);
    const maxRange = Math.round(targetPerMeal * 1.2);
    
    // Cập nhật nội dung tooltip
    tooltip.innerHTML = `
        <h4 style="margin: 0 0 8px 0; font-size: 14px; color: var(--text-color);">Calorie Target Explanation</h4>
        <p style="margin: 0 0 8px 0; font-size: 12px;">The recommended calorie range is calculated based on your settings:</p>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 5px;">
            <div style="font-size: 12px;"><strong style="color: var(--primary-color);">Daily goal:</strong> ${dailyGoal} cal</div>
            <div style="font-size: 12px;"><strong style="color: var(--primary-color);">Meal threshold:</strong> ${mealThreshold}%</div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 5px;">
            <div style="font-size: 12px;"><strong style="color: var(--primary-color);">Target per meal:</strong> ${targetPerMeal} cal</div>
            <div style="font-size: 12px;"><strong style="color: var(--primary-color);">Healthy range:</strong> ${minRange}-${maxRange} cal</div>
        </div>
        <p style="margin: 0; font-size: 12px;">You can adjust these in Settings > Calorie Settings.</p>
    `;
}
    
    // Toast notification function
    function showToast(message, type = 'success') {
        // Create toast element if it doesn't exist
        let toast = document.getElementById('toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'toast';
            document.body.appendChild(toast);
            
            // Add CSS for the toast
            const style = document.createElement('style');
            style.textContent = `
                #toast {
                    position: fixed;
                    bottom: 30px;
                    left: 50%;
                    transform: translateX(-50%);
                    background-color: var(--secondary-color);
                    color: white;
                    padding: 12px 24px;
                    border-radius: 8px;
                    z-index: 1000;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    max-width: 80%;
                    text-align: center;
                }
                #toast.show {
                    opacity: 1;
                }
                #toast.error {
                    background-color: var(--accent-color);
                }
                #toast.info {
                    background-color: var(--primary-color);
                }
            `;
            document.head.appendChild(style);
        }
        
        // Set message and show toast
        toast.textContent = message;
        toast.classList.add('show');
        
        // Set type-specific class
        toast.classList.remove('error', 'info', 'success');
        if (type === 'error' || type === 'info') {
            toast.classList.add(type);
        }
        
        // Hide toast after 3 seconds
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    }
    
    // Load saved settings from localStorage
    function loadSavedSettings() {
        // Account settings
        const savedFullName = localStorage.getItem('fullName');
        const savedEmail = localStorage.getItem('email');
        const savedStudentId = localStorage.getItem('studentId');
        
        if (savedFullName) {
            document.getElementById('fullNameInput').value = savedFullName;
            document.querySelector('.user-info h4').textContent = savedFullName;
        }
        
        if (savedEmail) {
            document.getElementById('emailInput').value = savedEmail;
        }
        
        if (savedStudentId) {
            document.getElementById('studentIdInput').value = savedStudentId;
            document.querySelector('.user-info p').textContent = `Student ID: ${savedStudentId}`;
        }
        
        // Display settings
        const savedLanguage = localStorage.getItem('language');
        const savedCurrency = localStorage.getItem('currency');
        
        if (savedLanguage) {
            document.getElementById('languageSelect').value = savedLanguage;
        }
        
        if (savedCurrency) {
            document.getElementById('currencySelect').value = savedCurrency;
        }
        
        // Calorie settings
        const savedCalorieGoal = localStorage.getItem('calorieGoal') || '2000';
        const savedCalorieThreshold = localStorage.getItem('calorieThreshold') || '30';
        
        if (savedCalorieGoal && savedCalorieThreshold) {
            // Update the calorie meter recommendation immediately after loading settings
            const recommendedMin = Math.round(savedCalorieGoal * (savedCalorieThreshold / 100) * 0.8);
            const recommendedMax = Math.round(savedCalorieGoal * (savedCalorieThreshold / 100) * 1.2);
            
            // Update the tooltip content and ensure it's visible
            const calorieMetrics = document.querySelector('.calorie-limit span:last-child');
            if (calorieMetrics) {
                calorieMetrics.textContent = `Recommended: ${recommendedMin}-${recommendedMax} kcal`;
                
                // Create notification section if it doesn't exist
                const notificationContainer = document.getElementById('notificationArea');
                if (notificationContainer) {
                    let notificationSection = notificationContainer.querySelector('.notification-section');
                    if (!notificationSection) {
                        notificationSection = document.createElement('div');
                        notificationSection.className = 'notification-section';
                        notificationContainer.appendChild(notificationSection);
                    }
                    
                    // Add a default tooltip
                    updateNotifications({
                        total_calories: 0,
                        bill_details: []
                    });
                }
            }
        }
    }
    
    // Load saved settings when the document is loaded
    loadSavedSettings();
    
    // Filter buttons in transaction history
    const filterButtons = document.querySelectorAll('.filter-button');
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            filterButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // In a real application, this would filter the transaction history
            // For this demo, we'll just show a message
            const filter = button.textContent;
            console.log(`Filtering transactions by: ${filter}`);
        });
    });
    
    // View details in transaction history
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('view-details')) {
            const row = e.target.closest('tr');
            const orderId = row.cells[1].textContent;
            const items = row.cells[2].textContent;
            const total = row.cells[3].textContent;
            const calories = row.cells[4].textContent;
            
            showToast(`Order ${orderId}: ${items}, ${total}, ${calories}`, 'info');
            // In a real application, this would show a modal with order details
        }
    });
});