# 🚗 License Plate Detection and Extraction

## 📌 Overview
This project implements a **license plate detection and extraction system** using OpenCV and NumPy. The code processes an image, applies **white balance correction**, detects **blue license plates**, and extracts them using **contour detection** and **perspective transformation**.

---

## ✨ Features
✅ **Gray World White Balance**: Adjusts the image's white balance to normalize color tones.
✅ **HSV Color Filtering**: Detects blue-colored license plates using HSV thresholds.
✅ **Morphological Processing**: Reduces noise using morphological operations.
✅ **Contour Detection**: Identifies potential license plates based on shape and size.
✅ **Perspective Transformation**: Warps the detected license plate to a frontal view.

---

## 🔧 Dependencies
Ensure you have the following Python packages installed:
```bash
pip install opencv-python matplotlib numpy
```

---

## 🚀 Usage
### 📥 1. Load an Image
Replace the image path in the following line with your own image:
```python
image = cv2.imread("path/to/license_plate_image.jpg")
```

### ⚙️ 2. Process the Image
The script performs the following steps automatically:
🔹 Reads the image
🔹 Applies white balance correction
🔹 Converts to HSV and filters for blue regions
🔹 Performs morphological operations
🔹 Detects contours and extracts the license plate

### 🖼️ 3. Display Results
After processing, the detected license plate is displayed alongside the original image.
```python
plt.show()
```

---

## 🛠️ Key Functions
### 🔹 `gray_world_white_balance(img)`
Adjusts the image's white balance using the **Gray World assumption**.

### 🔹 `sort_corners(points)`
Sorts the detected corner points in a specific order: **top-left, top-right, bottom-right, bottom-left**.

### 🔹 `transform(points, img)`
Performs a **perspective transformation** to rectify the license plate.

### 🔹 `averageBlueLight(wb_img)`
Calculates the **average blue channel brightness** to adapt to different lighting conditions.

---

## 📸 Output
📌 **Original Image with Detected License Plate**
📌 **Extracted License Plate Image**

---

## ⚠️ Notes
- 📍 This code works best with images where license plates have a distinct **blue color**.
- 📍 Adjust `lower_blue` and `upper_blue` thresholds for **different lighting conditions**.
- 📍 The algorithm assumes a **relatively clean background**; clutter may affect performance.

---

## 🔮 Future Improvements
🚀 **Support for multiple color plates**
🚀 **OCR integration for automatic license plate recognition**
🚀 **Improved robustness against varying lighting conditions**

---

## 👨‍💻 Author
This project was developed as part of a **license plate detection system** using OpenCV. 🏎️💨

