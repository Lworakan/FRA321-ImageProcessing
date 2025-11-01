# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


img_path = 'datasets/inside-the-box.jpg'
img = cv2.imread(img_path)

if img is None:
    print(f"Error: Could not read image from {img_path}")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"Original image shape: {img.shape}")
print(f"Original image dtype: {img.dtype}")
print(f"Original image min/max values: {img.min()}/{img.max()}")

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

def adjust_brightness_contrast(image, brightness=50, contrast=50):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            max_val = 255
        else:
            shadow = 0
            max_val = 255 + brightness
        alpha = (max_val - shadow) / 255
        gamma = shadow
        image = cv2.addWeighted(image, alpha, image, 0, gamma)

    if contrast != 0:
        alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        gamma = 127 * (1 - alpha)
        image = cv2.addWeighted(image, alpha, image, 0, gamma)

    return image

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_eq = cv2.equalizeHist(img_gray)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_gray)

def adjust_gamma(image, gamma=2.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

img_gamma = adjust_gamma(img, gamma=2.5)


img_enhanced = adjust_gamma(img, gamma=2.2)
img_enhanced = adjust_brightness_contrast(img_enhanced, brightness=300, contrast=100)

img_enhanced_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Image Enhancement Comparison', fontsize=16, fontweight='bold')

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original Image (Very Dark)', fontsize=12)
axes[0, 0].axis('off')

axes[0, 1].imshow(img_eq, cmap='gray')
axes[0, 1].set_title('Histogram Equalization', fontsize=12)
axes[0, 1].axis('off')

axes[0, 2].imshow(img_clahe, cmap='gray')
axes[0, 2].set_title('CLAHE', fontsize=12)
axes[0, 2].axis('off')

img_gamma_rgb = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2RGB)
axes[1, 0].imshow(img_gamma_rgb)
axes[1, 0].set_title('Gamma Correction (�=2.5)', fontsize=12)
axes[1, 0].axis('off')

axes[1, 1].imshow(img_enhanced_rgb)
axes[1, 1].set_title('Combined Enhancement (Best Result)', fontsize=12, fontweight='bold')
axes[1, 1].axis('off')

axes[1, 2].hist(img_gray.ravel(), 256, [0, 256], color='blue', alpha=0.5, label='Original')
axes[1, 2].hist(img_clahe.ravel(), 256, [0, 256], color='red', alpha=0.5, label='Enhanced')
axes[1, 2].set_title('Histogram Comparison', fontsize=12)
axes[1, 2].set_xlabel('Pixel Intensity')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('output/comparison.png', dpi=150, bbox_inches='tight')
print("\n Saved comparison image: output/comparison.png")

cv2.imwrite('output/enhanced_image.jpg', img_enhanced)
print(" Saved enhanced image: output/enhanced_image.jpg")

cv2.imwrite('output/histogram_equalization.jpg', img_eq)
cv2.imwrite('output/clahe.jpg', img_clahe)
cv2.imwrite('output/gamma_correction.jpg', img_gamma)

print("\n" + "="*60)
print("IMAGE ANALYSIS")
print("="*60)


gray_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_enhanced, 150, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    img_with_detection = img_enhanced.copy()
    cv2.rectangle(img_with_detection, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_with_detection, "Detected Object", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite('output/detected_object.jpg', img_with_detection)
    print(f"Detected object at position: x={x}, y={y}, width={w}, height={h}")
    print(f"Saved detection result: output/detected_object.jpg")

    roi = img_enhanced[y:y+h, x:x+w]
    cv2.imwrite('output/object_closeup.jpg', roi)
    print(f"Saved object closeup: output/object_closeup.jpg")

print("\n" + "="*60)
print("ENHANCEMENT METHODS USED:")
print("="*60)

plt.show()
