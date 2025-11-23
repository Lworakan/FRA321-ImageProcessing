import cv2
from matplotlib import pyplot as plt

# Load the dark image
image = cv2.imread("datasets/inside-the-box.jpg")

# Adjust brightness and contrast to reveal details
alpha = 1.9  # Increase contrast
beta = 130   # Increase brightness


# Enhance brightness and contrast
bright_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Convert to grayscale to see details better
gray_image = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)
# Convert the enhanced brightened image to grayscale first for CLAHE
gray_image_clahe = cv2.cvtColor(bright_image, cv2.COLOR_BGR2GRAY)

# Create a CLAHE object with a clip limit and tile grid size for local enhancement
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(13, 13))
clahe_image = clahe.apply(gray_image_clahe)

# Display the CLAHE enhanced image along with previous ones
plt.figure(figsize=(15, 5))

# Show original
plt.subplot(1, 4, 1)
plt.title("Original Dark Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Show enhanced bright image
plt.subplot(1, 4, 2)
plt.title("Enhanced Brightened Image")
plt.imshow(cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Show grayscale enhanced image
plt.subplot(1, 4, 3)
plt.title("Grayscale Enhanced Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

# Show CLAHE enhanced grayscale image
plt.subplot(1, 4, 4)
plt.title("CLAHE Enhanced Image")
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
