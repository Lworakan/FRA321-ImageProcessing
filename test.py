import cv2
import numpy as np
from realesrgan import RealESRGAN
from PIL import Image

# 1. Load original image
img = cv2.imread("datasets/inside-the-box.jpg")

# 2. Denoise first (keeps structure)
denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

# 3. Super-resolution upscale (2× or 4×)
model = RealESRGAN('cuda', scale=4)   # use 'cpu' if no GPU
model.load_weights('RealESRGAN_x4plus.pth')
sr_img = model.predict(Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)))
sr_img = cv2.cvtColor(np.array(sr_img), cv2.COLOR_RGB2BGR)

# 4. Edge-aware sharpening
gray = cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
lap = cv2.convertScaleAbs(lap)
sharpen_mask = cv2.merge([lap, lap, lap])
sharpened = cv2.addWeighted(sr_img, 1.25, sharpen_mask, 0.4, 0)

# 5. Local contrast boost (CLAHE)
lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
final_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

# Save + Show
cv2.imwrite("OWL_SemiAI_Final.png", final_img)
cv2.imshow("Final Result", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
