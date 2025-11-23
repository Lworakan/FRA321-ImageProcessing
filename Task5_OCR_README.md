# Task 5: OCR-Based 'O' Character Detection

This script uses Optical Character Recognition (OCR) to detect and locate the letter 'O' in text images, comparing results from both **Tesseract OCR** and **EasyOCR**.

## Features

- Detects 'O' characters using two different OCR engines
- Compares results between Tesseract and EasyOCR
- Provides confidence scores for each detection
- Generates comprehensive visualizations and statistics
- Creates a detailed PDF report with analysis
- Saves detection results to text file

## Requirements

### Python Libraries

```bash
# Core dependencies (required)
pip install opencv-python numpy matplotlib

# OCR libraries (at least one required)
pip install pytesseract  # For Tesseract OCR
pip install easyocr      # For EasyOCR
```

### System Dependencies

#### Tesseract OCR Binary

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**Windows:**
Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

#### EasyOCR

EasyOCR is Python-only and doesn't require system dependencies, but:
- First run will download models (~100MB)
- GPU support (optional): Requires CUDA-enabled PyTorch

## Installation

### Option 1: Install Both (Recommended)

```bash
# Install Python packages
pip install opencv-python numpy matplotlib pytesseract easyocr

# Install Tesseract binary (macOS)
brew install tesseract
```

### Option 2: Tesseract Only

```bash
pip install opencv-python numpy matplotlib pytesseract
brew install tesseract  # macOS
```

### Option 3: EasyOCR Only

```bash
pip install opencv-python numpy matplotlib easyocr
```

## Usage

### Basic Usage

```bash
python task5_ocr_O_detection.py
```

### Expected Output

The script will:
1. Load the image from `datasets/text_frombook.png`
2. Preprocess the image (grayscale, binary threshold)
3. Run Tesseract OCR detection (if available)
4. Run EasyOCR detection (if available)
5. Compare results and generate statistics
6. Create visualizations and save them to `output/` directory
7. Generate a comprehensive PDF report

### Output Files

All files are saved to the `output/` directory:

- `task5_ocr_comparison.png` - Side-by-side comparison of both methods
- `task5_ocr_statistics.png` - Confidence distribution and detection counts
- `task5_ocr_preprocessing.png` - Image preprocessing steps
- `Task5_OCR_O_Detection_Report.pdf` - Complete analysis report
- `task5_ocr_results.txt` - Detailed detection results in text format

## Method Comparison

### Tesseract OCR

**Advantages:**
- Fast and lightweight
- No GPU required
- Good for clean, printed text
- Open-source and widely supported

**Disadvantages:**
- Less accurate for varied fonts/quality
- Sensitive to image quality and orientation
- May struggle with handwritten text

**Best for:**
- Clean scanned documents
- Standard fonts
- High-quality images
- When speed is priority

### EasyOCR

**Advantages:**
- High accuracy across different fonts
- Robust to noise and image distortion
- Better handling of varied layouts
- Supports 80+ languages
- Deep learning-based

**Disadvantages:**
- Slower processing (especially without GPU)
- Larger model size (~100MB download)
- Higher memory requirements

**Best for:**
- Complex layouts
- Varied fonts and styles
- Lower quality images
- When accuracy is priority

## Troubleshooting

### Tesseract not found error

```
pytesseract.pytesseract.TesseractNotFoundError
```

**Solution:**
- Make sure Tesseract is installed: `brew install tesseract` (macOS)
- Or manually set path in script:
  ```python
  pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
  ```

### EasyOCR slow on first run

**Explanation:** EasyOCR downloads models (~100MB) on first run.

**Solution:** Wait for download to complete. Subsequent runs will be faster.

### Low detection accuracy

**Possible causes:**
- Image quality issues
- Font not well-supported
- Text too small or too large
- Poor contrast

**Solutions:**
- Adjust preprocessing (threshold values, denoising)
- Try different Tesseract PSM modes
- Increase image resolution
- Improve image contrast

### Memory issues with EasyOCR

**Solution:**
- Reduce image size before processing
- Use GPU mode if available (set `gpu=True`)
- Close other memory-intensive applications

## Configuration

### Adjusting Detection Thresholds

In the script, you can modify:

```python
# Tesseract confidence threshold
if conf > 30:  # Lower = more detections, higher = more accurate
```

```python
# Tesseract PSM mode
custom_config = r'--oem 3 --psm 6'
# PSM modes:
# 3 = Fully automatic page segmentation
# 6 = Assume a uniform block of text
# 11 = Sparse text, find as much as possible
```

### Enabling GPU for EasyOCR

```python
reader = easyocr.Reader(['en'], gpu=True)  # Requires CUDA
```

## Performance Comparison

Typical performance on a standard text image:

| Method    | Speed      | Accuracy | Memory |
|-----------|------------|----------|--------|
| Tesseract | Fast (~1s) | Good     | Low    |
| EasyOCR   | Slow (~5s) | Better   | High   |

*Note: Speed depends on image size and hardware*

## Algorithm Overview

### 1. Preprocessing
- Convert to grayscale
- Apply Otsu's binary thresholding
- Optional denoising

### 2. Tesseract Detection
- Configure PSM mode for text layout
- Extract character-level bounding boxes
- Filter for 'O' and 'o' characters
- Apply confidence threshold

### 3. EasyOCR Detection
- Initialize deep learning model
- Detect text regions
- Extract character positions
- Filter for 'O' and 'o' characters

### 4. Comparison
- Count detections from each method
- Calculate average confidence scores
- Generate comparative visualizations
- Create statistical analysis

## Example Output

```
================================================================================
TASK 5: OCR-BASED 'O' CHARACTER DETECTION
================================================================================

✓ Tesseract OCR: Available
✓ EasyOCR: Available

📷 กำลังอ่านภาพ: datasets/text_frombook.png
✓ อ่านภาพสำเร็จ: (800, 1200, 3)

================================================================================
STEP 2: TESSERACT OCR DETECTION
================================================================================
✓ Tesseract processed 342 text elements
  O #1: Position (123, 45), Size 24×28, Confidence: 95.2%
  O #2: Position (456, 78), Size 22×26, Confidence: 89.7%
...
✓ Tesseract found 15 'O' characters

================================================================================
STEP 3: EASYOCR DETECTION
================================================================================
✓ EasyOCR processed 28 text elements
  O #1: Position (125, 46), Size 23×27, Confidence: 97.3%
...
✓ EasyOCR found 14 'O' characters

Results Summary:
  Tesseract: 15 'O' characters detected
  EasyOCR:   14 'O' characters detected
```

## Comparison with Morphological Method

| Aspect              | OCR Method              | Morphological Method           |
|---------------------|-------------------------|--------------------------------|
| Approach            | Text recognition        | Shape analysis                 |
| Accuracy            | Very high               | Good (depends on criteria)     |
| Speed               | Medium-Slow             | Fast                           |
| False Positives     | Very low                | Can be higher                  |
| Font Variations     | Handles well            | May struggle                   |
| Language Support    | Built-in                | Not applicable                 |
| Dependencies        | OCR engines needed      | Only OpenCV                    |

## Use Cases

- **Tesseract**: Production OCR systems, document processing, quick scans
- **EasyOCR**: Academic research, multilingual documents, complex layouts
- **Morphological**: Real-time processing, embedded systems, specific shape detection

## References

- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- PyTesseract: https://github.com/madmaze/pytesseract

## License

This script is part of the FRA321 Image Processing course materials.
