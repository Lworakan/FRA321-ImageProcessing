# Image Processing GUI Task 2 What’s inside the box?


An adaptable, modern GUI for image processing with support for FFT, diffusion methods, and various enhancement techniques. All parameters are adjustable in real-time.

## Features

### **NEW: Adaptive Grid Enhancement** 

Perfect for images with uneven sharpness - where some parts are sharp (like your owl's right eye) but other details are blurry!

- **Adaptive Grid Enhancement**: Automatically detects and enhances only the blurry regions
- **Selective CLAHE**: Applies different contrast enhancement to sharp vs blurry areas
- **Guided Filter**: Advanced edge-preserving smoothing
- **Bilateral Grid**: Fast edge-aware filtering

### Core Processing Methods

1. **Basic Adjustments**
   - Gamma correction (0.1 - 5.0)
   - Brightness adjustment (-100 to +100)
   - Contrast adjustment (-100 to +100)
   - Saturation adjustment (0.0 - 3.0)

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Clip limit: 1.0 - 10.0
   - Tile size: 4 - 16
   - Works in LAB color space for better color preservation

3. **Denoising**
   - Non-local means denoising
   - Adjustable strength (1 - 30)
   - Supports both color and grayscale images

4. **FFT Filtering (Frequency Domain)**
   - **Filter Types:**
     - High-pass: Enhances edges and fine details
     - Low-pass: Smoothing and noise reduction
     - Band-pass: Keeps middle frequencies
     - Band-stop: Removes middle frequencies
   - **Adjustable Parameters:**
     - Radius: 5 - 100 (cutoff frequency)
     - Strength: 0.0 - 1.0 (filter intensity)

5. **Anisotropic Diffusion (Edge-Preserving Smoothing)**
   - Based on Perona-Malik algorithm
   - **Parameters:**
     - Iterations: 1 - 50 (number of diffusion steps)
     - Kappa: 1 - 100 (edge sensitivity, lower = more edge preservation)
     - Gamma: 0.01 - 0.5 (time step size)
     - Conduction function: Exponential (favors high-contrast) or Quadratic (favors wide regions)

6. **Reverse Diffusion (Denoising Diffusion)**
   - Simulates reverse diffusion process
   - **Parameters:**
     - Timesteps: 1 - 20 (number of diffusion steps)
     - Noise start: 0.01 - 0.5 (initial noise level)
     - Noise end: 0.001 - 0.1 (final noise level)
     - Optional FFT enhancement for detail recovery

7. **Sharpening (Unsharp Mask)**
   - Sigma: 0.1 - 5.0 (blur radius)
   - Strength: 0.0 - 3.0 (sharpening intensity)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository (if you haven't):
   ```bash
   git clone <repository-url>
   cd FRA321-ImageProcessing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the GUI

Run the GUI application:
```bash
python image_processing_gui.py
```

The application will start a local web server (default: http://localhost:7860) and automatically open in your browser.

### Using the Interface

1. **Upload Image**: Click on "Input Image" and upload an image file (JPG, PNG, etc.)

2. **Adjust Parameters**:
   - Expand any accordion section to reveal parameters
   - Use sliders to adjust values in real-time
   - The processed image updates automatically

3. **Quick Presets**:
   - **Brighten Dark Image**: Applies gamma correction + CLAHE for dark images
   - **Denoise**: Activates denoising + reverse diffusion
   - **Sharpen**: Enables sharpening + FFT high-pass filter
   - **Diffusion Enhancement**: Activates both anisotropic and reverse diffusion

4. **Save Result**: Click "Save Processed Image" to export the result to the `output/` folder

## Processing Pipeline Order

The image processing is applied in this order:
1. Gamma correction
2. Brightness & contrast adjustment
3. Saturation adjustment
4. CLAHE (if enabled)
5. Denoising (if enabled)
6. FFT filtering (if enabled)
7. Anisotropic diffusion (if enabled)
8. Reverse diffusion (if enabled)
9. Sharpening/unsharp mask (if enabled)

## Examples

### Example 1: Enhancing a Dark Image
1. Upload a dark image
2. Set Gamma to 3.5
3. Enable CLAHE with clip limit 4.0
4. Add brightness: +30

### Example 2: Detail Enhancement with FFT
1. Upload your image
2. Enable FFT Filtering
3. Select "high_pass" filter type
4. Set radius to 30, strength to 0.8
5. Enable Sharpening with strength 1.5

### Example 3: Edge-Preserving Enhancement
1. Upload your image
2. Enable Anisotropic Diffusion
3. Set iterations: 15, kappa: 30, gamma: 0.15
4. Enable Reverse Diffusion with FFT enhancement
5. Adjust gamma for final brightness

### Example 4: Noise Reduction
1. Upload noisy image
2. Enable Denoising with strength 15
3. Enable Reverse Diffusion
4. Set timesteps: 8, noise start: 0.15, noise end: 0.01

## Technical Details

### FFT Filtering
- Transforms image to frequency domain using Fast Fourier Transform
- Applies frequency-based filtering (removes or enhances certain frequencies)
- Useful for:
  - Removing periodic noise (low-pass)
  - Enhancing edges and details (high-pass)
  - Targeted frequency manipulation (band-pass/stop)

### Anisotropic Diffusion
- PDE-based smoothing that preserves edges
- Perona-Malik algorithm with two conduction functions:
  - **Exponential (option 1)**: c(∇I) = exp(-(∇I/κ)²) - favors high-contrast edges
  - **Quadratic (option 2)**: c(∇I) = 1/(1+(∇I/κ)²) - favors wide regions
- Applications:
  - Medical image processing
  - Edge detection preprocessing
  - Texture preservation

### Reverse Diffusion
- Simulates the reverse process of diffusion models
- Adds noise at multiple scales, then denoises
- Optional FFT enhancement for high-frequency detail recovery
- Based on denoising diffusion probabilistic models (DDPM)

## Output

All processed images are automatically saved with timestamps:
- Location: `output/processed_<timestamp>.png`
- Format: PNG (lossless)

## Performance Tips

1. **For large images**: Start with fewer iterations for diffusion methods
2. **Real-time adjustment**: Disable computationally expensive operations while fine-tuning other parameters
3. **Order matters**: Try different combinations by enabling/disabling different methods
4. **Presets**: Use presets as starting points, then fine-tune

## Troubleshooting

### GUI doesn't start
- Check if port 7860 is available
- Try changing the port in `image_processing_gui.py` (line at the end: `server_port=7860`)

### Processing is slow
- Reduce iterations for diffusion methods
- Disable reverse diffusion if not needed
- Process smaller images for testing

### Image quality issues
- Avoid stacking too many enhancement methods
- Use denoising before sharpening
- Be careful with extreme gamma values (>4.0)

## Advanced Usage

### Integrating with Other Scripts

You can import individual processing functions:

```python
from image_processing_gui import apply_fft_filter, apply_anisotropic_diffusion

# Load image with OpenCV
img = cv2.imread('your_image.jpg')

# Apply FFT filtering
filtered = apply_fft_filter(img, filter_type="high_pass", radius=30, strength=1.0)

# Apply anisotropic diffusion
enhanced = apply_anisotropic_diffusion(filtered, iterations=15, kappa=30, gamma=0.15)
```

### Batch Processing

For batch processing multiple images, you can create a script that uses the processing functions programmatically instead of the GUI.

## References

- Perona, P., & Malik, J. (1990). Scale-space and edge detection using anisotropic diffusion
- Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models
- Pizer, S. M., et al. (1987). Adaptive histogram equalization and its variations

## License

See LICENSE file in the repository.
