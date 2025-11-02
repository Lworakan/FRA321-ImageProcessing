#!/usr/bin/env python3
"""
Adaptable Image Processing GUI
Supports FFT, Diffusion, Grid Sampling, and various enhancement methods with adjustable parameters
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Create output directory
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)


# ===========================
# FFT PROCESSING FUNCTIONS
# ===========================

def apply_fft_filter(image, filter_type="high_pass", radius=30, strength=1.0):
    """
    Apply FFT-based frequency domain filtering

    Parameters:
    - filter_type: 'high_pass', 'low_pass', 'band_pass', 'band_stop'
    - radius: cutoff radius for filter
    - strength: filter strength (0.0 to 1.0)
    """
    if image is None:
        return None

    # Convert to grayscale for FFT
    if len(image.shape) == 3:
        # Process each channel separately
        channels = cv2.split(image)
        filtered_channels = []

        for channel in channels:
            filtered = _apply_fft_channel(channel, filter_type, radius, strength)
            filtered_channels.append(filtered)

        return cv2.merge(filtered_channels)
    else:
        return _apply_fft_channel(image, filter_type, radius, strength)


def _apply_fft_channel(channel, filter_type, radius, strength):
    """Apply FFT filter to a single channel"""
    # Convert to float
    img_float = channel.astype(np.float32)

    # Apply FFT
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Create filter mask
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2

    # Create coordinate arrays
    x, y = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)

    # Create filter based on type
    if filter_type == "high_pass":
        # High-pass: attenuate low frequencies
        mask = np.ones((rows, cols, 2), np.float32)
        mask[distance <= radius] = strength
    elif filter_type == "low_pass":
        # Low-pass: attenuate high frequencies
        mask = np.ones((rows, cols, 2), np.float32) * strength
        mask[distance <= radius] = 1.0
    elif filter_type == "band_pass":
        # Band-pass: keep middle frequencies
        mask = np.zeros((rows, cols, 2), np.float32)
        mask[(distance >= radius) & (distance <= radius * 2)] = 1.0
        mask += strength * 0.1
    elif filter_type == "band_stop":
        # Band-stop: remove middle frequencies
        mask = np.ones((rows, cols, 2), np.float32)
        mask[(distance >= radius) & (distance <= radius * 2)] = strength
    else:
        mask = np.ones((rows, cols, 2), np.float32)

    # Apply filter
    fshift = dft_shift * mask

    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # Normalize
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)

    return img_back.astype(np.uint8)


# ===========================
# GRID SAMPLING & ADAPTIVE ENHANCEMENT
# ===========================

def compute_local_variance(image, window_size=15):
    """
    Compute local variance map to identify sharp vs blurry regions
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_float = gray.astype(np.float64)

    # Compute local mean
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_mean = cv2.filter2D(gray_float, -1, kernel)

    # Compute local variance
    local_mean_sq = cv2.filter2D(gray_float ** 2, -1, kernel)
    local_variance = local_mean_sq - local_mean ** 2

    return local_variance


def compute_sharpness_map(image, window_size=15):
    """
    Compute sharpness map using Laplacian variance
    Higher values = sharper regions
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Laplacian for edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Compute local variance of Laplacian
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    lap_sq = laplacian ** 2
    sharpness = cv2.filter2D(lap_sq, -1, kernel)

    return sharpness


def apply_adaptive_grid_enhancement(image, grid_size=32, adaptive_strength=1.0,
                                   enhancement_type="sharpness", threshold_percentile=50):
    """
    Apply adaptive enhancement based on grid sampling

    Parameters:
    - grid_size: size of grid cells for analysis
    - adaptive_strength: strength of adaptive enhancement (0-3)
    - enhancement_type: 'sharpness', 'variance', or 'both'
    - threshold_percentile: percentile threshold for enhancement (0-100)
    """
    if image is None:
        return None

    h, w = image.shape[:2]

    # Compute sharpness/variance maps
    if enhancement_type in ['sharpness', 'both']:
        sharpness_map = compute_sharpness_map(image, window_size=15)
        # Normalize to 0-1
        sharpness_map = (sharpness_map - sharpness_map.min()) / (sharpness_map.max() - sharpness_map.min() + 1e-8)

    if enhancement_type in ['variance', 'both']:
        variance_map = compute_local_variance(image, window_size=15)
        # Normalize to 0-1
        variance_map = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min() + 1e-8)

    # Combine maps if needed
    if enhancement_type == 'sharpness':
        quality_map = sharpness_map
    elif enhancement_type == 'variance':
        quality_map = variance_map
    else:  # both
        quality_map = (sharpness_map + variance_map) / 2.0

    # Compute threshold
    threshold = np.percentile(quality_map, threshold_percentile)

    # Create enhancement mask (1.0 for blurry areas, 0.0 for sharp areas)
    enhancement_mask = np.clip((threshold - quality_map) / (threshold + 1e-8), 0, 1)
    enhancement_mask = (enhancement_mask * adaptive_strength).astype(np.float32)

    # Smooth the mask to avoid hard boundaries
    enhancement_mask = cv2.GaussianBlur(enhancement_mask, (21, 21), 5)

    # Apply adaptive sharpening
    # Only sharpen blurry regions
    if len(image.shape) == 3:
        enhancement_mask_3ch = np.stack([enhancement_mask] * 3, axis=2)
    else:
        enhancement_mask_3ch = enhancement_mask

    # Apply unsharp mask with adaptive strength
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image, 1.0, gaussian, -1.0, 0)

    # Blend based on mask
    result = image.astype(np.float32) + sharpened.astype(np.float32) * enhancement_mask_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def apply_bilateral_grid(image, spatial_sigma=8, range_sigma=0.1, grid_size=16):
    """
    Bilateral grid for fast edge-preserving filtering

    Parameters:
    - spatial_sigma: spatial smoothing (higher = more smoothing)
    - range_sigma: range smoothing (higher = less edge preservation)
    - grid_size: downsampling factor for grid
    """
    if image is None:
        return None

    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    h, w = img_float.shape[:2]

    # Downsample spatially
    small_h, small_w = h // grid_size, w // grid_size

    if len(img_float.shape) == 3:
        # Process each channel
        result = np.zeros_like(img_float)
        for c in range(3):
            channel = img_float[:, :, c]

            # Create bilateral grid
            grid_h = small_h
            grid_w = small_w
            grid_r = int(1.0 / range_sigma) + 1

            # Simplified bilateral filtering (full grid would be too slow)
            # Use approximate bilateral filter instead
            filtered = cv2.bilateralFilter(
                (channel * 255).astype(np.uint8),
                d=grid_size,
                sigmaColor=range_sigma * 255,
                sigmaSpace=spatial_sigma
            )
            result[:, :, c] = filtered.astype(np.float32) / 255.0
    else:
        filtered = cv2.bilateralFilter(
            (img_float * 255).astype(np.uint8),
            d=grid_size,
            sigmaColor=range_sigma * 255,
            sigmaSpace=spatial_sigma
        )
        result = filtered.astype(np.float32) / 255.0

    return (result * 255).astype(np.uint8)


def apply_guided_filter(image, radius=8, eps=0.01):
    """
    Guided filter for edge-preserving smoothing
    Better than bilateral filter for maintaining edges

    Parameters:
    - radius: filter radius
    - eps: regularization parameter (lower = more edge preservation)
    """
    if image is None:
        return None

    # Use guide image as input image
    guide = image.copy()

    if len(image.shape) == 3:
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        guide_float = guide.astype(np.float32) / 255.0

        result = np.zeros_like(img_float)
        for c in range(3):
            result[:, :, c] = _guided_filter_gray(
                img_float[:, :, c],
                cv2.cvtColor(guide_float, cv2.COLOR_BGR2GRAY),
                radius,
                eps
            )

        return (result * 255).astype(np.uint8)
    else:
        img_float = image.astype(np.float32) / 255.0
        guide_float = guide.astype(np.float32) / 255.0
        result = _guided_filter_gray(img_float, guide_float, radius, eps)
        return (result * 255).astype(np.uint8)


def _guided_filter_gray(p, guide, radius, eps):
    """
    Guided filter implementation for grayscale
    """
    mean_I = cv2.boxFilter(guide, -1, (radius, radius))
    mean_p = cv2.boxFilter(p, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(guide * p, -1, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    q = mean_a * guide + mean_b
    return q


def apply_selective_clahe(image, sharpness_threshold=50, clip_limit_blur=4.0,
                          clip_limit_sharp=2.0, tile_size=8):
    """
    Apply different CLAHE parameters to sharp vs blurry regions

    Parameters:
    - sharpness_threshold: percentile threshold (0-100)
    - clip_limit_blur: CLAHE clip limit for blurry regions
    - clip_limit_sharp: CLAHE clip limit for sharp regions
    - tile_size: CLAHE tile size
    """
    if image is None:
        return None

    # Compute sharpness map
    sharpness = compute_sharpness_map(image, window_size=15)
    threshold = np.percentile(sharpness, sharpness_threshold)

    # Create binary mask (True = blurry, False = sharp)
    blur_mask = sharpness < threshold
    blur_mask = cv2.GaussianBlur(blur_mask.astype(np.float32), (21, 21), 5) > 0.5

    # Convert to LAB
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply different CLAHE to different regions
        clahe_blur = cv2.createCLAHE(clipLimit=clip_limit_blur, tileGridSize=(tile_size, tile_size))
        clahe_sharp = cv2.createCLAHE(clipLimit=clip_limit_sharp, tileGridSize=(tile_size, tile_size))

        l_blur = clahe_blur.apply(l)
        l_sharp = clahe_sharp.apply(l)

        # Blend based on mask
        l_final = np.where(blur_mask, l_blur, l_sharp).astype(np.uint8)

        result = cv2.merge([l_final, a, b])
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    else:
        clahe_blur = cv2.createCLAHE(clipLimit=clip_limit_blur, tileGridSize=(tile_size, tile_size))
        clahe_sharp = cv2.createCLAHE(clipLimit=clip_limit_sharp, tileGridSize=(tile_size, tile_size))

        img_blur = clahe_blur.apply(image)
        img_sharp = clahe_sharp.apply(image)

        result = np.where(blur_mask, img_blur, img_sharp).astype(np.uint8)
        return result


# ===========================
# DIFFUSION FUNCTIONS
# ===========================

def apply_anisotropic_diffusion(image, iterations=10, kappa=30, gamma=0.15, option=1):
    """
    Perona-Malik anisotropic diffusion

    Parameters:
    - iterations: number of diffusion iterations
    - kappa: conduction coefficient (edge sensitivity)
    - gamma: integration constant (time step)
    - option: 1 for exponential, 2 for quadratic conduction function
    """
    if image is None:
        return None

    # Convert to float
    img_float = image.astype(np.float64) / 255.0

    # Apply diffusion to each channel
    if len(image.shape) == 3:
        diffused = np.zeros_like(img_float)
        for c in range(3):
            diffused[:, :, c] = _perona_malik_diffusion(
                img_float[:, :, c], iterations, kappa, gamma, option
            )
    else:
        diffused = _perona_malik_diffusion(img_float, iterations, kappa, gamma, option)

    # Convert back to uint8
    return np.clip(diffused * 255, 0, 255).astype(np.uint8)


def _perona_malik_diffusion(img, iterations, kappa, gamma, option):
    """
    Perona-Malik anisotropic diffusion implementation
    """
    img_out = img.copy()

    for i in range(iterations):
        # Calculate gradients in 4 directions
        deltaN = np.roll(img_out, 1, axis=0) - img_out
        deltaS = np.roll(img_out, -1, axis=0) - img_out
        deltaE = np.roll(img_out, -1, axis=1) - img_out
        deltaW = np.roll(img_out, 1, axis=1) - img_out

        # Calculate conduction coefficients
        if option == 1:
            # Exponential (favors high-contrast edges)
            cN = np.exp(-(deltaN / kappa) ** 2)
            cS = np.exp(-(deltaS / kappa) ** 2)
            cE = np.exp(-(deltaE / kappa) ** 2)
            cW = np.exp(-(deltaW / kappa) ** 2)
        else:
            # Quadratic (favors wide regions)
            cN = 1.0 / (1.0 + (deltaN / kappa) ** 2)
            cS = 1.0 / (1.0 + (deltaS / kappa) ** 2)
            cE = 1.0 / (1.0 + (deltaE / kappa) ** 2)
            cW = 1.0 / (1.0 + (deltaW / kappa) ** 2)

        # Update image
        img_out += gamma * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)

    return img_out


def apply_reverse_diffusion(image, timesteps=5, noise_start=0.1, noise_end=0.01, use_fft=True):
    """
    Reverse diffusion process for image enhancement

    Parameters:
    - timesteps: number of diffusion steps
    - noise_start: initial noise level
    - noise_end: final noise level
    - use_fft: whether to use FFT enhancement
    """
    if image is None:
        return None

    img_float = image.astype(np.float32) / 255.0
    noise_schedule = np.linspace(noise_start, noise_end, timesteps)

    result = img_float.copy()
    for t, noise_level in enumerate(noise_schedule):
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, result.shape).astype(np.float32)
        noisy = np.clip(result + noise, 0, 1)

        # Denoise with bilateral filter
        noisy_uint8 = (noisy * 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(noisy_uint8, 9, 75, 75)
        result = denoised.astype(np.float32) / 255.0

    enhanced = (result * 255).astype(np.uint8)

    # Optional FFT enhancement
    if use_fft and len(enhanced.shape) == 3:
        # Process luminance channel
        yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0].astype(np.float32)

        # Apply FFT
        dft = cv2.dft(y_channel, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # High-pass filter for detail enhancement
        rows, cols = y_channel.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.ones((rows, cols, 2), np.float32)
        r = 30
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - crow) ** 2 + (y - ccol) ** 2 <= r * r
        mask[mask_area] = 0.3

        # Apply and inverse FFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        yuv[:, :, 0] = img_back.astype(np.uint8)
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    return enhanced


# ===========================
# OTHER ENHANCEMENT FUNCTIONS
# ===========================

def adjust_gamma(image, gamma=1.0):
    """Gamma correction"""
    if image is None or gamma <= 0:
        return image

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_clahe(image, clip_limit=3.0, tile_size=8):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    if image is None:
        return None

    if len(image.shape) == 3:
        # Apply to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)


def apply_denoising(image, h=10, template_size=7, search_size=21):
    """Non-local means denoising"""
    if image is None:
        return None

    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_size, search_size)
    else:
        return cv2.fastNlMeansDenoising(image, None, h, template_size, search_size)


def apply_unsharp_mask(image, sigma=1.0, strength=1.5):
    """Unsharp masking for sharpening"""
    if image is None:
        return None

    gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """Adjust brightness and contrast"""
    if image is None:
        return None

    alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast)) if contrast != -127 else 1
    gamma = brightness

    return cv2.convertScaleAbs(image, alpha=alpha, beta=gamma)


def adjust_saturation(image, saturation=1.0):
    """Adjust color saturation"""
    if image is None or len(image.shape) != 3:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ===========================
# COMBINED PROCESSING PIPELINE
# ===========================

def process_image(
    image,
    # Basic adjustments
    gamma,
    brightness,
    contrast,
    saturation,
    # Enhancement methods
    enable_clahe, clahe_clip, clahe_tile,
    enable_denoise, denoise_strength,
    enable_sharpen, sharpen_sigma, sharpen_strength,
    # FFT
    enable_fft, fft_type, fft_radius, fft_strength,
    # Anisotropic diffusion
    enable_aniso, aniso_iter, aniso_kappa, aniso_gamma, aniso_option,
    # Reverse diffusion
    enable_reverse, reverse_steps, reverse_noise_start, reverse_noise_end, reverse_fft,
    # Grid sampling & adaptive enhancement
    enable_adaptive_grid, adaptive_strength, adaptive_type, adaptive_threshold,
    enable_selective_clahe, selective_threshold, selective_clip_blur, selective_clip_sharp,
    enable_guided_filter, guided_radius, guided_eps,
    enable_bilateral_grid, bilateral_spatial, bilateral_range
):
    """
    Complete image processing pipeline with all adjustable parameters
    """
    if image is None:
        return None

    # Convert PIL to OpenCV format (BGR)
    img = np.array(image)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Apply gamma correction
    if gamma != 1.0:
        img = adjust_gamma(img, gamma)

    # Apply brightness and contrast
    if brightness != 0 or contrast != 0:
        img = adjust_brightness_contrast(img, brightness, contrast)

    # Apply saturation
    if saturation != 1.0:
        img = adjust_saturation(img, saturation)

    # Apply guided filter (edge-preserving smoothing)
    if enable_guided_filter:
        img = apply_guided_filter(img, int(guided_radius), guided_eps)

    # Apply bilateral grid
    if enable_bilateral_grid:
        img = apply_bilateral_grid(img, bilateral_spatial, bilateral_range, 16)

    # Apply selective CLAHE (different params for sharp vs blurry regions)
    if enable_selective_clahe:
        img = apply_selective_clahe(
            img, selective_threshold, selective_clip_blur, selective_clip_sharp, 8
        )
    # Otherwise apply regular CLAHE
    elif enable_clahe:
        img = apply_clahe(img, clahe_clip, int(clahe_tile))

    # Apply denoising
    if enable_denoise:
        img = apply_denoising(img, h=int(denoise_strength))

    # Apply FFT filtering
    if enable_fft:
        img = apply_fft_filter(img, fft_type, int(fft_radius), fft_strength)

    # Apply anisotropic diffusion
    if enable_aniso:
        img = apply_anisotropic_diffusion(
            img, int(aniso_iter), aniso_kappa, aniso_gamma, int(aniso_option)
        )

    # Apply reverse diffusion
    if enable_reverse:
        img = apply_reverse_diffusion(
            img, int(reverse_steps), reverse_noise_start, reverse_noise_end, reverse_fft
        )

    # Apply adaptive grid enhancement (IMPORTANT: helps sharp right eye vs blurry left)
    if enable_adaptive_grid:
        img = apply_adaptive_grid_enhancement(
            img, 32, adaptive_strength, adaptive_type, int(adaptive_threshold)
        )

    # Apply sharpening (unsharp mask)
    if enable_sharpen:
        img = apply_unsharp_mask(img, sharpen_sigma, sharpen_strength)

    # Convert back to RGB for display
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img)


# ===========================
# GRADIO INTERFACE
# ===========================

def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(title="Image Processing GUI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Adaptable Image Processing GUI")
        gr.Markdown("Upload an image and adjust parameters in real-time. Supports FFT, Diffusion, Grid Sampling, and various enhancement methods.")

        with gr.Row():
            # LEFT SIDE: Images and Controls
            with gr.Column(scale=1):
                # Input image
                input_image = gr.Image(label="Input Image", type="pil", height=300)

                # Output image
                output_image = gr.Image(label="Processed Image", type="pil", height=300)

                # Preset buttons
                gr.Markdown("### Quick Presets")
                with gr.Row():
                    preset_brighten = gr.Button("Brighten", size="sm")
                    preset_denoise = gr.Button("Denoise", size="sm")
                with gr.Row():
                    preset_sharpen = gr.Button("Sharpen", size="sm")
                    preset_diffusion = gr.Button("Diffusion", size="sm")
                with gr.Row():
                    preset_adaptive = gr.Button("🎯 Adaptive", variant="secondary", size="sm")
                gr.Markdown("*🎯 Adaptive: Best for uneven sharpness*")

                # Save button
                save_btn = gr.Button("💾 Save Processed Image", variant="primary")
                save_status = gr.Textbox(label="Save Status", interactive=False, max_lines=1)

            # RIGHT SIDE: All Parameters
            with gr.Column(scale=2):
                # Parameters
                with gr.Accordion("Basic Adjustments", open=True):
                    with gr.Row():
                        gamma = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Gamma")
                        brightness = gr.Slider(-100, 100, value=0, step=1, label="Brightness")
                    with gr.Row():
                        contrast = gr.Slider(-100, 100, value=0, step=1, label="Contrast")
                        saturation = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Saturation")

                with gr.Accordion("CLAHE (Contrast Enhancement)", open=False):
                    enable_clahe = gr.Checkbox(label="Enable CLAHE", value=False)
                    with gr.Row():
                        clahe_clip = gr.Slider(1.0, 10.0, value=3.0, step=0.5, label="Clip Limit")
                        clahe_tile = gr.Slider(4, 16, value=8, step=2, label="Tile Size")

                with gr.Accordion("Denoising", open=False):
                    enable_denoise = gr.Checkbox(label="Enable Denoising", value=False)
                    denoise_strength = gr.Slider(1, 30, value=10, step=1, label="Strength")

                with gr.Accordion("Sharpening (Unsharp Mask)", open=False):
                    enable_sharpen = gr.Checkbox(label="Enable Sharpening", value=False)
                    with gr.Row():
                        sharpen_sigma = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Sigma")
                        sharpen_strength = gr.Slider(0.0, 3.0, value=1.5, step=0.1, label="Strength")

                with gr.Accordion("FFT Filtering (Frequency Domain)", open=False):
                    enable_fft = gr.Checkbox(label="Enable FFT Filtering", value=False)
                    fft_type = gr.Radio(
                        ["high_pass", "low_pass", "band_pass", "band_stop"],
                        value="high_pass",
                        label="Filter Type"
                    )
                    with gr.Row():
                        fft_radius = gr.Slider(5, 100, value=30, step=5, label="Radius")
                        fft_strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Strength")

                with gr.Accordion("Anisotropic Diffusion (Edge-Preserving)", open=False):
                    enable_aniso = gr.Checkbox(label="Enable Anisotropic Diffusion", value=False)
                    aniso_iter = gr.Slider(1, 50, value=10, step=1, label="Iterations")
                    with gr.Row():
                        aniso_kappa = gr.Slider(1, 100, value=30, step=1, label="Kappa (Edge Sensitivity)")
                        aniso_gamma = gr.Slider(0.01, 0.5, value=0.15, step=0.01, label="Gamma (Time Step)")
                    aniso_option = gr.Radio([1, 2], value=1, label="Conduction Function (1=Exponential, 2=Quadratic)")

                with gr.Accordion("Reverse Diffusion (Denoising Diffusion)", open=False):
                    enable_reverse = gr.Checkbox(label="Enable Reverse Diffusion", value=False)
                    reverse_steps = gr.Slider(1, 20, value=5, step=1, label="Timesteps")
                    with gr.Row():
                        reverse_noise_start = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="Noise Start")
                        reverse_noise_end = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Noise End")
                    reverse_fft = gr.Checkbox(label="Use FFT Enhancement", value=True)

                with gr.Accordion("🎯 Adaptive Grid Enhancement (Selective Sharpening)", open=True):
                    gr.Markdown("""
                    **This feature adaptively enhances blurry regions while preserving already-sharp areas.**
                    Perfect for images where some parts (like right eye) are sharp but others need enhancement.
                    """)
                    enable_adaptive_grid = gr.Checkbox(label="Enable Adaptive Grid Enhancement", value=False)
                    adaptive_strength = gr.Slider(0.0, 3.0, value=1.5, step=0.1,
                                                 label="Enhancement Strength (higher = more sharpening in blurry areas)")
                    adaptive_type = gr.Radio(
                        ["sharpness", "variance", "both"],
                        value="sharpness",
                        label="Analysis Type (sharpness = edge-based, variance = texture-based)"
                    )
                    adaptive_threshold = gr.Slider(0, 100, value=50, step=5,
                                                  label="Threshold Percentile (lower = enhance more areas)")

                with gr.Accordion("🔍 Selective CLAHE (Region-Aware Contrast)", open=False):
                    gr.Markdown("Apply different CLAHE parameters to sharp vs blurry regions automatically.")
                    enable_selective_clahe = gr.Checkbox(label="Enable Selective CLAHE", value=False)
                    selective_threshold = gr.Slider(0, 100, value=50, step=5,
                                                   label="Sharpness Threshold Percentile")
                    with gr.Row():
                        selective_clip_blur = gr.Slider(1.0, 10.0, value=5.0, step=0.5,
                                                        label="Clip Limit (Blurry Regions)")
                        selective_clip_sharp = gr.Slider(1.0, 10.0, value=2.0, step=0.5,
                                                         label="Clip Limit (Sharp Regions)")

                with gr.Accordion("🌟 Guided Filter (Edge-Preserving Smoothing)", open=False):
                    gr.Markdown("Better than bilateral filter for maintaining sharp edges while smoothing.")
                    enable_guided_filter = gr.Checkbox(label="Enable Guided Filter", value=False)
                    with gr.Row():
                        guided_radius = gr.Slider(1, 20, value=8, step=1, label="Radius")
                        guided_eps = gr.Slider(0.001, 0.5, value=0.01, step=0.001,
                                              label="Epsilon (lower = more edge preservation)")

                with gr.Accordion("🔲 Bilateral Grid (Fast Edge-Preserving)", open=False):
                    gr.Markdown("Fast approximate bilateral filtering using grid sampling.")
                    enable_bilateral_grid = gr.Checkbox(label="Enable Bilateral Grid", value=False)
                    with gr.Row():
                        bilateral_spatial = gr.Slider(1, 20, value=8, step=1,
                                                     label="Spatial Sigma (smoothing)")
                        bilateral_range = gr.Slider(0.01, 0.5, value=0.1, step=0.01,
                                                   label="Range Sigma (edge preservation)")

        # Collect all inputs
        all_inputs = [
            input_image,
            gamma, brightness, contrast, saturation,
            enable_clahe, clahe_clip, clahe_tile,
            enable_denoise, denoise_strength,
            enable_sharpen, sharpen_sigma, sharpen_strength,
            enable_fft, fft_type, fft_radius, fft_strength,
            enable_aniso, aniso_iter, aniso_kappa, aniso_gamma, aniso_option,
            enable_reverse, reverse_steps, reverse_noise_start, reverse_noise_end, reverse_fft,
            enable_adaptive_grid, adaptive_strength, adaptive_type, adaptive_threshold,
            enable_selective_clahe, selective_threshold, selective_clip_blur, selective_clip_sharp,
            enable_guided_filter, guided_radius, guided_eps,
            enable_bilateral_grid, bilateral_spatial, bilateral_range
        ]

        # Auto-update on parameter change
        for inp in all_inputs:
            inp.change(fn=process_image, inputs=all_inputs, outputs=output_image)

        # Preset functions
        def apply_brighten_preset():
            return {
                gamma: 3.5,
                enable_clahe: True,
                clahe_clip: 4.0,
                brightness: 30,
                contrast: 20
            }

        def apply_denoise_preset():
            return {
                enable_denoise: True,
                denoise_strength: 15,
                enable_reverse: True,
                reverse_steps: 5
            }

        def apply_sharpen_preset():
            return {
                enable_sharpen: True,
                sharpen_strength: 2.0,
                enable_fft: True,
                fft_type: "high_pass"
            }

        def apply_diffusion_preset():
            return {
                enable_aniso: True,
                aniso_iter: 15,
                aniso_kappa: 30,
                enable_reverse: True,
                reverse_fft: True
            }

        def apply_adaptive_preset():
            """Preset for images with uneven sharpness (sharp in some areas, blurry in others)"""
            return {
                enable_adaptive_grid: True,
                adaptive_strength: 2.0,
                adaptive_type: "sharpness",
                adaptive_threshold: 50,
                enable_selective_clahe: True,
                selective_threshold: 50,
                selective_clip_blur: 5.0,
                selective_clip_sharp: 2.0,
                gamma: 2.0
            }

        preset_brighten.click(
            fn=apply_brighten_preset,
            outputs=[gamma, enable_clahe, clahe_clip, brightness, contrast]
        )
        preset_denoise.click(
            fn=apply_denoise_preset,
            outputs=[enable_denoise, denoise_strength, enable_reverse, reverse_steps]
        )
        preset_sharpen.click(
            fn=apply_sharpen_preset,
            outputs=[enable_sharpen, sharpen_strength, enable_fft, fft_type]
        )
        preset_diffusion.click(
            fn=apply_diffusion_preset,
            outputs=[enable_aniso, aniso_iter, aniso_kappa, enable_reverse, reverse_fft]
        )
        preset_adaptive.click(
            fn=apply_adaptive_preset,
            outputs=[
                enable_adaptive_grid, adaptive_strength, adaptive_type, adaptive_threshold,
                enable_selective_clahe, selective_threshold, selective_clip_blur, selective_clip_sharp,
                gamma
            ]
        )

        # Save function
        def save_image(img):
            if img is None:
                return "No image to save"

            import time
            filename = f"output/processed_{int(time.time())}.png"
            img.save(filename)
            return f"Image saved to {filename}"

        save_btn.click(fn=save_image, inputs=output_image, outputs=save_status)

    return demo


# ===========================
# MAIN
# ===========================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
