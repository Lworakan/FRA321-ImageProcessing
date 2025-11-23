#!/usr/bin/env python3

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

output_dir = Path('output')
output_dir.mkdir(exist_ok=True)


def apply_fft_filter(image, filter_type="high_pass", radius=30, strength=1.0):
    if image is None:
        return None

    if len(image.shape) == 3:
        channels = cv2.split(image)
        filtered_channels = []

        for channel in channels:
            filtered = _apply_fft_channel(channel, filter_type, radius, strength)
            filtered_channels.append(filtered)

        return cv2.merge(filtered_channels)
    else:
        return _apply_fft_channel(image, filter_type, radius, strength)


def _apply_fft_channel(channel, filter_type, radius, strength):
    img_float = channel.astype(np.float32)

    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2

    x, y = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)

    if filter_type == "high_pass":
        mask = np.ones((rows, cols, 2), np.float32)
        mask[distance <= radius] = strength
    elif filter_type == "low_pass":
        mask = np.ones((rows, cols, 2), np.float32) * strength
        mask[distance <= radius] = 1.0
    elif filter_type == "band_pass":
        mask = np.zeros((rows, cols, 2), np.float32)
        mask[(distance >= radius) & (distance <= radius * 2)] = 1.0
        mask += strength * 0.1
    elif filter_type == "band_stop":
        mask = np.ones((rows, cols, 2), np.float32)
        mask[(distance >= radius) & (distance <= radius * 2)] = strength
    else:
        mask = np.ones((rows, cols, 2), np.float32)

    fshift = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)

    return img_back.astype(np.uint8)


def compute_local_variance(image, window_size=15):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray_float = gray.astype(np.float64)

    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    local_mean = cv2.filter2D(gray_float, -1, kernel)

    local_mean_sq = cv2.filter2D(gray_float ** 2, -1, kernel)
    local_variance = local_mean_sq - local_mean ** 2

    return local_variance


def compute_sharpness_map(image, window_size=15):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    lap_sq = laplacian ** 2
    sharpness = cv2.filter2D(lap_sq, -1, kernel)

    return sharpness


def apply_adaptive_grid_enhancement(image, grid_size=32, adaptive_strength=1.0,
                                   enhancement_type="sharpness", threshold_percentile=50):
    if image is None:
        return None

    h, w = image.shape[:2]

    if enhancement_type in ['sharpness', 'both']:
        sharpness_map = compute_sharpness_map(image, window_size=15)
        sharpness_map = (sharpness_map - sharpness_map.min()) / (sharpness_map.max() - sharpness_map.min() + 1e-8)

    if enhancement_type in ['variance', 'both']:
        variance_map = compute_local_variance(image, window_size=15)
        variance_map = (variance_map - variance_map.min()) / (variance_map.max() - variance_map.min() + 1e-8)

    if enhancement_type == 'sharpness':
        quality_map = sharpness_map
    elif enhancement_type == 'variance':
        quality_map = variance_map
    else:
        quality_map = (sharpness_map + variance_map) / 2.0

    threshold = np.percentile(quality_map, threshold_percentile)

    enhancement_mask = np.clip((threshold - quality_map) / (threshold + 1e-8), 0, 1)
    enhancement_mask = (enhancement_mask * adaptive_strength).astype(np.float32)

    enhancement_mask = cv2.GaussianBlur(enhancement_mask, (21, 21), 5)

    if len(image.shape) == 3:
        enhancement_mask_3ch = cv2.merge([enhancement_mask] * 3)
    else:
        enhancement_mask_3ch = enhancement_mask

    kernel_size = int(5 + enhancement_mask.max() * 2)
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    result = image.astype(np.float32) + (sharpened.astype(np.float32) - image.astype(np.float32)) * enhancement_mask_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def apply_selective_clahe(image, threshold_percentile=50, clip_limit_blur=5.0, clip_limit_sharp=2.0):
    if image is None:
        return None

    sharpness_map = compute_sharpness_map(image, window_size=15)
    sharpness_map = (sharpness_map - sharpness_map.min()) / (sharpness_map.max() - sharpness_map.min() + 1e-8)

    threshold = np.percentile(sharpness_map, threshold_percentile)

    blur_mask = (sharpness_map < threshold).astype(np.uint8)
    sharp_mask = (sharpness_map >= threshold).astype(np.uint8)

    blur_mask = cv2.GaussianBlur(blur_mask.astype(np.float32), (21, 21), 5)
    sharp_mask = cv2.GaussianBlur(sharp_mask.astype(np.float32), (21, 21), 5)

    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        clahe_blur = cv2.createCLAHE(clipLimit=clip_limit_blur, tileGridSize=(8, 8))
        clahe_sharp = cv2.createCLAHE(clipLimit=clip_limit_sharp, tileGridSize=(8, 8))

        l_blur = clahe_blur.apply(l_channel)
        l_sharp = clahe_sharp.apply(l_channel)

        l_result = (l_blur.astype(np.float32) * blur_mask +
                   l_sharp.astype(np.float32) * sharp_mask)
        l_result = np.clip(l_result, 0, 255).astype(np.uint8)

        lab[:, :, 0] = l_result
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe_blur = cv2.createCLAHE(clipLimit=clip_limit_blur, tileGridSize=(8, 8))
        clahe_sharp = cv2.createCLAHE(clipLimit=clip_limit_sharp, tileGridSize=(8, 8))

        img_blur = clahe_blur.apply(image)
        img_sharp = clahe_sharp.apply(image)

        result = (img_blur.astype(np.float32) * blur_mask +
                 img_sharp.astype(np.float32) * sharp_mask)
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def guided_filter(guide, src, radius=8, eps=0.01):
    guide = guide.astype(np.float32) / 255.0
    src = src.astype(np.float32) / 255.0

    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_src = cv2.boxFilter(src, -1, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))

    cov_guide_src = mean_guide_src - mean_guide * mean_src

    mean_guide_sq = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_guide = mean_guide_sq - mean_guide * mean_guide

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    output = mean_a * guide + mean_b
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    return output


def apply_guided_filter(image, radius=8, eps=0.01):
    if image is None:
        return None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        channels = cv2.split(image)
        filtered_channels = []

        for channel in channels:
            filtered = guided_filter(gray, channel, radius, eps)
            filtered_channels.append(filtered)

        return cv2.merge(filtered_channels)
    else:
        return guided_filter(image, image, radius, eps)


def bilateral_grid_filter(image, spatial_sigma=8, range_sigma=0.1, grid_step=8):
    if image is None:
        return None

    img_float = image.astype(np.float32) / 255.0

    h, w = img_float.shape[:2]
    grid_h = (h + grid_step - 1) // grid_step
    grid_w = (w + grid_step - 1) // grid_step

    if len(img_float.shape) == 3:
        num_channels = img_float.shape[2]
    else:
        num_channels = 1
        img_float = img_float[:, :, np.newaxis]

    intensity_bins = 32
    grid = np.zeros((grid_h, grid_w, intensity_bins, num_channels), dtype=np.float32)
    weights = np.zeros((grid_h, grid_w, intensity_bins), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            gi = i // grid_step
            gj = j // grid_step

            if len(image.shape) == 3:
                intensity = np.mean(img_float[i, j])
            else:
                intensity = img_float[i, j, 0]

            intensity_idx = int(intensity * (intensity_bins - 1))
            intensity_idx = min(intensity_idx, intensity_bins - 1)

            spatial_weight = np.exp(-((i % grid_step) ** 2 + (j % grid_step) ** 2) / (2 * spatial_sigma ** 2))

            grid[gi, gj, intensity_idx] += img_float[i, j] * spatial_weight
            weights[gi, gj, intensity_idx] += spatial_weight

    weights = np.maximum(weights, 1e-8)
    for c in range(num_channels):
        grid[:, :, :, c] /= weights

    result = np.zeros_like(img_float)

    for i in range(h):
        for j in range(w):
            gi = i // grid_step
            gj = j // grid_step

            if len(image.shape) == 3:
                intensity = np.mean(img_float[i, j])
            else:
                intensity = img_float[i, j, 0]

            intensity_idx = int(intensity * (intensity_bins - 1))
            intensity_idx = min(intensity_idx, intensity_bins - 1)

            range_weight = np.exp(-(intensity - intensity_idx / (intensity_bins - 1)) ** 2 / (2 * range_sigma ** 2))

            result[i, j] = grid[gi, gj, intensity_idx] * range_weight

    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    if num_channels == 1:
        result = result[:, :, 0]

    return result


def anisotropic_diffusion(image, iterations=10, kappa=30, gamma=0.15, option=1):
    if image is None:
        return None

    img = image.astype(np.float32)

    if len(img.shape) == 3:
        channels = cv2.split(img)
        result_channels = []
        for channel in channels:
            result = _anisotropic_diffusion_channel(channel, iterations, kappa, gamma, option)
            result_channels.append(result)
        return cv2.merge(result_channels)
    else:
        return _anisotropic_diffusion_channel(img, iterations, kappa, gamma, option)


def _anisotropic_diffusion_channel(img, iterations, kappa, gamma, option):
    img = img.copy()

    for _ in range(iterations):
        deltaS = np.roll(img, 1, axis=0) - img
        deltaN = np.roll(img, -1, axis=0) - img
        deltaE = np.roll(img, 1, axis=1) - img
        deltaW = np.roll(img, -1, axis=1) - img

        if option == 1:
            cS = np.exp(-(deltaS / kappa) ** 2)
            cN = np.exp(-(deltaN / kappa) ** 2)
            cE = np.exp(-(deltaE / kappa) ** 2)
            cW = np.exp(-(deltaW / kappa) ** 2)
        else:
            cS = 1.0 / (1.0 + (deltaS / kappa) ** 2)
            cN = 1.0 / (1.0 + (deltaN / kappa) ** 2)
            cE = 1.0 / (1.0 + (deltaE / kappa) ** 2)
            cW = 1.0 / (1.0 + (deltaW / kappa) ** 2)

        img += gamma * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)

    return np.clip(img, 0, 255).astype(np.uint8)


def reverse_diffusion_denoise(image, timesteps=5, noise_start=0.1, noise_end=0.01, use_fft=True):
    if image is None:
        return None

    img = image.astype(np.float32) / 255.0

    noise_schedule = np.linspace(noise_start, noise_end, timesteps)

    for t in range(timesteps):
        noise_level = noise_schedule[t]

        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        noisy_img = np.clip(img + noise, 0, 1)

        if use_fft:
            denoised = noisy_img * 255
            if len(denoised.shape) == 3:
                denoised = denoised.astype(np.uint8)
                denoised = apply_fft_filter(denoised, filter_type="low_pass", radius=30, strength=0.5)
                denoised = denoised.astype(np.float32) / 255.0
            else:
                denoised = denoised.astype(np.uint8)
                denoised = apply_fft_filter(denoised, filter_type="low_pass", radius=30, strength=0.5)
                denoised = denoised.astype(np.float32) / 255.0
        else:
            denoised = cv2.GaussianBlur(noisy_img, (5, 5), 1.0)

        img = img + (img - noisy_img) * 0.5 + (denoised - noisy_img) * 0.5

        img = np.clip(img, 0, 1)

    result = (img * 255).astype(np.uint8)
    return result


def apply_gamma_correction(image, gamma=1.0):
    if gamma == 1.0:
        return image
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness == 0 and contrast == 0:
        return image

    brightness = int((brightness / 100) * 255)
    contrast = contrast / 100

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    return image


def adjust_saturation(image, saturation=0):
    if saturation == 0:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 + saturation / 100)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_clahe(image, clip_limit=2.0, tile_size=8):
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)


def denoise_image(image, strength=10):
    if len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def apply_unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def process_image(
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
):
    if input_image is None:
        return None

    img = np.array(input_image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = apply_gamma_correction(img, gamma)

    img = adjust_brightness_contrast(img, brightness, contrast)

    img = adjust_saturation(img, saturation)

    if enable_clahe:
        img = apply_clahe(img, clahe_clip, clahe_tile)

    if enable_denoise:
        img = denoise_image(img, denoise_strength)

    if enable_aniso:
        img = anisotropic_diffusion(img, aniso_iter, aniso_kappa, aniso_gamma, aniso_option)

    if enable_reverse:
        img = reverse_diffusion_denoise(img, reverse_steps, reverse_noise_start, reverse_noise_end, reverse_fft)

    if enable_adaptive_grid:
        img = apply_adaptive_grid_enhancement(img, adaptive_strength=adaptive_strength,
                                             enhancement_type=adaptive_type,
                                             threshold_percentile=adaptive_threshold)

    if enable_selective_clahe:
        img = apply_selective_clahe(img, selective_threshold, selective_clip_blur, selective_clip_sharp)

    if enable_guided_filter:
        img = apply_guided_filter(img, guided_radius, guided_eps)

    if enable_bilateral_grid:
        img = bilateral_grid_filter(img, bilateral_spatial, bilateral_range)

    if enable_sharpen:
        img = apply_unsharp_mask(img, sharpen_sigma, sharpen_strength)

    if enable_fft:
        img = apply_fft_filter(img, fft_type, fft_radius, fft_strength)

    return Image.fromarray(img)


def create_interface():
    with gr.Blocks(title="Advanced Image Processing") as demo:
        gr.Markdown("# Advanced Image Processing Suite")
        gr.Markdown("Upload an image and adjust parameters to enhance it using various techniques.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")

                gr.Markdown("### Quick Presets")
                with gr.Row():
                    preset_brighten = gr.Button("✨ Brighten", size="sm")
                    preset_denoise = gr.Button("🔇 Denoise", size="sm")
                with gr.Row():
                    preset_sharpen = gr.Button("🔪 Sharpen", size="sm")
                    preset_diffusion = gr.Button("🌊 Diffusion", size="sm")
                with gr.Row():
                    preset_adaptive = gr.Button("🎯 Adaptive", size="sm")

            with gr.Column(scale=1):
                output_image = gr.Image(type="pil", label="Processed Image")
                with gr.Row():
                    save_btn = gr.Button("💾 Save Image")
                    save_status = gr.Textbox(label="Status", interactive=False)

        with gr.Accordion("⚙️ Processing Parameters", open=True):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Basic Adjustments")
                    gamma = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Gamma Correction")
                    brightness = gr.Slider(-100, 100, value=0, step=1, label="Brightness")
                    contrast = gr.Slider(-100, 100, value=0, step=1, label="Contrast")
                    saturation = gr.Slider(-100, 100, value=0, step=1, label="Saturation")

                with gr.Column():
                    with gr.Accordion("CLAHE (Adaptive Histogram Equalization)", open=False):
                        enable_clahe = gr.Checkbox(label="Enable CLAHE", value=False)
                        with gr.Row():
                            clahe_clip = gr.Slider(1.0, 10.0, value=2.0, step=0.5, label="Clip Limit")
                            clahe_tile = gr.Slider(2, 16, value=8, step=2, label="Tile Size")

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

        for inp in all_inputs:
            inp.change(fn=process_image, inputs=all_inputs, outputs=output_image)

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

        def save_image(img):
            if img is None:
                return "No image to save"

            import time
            filename = f"output/processed_{int(time.time())}.png"
            img.save(filename)
            return f"Image saved to {filename}"

        save_btn.click(fn=save_image, inputs=output_image, outputs=save_status)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
