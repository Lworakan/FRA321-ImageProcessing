#!/usr/bin/env python3
"""
Task 3: Histogram Equalization
รูปภาพขนาด 5x5 พิกเซล ข้อมูลแต่ละพิกเซลมีขนาด 3 บิต
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ตั้งค่าให้แสดงภาษาไทยได้ (ถ้ามี font ที่รองรับ)
rcParams['font.family'] = 'sans-serif'

# ข้อมูลต้นฉบับ
original_data = [5,3,1,0,1,0,2,1,0,5,1,5,0,1,2,4,2,6,2,1,6,2,0,1,5]
image_size = 5  # 5x5
bit_depth = 3   # 3 bits
L = 2**bit_depth  # จำนวน gray levels = 8 (0-7)

print("="*80)
print("HISTOGRAM EQUALIZATION - ขั้นตอนการคำนวณ")
print("="*80)
print(f"\nข้อมูลเริ่มต้น:")
print(f"- ขนาดภาพ: {image_size}x{image_size} = {image_size**2} pixels")
print(f"- Bit depth: {bit_depth} bits")
print(f"- Gray levels: 0 ถึง {L-1}")
print(f"\nค่า pixel ต้นฉบับ:")
print(original_data)

# แปลงเป็น numpy array และ reshape เป็น 5x5
original_image = np.array(original_data).reshape(image_size, image_size)
print(f"\nรูปภาพ 5x5:")
print(original_image)

# ============================================
# STEP 1: สร้าง Histogram (นับความถี่)
# ============================================
print("\n" + "="*80)
print("STEP 1: สร้าง Histogram (นับความถี่ของแต่ละระดับความเข้ม)")
print("="*80)

histogram = np.zeros(L, dtype=int)
for pixel_value in original_data:
    histogram[pixel_value] += 1

print("\nความถี่ของแต่ละระดับ:")
print(f"{'Intensity (k)':<15} {'Frequency h(k)':<20} {'Pixels'}")
print("-" * 80)
total_check = 0
for k in range(L):
    positions = [i for i, v in enumerate(original_data) if v == k]
    if histogram[k] > 0:
        print(f"{k:<15} {histogram[k]:<20} {positions}")
    else:
        print(f"{k:<15} {histogram[k]:<20} -")
    total_check += histogram[k]

print(f"\nTotal pixels: {total_check}")

# ============================================
# STEP 2: คำนวณ Probability (ความน่าจะเป็น)
# ============================================
print("\n" + "="*80)
print("STEP 2: คำนวณ Probability P(k) = h(k) / (MxN)")
print("="*80)

n_pixels = image_size ** 2
probability = histogram / n_pixels

print(f"\n{'Intensity (k)':<15} {'h(k)':<15} {'P(k) = h(k)/{n_pixels}':<25} {'P(k) decimal'}")
print("-" * 80)
for k in range(L):
    print(f"{k:<15} {histogram[k]:<15} {histogram[k]}/{n_pixels:<20} {probability[k]:.4f}")

# ============================================
# STEP 3: คำนวณ CDF (Cumulative Distribution Function)
# ============================================
print("\n" + "="*80)
print("STEP 3: คำนวณ CDF (Cumulative Distribution Function)")
print("="*80)

cdf = np.cumsum(probability)

print(f"\n{'Intensity (k)':<15} {'P(k)':<15} {'CDF(k) = Σ P(i) for i=0 to k':<40} {'CDF decimal'}")
print("-" * 80)
for k in range(L):
    cdf_formula = " + ".join([f"{probability[i]:.4f}" for i in range(k+1) if probability[i] > 0])
    if not cdf_formula:
        cdf_formula = "0"
    print(f"{k:<15} {probability[k]:<15.4f} {cdf_formula:<40} {cdf[k]:.4f}")

# ============================================
# STEP 4: คำนวณค่าใหม่ด้วยสูตร Histogram Equalization
# ============================================
print("\n" + "="*80)
print("STEP 4: คำนวณค่าใหม่ด้วยสูตร: new(k) = round((L-1) × CDF(k))")
print(f"        โดย L = {L}, ดังนั้น (L-1) = {L-1}")
print("="*80)

new_values = np.round((L - 1) * cdf).astype(int)

print(f"\n{'Old Intensity':<20} {'CDF(k)':<15} {'(L-1) × CDF(k)':<25} {'New Intensity (rounded)'}")
print("-" * 80)
for k in range(L):
    calculation = (L - 1) * cdf[k]
    print(f"{k:<20} {cdf[k]:<15.4f} {L-1} × {cdf[k]:.4f} = {calculation:<10.2f} {new_values[k]}")

# สร้างตาราง mapping
print("\n" + "="*80)
print("ตาราง Mapping (Old → New):")
print("="*80)
print(f"{'Old Intensity':<20} {'→':<5} {'New Intensity'}")
print("-" * 50)
for k in range(L):
    if histogram[k] > 0:  # แสดงเฉพาะค่าที่มีในภาพ
        print(f"{k:<20} → {new_values[k]}")

# ============================================
# STEP 5: Apply mapping ไปยังภาพต้นฉบับ
# ============================================
print("\n" + "="*80)
print("STEP 5: นำ mapping ไปใช้กับภาพต้นฉบับ")
print("="*80)

equalized_data = [new_values[pixel] for pixel in original_data]
equalized_image = np.array(equalized_data).reshape(image_size, image_size)

print("\nการแปลงแต่ละ pixel:")
print(f"{'Position':<12} {'Original':<12} {'→':<5} {'Equalized'}")
print("-" * 50)
for i, (orig, eq) in enumerate(zip(original_data, equalized_data)):
    row, col = i // image_size, i % image_size
    print(f"({row},{col}){'':<5} {orig:<12} → {eq}")

print("\n" + "="*80)
print("ผลลัพธ์สุดท้าย:")
print("="*80)

print("\nภาพต้นฉบับ (5x5):")
print(original_image)

print("\nภาพหลัง Histogram Equalization (5x5):")
print(equalized_image)

print("\nค่า pixel แบบ array:")
print(f"Original:   {original_data}")
print(f"Equalized:  {equalized_data}")

# ============================================
# สร้างกราฟเปรียบเทียบ
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Histogram Equalization - Step by Step Visualization', fontsize=16, fontweight='bold')

# Original image
axes[0, 0].imshow(original_image, cmap='gray', vmin=0, vmax=L-1)
axes[0, 0].set_title('Original Image (5x5)', fontweight='bold')
axes[0, 0].axis('off')
# Add pixel values as text
for i in range(image_size):
    for j in range(image_size):
        axes[0, 0].text(j, i, str(original_image[i, j]),
                       ha="center", va="center", color="red", fontsize=12, fontweight='bold')

# Original histogram
axes[0, 1].bar(range(L), histogram, color='blue', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Intensity Level (k)')
axes[0, 1].set_ylabel('Frequency h(k)')
axes[0, 1].set_title('Original Histogram', fontweight='bold')
axes[0, 1].set_xticks(range(L))
axes[0, 1].grid(axis='y', alpha=0.3)
# Add value labels on bars
for i, v in enumerate(histogram):
    if v > 0:
        axes[0, 1].text(i, v + 0.2, str(v), ha='center', fontweight='bold')

# CDF
axes[0, 2].plot(range(L), cdf, marker='o', linewidth=2, markersize=8, color='green')
axes[0, 2].set_xlabel('Intensity Level (k)')
axes[0, 2].set_ylabel('CDF(k)')
axes[0, 2].set_title('Cumulative Distribution Function', fontweight='bold')
axes[0, 2].set_xticks(range(L))
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_ylim([0, 1.1])
# Add value labels
for i, v in enumerate(cdf):
    axes[0, 2].text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=8)

# Equalized image
axes[1, 0].imshow(equalized_image, cmap='gray', vmin=0, vmax=L-1)
axes[1, 0].set_title('Equalized Image (5x5)', fontweight='bold')
axes[1, 0].axis('off')
# Add pixel values as text
for i in range(image_size):
    for j in range(image_size):
        axes[1, 0].text(j, i, str(equalized_image[i, j]),
                       ha="center", va="center", color="red", fontsize=12, fontweight='bold')

# Equalized histogram
equalized_histogram = np.zeros(L, dtype=int)
for pixel_value in equalized_data:
    equalized_histogram[pixel_value] += 1

axes[1, 1].bar(range(L), equalized_histogram, color='orange', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Intensity Level (k)')
axes[1, 1].set_ylabel('Frequency h(k)')
axes[1, 1].set_title('Equalized Histogram', fontweight='bold')
axes[1, 1].set_xticks(range(L))
axes[1, 1].grid(axis='y', alpha=0.3)
# Add value labels on bars
for i, v in enumerate(equalized_histogram):
    if v > 0:
        axes[1, 1].text(i, v + 0.2, str(v), ha='center', fontweight='bold')

# Mapping visualization
axes[1, 2].plot(range(L), new_values, marker='o', linewidth=2, markersize=10, color='red')
axes[1, 2].plot([0, L-1], [0, L-1], 'k--', alpha=0.3, label='Identity (y=x)')
axes[1, 2].set_xlabel('Original Intensity')
axes[1, 2].set_ylabel('New Intensity')
axes[1, 2].set_title('Transformation Function', fontweight='bold')
axes[1, 2].set_xticks(range(L))
axes[1, 2].set_yticks(range(L))
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].legend()
# Add value labels
for i, v in enumerate(new_values):
    axes[1, 2].text(i, v + 0.2, f'{i}→{v}', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('output/task3_histogram_equalization.png', dpi=150, bbox_inches='tight')
print("\n📊 บันทึกกราฟ: output/task3_histogram_equalization.png")

# สร้างตารางสรุปผล
print("\n" + "="*80)
print("สรุปผลการเปรียบเทียบ")
print("="*80)
print(f"\nOriginal image statistics:")
print(f"  Min: {original_image.min()}, Max: {original_image.max()}, Mean: {original_image.mean():.2f}, Std: {original_image.std():.2f}")
print(f"\nEqualized image statistics:")
print(f"  Min: {equalized_image.min()}, Max: {equalized_image.max()}, Mean: {equalized_image.mean():.2f}, Std: {equalized_image.std():.2f}")

print("\n" + "="*80)
print("✅ Histogram Equalization เสร็จสมบูรณ์!")
print("="*80)

plt.show()
