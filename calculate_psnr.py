import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

output_images_path = "test_out"
gt_masks_path = "real_data/mask_gt"
gt_images_path=  "real_data/render_img_gt"

psnr_values = []

for output_image_name in os.listdir(output_images_path):
    if output_image_name[-4:] != ".jpg":
        continue
    image_number = output_image_name[:4]
    gt_name = image_number + ".png"
    output_image = np.array(Image.open(os.path.join(output_images_path, output_image_name))).astype(float)[:, :, :3] / 255.0
    gt_image = np.array(Image.open(os.path.join(gt_images_path, gt_name))).astype(float)[:, :, :3] / 255.0
    gt_mask = np.array(Image.open(os.path.join(gt_masks_path, gt_name))).astype(bool)[:, :, :3]

    diff = output_image[gt_mask] - gt_image[gt_mask]
    mse = (diff * diff).mean()
    psnr = 10 * np.log10((1.0 ** 2) / mse)
    psnr_values.append(psnr)

print(f"Collected {len(psnr_values)} PSNR values")
print(f"Mean PSNR:   {np.mean(psnr_values):.2f} dB")
print(f"Median PSNR: {np.median(psnr_values):.2f} dB")
print(f"Min PSNR:    {np.min(psnr_values):.2f} dB")
print(f"Max PSNR:    {np.max(psnr_values):.2f} dB")

sorted_vals = np.sort(psnr_values)
cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sorted_vals, cdf, color="steelblue", linewidth=2)

ax.axvline(np.mean(psnr_values), color="tomato", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(psnr_values):.2f} dB")
ax.axvline(np.median(psnr_values), color="orange", linestyle="--", linewidth=1.5, label=f"Median: {np.median(psnr_values):.2f} dB")

ax.set_xlabel("PSNR (dB)", fontsize=12)
ax.set_ylabel("Cumulative Fraction", fontsize=12)
ax.set_title("PSNR CDF", fontsize=14)
ax.set_ylim(0, 1)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("psnr_cdf.png", dpi=150)
plt.show()
print("Saved CDF to psnr_cdf.png")
