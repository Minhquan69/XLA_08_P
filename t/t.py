import cv2
import numpy as np
import os

# ĐƯỜNG DẪN
path_input = r"D:\XuLyAnh\HM\bieu-cam-cuoi-tu-nhien-khi-chup-anh-768x457.webp"
out_dir = r"D:\XuLyAnh\HM\t"

os.makedirs(out_dir, exist_ok=True)

# Load ảnh gốc grayscale
img = cv2.imread(path_input, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

# Hàm tạo nhiễu
def add_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1)

# ====== 3 mức nhiễu ======
noise_levels = {
    "nhieu_nhe" : 0.05,   # nhiễu nhẹ
    "nhieu_vua" : 0.10,   # nhiễu trung bình (khuyên dùng)
    "nhieu_manh": 0.18    # nhiễu mạnh
}

for name, sigma in noise_levels.items():
    noisy = add_noise(img, sigma)
    output_path = os.path.join(out_dir, f"{name}.png")
    cv2.imwrite(output_path, (noisy * 255).astype(np.uint8))
    print(f"Đã tạo: {output_path}  (sigma={sigma})")
