import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def make_grids(shape):
    M, N = shape
    u = np.arange(-M // 2, M // 2)
    v = np.arange(-N // 2, N // 2)
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    return D

def H_ideal_lp(shape, D0):
    D = make_grids(shape)
    return (D <= D0).astype(np.float32)

def H_gauss_lp(shape, D0):
    D = make_grids(shape)
    return np.exp(-(D**2)/(2*(D0**2)))

def H_butter_lp(shape, D0, n=2):
    D = make_grids(shape)
    return 1 / (1 + (D / D0)**(2*n))

def H_ideal_hp(shape, D0):
    D = make_grids(shape)
    return (D > D0).astype(np.float32)

def H_gauss_hp(shape, D0):
    D = make_grids(shape)
    return 1 - np.exp(-(D**2)/(2*(D0**2)))

def H_butter_hp(shape, D0, n=2):
    D = make_grids(shape)
    return 1 / (1 + (D0 / D)**(2*n) + 1e-6)

def freq_filter(img, H):
    F = fftshift(fft2(img))
    G = F * H
    out = np.real(ifft2(ifftshift(G)))
    return np.clip(out, 0, 1)


# ====== Load ảnh gốc và nhiễu ======
img_noisy = cv2.imread(r"D:\XuLyAnh\HM\t\nhieu_manh.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
img_ref   = cv2.imread(r"D:\XuLyAnh\HM\bieu-cam-cuoi-tu-nhien-khi-chup-anh-768x457.webp", cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0

img_ref = cv2.resize(img_ref, (img_noisy.shape[1], img_noisy.shape[0]))

# ====== Chọn D0 ======
D0 = 45   # hoặc bạn chọn giá trị nào bạn thấy tối ưu


# ====== Lọc ảnh ======
out_ideal_lp = freq_filter(img_noisy, H_ideal_lp(img_noisy.shape, D0))
out_gauss_lp = freq_filter(img_noisy, H_gauss_lp(img_noisy.shape, D0))
out_butt2_lp = freq_filter(img_noisy, H_butter_lp(img_noisy.shape, D0, n=2))
out_butt4_lp = freq_filter(img_noisy, H_butter_lp(img_noisy.shape, D0, n=4))

out_ideal_hp = freq_filter(img_noisy, H_ideal_hp(img_noisy.shape, D0))
out_gauss_hp = freq_filter(img_noisy, H_gauss_hp(img_noisy.shape, D0))
out_butt2_hp = freq_filter(img_noisy, H_butter_hp(img_noisy.shape, D0, n=2))
out_butt4_hp = freq_filter(img_noisy, H_butter_hp(img_noisy.shape, D0, n=4))


# ====== In PSNR + SSIM ======
print("\n--- LOW PASS FILTERS ---")
print("Ideal LPF:     ", psnr(img_ref, out_ideal_lp), ssim(img_ref, out_ideal_lp, data_range=1.0))
print("Gaussian LPF:  ", psnr(img_ref, out_gauss_lp), ssim(img_ref, out_gauss_lp, data_range=1.0))
print("Butter LPF n=2:", psnr(img_ref, out_butt2_lp), ssim(img_ref, out_butt2_lp, data_range=1.0))
print("Butter LPF n=4:", psnr(img_ref, out_butt4_lp), ssim(img_ref, out_butt4_lp, data_range=1.0))

print("\n--- HIGH PASS FILTERS ---")
print("Ideal HPF:     ", psnr(img_ref, out_ideal_hp), ssim(img_ref, out_ideal_hp, data_range=1.0))
print("Gaussian HPF:  ", psnr(img_ref, out_gauss_hp), ssim(img_ref, out_gauss_hp, data_range=1.0))
print("Butter HPF n=2:", psnr(img_ref, out_butt2_hp), ssim(img_ref, out_butt2_hp, data_range=1.0))
print("Butter HPF n=4:", psnr(img_ref, out_butt4_hp), ssim(img_ref, out_butt4_hp, data_range=1.0))


# ====== HIỂN THỊ TẤT CẢ 10 ẢNH ======
plt.figure(figsize=(14, 10))

titles = [
    "Ảnh gốc",
    "Ảnh nhiễu",
    "Ideal LPF",
    "Gaussian LPF",
    "Butterworth LPF n=2",
    "Ideal HPF",
    "Gaussian HPF",
    "Butterworth HPF n=2"
]

imgs = [
    img_ref, img_noisy,
    out_ideal_lp, out_gauss_lp, out_butt2_lp,
    out_ideal_hp, out_gauss_hp, out_butt2_hp
]

for i, (im, t) in enumerate(zip(imgs, titles), 1):
    plt.subplot(2, 4, i)
    plt.imshow(im, cmap='gray')
    plt.title(t)
    plt.axis("off")

plt.tight_layout()
plt.show()
