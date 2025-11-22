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

img_noisy = cv2.imread(r"t\nhieu_manh.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
img_ref   = cv2.imread(r"t\bieu-cam-cuoi-tu-nhien-khi-chup-anh-768x457.webp", cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
img_ref = cv2.resize(img_ref, (img_noisy.shape[1], img_noisy.shape[0]))

D0_values = [10, 20, 40, 60]

for D0 in D0_values:
    print("\n============================")
    print(f"      D0 = {D0}")
    print("============================")

    out_ideal_lp = freq_filter(img_noisy, H_ideal_lp(img_noisy.shape, D0))
    out_gauss_lp = freq_filter(img_noisy, H_gauss_lp(img_noisy.shape, D0))
    out_butt2_lp = freq_filter(img_noisy, H_butter_lp(img_noisy.shape, D0, n=2))
    out_butt4_lp = freq_filter(img_noisy, H_butter_lp(img_noisy.shape, D0, n=4))

    out_ideal_hp = freq_filter(img_noisy, H_ideal_hp(img_noisy.shape, D0))
    out_gauss_hp = freq_filter(img_noisy, H_gauss_hp(img_noisy.shape, D0))
    out_butt2_hp = freq_filter(img_noisy, H_butter_hp(img_noisy.shape, D0, n=2))
    out_butt4_hp = freq_filter(img_noisy, H_butter_hp(img_noisy.shape, D0, n=4))

    print("\n--- Low Pass Filters ---")
    print("Ideal LPF:     PSNR =", psnr(img_ref, out_ideal_lp), 
          " SSIM =", ssim(img_ref, out_ideal_lp, data_range=1.0))
    print("Gaussian LPF:  PSNR =", psnr(img_ref, out_gauss_lp), 
          " SSIM =", ssim(img_ref, out_gauss_lp, data_range=1.0))
    print("Butter LPF n=2:", psnr(img_ref, out_butt2_lp), 
          " SSIM =", ssim(img_ref, out_butt2_lp, data_range=1.0))
    print("Butter LPF n=4:", psnr(img_ref, out_butt4_lp), 
          " SSIM =", ssim(img_ref, out_butt4_lp, data_range=1.0))

    print("\n--- High Pass Filters ---")
    print("Ideal HPF:     PSNR =", psnr(img_ref, out_ideal_hp), 
          " SSIM =", ssim(img_ref, out_ideal_hp, data_range=1.0))
    print("Gaussian HPF:  PSNR =", psnr(img_ref, out_gauss_hp), 
          " SSIM =", ssim(img_ref, out_gauss_hp, data_range=1.0))
    print("Butter HPF n=2:", psnr(img_ref, out_butt2_hp), 
          " SSIM =", ssim(img_ref, out_butt2_hp, data_range=1.0))
    print("Butter HPF n=4:", psnr(img_ref, out_butt4_hp), 
          " SSIM =", ssim(img_ref, out_butt4_hp, data_range=1.0))

    plt.figure(figsize=(14, 8))
    plt.suptitle(f"Filtering Results (D0={D0})", fontsize=14)

    imgs = [
        out_ideal_lp, out_gauss_lp, out_butt2_lp, out_butt4_lp,
        out_ideal_hp, out_gauss_hp, out_butt2_hp, out_butt4_hp
    ]

    titles = [
        "Ideal LPF", "Gaussian LPF", "Butter LPF n=2", "Butter LPF n=4",
        "Ideal HPF", "Gaussian HPF", "Butter HPF n=2", "Butter HPF n=4"
    ]

    for i, (im, t) in enumerate(zip(imgs, titles), 1):
        plt.subplot(2, 4, i)
        plt.title(t)
        plt.imshow(im, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
cv2.imwrite(os.path.join(save_dir, f"ideal_lpf_D0_{D0}.png"), (out_ideal_lp * 255).astype(np.uint8))
cv2.imwrite(os.path.join(save_dir, f"gauss_lpf_D0_{D0}.png"), (out_gauss_lp * 255).astype(np.uint8))
cv2.imwrite(os.path.join(save_dir, f"butter2_lpf_D0_{D0}.png"), (out_butt2_lp * 255).astype(np.uint8))
cv2.imwrite(os.path.join(save_dir, f"butter4_lpf_D0_{D0}.png"), (out_butt4_lp * 255).astype(np.uint8))

cv2.imwrite(os.path.join(save_dir, f"ideal_hpf_D0_{D0}.png"), (out_ideal_hp * 255).astype(np.uint8))
cv2.imwrite(os.path.join(save_dir, f"gauss_hpf_D0_{D0}.png"), (out_gauss_hp * 255).astype(np.uint8))
cv2.imwrite(os.path.join(save_dir, f"butter2_hpf_D0_{D0}.png"), (out_butt2_hp * 255).astype(np.uint8))
cv2.imwrite(os.path.join(save_dir, f"butter4_hpf_D0_{D0}.png"), (out_butt4_hp * 255).astype(np.uint8))

print(f">>> Đã lưu ảnh kết quả cho D0 = {D0} vào thư mục /{save_dir}")