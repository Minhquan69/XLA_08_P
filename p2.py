import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2, max_val=1.0):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    return 10 * np.log10((max_val ** 2) / mse_val)

def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.std(img1)
    sigma2 = np.std(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))

def make_grids(shape):
    M, N = shape
    u = np.arange(-M//2, M//2)
    v = np.arange(-N//2, N//2)
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
    return 1.0 / (1.0 + (D/D0)**(2*n))

def freq_filter(img, H):
    F = fftshift(fft2(img))
    G = F * H
    out = np.abs(ifft2(ifftshift(G)))
    out = out / out.max() if out.max() > 0 else out
    return np.clip(out, 0, 1)

def load_gray(path):
    img = plt.imread(path)
    if img.ndim == 3:
        gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img
    if gray.dtype == np.uint8:
        gray = gray.astype(np.float32) / 255.0
    return gray

noisy_img = load_gray(r't\nhieu_manh.png')
clean_img = load_gray(r't\bieu-cam-cuoi-tu-nhien-khi-chup-anh-768x457.webp')

if noisy_img.shape != clean_img.shape:
    h_diff = noisy_img.shape[0] - clean_img.shape[0]
    w_diff = noisy_img.shape[1] - clean_img.shape[1]
    start_h = max(h_diff // 2, 0)
    start_w = max(w_diff // 2, 0)
    end_h = start_h + min(noisy_img.shape[0], clean_img.shape[0])
    end_w = start_w + min(noisy_img.shape[1], clean_img.shape[1])
    noisy_img = noisy_img[start_h:end_h, start_w:end_w]

D0_values = [10, 20, 30, 50, 80, 120]
best_psnr = 0
best_D0 = 30
best_out = None

for D0 in D0_values:
    out = freq_filter(noisy_img, H_gauss_lp(noisy_img.shape, D0))
    p = psnr(clean_img, out)
    if p > best_psnr:
        best_psnr = p
        best_D0 = D0
        best_out = out
print(f"Best D0 tự động chọn: {best_D0} với PSNR: {best_psnr:.2f}")

out_ideal = freq_filter(noisy_img, H_ideal_lp(noisy_img.shape, best_D0))
out_gauss = freq_filter(noisy_img, H_gauss_lp(noisy_img.shape, best_D0))
out_butt2 = freq_filter(noisy_img, H_butter_lp(noisy_img.shape, best_D0, n=2))
out_butt4 = freq_filter(noisy_img, H_butter_lp(noisy_img.shape, best_D0, n=4))

plt.figure(figsize=(15, 8))
titles = ['Noisy', 'Ideal LPF', 'Gaussian LPF', 'Butter n=2', 'Butter n=4']
imgs = [noisy_img, out_ideal, out_gauss, out_butt2, out_butt4]
for i, (im, t) in enumerate(zip(imgs, titles), 1):
    plt.subplot(2, 3, i)
    plt.title(f'{t} (D0={best_D0})')
    plt.imshow(im, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()

plt.imsave('denoised_ideal.jpg', out_ideal, cmap='gray')
plt.imsave('denoised_gauss.jpg', out_gauss, cmap='gray')
plt.imsave('denoised_butt2.jpg', out_butt2, cmap='gray')
plt.imsave('denoised_butt4.jpg', out_butt4, cmap='gray')