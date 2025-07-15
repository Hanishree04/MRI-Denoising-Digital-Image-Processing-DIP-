import matplotlib.pyplot as plt
from skimage import io, img_as_float, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim, mean_squared_error
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma
import numpy as np

# Load images
noisy_img = img_as_float(io.imread("images/MRI_images/MRI_noisy.tif", as_gray=True))
ref_img = img_as_float(io.imread("images/MRI_images/MRI_clean.tif", as_gray=True))

# ---- GAUSSIAN DENOISING ----
gaussian_img = gaussian_filter(noisy_img, sigma=5)
gaussian_psnr = peak_signal_noise_ratio(ref_img, gaussian_img)
gaussian_ssim = ssim(ref_img, gaussian_img)
gaussian_mse = mean_squared_error(ref_img, gaussian_img)
plt.imsave("images/MRI_images/Gaussian_smoothed.tif", gaussian_img, cmap='gray')
print(f"Gaussian PSNR: {gaussian_psnr:.2f}")
print(f"Gaussian SSIM: {gaussian_ssim:.4f}")
print(f"Gaussian MSE: {gaussian_mse:.6f}")

# ---- TV DENOISING ----
tv_denoised = denoise_tv_chambolle(noisy_img, weight=0.3, multichannel=False)
tv_psnr = peak_signal_noise_ratio(ref_img, tv_denoised)
tv_ssim = ssim(ref_img, tv_denoised)
tv_mse = mean_squared_error(ref_img, tv_denoised)
plt.imsave("images/MRI_images/TV_smoothed.tif", tv_denoised, cmap='gray')
print(f"TV Denoising PSNR: {tv_psnr:.2f}")
print(f"TV Denoising SSIM: {tv_ssim:.4f}")
print(f"TV Denoising MSE: {tv_mse:.6f}")

# ---- NLM DENOISING (skimage) ----
sigma_est = np.mean(estimate_sigma(noisy_img, multichannel=False))
nlm_denoised = denoise_nl_means(noisy_img, h=1.15 * sigma_est, fast_mode=True,
                                 patch_size=9, patch_distance=5, multichannel=False)
nlm_psnr = peak_signal_noise_ratio(ref_img, nlm_denoised)
nlm_ssim = ssim(ref_img, nlm_denoised)
nlm_mse = mean_squared_error(ref_img, nlm_denoised)
plt.imsave("images/MRI_images/NLM_skimage_denoised.tif", img_as_ubyte(nlm_denoised), cmap='gray')
print(f"NLM PSNR: {nlm_psnr:.2f}")
print(f"NLM SSIM: {nlm_ssim:.4f}")
print(f"NLM MSE: {nlm_mse:.6f}")
