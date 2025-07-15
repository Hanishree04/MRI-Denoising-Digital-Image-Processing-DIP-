import streamlit as st
import numpy as np
import cv2
from skimage import img_as_float, img_as_ubyte
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
from PIL import Image
import io

# App configuration
st.set_page_config(page_title="MRI Denoising App", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'> MRI Denoising using DIP Techniques</h1>", unsafe_allow_html=True)
st.markdown("### Upload an MRI image ")

uploaded_file = st.file_uploader("")

# Utility Functions
def convert_to_downloadable(image):
    img = Image.fromarray(img_as_ubyte(image))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def calculate_metrics(original, denoised):
    mse_val = mse(original, denoised)
    psnr_val = psnr(original, denoised)
    ssim_val = ssim(original, denoised, data_range=denoised.max() - denoised.min())
    return mse_val, psnr_val, ssim_val

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("L")
        img_np = img_as_float(np.array(image))
        st.image(image, caption="Original MRI Image",use_container_width=True)
    except Exception as e:
        st.error(f"Error opening image: {e}")
        st.stop()

    with st.sidebar:
        st.header("Filter Settings")
        gaussian_kernel_size = st.slider("Gaussian Kernel Size", 3, 15, 5, step=2)
        tv_weight = st.slider("TV Denoising Weight", 0.01, 1.0, 0.1)
        patch_size = st.slider("NLM Patch Size", 3, 15, 5, step=2)
        patch_distance = st.slider("NLM Patch Distance", 3, 15, 3, step=2)

    if st.button("Apply Denoising Filters"):
        with st.spinner('Processing filters...'):

            # Gaussian Filter
            gaussian = cv2.GaussianBlur(img_np, (gaussian_kernel_size, gaussian_kernel_size), 1)
            mse_gauss, psnr_gauss, ssim_gauss = calculate_metrics(img_np, gaussian)

            # Total Variation
            tv = denoise_tv_chambolle(img_np, weight=tv_weight, channel_axis=None)
            mse_tv, psnr_tv, ssim_tv = calculate_metrics(img_np, tv)

            # NLM
            sigma_est = np.mean(estimate_sigma(img_np, channel_axis=None))
            nlm = denoise_nl_means(
                img_np,
                h=1.15 * sigma_est,
                fast_mode=True,
                patch_size=patch_size,
                patch_distance=patch_distance,
                channel_axis=None
            )
            mse_nlm, psnr_nlm, ssim_nlm = calculate_metrics(img_np, nlm)

        st.markdown("## Denoised Results & Evaluation")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Gaussian Filter")
            st.image(img_as_ubyte(gaussian), clamp=True, caption="Gaussian Denoised")
            st.metric("MSE", f"{mse_gauss:.4f}")
            st.metric("PSNR", f"{psnr_gauss:.2f} dB")
            st.metric("SSIM", f"{ssim_gauss:.4f}")
            st.download_button("Download", convert_to_downloadable(gaussian), "gaussian_denoised.png")

        with col2:
            st.markdown("### Total Variation")
            st.image(img_as_ubyte(tv), clamp=True, caption="TV Denoised")
            st.metric("MSE", f"{mse_tv:.4f}")
            st.metric("PSNR", f"{psnr_tv:.2f} dB")
            st.metric("SSIM", f"{ssim_tv:.4f}")
            st.download_button("Download", convert_to_downloadable(tv), "tv_denoised.png")

        with col3:
            st.markdown("### NLM")
            st.image(img_as_ubyte(nlm), clamp=True, caption="NLM Denoised")
            st.metric("MSE", f"{mse_nlm:.4f}")
            st.metric("PSNR", f"{psnr_nlm:.2f} dB")
            st.metric("SSIM", f"{ssim_nlm:.4f}")
            st.download_button("Download", convert_to_downloadable(nlm), "nlm_denoised.png")
