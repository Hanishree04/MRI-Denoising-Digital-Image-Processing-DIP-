**MRI Denoising: Digital Image Processing (DIP)**
 
**Project Description**

This project focuses on the application of **Digital Image Processing (DIP)** techniques for **denoising MRI (Magnetic Resonance Imaging)** scans. The goal is to reduce the effects of noise and improve the quality of MRI images, which is crucial for medical professionals who rely on accurate imaging for diagnoses. We apply various **denoising filters** to the MRI scans and evaluate their effectiveness using standard image quality metrics: **PSNR (Peak Signal-to-Noise Ratio)**, **SSIM (Structural Similarity Index Measure)**, and **MSE (Mean Squared Error)**.

The key aim is to simulate **real-world MRI noise** and apply different image restoration methods to recover a clean version of the image while preserving essential details that are important for medical analysis.

---

## Problem Statement

MRI images are often degraded by noise, either from the scanning process or external interferences. This noise can reduce the accuracy and reliability of medical diagnoses. Specifically, in lower-quality or older scanners, noise becomes more prominent. By applying denoising techniques, we aim to improve the clarity and resolution of these images, ensuring that radiologists and healthcare providers can make better-informed decisions.

---

##  Real-World Relevance

 **Medical Imaging Quality Improvement**: Ensures that MRI scans are clear, allowing better diagnosis of conditions such as brain tumors, spinal cord injuries, etc.
*  **Enhances Radiology Workflow**: Reduces the time radiologists spend dealing with noisy images, helping them focus on diagnosing.
*  **Reduces Image Artifacts**: Denoising minimizes the visual artifacts caused by MRI scanners, leading to more reliable results.
   **Advancement of Medical Research**: By improving MRI imaging, researchers can obtain more precise data for understanding diseases and developing treatments.

---

## Dataset Used

The images used in this project are **MRI scans** that have been sourced from publicly available dataset, such as those provided by medical imaging repositories. These images are corrupted with noise (simulating real-world scanning artifacts) to test the effectiveness of the denoising techniques. Various levels of noise are introduced to replicate the types of distortions seen in actual MRI scans.

---

## Technologies Used

* **Python 3.x**: The primary programming language used to implement the solution.
* **OpenCV**: For image processing and applying denoising filters.
* **NumPy**: For handling and manipulating image data arrays.
* **scikit-image**: For additional image processing functions, including denoising and quality metrics (PSNR, SSIM).
* **Streamlit**: For building an interactive user interface (UI) to showcase the denoising techniques and results.
* **Pillow**: For image manipulation and saving denoised results.

---

## Techniques Implemented

The following denoising techniques were implemented and tested in the project:

* **Gaussian Filter**: A basic filter used to smooth the image and reduce noise.
* **Total Variation (TV) Denoising**: A regularization method aimed at reducing noise while preserving image edges.
* **Non-Local Means (NLM)**: A powerful denoising technique based on the self-similarity of image patches.

---

##  Evaluation Results

The effectiveness of each denoising technique is evaluated using three common image quality metrics: **MSE (Mean Squared Error)**, **PSNR (Peak Signal-to-Noise Ratio)**, and **SSIM (Structural Similarity Index)**. Below are the results for each technique:

### Performance Evaluation for Y20

| Method                    | MSE    | PSNR (dB) | SSIM   |
| ------------------------- | ------ | --------- | ------ |
| **Gaussian Filter**       | 0.0008 | 0.0010     | 0.0000 |
| **Total Variation (TV)**  | 30.90 dB | 30.03dB     | 46.65dB |
| **Non-Local Means (NLM)** | 0.9274 | 0.8748     | 0.9912 |

* **MSE (Mean Squared Error)**: Lower values are better. Indicates the overall difference between the original and denoised images.
* **PSNR (Peak Signal-to-Noise Ratio)**: Higher values indicate better denoising performance.
* **SSIM (Structural Similarity Index)**: Closer to 1 means better preservation of the image structure.

From the table above, we can conclude that **Non-Local Means (NLM)** provides the best performance in terms of both PSNR and SSIM.

---

## Project Structure

```plaintext
MRI-Denoising/
├── image.jpeg
├── app.py
├── denoising_mri.py
├── README.md

```

---

## How to Run

To run this project locally, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/MRI-Denoising.git
   ```

2. **Run the app**:

   ```bash
   streamlit run app.py
   ```

   This will open the app in your browser, where you can upload an MRI image, apply denoising filters, and evaluate the results.

---

## Team Members

* **Ananya (4SO22CD004)**

  *  Email: [22j04.ananya@sjec.ac.in]
  *  GitHub: [https://github.com/anany20]

* **Chaitra Suvarna (4SO22CD014)**

  *  Email: [22j14.chaitra@sjec.ac.in]
  *  GitHub: [https://github.com/CSuvarna23]

* **Hanishree (4SO22CD022)**

  *  Email: [22j22.hanishree@sjec.ac.in]
  *  GitHub: [https://github.com/Hanishree04]

---

## Acknowledgments

Special thanks to the following libraries and resources:

* [scikit-image](https://scikit-image.org/)
* [OpenCV](https://opencv.org/)
* [Streamlit](https://streamlit.io/)
* [NumPy](https://numpy.org/)
* [Pillow](https://python-pillow.org/)

