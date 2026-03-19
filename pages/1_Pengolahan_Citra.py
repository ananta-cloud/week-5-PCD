import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Pengaturan Halaman
st.set_page_config(page_title="Pengolahan Citra", page_icon="🖼️")
st.title("Pengolahan Citra Interaktif")
st.write("Eksplorasi penambahan noise, denoising, dan sharpening pada citra.")

# --- FUNGSI PENGOLAHAN CITRA ---

def add_salt_and_pepper_noise(image, prob):
    """Menambahkan Salt and Pepper noise ke gambar RGB."""
    output = np.copy(image)
    # Buat matriks probabilitas acak sesuai dimensi gambar (tinggi x lebar)
    rnd = np.random.rand(output.shape[0], output.shape[1])
    
    # Pepper (Hitam)
    output[rnd < (prob / 2)] = [0, 0, 0]
    # Salt (Putih)
    output[rnd > 1 - (prob / 2)] = [255, 255, 255]
    
    return output

def denoise_image(image, method, ksize):
    """Menghilangkan noise menggunakan Median atau Gaussian Blur."""
    if method == "Median Blur":
        # Median blur sangat efektif untuk Salt & Pepper noise
        return cv2.medianBlur(image, ksize)
    else:
        # Gaussian blur untuk noise yang lebih halus
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

def sharpen_image(image):
    """Menajamkan gambar menggunakan filter 2D (Laplacian approximation)."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# --- ANTARMUKA STREAMLIT ---

# 1. Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar dari komputer Anda...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar menggunakan PIL dan konversi ke Numpy Array (Format RGB)
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.subheader("Gambar Asli")
    st.image(img_array, use_container_width=True)
    
    st.markdown("---")
    
    # Membagi layout menjadi 3 kolom untuk menampilkan hasil
    col1, col2, col3 = st.columns(3)
    
    # Parameter Kontrol di Sidebar
    st.sidebar.header("Pengaturan Parameter")
    noise_prob = st.sidebar.slider("Intensitas Noise S&P", 0.0, 0.2, 0.05, 0.01)
    blur_method = st.sidebar.selectbox("Metode Penghilang Noise", ["Median Blur", "Gaussian Blur"])
    ksize = st.sidebar.slider("Ukuran Kernel Blur (Ganjil)", 3, 11, 3, step=2)
    
    # a. Proses Menambahkan Noise
    noisy_img = add_salt_and_pepper_noise(img_array, noise_prob)
    with col1:
        st.write("**1. Citra + Noise**")
        st.image(noisy_img, use_container_width=True)
        st.caption(f"Noise probabilitas: {noise_prob}")
        
    # b. Proses Menghilangkan Noise
    denoised_img = denoise_image(noisy_img, blur_method, ksize)
    with col2:
        st.write("**2. Denoised (Penghilangan Noise)**")
        st.image(denoised_img, use_container_width=True)
        st.caption(f"Metode: {blur_method} (k={ksize})")
        
    # c. Proses Penajaman Citra
    sharpened_img = sharpen_image(denoised_img)
    with col3:
        st.write("**3. Sharpened (Penajaman)**")
        st.image(sharpened_img, use_container_width=True)
        st.caption("Kernel: [[0,-1,0], [-1,5,-1], [0,-1,0]]")

else:
    st.info("Silakan unggah gambar terlebih dahulu untuk memulai pengolahan citra.")