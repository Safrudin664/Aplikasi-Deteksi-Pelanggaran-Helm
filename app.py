import os
# Bypass sistem keamanan PyTorch 2.6+ (Sangat penting untuk Windows)
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import streamlit as st
import cv2
import torch
import numpy as np
import time
import tempfile
import glob
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# ----------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & HIDE SIDEBAR
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sistem Deteksi Pelanggaran Helm", 
    page_icon="🛵", 
    layout="wide",
    initial_sidebar_state="collapsed" # Menyembunyikan sidebar bawaan
)

# Kustomisasi CSS untuk menyembunyikan tombol panah sidebar bawaan Streamlit
st.markdown("""
    <style>
    [data-testid="collapsedControl"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Folder untuk menampung gambar screenshot pelanggar
folder_simpan = 'no-helm'
if not os.path.exists(folder_simpan):
    os.makedirs(folder_simpan)

# ----------------------------------------------------------------------
# 2. INISIALISASI SESSION STATE (MANAJEMEN NAVIGASI & KAMERA)
# ----------------------------------------------------------------------
# Memori untuk melacak kita sedang berada di halaman mana
if 'halaman_aktif' not in st.session_state:
    st.session_state.halaman_aktif = 'Home'

if 'kamera_berjalan' not in st.session_state:
    st.session_state.kamera_berjalan = False
if 'waktu_jepret' not in st.session_state:
    st.session_state.waktu_jepret = 0

# Fungsi ganti halaman
def pindah_halaman(nama_halaman):
    st.session_state.halaman_aktif = nama_halaman
    # Matikan kamera otomatis jika user keluar dari halaman deteksi realtime
    if nama_halaman != 'Deteksi Real-Time':
        st.session_state.kamera_berjalan = False

# ----------------------------------------------------------------------
# 3. PEMUATAN MODEL YOLOV7
# ----------------------------------------------------------------------
@st.cache_resource
def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(weights_path, map_location=device)
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    return model, device, stride, names

weights = 'best.pt'
try:
    model, device, stride, names = load_model(weights)
except Exception as e:
    st.error(f"Gagal memuat model 'best.pt': {e}")
    st.stop()

# ----------------------------------------------------------------------
# 4. FUNGSI LOGIKA INTI DETEKSI
# ----------------------------------------------------------------------
def proses_deteksi_frame(img0, last_capture_time):
    img = letterbox(img0, 320, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)
    
    jumlah_no_helm = 0
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label_name = names[int(cls)]
                confidence = float(conf)
                
                if label_name.lower() == "helm":
                    color = (0, 255, 0)
                    label_text = f"Helm {confidence:.2f}"
                elif label_name.lower() == "no - helm" or label_name.lower() == "no-helmet":
                    color = (0, 0, 255)
                    label_text = f"No-Helm {confidence:.2f}"
                    jumlah_no_helm += 1
                    
                    current_time = time.time()
                    if current_time - last_capture_time > 2.0:
                        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                        screenshot_path = os.path.join(folder_simpan, f"pelanggar_{timestamp_str}.jpg")
                        cv2.imwrite(screenshot_path, img0)
                        last_capture_time = current_time
                else:
                    color = (255, 255, 255)
                    label_text = f"{label_name} {confidence:.2f}"
                
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img0, (x1, y1), (x2, y2), color, 3)
                t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img0, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), color, -1)
                cv2.putText(img0, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
    return img0, last_capture_time, jumlah_no_helm


# ======================================================================
# PENGATURAN ROUTING HALAMAN (TAMPILAN UI)
# ======================================================================

# ----------------------------------------------------------------------
# HALAMAN 1: HOME (PUSAT NAVIGASI)
# ----------------------------------------------------------------------
if st.session_state.halaman_aktif == 'Home':
    st.title("🏠 Sistem Deteksi Pelnggaran Helm")
    st.markdown("Selamat datang di **Sistem Deteksi Pelanggaran Helm**. Silakan pilih menu operasional di bawah ini:")
    st.markdown("---")
    
    total_pelanggaran = len(glob.glob(f"{folder_simpan}/*.jpg"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 📸 Deteksi Real-Time\nLakukan pendeteksian secara langsung menggunakan sensor kamera web / perangkat lokal.")
        if st.button("Buka Menu Deteksi Real-Time ➔", use_container_width=True):
            pindah_halaman('Deteksi Real-Time')
            st.rerun()
            
    with col2:
        st.success("### 📂 Upload Video\nLakukan analisis cerdas menggunakan berkas video rekaman lalu lintas (MP4/AVI).")
        if st.button("Buka Menu Upload Video ➔", use_container_width=True):
            pindah_halaman('Upload Video')
            st.rerun()
            
    st.write("") 
    col3, col4 = st.columns(2)
    
    with col3:
        st.warning(f"### 📊 Bukti Pelanggaran\nLihat dan kelola arsip bukti pelanggaran. (Total terekam: **{total_pelanggaran}** foto)")
        if st.button("Buka Menu Bukti Pelanggaran ➔", use_container_width=True):
            pindah_halaman('Bukti Pelanggaran')
            st.rerun()
            
    with col4:
        st.error("### ℹ️ Tentang Sistem\nInformasi mengenai pengembangan website, tujuan, serta teknologi yang digunakan.")
        if st.button("Buka Menu Tentang ➔", use_container_width=True):
            pindah_halaman('Tentang')
            st.rerun()

# ----------------------------------------------------------------------
# HALAMAN 2: DETEKSI REAL-TIME
# ----------------------------------------------------------------------
elif st.session_state.halaman_aktif == 'Deteksi Real-Time':
    if st.button("⬅️ Kembali ke Home"):
        pindah_halaman('Home')
        st.rerun()
        
    st.title("📸 Menu Deteksi Real-Time")
    st.write("Gunakan menu ini untuk mendeteksi pelanggaran secara langsung melalui sensor kamera.")
    
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        if st.button("🟢 Buka Kamera", use_container_width=True):
            st.session_state.kamera_berjalan = True
            st.rerun()
    with col_btn2:
        if st.button("🔴 Tutup Kamera", use_container_width=True):
            st.session_state.kamera_berjalan = False
            st.rerun()
            
    st.markdown("---")
    
    if st.session_state.kamera_berjalan:
        st.success("Status Kamera: **AKTIF** (Sedang Melakukan Proses Deteksi...)")
        layar_kamera = st.image([])
        cap = cv2.VideoCapture(0)
        
        while st.session_state.kamera_berjalan:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membaca hardware kamera.")
                break
                
            frame_hasil, st.session_state.waktu_jepret, _ = proses_deteksi_frame(frame, st.session_state.waktu_jepret)
            frame_rgb = cv2.cvtColor(frame_hasil, cv2.COLOR_BGR2RGB)
            layar_kamera.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()
        layar_kamera.empty()
    else:
        st.info("Status Kamera: **NON-AKTIF**. Silakan klik 'Buka Kamera'.")

# ----------------------------------------------------------------------
# HALAMAN 3: UPLOAD VIDEO
# ----------------------------------------------------------------------
elif st.session_state.halaman_aktif == 'Upload Video':
    if st.button("⬅️ Kembali ke Home"):
        pindah_halaman('Home')
        st.rerun()
        
    st.title("📂 Menu Upload Video")
    st.write("Gunakan menu ini untuk mendeteksi pelanggaran helm melalui berkas video rekaman lalu lintas.")
    
    if 'video_key' not in st.session_state:
        st.session_state.video_key = 0
        
    file_video = st.file_uploader("Unggah File Video:", type=["mp4", "avi", "mov"], key=f"uploader_{st.session_state.video_key}")
    
    col_v1, col_v2, _ = st.columns([1, 1, 4])
    
    if file_video is not None:
        with col_v1:
            mulai_deteksi = st.button("🚀 Deteksi Video", use_container_width=True)
        with col_v2:
            if st.button("🗑️ Hapus Video", use_container_width=True):
                st.session_state.video_key += 1 
                st.rerun()
                
        st.markdown("---")
        
        if mulai_deteksi:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file_video.read())
            
            layar_video = st.image([])
            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.success("✅ Seluruh bingkai berkas video berhasil dianalisis penuh!")
                    break
                    
                frame_hasil, st.session_state.waktu_jepret, _ = proses_deteksi_frame(frame, st.session_state.waktu_jepret)
                frame_rgb = cv2.cvtColor(frame_hasil, cv2.COLOR_BGR2RGB)
                layar_video.image(frame_rgb, channels="RGB", use_container_width=True)
            cap.release()

# ----------------------------------------------------------------------
# HALAMAN 4: BUKTI PELANGGARAN
# ----------------------------------------------------------------------
elif st.session_state.halaman_aktif == 'Bukti Pelanggaran':
    if st.button("⬅️ Kembali ke Home"):
        pindah_halaman('Home')
        st.rerun()
        
    st.title("📊 Menu Bukti Pelanggaran")
    st.write("Halaman ini menampilkan dokumentasi bukti pelanggaran yang ditangkap otomatis oleh sistem.")
    
    daftar_foto = glob.glob(f"{folder_simpan}/*.jpg")
    daftar_foto.sort(key=os.path.getmtime, reverse=True)
    
    # --- FITUR HAPUS BUKTI PELANGGARAN (SELEKTIF) ---
    st.subheader("🗑️ Hapus Bukti Pelanggaran")
    if daftar_foto:
        # Mengambil nama file saja untuk ditampilkan di dropdown pilihan
        nama_file_saja = [os.path.basename(f) for f in daftar_foto]
        
        pilihan_hapus = st.multiselect(
            "Pilih satu atau beberapa gambar yang ingin dihapus:", 
            options=nama_file_saja
        )
        
        if st.button("🚨 Hapus Gambar Terpilih", use_container_width=True):
            if pilihan_hapus:
                for nama_file in pilihan_hapus:
                    target_path = os.path.join(folder_simpan, nama_file)
                    if os.path.exists(target_path):
                        os.remove(target_path)
                
                st.success(f"Berhasil menghapus {len(pilihan_hapus)} gambar terpilih dari sistem!")
                time.sleep(1) # Jeda agar pesan sukses terlihat oleh pengguna
                st.rerun()
            else:
                st.warning("Silakan pilih setidaknya satu gambar dari kolom di atas sebelum menekan tombol hapus.")
    else:
        st.info("Belum ada data bukti pelanggaran yang bisa dihapus.")

    st.markdown("---")
    
    # --- FITUR LIHAT BUKTI PELANGGARAN ---
    st.subheader("📁 Lihat Bukti Pelanggaran")
    if not daftar_foto:
        st.info("Belum ada data tangkapan layar pelanggaran yang tersimpan.")
    else:
        st.caption(f"Menampilkan total {len(daftar_foto)} foto bukti pelanggaran:")
        kolom_grid = st.columns(4)
        for idx, path_gambar in enumerate(daftar_foto):
            nama_berkas = os.path.basename(path_gambar)
            with kolom_grid[idx % 4]:
                st.image(path_gambar, caption=nama_berkas, use_container_width=True)

# ----------------------------------------------------------------------
# HALAMAN 5: TENTANG
# ----------------------------------------------------------------------
elif st.session_state.halaman_aktif == 'Tentang':
    if st.button("⬅️ Kembali ke Home"):
        pindah_halaman('Home')
        st.rerun()
        
    st.title("ℹ️ Tentang Sistem")
    st.markdown("""
    Website ini merupakan sistem deteksi pelanggaran penggunaan helm pada pengendara sepeda motor secara real-time yang dikembangkan sebagai implementasi algoritma YOLOv7 berbasis Convolutional Neural Network (CNN). Sistem ini dirancang untuk membantu proses pemantauan penggunaan helm secara otomatis melalui kamera maupun video yang diunggah oleh pengguna.

Sistem mampu mendeteksi objek pengendara yang menggunakan helm (Helmet) dan tidak menggunakan helm (No-Helmet) secara cepat dan akurat. Apabila terdeteksi pengendara yang tidak menggunakan helm, sistem akan menampilkan hasil deteksi berupa bounding box, label kelas, confidence score, serta menyimpan bukti pelanggaran secara otomatis.

Website ini menyediakan beberapa fitur, yaitu Deteksi Real-Time, Upload Video, Bukti Pelanggaran, dan Tentang. Melalui fitur-fitur tersebut, pengguna dapat melakukan proses deteksi, melihat hasil deteksi, serta mengakses informasi mengenai sistem yang dikembangkan.

Sistem ini dibangun menggunakan Python, Flask, YOLOv7, OpenCV, dan Roboflow sebagai pendukung proses pengolahan data, pelatihan model, serta implementasi website.
    """)