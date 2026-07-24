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
import threading
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    [data-testid="collapsedControl"] {display: none;}
    </style>
""", unsafe_allow_html=True)

folder_simpan = 'no-helm'
if not os.path.exists(folder_simpan):
    os.makedirs(folder_simpan)

# Konfigurasi ICE server (STUN + TURN).
#
# Streamlit Community Cloud diketahui sering memblokir paket WebRTC secara
# langsung, sehingga TURN server WAJIB dipakai (bukan cuma STUN), dan harus
# yang stabil. Open Relay Project (gratis) sering down, jadi di sini kita
# pakai Twilio Network Traversal Service — ini rekomendasi resmi dari
# pembuat streamlit-webrtc, dan dipakai juga di demo resminya di Community
# Cloud. Daftar gratis (ada trial credit) di https://www.twilio.com/try-twilio
#
# Simpan TWILIO_ACCOUNT_SID & TWILIO_AUTH_TOKEN di Streamlit Secrets
# (menu Settings > Secrets di dashboard Community Cloud), JANGAN ditulis
# langsung di kode.
@st.cache_data(ttl=3000)  # token Twilio berlaku ~1 jam, cache 50 menit
def get_ice_servers():
    try:
        from twilio.rest import Client

        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        # Fallback ke STUN publik saja jika Twilio belum dikonfigurasi
        # (deteksi realtime mungkin tidak akan berfungsi di Community Cloud
        # tanpa TURN yang stabil, tapi ini mencegah aplikasi crash total)
        st.warning(
            "TURN server (Twilio) belum dikonfigurasi dengan benar, "
            "menggunakan STUN saja. Deteksi realtime mungkin gagal connect "
            "di Streamlit Community Cloud. Cek Settings > Secrets."
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration({"iceServers": get_ice_servers()})

# ----------------------------------------------------------------------
# 2. INISIALISASI SESSION STATE
# ----------------------------------------------------------------------
if 'halaman_aktif' not in st.session_state:
    st.session_state.halaman_aktif = 'Home'

def pindah_halaman(nama_halaman):
    st.session_state.halaman_aktif = nama_halaman

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
# 4. FUNGSI LOGIKA INTI DETEKSI (dipakai bersama oleh semua halaman)
# ----------------------------------------------------------------------
# Dipecah jadi 2 tahap:
#   1) jalankan_deteksi()  -> tahap MAHAL (inferensi YOLOv7). Hanya perlu
#      dipanggil sesekali (tidak wajib tiap frame) supaya CPU tidak overload.
#   2) gambar_hasil_deteksi() -> tahap MURAH (cuma cv2.rectangle/putText).
#      Aman dipanggil tiap frame, dipakai untuk "menghias ulang" frame baru
#      dengan hasil deteksi terakhir saat sedang skip inferensi.
def jalankan_deteksi(img0, last_capture_time):
    # Menggunakan resolusi 320 agar proses di cloud (CPU) lebih lancar
    img = letterbox(img0, 320, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45)

    daftar_deteksi = []  # list of (x1, y1, x2, y2, label_text, color, is_no_helm)
    jumlah_no_helm = 0
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label_name = names[int(cls)]
                confidence = float(conf)
                is_no_helm = False

                if label_name.lower() == "helm":
                    color = (0, 255, 0)
                    label_text = f"Helm {confidence:.2f}"
                elif label_name.lower() == "no - helm" or label_name.lower() == "no-helmet":
                    color = (0, 0, 255)
                    label_text = f"No-Helm {confidence:.2f}"
                    jumlah_no_helm += 1
                    is_no_helm = True

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
                daftar_deteksi.append((x1, y1, x2, y2, label_text, color, is_no_helm))

    return daftar_deteksi, last_capture_time, jumlah_no_helm


def gambar_hasil_deteksi(img0, daftar_deteksi):
    for x1, y1, x2, y2, label_text, color, _ in daftar_deteksi:
        cv2.rectangle(img0, (x1, y1), (x2, y2), color, 3)
        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img0, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), color, -1)
        cv2.putText(img0, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img0


def proses_deteksi_frame(img0, last_capture_time):
    """Dipakai oleh halaman Upload Video: deteksi + gambar sekaligus, tiap frame."""
    daftar_deteksi, last_capture_time, jumlah_no_helm = jalankan_deteksi(img0, last_capture_time)
    img0 = gambar_hasil_deteksi(img0, daftar_deteksi)
    return img0, last_capture_time, jumlah_no_helm


# ----------------------------------------------------------------------
# 4b. VIDEO PROCESSOR UNTUK STREAMLIT-WEBRTC
# ----------------------------------------------------------------------
# recv() dijalankan di thread terpisah oleh streamlit-webrtc untuk setiap
# frame yang masuk dari kamera, sehingga streaming jauh lebih mulus
# dibanding pendekatan rerun-per-frame seperti camera_input_live.
#
# ANTI-LAG: inferensi YOLOv7 di CPU jauh lebih lambat dari kecepatan kamera
# mengirim frame, sehingga kalau dijalankan di SETIAP frame, frame akan
# menumpuk (backpressure) dan video terasa patah-patah. Solusinya: jalankan
# inferensi hanya tiap PROSES_TIAP_N_FRAME frame, dan di frame-frame lain
# cukup gambar ulang kotak deteksi terakhir (murah, tidak butuh inferensi)
# supaya frame rate video tetap tinggi dan mulus.
PROSES_TIAP_N_FRAME = 3  # naikkan angka ini jika masih lag, turunkan jika ingin kotak lebih responsif

class HelmDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_capture_time = 0
        self.jumlah_no_helm_terakhir = 0
        self.frame_counter = 0
        self.daftar_deteksi_terakhir = []
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img0 = frame.to_ndarray(format="bgr24")

        with self.lock:
            self.frame_counter += 1
            jalankan_inferensi = (self.frame_counter % PROSES_TIAP_N_FRAME == 0)
            last_capture_time = self.last_capture_time
            daftar_deteksi = self.daftar_deteksi_terakhir

        if jalankan_inferensi:
            daftar_deteksi, last_capture_time, jumlah_no_helm = jalankan_deteksi(img0, last_capture_time)
            with self.lock:
                self.last_capture_time = last_capture_time
                self.daftar_deteksi_terakhir = daftar_deteksi
                self.jumlah_no_helm_terakhir = jumlah_no_helm

        img_hasil = gambar_hasil_deteksi(img0, daftar_deteksi)
        return av.VideoFrame.from_ndarray(img_hasil, format="bgr24")


# ======================================================================
# PENGATURAN ROUTING HALAMAN (TAMPILAN UI)
# ======================================================================

# ----------------------------------------------------------------------
# HALAMAN 1: HOME
# ----------------------------------------------------------------------
if st.session_state.halaman_aktif == 'Home':
    st.title("🏠 Halaman Utama (Home)")
    st.markdown("Selamat datang di **Sistem Inteligensia Pemantau Ketertiban Lalu Lintas**. Silakan pilih menu operasional di bawah ini:")
    st.markdown("---")

    total_pelanggaran = len(glob.glob(f"{folder_simpan}/*.jpg"))

    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📸 Deteksi Real-Time\nLakukan pendeteksian langsung melalui web browser dengan streaming WebRTC yang mulus.")
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
# HALAMAN 2: DETEKSI REAL-TIME (STREAMLIT-WEBRTC)
# ----------------------------------------------------------------------
elif st.session_state.halaman_aktif == 'Deteksi Real-Time':
    if st.button("⬅️ Kembali ke Home"):
        pindah_halaman('Home')
        st.rerun()

    st.title("📸 Menu Deteksi Real-Time (WebRTC)")
    st.write("Gunakan menu ini untuk mendeteksi pelanggaran secara langsung melalui sensor kamera. Klik **START** di bawah untuk memulai streaming.")
    st.markdown("---")

    webrtc_ctx = webrtc_streamer(
        key="deteksi-helm-realtime",
        video_processor_factory=HelmDetectionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        st.success("Status Kamera: **AKTIF** — streaming & deteksi berjalan.")
    else:
        st.info("Status Kamera: **NON-AKTIF**. Klik tombol **START** pada widget di atas untuk menyalakan sensor.")

    st.markdown("---")
    st.caption("Catatan: Mode ini menggunakan WebRTC sehingga frame diproses langsung di thread terpisah (bukan rerun per-frame), hasilnya streaming lebih mulus dan tidak patah-patah, termasuk saat diakses melalui jaringan cloud.")

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
    if 'waktu_jepret_video' not in st.session_state:
        st.session_state.waktu_jepret_video = 0

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

                frame_hasil, st.session_state.waktu_jepret_video, _ = proses_deteksi_frame(frame, st.session_state.waktu_jepret_video)
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

    st.subheader("🗑️ Fitur: Hapus Bukti Pelanggaran")
    if daftar_foto:
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
                time.sleep(1)
                st.rerun()
            else:
                st.warning("Silakan pilih setidaknya satu gambar dari kolom di atas.")
    else:
        st.info("Belum ada data bukti pelanggaran yang bisa dihapus.")

    st.markdown("---")

    st.subheader("📁 Fitur: Lihat Bukti Pelanggaran")
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
    ### 🔬 Tujuan Pengembangan
    Aplikasi web ini dibangun sebagai bagian dari penelitian sistem pakar untuk mendeteksi pelanggaran penggunaan helm pada pengendara sepeda motor secara *real-time*. Tujuannya adalah membantu mengotomatisasi sistem peringatan dini di jalan raya guna meningkatkan kepatuhan lalu lintas.

    ### 🛠️ Teknologi (Tech-Stack)
    * **Model AI:** YOLOv7 (You Only Look Once Version 7)
    * **Pemrograman:** Python 3.10+
    * **Komputasi Visual:** OpenCV, PyTorch, & streamlit-webrtc
    * **Web Framework:** Streamlit
    """)
