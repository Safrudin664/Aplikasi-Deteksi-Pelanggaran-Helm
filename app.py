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
import av
from twilio.rest import Client
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# ----------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & UI CSS CUSTOM
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sistem Deteksi Pelanggaran Helm", 
    page_icon="🛵", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk membuat tampilan persis seperti gambar desain Anda
st.markdown("""
    <style>
    .card-realtime {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        color: white;
    }
    .card-log {
        background-color: #3F4222;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #A3E635;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

folder_simpan = 'no-helm'
if not os.path.exists(folder_simpan):
    os.makedirs(folder_simpan)

# ----------------------------------------------------------------------
# 2. INISIALISASI SESSION STATE & TWILIO TURN SERVER
# ----------------------------------------------------------------------
if 'waktu_jepret' not in st.session_state:
    st.session_state.waktu_jepret = 0

# --- KONFIGURASI TWILIO ---
# PERHATIAN: Pastikan Token dicopy SETELAH mengklik tombol "Show" di dashboard Twilio
TWILIO_ACCOUNT_SID = "AC0d854a7b87db93f735d506b9e7f7a900" 
TWILIO_AUTH_TOKEN = "aa8a2f0a7dba5264bf0530b2c284f1ab" # Ganti dengan Token Asli yang sudah di-Show!

def get_ice_servers():
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        token = client.tokens.create()
        return token.ice_servers
    except Exception as e:
        st.sidebar.error(f"❌ Twilio Error: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": get_ice_servers()}
)

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
# 4. FUNGSI LOGIKA INTI DETEKSI (DIPERCEPAT UNTUK ANTI-LAG)
# ----------------------------------------------------------------------
def proses_deteksi_frame(img0, last_capture_time):
    # Menggunakan resolusi 320 agar lebih ringan di CPU Cloud
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

# --- KELAS PROSESOR WEBRTC (DENGAN ASYNC FRAME DROPPER) ---
class YoloVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.waktu_jepret = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Proses AI langsung di backend
        img_hasil, self.waktu_jepret, _ = proses_deteksi_frame(img, self.waktu_jepret)
        return av.VideoFrame.from_ndarray(img_hasil, format="bgr24")


# ======================================================================
# PENGATURAN ROUTING HALAMAN & SIDEBAR
# ======================================================================

st.sidebar.markdown("### 🛵 Pusat Navigasi")
menu_utama = st.sidebar.radio(
    "Pilih Menu Halaman:", 
    ["🏠 Home", "📸 Deteksi Real-Time", "📂 Upload Video", "📊 Bukti Pelanggaran", "ℹ️ Tentang"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Sistem Informasi Deteksi Pelanggaran Lalu Lintas v1.0")

# ----------------------------------------------------------------------
# HALAMAN 1: HOME
# ----------------------------------------------------------------------
if menu_utama == '🏠 Home':
    st.markdown("<h1>🏠 Halaman Utama (Home)</h1>", unsafe_allow_html=True)
    st.write("Selamat datang di **Sistem Inteligensia Pemantau Ketertiban Lalu Lintas**. Halaman ini merupakan pusat navigasi utama dari sistem deteksi pelanggaran helm pengendara sepeda motor.")
    
    st.write("<br>", unsafe_allow_html=True)
    total_pelanggaran = len(glob.glob(f"{folder_simpan}/*.jpg"))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="card-realtime">
            <h3>📸 Sensor Real-Time</h3>
            <p>Siap memproses data streaming langsung dari kamera video lokal maupun online.</p>
        </div>
        """, unsafe_allow_html=True)
            
    with col2:
        st.markdown(f"""
        <div class="card-log">
            <h3>📦 Log Pelanggaran</h3>
            <p>Telah mengarsipkan sebanyak <b>{total_pelanggaran}</b> foto bukti pelanggaran hukum.</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HALAMAN 2: DETEKSI REAL-TIME (WEBRTC ONLINE)
# ----------------------------------------------------------------------
elif menu_utama == '📸 Deteksi Real-Time':
    st.markdown("<h1>📸 Menu Deteksi Real-Time (Cloud)</h1>", unsafe_allow_html=True)
    st.write("Silakan klik tombol **'START'** di bawah ini dan berikan izin pada browser Anda untuk mengakses kamera. Video akan berjalan mulus melalui teknologi WebRTC.")
    
    st.markdown("---")
    
    # Menjalankan stream WebRTC (async_processing=True adalah kunci Anti-Lag)
    webrtc_streamer(
        key="deteksi-realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=YoloVideoProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True 
    )
    
    st.markdown("---")
    st.caption("Mode WebRTC P2P diaktifkan. Algoritma Async Dropper memastikan aliran video dari kamera tetap lancar tanpa terhalang proses komputasi AI di backend.")

# ----------------------------------------------------------------------
# HALAMAN 3: UPLOAD VIDEO
# ----------------------------------------------------------------------
elif menu_utama == '📂 Upload Video':
    st.markdown("<h1>📂 Menu Upload Video</h1>", unsafe_allow_html=True)
    st.write("Gunakan menu ini untuk mendeteksi pelanggaran helm melalui berkas video rekaman lalu lintas.")
    
    if 'video_key' not in st.session_state:
        st.session_state.video_key = 0
        
    st.write("Unggah File Video:")
    file_video = st.file_uploader("", type=["mp4", "avi", "mov"], key=f"uploader_{st.session_state.video_key}", label_visibility="collapsed")
    
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
elif menu_utama == '📊 Bukti Pelanggaran':
    st.markdown("<h1>📊 Menu Bukti Pelanggaran</h1>", unsafe_allow_html=True)
    st.write("Halaman ini menampilkan dokumentasi bukti pelanggaran yang ditangkap otomatis oleh sistem.")
    
    daftar_foto = glob.glob(f"{folder_simpan}/*.jpg")
    daftar_foto.sort(key=os.path.getmtime, reverse=True)
    
    st.subheader("🗑️ Hapus Bukti Pelanggaran")
    if daftar_foto:
        nama_file_saja = [os.path.basename(f) for f in daftar_foto]
        pilihan_hapus = st.multiselect(
            "Pilih satu atau beberapa gambar yang ingin dihapus:", 
            options=nama_file_saja,
            label_visibility="collapsed"
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
elif menu_utama == 'ℹ️ Tentang':
    st.markdown("<h1>ℹ️ Tentang Sistem</h1>", unsafe_allow_html=True)
    st.markdown("""
    Website ini merupakan sistem deteksi pelanggaran penggunaan helm pada pengendara sepeda motor secara real-time yang dikembangkan sebagai implementasi algoritma YOLOv7 berbasis Convolutional Neural Network (CNN). Sistem ini dirancang untuk membantu proses pemantauan penggunaan helm secara otomatis melalui kamera maupun video yang diunggah oleh pengguna.
    
    Sistem mampu mendeteksi objek pengendara yang menggunakan helm (Helmet) dan tidak menggunakan helm (No-Helmet) secara cepat dan akurat. Apabila terdeteksi pengendara yang tidak menggunakan helm, sistem akan menampilkan hasil deteksi berupa bounding box, label kelas, confidence score, serta menyimpan bukti pelanggaran secara otomatis.
    
    Website ini menyediakan beberapa fitur, yaitu Deteksi Real-Time, Upload Video, Bukti Pelanggaran, dan Tentang. Melalui fitur-fitur tersebut, pengguna dapat melakukan proses deteksi, melihat hasil deteksi, serta mengakses informasi mengenai sistem yang dikembangkan.
    
    Sistem ini dibangun menggunakan Python, Flask, YOLOv7, OpenCV, dan Roboflow sebagai pendukung proses pengolahan data, pelatihan model, serta implementasi website.
    """)
