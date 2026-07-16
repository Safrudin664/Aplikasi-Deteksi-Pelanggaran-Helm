import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import cv2
import torch
import numpy as np
import time
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

def main():
    # 1. Konfigurasi Awal
    weights = 'best.pt' 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Membuat folder 'bukti_pelanggaran' secara otomatis ---
    folder_simpan = 'bukti_pelanggaran'
    if not os.path.exists(folder_simpan):
        os.makedirs(folder_simpan)
        print(f"📁 Folder '{folder_simpan}' berhasil dibuat otomatis oleh sistem.")

    # 2. Memuat Model
    print(f"Memuat model dari {weights} menggunakan {device}...")
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  
    names = model.module.names if hasattr(model, 'module') else model.names 
    
    # 3. Inisialisasi Kamera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
    
    print("Memulai deteksi real-time... Tekan 'q' untuk keluar.")
    
    # --- PERUBAHAN: Variabel jeda waktu untuk Anti-Spam Screenshot ---
    last_capture_time = 0 
    
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
            
        img0 = frame.copy() 
        
        # 4. Pra-pemrosesan Citra
        img = letterbox(img0, 320, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(device)
        img = img.float()  
        img /= 255.0       
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # 5. Inferensi
        with torch.no_grad():
            pred = model(img, augment=False)[0]
            
        # 6. Menerapkan NMS
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.45, classes=None, agnostic=False)
        
        # 7. Memproses Hasil Deteksi
        for i, det in enumerate(pred):  
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                
                for *xyxy, conf, cls in reversed(det):
                    label_name = names[int(cls)]
                    confidence = float(conf)
                    
                    if label_name.lower() == "helm":
                        color = (0, 255, 0) # Hijau
                        label_text = f"Helm {confidence:.2f}"
                        
                    elif label_name.lower() == "no - helm":
                        color = (0, 0, 255) # Merah
                        label_text = f"No-Helm {confidence:.2f}"
                        
                        #  Logika Screenshot ke dalam folder 'bukti_pelanggaran'
                        current_time = time.time()
                        
                        # Memastikan jarak antar foto minimal 3 detik
                        if current_time - last_capture_time > 3.0:
                            # Membuat nama file unik berdasarkan jam, menit, detik
                            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                            screenshot_path = os.path.join(folder_simpan, f"pelanggar_{timestamp_str}.jpg")
                            
                            # Menyimpan gambar
                            cv2.imwrite(screenshot_path, img0)
                            print(f"📸 Cekrek! Pelanggar terdeteksi. Disimpan di: {screenshot_path}")
                            
                            # Mengatur ulang waktu jepretan terakhir
                            last_capture_time = current_time
                            
                    else:
                        color = (255, 255, 255) 
                        label_text = f"{label_name} {confidence:.2f}"
                    
                    # Menggambar Bounding Box
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img0, (x1, y1), (x2, y2), color, 2)
                    
                    # Menggambar Teks
                    t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(img0, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), color, -1)
                    cv2.putText(img0, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # FPS Counter
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(img0, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 8. Menampilkan Hasil
        cv2.imshow("Deteksi Pelanggaran Helm YOLOv7", img0)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()