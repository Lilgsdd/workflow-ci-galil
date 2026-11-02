import mlflow
import dagshub
from ultralytics import YOLO
import os

# 1. Inisialisasi DagsHub untuk standar Advance
# Ganti dengan username dan nama repositori DagsHub kamu
dagshub.init(repo_owner='<nama_owner_github>', repo_name='<nama_repo_dagshub>', mlflow=True)

# 2. Konfigurasi MLflow ke localhost sesuai instruksi
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Latihan_Fall_Detection")

def train_model():
    with mlflow.start_run():
        print("Memulai pelatihan YOLOv8...")
        
        # Inisialisasi model YOLO
        model = YOLO('yolov8n.pt')
        
        # Proses training (YOLO biasanya otomatis mencatat metrik ke MLflow aktif)
        results = model.train(data='data.yaml', epochs=50, imgsz=640)
        
        # Log model terbaik secara manual sebagai artefak (opsional untuk memastikan masuk ke DagsHub)
        best_model_path = "runs/detect/train/weights/best.pt"
        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path, artifact_path="model")
            
        print("Pelatihan selesai dan artefak berhasil disimpan ke MLflow/DagsHub.")

if __name__ == "__main__":
    train_model()