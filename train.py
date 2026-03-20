from ultralytics import YOLO
import os

# ===== CONFIG =====
DATA_YAML   = os.path.join(os.path.dirname(__file__), "dataset", "data.yaml")
MODEL       = "yolo11n.pt"   # nano = เร็วที่สุด  |  เปลี่ยนเป็น yolo11s.pt / yolo11m.pt ถ้าต้องการแม่นขึ้น
EPOCHS      = 50             # รอบการ train  (เพิ่มได้ถ้ามีเวลา)
IMG_SIZE    = 640
BATCH       = 16             # ลดเป็น 8 ถ้า GPU/RAM ไม่พอ
PROJECT     = "runs/train"
NAME        = "doggycatscaner"
# ==================

def main():
    print("=" * 50)
    print("  DoggyCatScaner — YOLO11 Training")
    print("=" * 50)
    print(f"  Model   : {MODEL}")
    print(f"  Data    : {DATA_YAML}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Batch   : {BATCH}")
    print("=" * 50)

    model = YOLO(MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        patience=15,        # หยุด early ถ้าไม่ดีขึ้นใน 15 epoch
        save=True,
        plots=True,
        verbose=True,
    )

    print("\n✅ Training เสร็จแล้ว!")
    print(f"📁 ผลลัพธ์อยู่ที่: {PROJECT}/{NAME}/")
    print(f"🏆 Best model  : {PROJECT}/{NAME}/weights/best.pt")

    # Validate บน test set
    print("\n📊 กำลัง Validate บน test set...")
    metrics = model.val(data=DATA_YAML, split="test")
    print(f"mAP50     : {metrics.box.map50:.3f}")
    print(f"mAP50-95  : {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
