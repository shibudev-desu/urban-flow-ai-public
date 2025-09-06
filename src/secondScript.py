import torch, ultralytics, os, shutil

def train_model():
  version = "models/best.pt"
  ultralytics.settings.update({"datasets_dir": ""})
  torch.cuda.empty_cache()

  if os.path.exists(version):
    model = ultralytics.YOLO(version)
  else:
    model = ultralytics.YOLO("models/yolov8n.pt")

  if torch.cuda.is_available():
    option = "cuda"
  else:
    option = "cpu"
  
  if os.path.exists("models/vehicle-detector"):
    shutil.rmtree("models/vehicle-detector")

  model.train(
    name="vehicle-detector",
    project="models",
    data="datasets/1/data.yaml",
    device=option,
    epochs=8,
    imgsz=640,
    batch=8,
    cache=False,
    amp=False
  )

  if os.path.exists("models/vehicle-detector/weights/best.pt"):
    os.remove(version)
    shutil.move("models/vehicle-detector/weights/best.pt", "models/best.pt")

if __name__ == "__main__":
  train_model()