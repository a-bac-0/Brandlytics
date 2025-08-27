import torch
from ultralytics import YOLO

# Carga el modelo pre-entrenado
model = YOLO("yolov5s.pt")  # Usa el modelo 's' para empezar, es más pequeño y rápido

# Inicia el entrenamiento con tus datos
results = model.train(data='../data/marcas.yaml', epochs=100, imgsz=640)

# El modelo entrenado se guardará en la carpeta 'runs'
print("Entrenamiento completado. Los resultados se guardaron en la carpeta 'runs'.")