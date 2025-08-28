import cv2
from ultralytics import YOLO

def run_detection(model_path, image_path):
    """
    Ejecuta la detección de objetos en una imagen y muestra el resultado.
    """
    # Cargar el modelo entrenado
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Cargar la imagen
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen desde la ruta: {image_path}")
        return

    # Realizar la predicción
    results = model(img)[0]  # Obtenemos los resultados de la imagen

    # Dibujar los resultados en la imagen
    for box in results.boxes:
        # Obtener coordenadas del bounding box
        x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
        
        # Obtener la confianza y la clase
        confidence = box.conf[0]
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        # Formatear el texto de la etiqueta
        label = f"{class_name}: {confidence:.2f}"
        
        # Dibujar el rectángulo
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Dibujar el texto de la etiqueta
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar la imagen con las detecciones
    cv2.imshow("Deteccion de Marcas", img)
    print("Presiona cualquier tecla para cerrar la ventana.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ruta al modelo entrenado
    MODEL_PATH = "../models/best2.pt"  
    
    # Ruta a la imagen a probar
    IMAGE_TO_TEST = "../data/raw/images/BrandDetection.v6-version6.yolov8/test/images/imgi_26_default_jpg.rf.088bae248b0e469279bf0eb066bc1b00.jpg"
    # ---------------------

    run_detection(MODEL_PATH, IMAGE_TO_TEST)