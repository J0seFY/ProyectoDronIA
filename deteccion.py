import cv2
from ultralytics import YOLO

# cargar el modelo entrenado
model = YOLO('runs/detect/gestos_manos/weights/best.pt')

# Abrir la camara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza la detección pasando el frame al modelo
    results = model(frame)

    # Dibuja las cajas y etiquetas en el frame
    annotated_frame = results[0].plot()

    # Muestra la imagen con las detecciones
    cv2.imshow('Detección YOLOv8 + OpenCV', annotated_frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
