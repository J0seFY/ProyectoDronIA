from djitellopy import Tello
import cv2
from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO('runs/detect/gestos_manos/weights/best.pt')

# Inicializar el dron Tello
tello = Tello()
tello.connect()

print("Bateria:", tello.get_battery())

# Iniciar la transmisión de video
tello.streamon()

# Capturar el video del dron
while True:
    frame = tello.get_frame_read().frame
    if frame is None:
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

# Detener la transmisión de video y liberar recursos
tello.streamoff()
tello.end()
# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()
# Finalizar la conexión con el dron