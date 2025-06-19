import cv2
from ultralytics import YOLO
from djitellopy import Tello
import time

# Inicializar el dron
tello = Tello()
tello.connect()
print("Batería:", tello.get_battery())

# Despegar el dron
tello.takeoff()
tello.streamon()

# Cargar el modelo YOLOv8
model = YOLO('runs/detect/gestos_manos/weights/best.pt')

# Diccionario de acciones según el nombre del gesto
acciones = {
    'Up': lambda: tello.move_up(30),
    'Down': lambda: tello.move_down(30),
    'Left': lambda: tello.move_left(30),
    'Right': lambda: tello.move_right(30),
    'Thumbs up': lambda: tello.move_forward(30),
    'Thumbs Down': lambda: tello.move_back(30),
    'Stop': lambda: tello.hover()
}

# Variables de control
ultimo_gesto = None
cooldown = 2  # segundos
ultimo_tiempo = time.time()

# Captura desde la cámara del dron
while True:
    frame = tello.get_frame_read().frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detección de gestos

    results = model(frame_rgb)
    annotated_frame = results[0].plot()

    # Procesar solo si hay detecciones  
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Solo actuar si hay suficiente confianza
            if conf > 0.35:
                gesto = model.names[cls_id]
                tiempo_actual = time.time()

                if gesto != ultimo_gesto and (tiempo_actual - ultimo_tiempo > cooldown):
                    print(f"Gesto detectado: {gesto}")
                    accion = acciones.get(gesto, None)
                    if accion:
                        accion()
                        ultimo_gesto = gesto
                        ultimo_tiempo = tiempo_actual
                    break  # Solo una acción por frame

    # Mostrar frame con anotaciones
    cv2.imshow('Detección de gestos', annotated_frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Finalizar
tello.land()
cv2.destroyAllWindows()
