import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Configuración MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parámetros
CAM_WIDTH, CAM_HEIGHT = 960, 720  # reducción para mejorar FPS si tu máquina es lenta
DRAW_COLOR = (0, 255, 0)
BRUSH_BASE = 8

# Utilidades
def normalized_to_pixel(coord, w, h):
    return int(coord.x * w), int(coord.y * h)

def euclidean(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def hand_size(landmarks, w, h):
    # Usaremos distancia entre muñeca (0) y dedo medio MCP (9) como referencia
    p0 = normalized_to_pixel(landmarks[0], w, h)
    p9 = normalized_to_pixel(landmarks[9], w, h)
    return euclidean(p0, p9)

def fingers_up(landmarks, w, h, handedness='Right'):
    """
    Devuelve lista booleana [thumb, index, middle, ring, pinky] indicando si dedo está extendido.
    Heurística basada en la posición relativa tip vs pip.
    handedness: 'Right' o 'Left' influye en la interpretación del pulgar.
    """
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]  # para pulgar usamos 3 (IP) y 4 (tip)
    fingers = []
    # Convertir a pixels para comparar
    pts = [normalized_to_pixel(lm, w, h) for lm in landmarks]
    # Pulgar (comparar x porque se abre lateralmente)
    if handedness == 'Right':
        fingers.append(pts[4][0] > pts[3][0])  # tip.x > ip.x -> extendido hacia la derecha
    else:
        fingers.append(pts[4][0] < pts[3][0])  # mano izquierda espejo
    # Otros 4 dedos (comparar y: tip arriba de pip => extendido)
    for tip, pip in zip(tips_ids[1:], pip_ids[1:]):
        fingers.append(pts[tip][1] < pts[pip][1])
    return fingers

def classify_gesture(landmarks, w, h, handedness='Right'):
    """
    Regla simple para clasificar gestos:
      - Open palm: 5 dedos true
      - Fist: 0 dedos true
      - Victory: index+middle true
      - Pointing: index true, others false
      - ThumbsUp: thumb true + others false (y orientación)
      - Pinch: distancia index_tip-thumb_tip < threshold
      - OK: index tip near thumb tip but other fingers extended? (simple check)
    """
    fingers = fingers_up(landmarks, w, h, handedness)
    cnt = sum(fingers)
    # Obtener puntos relevantes
    pts = [normalized_to_pixel(lm, w, h) for lm in landmarks]
    hand_ref = hand_size(landmarks, w, h)
    thumb_idx_dist = euclidean(pts[4], pts[8]) / (hand_ref + 1e-6)

    # umbral de pinch: ajustar experimentalmente
    PINCH_THRESH = 0.22
    if thumb_idx_dist < PINCH_THRESH:
        return 'PINCH', fingers

    if cnt == 5:
        return 'OPEN', fingers
    if cnt == 0:
        return 'FIST', fingers
    # Victory (V)
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return 'VICTORY', fingers
    # Pointing
    if fingers[1] and not any([fingers[2], fingers[3], fingers[4]]):
        return 'POINT', fingers
    # Thumbs up (pulgar levantado y resto cerrado)
    if fingers[0] and not any(fingers[1:]):
        # Adicional: chequear que el pulgar apunta arriba comparando su tip y muñeca
        if pts[4][1] < pts[0][1]:
            return 'THUMB_UP', fingers
    # OK sign: índice y pulgar cerca, otros extendidos o no
    if thumb_idx_dist < 0.32 and fingers[2] and fingers[3] and fingers[4]:
        return 'OK', fingers

    # Default: devolver conteo
    return f'{cnt}_FINGERS', fingers

# Visual mapping: gesto -> acción
# Ejemplo para un "Gesture Painter" (minijuego / interfaz):
#  - POINT: dibujar con índice (seguir punta índice)
#  - PINCH: alternar modo "cambiar color"
#  - FIST: limpiar pantalla
#  - OPEN: pausar/reanudar
#  - THUMB_UP: aumentar grosor
#  - VICTORY: disminuir grosor
#  - OK: guardar snapshot (imagen)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # Canvas para dibujar
    canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5) as hands:

        mode_paint = True
        drawing = False
        color_idx = 0
        colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,255,255)]
        brush = BRUSH_BASE
        last_pinch_time = 0
        pinch_cooldown = 0.4
        pts_deque = deque(maxlen=512)  # para trazar el rastro
        prev_gesture = None
        fps_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Mirror image for intuitive interaction
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            gesture = None
            handedness_str = 'Right'

            if results.multi_hand_landmarks:
                # Tomamos la primera mano detectada
                hand_landmarks = results.multi_hand_landmarks[0]
                # Determinar handedness (si está disponible)
                if results.multi_handedness:
                    try:
                        handedness_str = results.multi_handedness[0].classification[0].label
                    except:
                        handedness_str = 'Right'

                # Dibujar landmarks en la imagen (opcional)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Clasificar gesto
                landmarks = hand_landmarks.landmark
                gesture, fingers = classify_gesture(landmarks, CAM_WIDTH, CAM_HEIGHT, handedness_str)

                # Acción por gesto
                # Obtener coordenadas punta índice y pulgar
                pts = [normalized_to_pixel(lm, CAM_WIDTH, CAM_HEIGHT) for lm in landmarks]
                idx_tip = pts[8]
                thumb_tip = pts[4]
                # Normalizar distancia de pellizco
                href = hand_size(landmarks, CAM_WIDTH, CAM_HEIGHT)
                pinch_dist = euclidean(idx_tip, thumb_tip) / (href + 1e-6)

                # Mapeo:
                if gesture == 'POINT' and mode_paint:
                    # dibujar en canvas siguiendo el índice
                    pts_deque.append(idx_tip)
                    drawing = True
                else:
                    # cerrar el trazo (solo para cuando dejamos de point)
                    drawing = False
                    pts_deque.clear()

                if gesture == 'PINCH':
                    now = time.time()
                    if now - last_pinch_time > pinch_cooldown:
                        color_idx = (color_idx + 1) % len(colors)
                        last_pinch_time = now

                if gesture == 'FIST':
                    canvas[:] = 0  # limpiar
                if gesture == 'OPEN':
                    mode_paint = not mode_paint  # alternar
                if gesture == 'THUMB_UP':
                    brush += 2
                if gesture == 'VICTORY':
                    brush = max(1, brush - 2)
                if gesture == 'OK':
                    # guardar snapshot
                    timestamp = int(time.time())
                    cv2.imwrite(f'gesture_snapshot_{timestamp}.png', canvas)
                    # peque�a retroalimentación visual
                    cv2.putText(frame, "SAVED", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

                # Dibujar rastro del pincel
                if pts_deque:
                    for i in range(1, len(pts_deque)):
                        if pts_deque[i-1] is None or pts_deque[i] is None:
                            continue
                        cv2.line(canvas, pts_deque[i-1], pts_deque[i], colors[color_idx], brush)

                # Mostrar distancia normalizada y conteo
                cv2.putText(frame, f'Gesture: {gesture}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame, f'PinchDist: {pinch_dist:.2f}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Fusionar canvas y cámara
            # Asegurar mismo tamaño que el canvas
            frame_resized = cv2.resize(frame, (canvas.shape[1], canvas.shape[0]))
            overlay = cv2.addWeighted(frame_resized, 0.6, canvas, 0.4, 0)
            # UI info
            cv2.putText(overlay, f'Brush: {brush}', (CAM_WIDTH-220,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(overlay, f'Color: {color_idx}', (CAM_WIDTH-220,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(overlay, f'Mode Paint: {mode_paint}', (CAM_WIDTH-220,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Calcular FPS
            now = time.time()
            fps = 1.0 / (now - fps_time) if now != fps_time else 0.0
            fps_time = now
            cv2.putText(overlay, f'FPS: {int(fps)}', (10, CAM_HEIGHT-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow('Gesture Painter (MediaPipe Hands)', overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC para salir
                break
            if key == ord('c'):
                canvas[:] = 0  # limpiar con tecla c

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
