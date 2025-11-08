import cv2, time, threading, queue, math, os
import numpy as np

# --- Ajusta backend de voz: 'google' o 'vosk' ---
VOICE_BACKEND = 'vosk'  # 'vosk' para offline (configurar modelo_vosk_path)
MODEL_VOSK_PATH = 'models/vosk-model-small-es-0.42'  # si usas vosk

# --- Paquetes de voz import en try/except para evitar crash si no instalados ---


from vosk import Model, KaldiRecognizer
import sounddevice as sd

import mediapipe as mp

# --- Utils y estructuras ---
Event = dict  # simple alias; cada evento tendrá {'type': 'gesture'|'speech', 'name':..., 'ts':..., ...}

event_q = queue.Queue()  # cola principal de eventos

# Gestor de estado / UI
state = {
    'mode_paint': True,
    'color': (0, 255, 0),
    'brush': 8,
    'drawing': False,
    'last_action_ts': 0,
}

# ----- VOICE THREADS -----
def speech_thread_vosk(stop_event):
    """Reconocimiento con VOSK (offline). Emite resultados parciales y finales."""
    if Model is None or sd is None:
        print("VOSK o sounddevice no disponible.")
        return
    model = Model(MODEL_VOSK_PATH)
    # Configura audio stream
    samplerate = 16000
    rec = KaldiRecognizer(model, samplerate)
    def callback(indata, frames, time_info, status):
        if stop_event.is_set():
            raise sd.CallbackStop()
        data = bytes(indata)  # convierte el buffer a bytes de forma segura
        if rec.AcceptWaveform(data):
            res = rec.Result()
            try:
                import json
                text = json.loads(res).get('text', '')
                if text.strip():
                    event_q.put({'type':'speech', 'text': text.lower(), 'ts': time.time(), 'conf': None})
                    print("[VOICE VOSK]", text)
            except Exception:
                pass
        # else partial -> opcional
    with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, dtype='int16',
                           channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)

# ----- GESTURE DETECTION (MediaPipe) -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalized_to_pixel(coord, w, h):
    return int(coord.x * w), int(coord.y * h)

def euclidean(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def fingers_up(landmarks, w, h, handedness='Right'):
    tips_ids = [4,8,12,16,20]
    pip_ids = [3,6,10,14,18]
    pts = [normalized_to_pixel(lm, w, h) for lm in landmarks]
    fingers = []
    if handedness == 'Right':
        fingers.append(pts[4][0] > pts[3][0])
    else:
        fingers.append(pts[4][0] < pts[3][0])
    for tip,pip in zip(tips_ids[1:], pip_ids[1:]):
        fingers.append(pts[tip][1] < pts[pip][1])
    return fingers

def classify_gesture(landmarks, w, h, handedness='Right'):
    fingers = fingers_up(landmarks,w,h,handedness)
    cnt = sum(fingers)
    pts = [normalized_to_pixel(lm,w,h) for lm in landmarks]
    hand_ref = euclidean(pts[0], pts[9]) + 1e-6
    thumb_idx_dist = euclidean(pts[4], pts[8]) / hand_ref
    PINCH_THRESH = 0.22
    if thumb_idx_dist < PINCH_THRESH:
        return 'PINCH', fingers
    if cnt == 5:
        return 'OPEN', fingers
    if cnt == 0:
        return 'FIST', fingers
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return 'VICTORY', fingers
    if fingers[1] and not any([fingers[2],fingers[3],fingers[4]]):
        return 'POINT', fingers
    if fingers[0] and not any(fingers[1:]):
        # extra check: pulgar hacia arriba
        if pts[4][1] < pts[0][1]:
            return 'THUMB_UP', fingers
    return f'{cnt}_FINGERS', fingers

# ----- FUSION & DECISION ENGINE -----
# Ventana temporal de fusión (s)
FUSION_WINDOW = 0.8

def fusion_engine_worker(stop_event):
    """
    Consume eventos desde event_q y aplica reglas compuestas.
    Mantiene una pequeña lista de eventos recientes para matching dentro de FUSION_WINDOW.
    """
    recent = []  # lista de eventos {..}
    while not stop_event.is_set():
        try:
            ev = event_q.get(timeout=0.1)
        except queue.Empty:
            # limpiar eventos viejos de recent
            tnow = time.time()
            recent = [e for e in recent if tnow - e['ts'] <= FUSION_WINDOW]
            continue
        # Añadir a recent y procesar
        recent.append(ev)
        # Purga vieja
        tnow = time.time()
        recent = [e for e in recent if tnow - e['ts'] <= FUSION_WINDOW]

        # Reglas: buscar combinaciones voice+gesture
        if ev['type'] == 'speech':
            text = ev['text']
            # ejemplo: detectar "color rojo" -> buscar gesto PINCH en recent
            if 'color' in text or any(w in text for w in ['rojo','azul','verde','amarillo']):
                # identificar nombre de color en la oración
                color_map = {'rojo':'(0,0,255)','azul':'(255,0,0)','verde':'(0,255,0)','amarillo':'(0,255,255)'}
                chosen = None
                for k in color_map:
                    if k in text:
                        chosen = k; break
                # buscar PINCH en recent (puede ser anterior o posterior)
                pinch = next((e for e in recent if e['type']=='gesture' and e['name']=='PINCH'), None)
                if pinch:
                    # ejecutar acción compuesta: set color
                    execute_action({'action':'set_color','color_name':chosen, 'source':'voice+pinch'})
                    continue
                else:
                    # si no hay pinch, podemos asignar color directamente (vío solo por voz)
                    execute_action({'action':'set_color','color_name':chosen, 'source':'voice_only'})
                    continue
            # otros comandos por voz
            if 'guardar' in text or 'guardar' in text or 'save' in text:
                # buscar OK gesture
                ok = next((e for e in recent if e['type']=='gesture' and e['name']=='OK'), None)
                if ok:
                    execute_action({'action':'save', 'source':'voice+ok'})
                else:
                    execute_action({'action':'save', 'source':'voice_only'})
                continue
            if 'borrar' in text or 'limpiar' in text or 'clear' in text:
                execute_action({'action':'clear','source':'voice'})
                continue

        if ev['type'] == 'gesture':
            # mapeo simple por gesto
            g = ev['name']
            if g == 'POINT':
                execute_action({'action':'start_paint','pos':ev.get('pos'),'source':'gesture'})
            elif g == 'FIST':
                # Si voice recientes contiene "borrar", priorizar voice; si no, limpiar
                recent_voice = next((e for e in recent if e['type']=='speech' and any(w in e['text'] for w in ['borrar','limpiar','clear'])), None)
                if recent_voice:
                    execute_action({'action':'clear','source':'voice+gesture'})
                else:
                    execute_action({'action':'clear','source':'gesture'})
            elif g == 'PINCH':
                execute_action({'action':'pinch','source':'gesture'})
            elif g == 'THUMB_UP':
                execute_action({'action':'increase_brush','source':'gesture'})
            elif g == 'VICTORY':
                execute_action({'action':'decrease_brush','source':'gesture'})

        # marcar task done
        event_q.task_done()

# ----- ACTION EXECUTOR (UI updates) -----
# este executor actualiza el canvas y dibuja overlays
CANVAS_W, CANVAS_H = 960, 720
canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

def execute_action(act):
    """Centraliza la ejecución de acciones con retroalimentación visual."""
    now = time.time()
    state['last_action_ts'] = now
    a = act.get('action')
    src = act.get('source')
    print("EXECUTE:", a, act)
    if a == 'set_color':
        name = act.get('color_name')
        map_colors = {'rojo':(0,0,255),'azul':(255,0,0),'verde':(0,255,0),'amarillo':(0,255,255)}
        if name in map_colors:
            state['color'] = map_colors[name]
    elif a == 'save':
        fname = f"multimodal_snapshot_{int(now)}.png"
        cv2.imwrite(fname, canvas)
        print("Saved", fname)
    elif a == 'clear':
        canvas[:] = 0
    elif a == 'start_paint':
        # activar estado de dibujado (seguido por la parte de gestures)
        state['drawing'] = True
    elif a == 'pinch':
        # cambiar paleta (ciclo)
        state['color'] = (int(time.time()*100) % 256, int(time.time()*50) % 256, int(time.time()*10) % 256)
    elif a == 'increase_brush':
        state['brush'] += 2
    elif a == 'decrease_brush':
        state['brush'] = max(1, state['brush'] - 2)
    # podrías emitir señales sonoras o visuales aquí

# ----- MAIN: captura camera, detecta gestos y pone eventos en cola -----
def main():
    cap = cv2.VideoCapture(0)
    CAP_W, CAP_H = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    # Thread stop events
    stop_event = threading.Event()
    # Inicia speech thread según backend
    if VOICE_BACKEND == 'vosk' and Model is not None:
        t_speech = threading.Thread(target=speech_thread_vosk, args=(stop_event,), daemon=True)
        t_speech.start()
    else:
        print("No speech backend iniciado. Ajusta VOICE_BACKEND o instala dependencias.")

    # Inicia fusion engine thread
    t_fusion = threading.Thread(target=fusion_engine_worker, args=(stop_event,), daemon=True)
    t_fusion.start()

    # MediaPipe hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.55, min_tracking_confidence=0.5)
    last_point = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)

            # Ajustar tamaño del canvas al final para mezclar (evitar error addWeighted)
            frame_resized = cv2.resize(frame, (CANVAS_W, CANVAS_H))

            if res.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(res.multi_hand_landmarks):
                    # handedness si está disponible
                    handedness_str = 'Right'
                    # extraer landmarks en pixeles
                    lm = hand_landmarks.landmark
                    gesture_name, fingers = classify_gesture(lm, CANVAS_W, CANVAS_H, handedness_str)
                    # Ubicación punta índice
                    pts = [normalized_to_pixel(l, CANVAS_W, CANVAS_H) for l in lm]
                    idx_tip = pts[8]
                    # Generar evento de gesto (incluir pos para acciones como dibujar)
                    g_event = {'type':'gesture','name':gesture_name,'ts':time.time(),'pos':idx_tip,'fingers':fingers}
                    event_q.put(g_event)

                    # Dibujar landmarks
                    mp_drawing.draw_landmarks(frame_resized, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Si gesture POINT y mode_paint -> dibujar en canvas
                    if gesture_name == 'POINT' and state['mode_paint']:
                        # trazar linea
                        if last_point is None:
                            last_point = idx_tip
                        cv2.line(canvas, last_point, idx_tip, state['color'], state['brush'])
                        last_point = idx_tip
                    else:
                        last_point = None
            else:
                last_point = None

            # Overlay: mezcla frame_resized y canvas
            overlay = cv2.addWeighted(frame_resized, 0.6, canvas, 0.4, 0)

            # UI: mostrar último evento y estado
            cv2.putText(overlay, f"Color: {state['color']}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
            cv2.putText(overlay, f"Brush: {state['brush']}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
            cv2.putText(overlay, f"Drawing: {state['drawing']}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

            cv2.imshow("Multimodal Voz+Gestos", overlay)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if k == ord('c'):
                canvas[:] = 0

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        time.sleep(0.2)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
