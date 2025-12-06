# -*- coding: utf-8 -*-
"""
Subsistema Multimodal Simple - Voz + Gestos + EEG
Di colores/formas, usa gestos, ve efectos EEG en tiempo real
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import speech_recognition as sr
import threading
import time
import random
import os
import urllib.request

class MultimodalSystem:
    """Sistema multimodal simple y fácil de usar"""
    
    def __init__(self):
        # Colores disponibles
        self.colors = {
            'rojo': (0, 0, 255),
            'azul': (255, 0, 0), 
            'verde': (0, 255, 0),
            'amarillo': (0, 255, 255),
            'morado': (255, 0, 255),
            'naranja': (0, 165, 255)
        }
        
        # Formas disponibles
        self.shapes = ['circulo', 'cuadrado', 'triangulo', 'estrella']
        
        # Estado actual
        self.current_color = 'rojo'
        self.current_shape = 'circulo'
        self.filter_active = False
        self.eeg_level = 0.5
        self.last_gesture_time = 0  # Para evitar detecciones múltiples
        
        # Métricas simples
        self.start_time = time.time()
        self.frame_count = 0
        self.fps_values = []
        self.voice_detections = 0
        self.gesture_detections = 0
        self.eeg_changes = 0
        
        # Componentes
        self.voice_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.voice_active = True
        
        # Modelo MediaPipe Hand Landmarker
        self.setup_hand_detector()
        
        print("🎯 Sistema Multimodal Listo!")
        print("🎤 Di colores: rojo, azul, verde, amarillo, morado, naranja")
        print("🎤 Di formas: circulo, cuadrado, triangulo, estrella") 
        print("🤏 Gesto PINCH: activar/desactivar filtro")
        print("🧠 EEG simulado afecta intensidad visual")
    
    def setup_hand_detector(self):
        """Configura el detector de manos con modelo preentrenado"""
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("📥 Descargando modelo hand_landmarker.task...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            print("✅ Modelo descargado")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
    
    def listen_for_voice(self):
        """Escucha voz en segundo plano"""
        while self.voice_active:
            try:
                with self.microphone as source:
                    self.voice_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.voice_recognizer.listen(source, timeout=1, phrase_time_limit=2)
                
                text = self.voice_recognizer.recognize_google(audio, language='es-ES').lower()
                
                # Detectar colores
                for color in self.colors.keys():
                    if color in text:
                        self.current_color = color
                        print(f"🎨 Color cambiado a: {color}")
                        self.voice_detections += 1
                        break

                # Detectar formas
                for shape in self.shapes:
                    if shape in text:
                        self.current_shape = shape
                        print(f"🔷 Forma cambiada a: {shape}")
                        self.voice_detections += 1
                        break

            except:
                pass  # Ignorar errores de voz
            
            time.sleep(0.5)
    
    def detect_gesture(self, frame):
        """Detecta gestos usando modelo preentrenado"""
        # Evitar detecciones demasiado frecuentes
        current_time = time.time()
        if current_time - self.last_gesture_time < 1.0:  # 1 segundo de cooldown
            return False
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.hand_detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Dibujar landmarks
                self.draw_hand_landmarks(frame, hand_landmarks)
                
                # Detectar pinch (pulgar e índice cerca)
                if len(hand_landmarks) >= 9:
                    thumb_tip = hand_landmarks[4]
                    index_tip = hand_landmarks[8]
                    
                    # Convertir a coordenadas de frame
                    frame_height, frame_width = frame.shape[:2]
                    thumb_x = int(thumb_tip.x * frame_width)
                    thumb_y = int(thumb_tip.y * frame_height)
                    index_x = int(index_tip.x * frame_width)
                    index_y = int(index_tip.y * frame_height)
                    
                    # Calcular distancia euclidiana
                    distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                    
                    # Si están cerca (pinch), alternar filtro
                    if distance < 50:  # Umbral de distancia en pixels
                        self.filter_active = not self.filter_active
                        self.last_gesture_time = current_time
                        print(f"🤏 Filtro {'ACTIVADO' if self.filter_active else 'DESACTIVADO'}")
                        return True
                        
        return False
    
    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Dibuja los landmarks de la mano en el frame"""
        frame_height, frame_width = frame.shape[:2]
        
        # Dibujar puntos
        for landmark in hand_landmarks:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Dibujar conexiones
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
            (0, 5), (5, 6), (6, 7), (7, 8),  # Índice
            (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
            (0, 13), (13, 14), (14, 15), (15, 16),  # Anular
            (0, 17), (17, 18), (18, 19), (19, 20),  # Meñique
            (5, 9), (9, 13), (13, 17)  # Palma
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                
                start_point = (int(start.x * frame_width), int(start.y * frame_height))
                end_point = (int(end.x * frame_width), int(end.y * frame_height))
                
                cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
    
    def apply_visual_effects(self, frame):
        """Aplica efectos visuales basados en estado"""
        # Efecto de color
        color_bgr = self.colors[self.current_color]
        overlay = np.full_like(frame, color_bgr)
        
        # Mezclar con intensidad basada en EEG
        intensity = self.eeg_level
        frame = cv2.addWeighted(frame, 1 - intensity * 0.3, overlay, intensity * 0.3, 0)
        
        # Filtro adicional si está activo
        if self.filter_active:
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        
        # Dibujar forma
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        
        if self.current_shape == 'circulo':
            cv2.circle(frame, center, 80, (255, 255, 255), 3)
        elif self.current_shape == 'cuadrado':
            cv2.rectangle(frame, (center[0]-60, center[1]-60), (center[0]+60, center[1]+60), (255, 255, 255), 3)
        elif self.current_shape == 'triangulo':
            pts = np.array([[center[0], center[1]-70], [center[0]-70, center[1]+70], [center[0]+70, center[1]+70]])
            cv2.polylines(frame, [pts], True, (255, 255, 255), 3)
        elif self.current_shape == 'estrella':
            # Estrella simple de 5 puntas
            for i in range(5):
                angle = i * 72
                x1 = int(center[0] + 60 * np.cos(np.radians(angle)))
                y1 = int(center[1] + 60 * np.sin(np.radians(angle)))
                x2 = int(center[0] + 30 * np.cos(np.radians(angle + 36)))
                y2 = int(center[1] + 30 * np.sin(np.radians(angle + 36)))
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        return frame
    
    def update_eeg_simulation(self):
        """Simula EEG con cambios graduales"""
        change = random.uniform(-0.05, 0.05)
        self.eeg_level = np.clip(self.eeg_level + change, 0.0, 1.0)
    
    def draw_ui(self, frame):
        """Dibuja interfaz de usuario simple"""
        # Información en pantalla
        cv2.putText(frame, f"Color: {self.current_color.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors[self.current_color], 2)
        
        cv2.putText(frame, f"Forma: {self.current_shape.upper()}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Filtro: {'ON' if self.filter_active else 'OFF'}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if self.filter_active else (0, 0, 255), 2)
        
        cv2.putText(frame, f"EEG: {self.eeg_level:.2f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        # Barra de EEG
        bar_width = int(self.eeg_level * 200)
        cv2.rectangle(frame, (10, 130), (10 + bar_width, 140), (255, 0, 255), -1)
        cv2.rectangle(frame, (10, 130), (210, 140), (255, 255, 255), 1)
        
        # Instrucciones
        cv2.putText(frame, "Q: Salir | Di colores/formas | Pinch: filtro", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def save_metrics(self):
        """Guardar métricas en archivo de texto"""
        os.makedirs('metrics', exist_ok=True)
        
        if self.fps_values:
            avg_fps = sum(self.fps_values) / len(self.fps_values)
            min_fps = min(self.fps_values)
            max_fps = max(self.fps_values)
        else:
            avg_fps = min_fps = max_fps = 0
            
        duration = time.time() - self.start_time
        
        with open('metrics/multimodal_metrics.txt', 'w') as f:
            f.write("METRICAS DEL SUBSISTEMA MULTIMODAL\n")
            f.write("=" * 40 + "\n")
            f.write(f"Duracion sesion: {duration:.1f} segundos\n")
            f.write(f"Frames procesados: {self.frame_count}\n")
            f.write(f"FPS Promedio: {avg_fps:.2f}\n")
            f.write(f"FPS Minimo: {min_fps:.2f}\n")
            f.write(f"FPS Maximo: {max_fps:.2f}\n")
            f.write(f"Detecciones de voz: {self.voice_detections}\n")
            f.write(f"Detecciones de gestos: {self.gesture_detections}\n")
            f.write(f"Cambios de EEG: {self.eeg_changes}\n")
            f.write("\n")
            f.write("MODALIDADES ACTIVAS:\n")
            f.write("- Voz: Deteccion de colores y formas\n")
            f.write("- Gestos: Pinch para activar/desactivar filtro\n")
            f.write("- EEG: Simulacion con efectos visuales\n")
        
        print(f"📊 Metricas guardadas en metrics/multimodal_metrics.txt")
    
    def print_summary(self):
        """Imprimir resumen de métricas"""
        if self.fps_values:
            avg_fps = sum(self.fps_values) / len(self.fps_values)
        else:
            avg_fps = 0
            
        duration = time.time() - self.start_time
        
        print("\n" + "=" * 50)
        print("RESUMEN DEL SUBSISTEMA MULTIMODAL")
        print("=" * 50)
        print(f"Duracion: {duration:.1f} segundos")
        print(f"Frames procesados: {self.frame_count}")
        print(f"FPS Promedio: {avg_fps:.2f}")
        print(f"Detecciones Voz: {self.voice_detections}")
        print(f"Detecciones Gestos: {self.gesture_detections}")
        print(f"Cambios EEG: {self.eeg_changes}")
        print("=" * 50)
    
    def run(self):
        """Ejecuta el sistema multimodal"""
        # Iniciar hilo de voz
        voice_thread = threading.Thread(target=self.listen_for_voice)
        voice_thread.daemon = True
        voice_thread.start()

        # Abrir webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: No se puede abrir la webcam")
            return

        print("📹 Webcam abierta. Presiona 'q' para salir.")

        try:
            while True:
                start_frame_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1

                # Procesar gestos
                gesture_detected = self.detect_gesture(frame)
                if gesture_detected:
                    self.gesture_detections += 1

                # Actualizar EEG
                old_eeg_state = getattr(self, 'current_eeg_state', None)
                self.update_eeg_simulation()
                if hasattr(self, 'current_eeg_state') and old_eeg_state != self.current_eeg_state:
                    self.eeg_changes += 1

                # Aplicar efectos visuales
                frame = self.apply_visual_effects(frame)

                # Dibujar UI
                self.draw_ui(frame)

                # Calcular y guardar FPS
                frame_time = time.time() - start_frame_time
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_values.append(current_fps)

                # Mostrar frame
                cv2.imshow('Sistema Multimodal Simple', frame)

                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.close()
            self.voice_active = False

            # Guardar métricas finales
            self.save_metrics()
            self.print_summary()

            print("✅ Sistema finalizado")

def main():
    system = MultimodalSystem()
    system.run()

if __name__ == "__main__":
    main()
