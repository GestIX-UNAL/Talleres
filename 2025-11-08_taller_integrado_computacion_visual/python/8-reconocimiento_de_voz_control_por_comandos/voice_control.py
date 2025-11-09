# -*- coding: utf-8 -*-
# Requisitos: SpeechRecognition, pyaudio, pyttsx3, pygame, numpy
import threading, queue, time, math, random
import pygame
import speech_recognition as sr
import pyttsx3

# ---------- Config ----------
LANG = "es-ES"
FEEDBACK_TTS = True
COMMANDS = {
    # texto -> función lógica básica
    "color rojo": ("color", (255, 64, 64)),
    "color azul": ("color", (64, 128, 255)),
    "color verde": ("color", (64, 200, 120)),
    "color blanco": ("color", (230, 230, 230)),
    "rotar": ("rotate_on", True),
    "parar": ("rotate_on", False),
    "luz arriba": ("bright_delta", +0.1),
    "luz abajo": ("bright_delta", -0.1),
    "más grande": ("scale_delta", +0.1),
    "más pequeño": ("scale_delta", -0.1),
    "reset": ("reset", True),
}

# ---------- Estado visual ----------
class SceneState:
    def __init__(self):
        self.color = (64, 128, 255)
        self.rotate_on = True
        self.brightness = 1.0        # factor multiplicativo
        self.scale = 1.0             # tamaño relativo
        self.angle = 0.0             # grados

    def apply(self, op, val):
        if op == "color":
            self.color = val
        elif op == "rotate_on":
            self.rotate_on = bool(val)
        elif op == "bright_delta":
            self.brightness = max(0.1, min(2.5, self.brightness + val))
        elif op == "scale_delta":
            self.scale = max(0.3, min(2.5, self.scale + val))
        elif op == "reset":
            self.__init__()

state = SceneState()

# ---------- TTS ----------
tts = pyttsx3.init()
def say(msg):
    if FEEDBACK_TTS:
        tts.say(msg); tts.runAndWait()

# ---------- Reconocimiento ----------
audio_q = queue.Queue()
r = sr.Recognizer()

def mic_worker():
    with sr.Microphone() as mic:
        r.adjust_for_ambient_noise(mic, duration=0.8)
        say("Voz lista")
        while True:
            try:
                audio = r.listen(mic, phrase_time_limit=3)
                audio_q.put(audio)
            except Exception:
                pass

def asr_worker():
    while True:
        audio = audio_q.get()
        try:
            # Online (simple y robusto). Si requieres offline, cambia a recognize_sphinx(language="es-ES")
            text = r.recognize_google(audio, language=LANG).lower().strip()
            print("Heard:", text)
            executed = False
            for key, (op, val) in COMMANDS.items():
                if key in text:
                    state.apply(op, val)
                    say(key)
                    executed = True
            if not executed:
                say("No entendido")
        except Exception as e:
            print("ASR error:", e)

# ---------- Visual (PyGame) ----------
def tint(col, factor):
    r,g,b = col
    return (int(max(0,min(255,r*factor))),
            int(max(0,min(255,g*factor))),
            int(max(0,min(255,b*factor))))

def run_visual():
    pygame.init()
    W, H = 900, 600
    screen = pygame.display.set_mode((W,H))
    pygame.display.set_caption("Voz → Comandos visuales")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # “Cubo” 2D estilizado a partir de un rectángulo rotado
    base_size = 160

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return

        screen.fill((15, 15, 20))
        if state.rotate_on:
            state.angle += 0.6

        size = int(base_size * state.scale)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surf, tint(state.color, state.brightness), (0,0,size,size), border_radius=16)
        # borde
        pygame.draw.rect(surf, (255,255,255,40), (3,3,size-6,size-6), width=3, border_radius=14)

        rot = pygame.transform.rotozoom(surf, state.angle, 1.0)
        rect = rot.get_rect(center=(W//2, H//2))
        screen.blit(rot, rect)

        # HUD
        text = f"Comandos: color rojo/azul/verde/blanco | rotar/parar | luz arriba/abajo | más grande/pequeño | reset"
        screen.blit(font.render(text, True, (220,220,220)), (20, 20))
        status = f"color={state.color} rotar={state.rotate_on} brillo={state.brightness:.2f} escala={state.scale:.2f}"
        screen.blit(font.render(status, True, (170,170,170)), (20, 48))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    threading.Thread(target=mic_worker, daemon=True).start()
    threading.Thread(target=asr_worker, daemon=True).start()
    run_visual()
