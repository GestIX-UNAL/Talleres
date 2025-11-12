
# Taller Integral de Computación Visual

## 1. Materiales, luz y color (PBR y modelos cromáticos)

### 🎯 Concepto

Explorar materiales PBR (albedo, roughness, metalness, normal maps) y técnicas de iluminación para crear escenas coherentes en términos de color y respuesta a la luz. Se incluyen conversiones y justificaciones cromáticas (RGB/HSV/CIE Lab) para selección de paletas y contraste.

---

### ⚙️ Funcionalidades principales

- Creación de materiales PBR con mapas: albedo, roughness, metalness, normal.  
- Uso de HDRI para iluminación global y luces puntuales (key, fill, rim).  
- Visualización de canales de color y conversión a CIELAB para análisis de contraste.

---

### 🧰 Dependencias e instalación

Depende del entorno (Three.js/Unity/Python). Ejemplo para Three.js:

```bash
npm install three @react-three/fiber @react-three/drei
```

Para pruebas en Python (inspección de texturas):

```bash
pip install numpy opencv-python matplotlib
```

---

### ▶️ Evidencia esperada

Coloca capturas/GIFs en `threejs/01_pbr_luz_color/evidencias/gifs/` y refiérelas aquí. Ejemplo:

![PBR resultado](threejs/01_pbr_luz_color/evidencias/gifs/01_pbr_luz_color.gif)

---

## 2. Modelado procedural desde código

### 🎯 Concepto

Generar geometría por algoritmos: rejillas, espirales, superficies paramétricas y patrones fractales simples, controlados por parámetros en código para producir variaciones y animaciones.

---

### ⚙️ Funcionalidades principales

- Generación de mallas a partir de fórmulas (paramétricas, ruido Perlin/simplex).  
- Exportación a OBJ/GLTF para visualización.  
- Animaciones por modificación de vértices en tiempo real.

---

### 🧰 Dependencias e instalación

Para Python:

```bash
pip install numpy trimesh vedo
```

Para Three.js: librería base y utilidades (ver sección 1).

---

### ▶️ Evidencia esperada

Guarda capturas y/o modelos en `threejs/02_procedural/evidencias/`.

![Procedural resultado](threejs/02_procedural/evidencias/gifs/demo_modelado_procedural.gif)

---

## 3. Shaders personalizados y efectos

### 🎯 Concepto

Implementar shaders (GLSL/ShaderGraph) que modifiquen color y forma en función de posición, tiempo e interacción. Incluye toon shading, noise-based deformation y efectos UV.

---

### ⚙️ Funcionalidades principales

- Fragment y vertex shaders personalizados.  
- Parámetros uniformes para time, mouse/gestures y textures.  
- Efectos: toon, wireframe overlay, dissolving, normal perturbation.

---

### 🧰 Dependencias e instalación

Para proyectos web:

```bash
npm install three glslify
```

En Unity usar Shader Graph (LTS) o HLSL para shaders escritos.

---

### ▶️ Evidencia esperada

Capturas y GIFs en `threejs/03_shaders/evidencias/`.

![Shaders resultado](threejs/03_shaders/Evidencias/gift/03_shaders.gif)

---

## 4. Texturizado dinámico y partículas

### 🎯 Concepto

Materiales reactivos al tiempo y a la interacción con texturas animadas, mapas emisivos y sistemas de partículas que responden a entradas (audio, gestos, parámetros).

---

### ⚙️ Funcionalidades principales

- Texturas animadas y mezcla de mapas (emissive, normal, offset UV).  
- Sistemas de partículas sincronizados con eventos y materiales.  
- Exportación de secuencias para evidencia.

---

### 🧰 Dependencias e instalación

Dependencias según entorno; ejemplos:

```bash
# Three.js
npm install three @react-three/fiber @react-three/drei

# Python (para preprocesado de texturas)
pip install numpy opencv-python
```

---

### ▶️ Evidencia esperada

Guarda capturas/GIFs en `threejs/04_texturas_particulas/evidencias/`.

![Texturas & partículas](threejs/04_texturas_particulas/Evidencias/gift/04_texturas_particulas.gif)

---

## 5. Visualización de imágenes y video 360°

### 🎯 Concepto

Un **visor inmersivo** que permite explorar imágenes o videos 360° dentro de una esfera virtual usando **Three.js** y **React Three Fiber**, simulando una experiencia de realidad virtual básica en el navegador.

---

### ⚙️ Funcionalidades principales

- Renderiza una **imagen HDRI panorámica** o un **video 360°** como fondo.  
- Control de cámara con **OrbitControls** (rotación libre).  
- Botones para cambiar entre modo imagen y modo video.  
- Video 360° proyectado internamente sobre una esfera invertida (`BackSide`).

---

### 🧰 Dependencias e instalación

```bash
npm install three @react-three/fiber @react-three/drei
```

---

### ▶️ Ejecución

En un proyecto React con Vite o Create React App:

```bash
npm run dev
```

Coloca tus archivos multimedia en `/public`:
- `/bloem_field_sunrise_4k.hdr`  
- `/20257855-hd_1920_1080_60fps.mp4`

---

### 🧠 Fragmento clave

```jsx
<mesh ref={meshRef} scale={[-1, 1, 1]}>
  <sphereGeometry args={[500, 60, 40]} />
  <meshBasicMaterial side={THREE.BackSide} />
</mesh>
```

---

### 📸 Evidencia gráfica

![Video actividad 5](/2025-11-08_taller_integrado_computacion_visual/media/actividad-5/actividad_5.gif)

---

### 💡 Reflexión

**Aprendizajes:** manejo de texturas HDR y videos como `VideoTexture`, control de cámara con OrbitControls.

**Retos técnicos:** sincronización de texturas de video y rendimiento en navegadores.

**Mejoras posibles:** agregar puntos interactivos (hotspots), audio espacial y soporte para visores VR/WebXR.

---

## 6. Entrada e interacción (UI, input y colisiones)

### 🎯 Concepto

Captura de entradas (teclado, mouse, touch) y UI para manipular escenas: sliders, botones y eventos que disparan animaciones o cambian parámetros de materiales.

---

### ⚙️ Funcionalidades principales

- UI HTML/Canvas o Unity UI para controles en tiempo real.  
- Detección de colisiones/triggers para activar efectos.  
- Soporte para dispositivos táctiles y gamepads.

---

### 🧰 Dependencias e instalación

Ejemplo web:

```bash
npm install react @react-three/fiber leva
```

Ejemplo Unity: usar UI Toolkit o Canvas.

---

### ▶️ Evidencia esperada

Capturas e instrucciones en `threejs/06_interaccion/evidencias/`.

![Interacción resultado](threejs/06_interaccion/gift/06_interaccion.gif)

---


## 7. Gestos con cámara web (MediaPipe Hands)

### 🎯 Concepto

Un **experimento visual interactivo** basado en visión por computadora, donde el usuario puede **dibujar en el aire** usando los gestos de su mano detectados por la cámara. Cada gesto realiza una acción sobre el lienzo digital.

---

### ⚙️ Funcionalidades principales

- Detección en tiempo real de la mano con **MediaPipe Hands**.  
- Clasificación de gestos: ✋ `OPEN`, 👊 `FIST`, ☝️ `POINT`, ✌️ `VICTORY`, 👍 `THUMB_UP`, 🤏 `PINCH`, 👌 `OK`.  
- **Mapeo visual interactivo**:
  - `POINT`: dibujar con el índice.  
  - `PINCH`: cambiar color.  
  - `FIST`: limpiar pantalla.  
  - `OPEN`: pausar/reanudar.  
  - `THUMB_UP`: aumentar grosor del pincel.  
  - `VICTORY`: disminuir grosor.  
  - `OK`: guardar snapshot.

---

### 🧰 Dependencias e instalación

```bash
pip install opencv-python mediapipe numpy
```

---

### ▶️ Ejecución

```bash
python gestos_con_camara_web.py
```

- Presiona `ESC` para salir.  
- Presiona `C` para limpiar el lienzo manualmente.  

---

### 🧠 Fragmento clave

```python
if gesture == 'POINT' and mode_paint:
    pts_deque.append(idx_tip)
elif gesture == 'PINCH':
    color_idx = (color_idx + 1) % len(colors)
elif gesture == 'FIST':
    canvas[:] = 0
```

---

### 📸 Evidencia gráfica

![Video actividad 7](/2025-11-08_taller_integrado_computacion_visual/media/actividad-7/actividad_7.gif)

---

### 💡 Reflexión

Este experimento permitió comprender la **traducción de señales corporales a acciones digitales**.  
**Aprendizajes:** procesamiento de landmarks, suavizado temporal, y calibración de umbrales de detección.  
**Retos técnicos:** la variabilidad de iluminación y la velocidad de procesamiento en tiempo real.  
**Posibles mejoras:** detección de múltiples manos y uso de modelos de aprendizaje profundo para reconocimiento dinámico de gestos.

---
## 8. Reconocimiento de voz y control por comandos

### 🧰 Dependencias e instalación
```bash
pip install SpeechRecognition pyaudio pyttsx3 pygame numpy
```
---
### ▶️ Ejecución
```bash
python voice_control.py
```
Asegúrate de tener un micrófono conectado y configurado correctamente.
---
### 🧠 Fragmento clave
```python
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
```
---

### 💡 Reflexión

La implementación de reconocimiento de voz permite una interacción más natural y fluida con el sistema.  
**Aprendizajes:** integración de bibliotecas de reconocimiento de voz, manejo de excepciones y control visual mediante comandos de voz.  
**Retos técnicos:** variabilidad en la calidad del audio y la precisión del reconocimiento.  
**Mejoras posibles:** agregar soporte para múltiples idiomas y comandos personalizados.

---

### Evidencia gráfica
![Video actividad 8](media/actividad-8/actividad_8.gif)

---
## 9. Interfaces multimodales (voz + gestos)

### 🎯 Concepto

Extiende el experimento anterior integrando **comandos de voz (Vosk)** y **gestos simultáneos** para crear una **interfaz multimodal** donde ambos canales (visual y auditivo) se fusionan para controlar el lienzo.

---

### ⚙️ Funcionalidades principales

- Reconocimiento de voz **offline** con Vosk.  
- Detección de gestos con **MediaPipe Hands**.  
- **Fusión temporal** de eventos (`voz + gesto`) para ejecutar acciones combinadas.  
- **Canvas interactivo** que responde a comandos:
  - “Color rojo” + gesto `PINCH` → cambia color.  
  - “Guardar” + gesto `OK` → guarda snapshot.  
  - “Borrar” + `FIST` → limpia lienzo.  
  - `THUMB_UP` / `VICTORY` → aumenta o disminuye el pincel.  

---

### 🧰 Dependencias e instalación

```bash
pip install opencv-python mediapipe numpy vosk sounddevice
```

> ⚠️ Descarga el modelo de voz Vosk en español:
```bash
mkdir -p models
wget https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip vosk-model-small-es-0.42.zip -d models/
```

---

### ▶️ Ejecución

```bash
python main.py
```

Requiere cámara y micrófono activos.

---

### 🧠 Fragmento clave

```python
if 'color' in text:
    pinch = next((e for e in recent if e['name']=='PINCH'), None)
    if pinch:
        execute_action({'action':'set_color','color_name':chosen, 'source':'voice+pinch'})
```

---

### 📸 Evidencia gráfica
![Video actividad 9](/2025-11-08_taller_integrado_computacion_visual/media/actividad-9/actividad_9.gif)

---

### 💡 Reflexión

Combinar voz y gestos introduce **sinergia cognitiva** en la interacción hombre-máquina.  
**Aprendizajes:** uso de hilos para reconocimiento en paralelo, sincronización de eventos y arquitectura multimodal.  
**Retos técnicos:** latencia en la sincronización voz-gesto y manejo concurrente del micrófono y la cámara.  
**Mejoras futuras:** integrar un módulo de contexto para aprender patrones de interacción del usuario o comandos personalizados.

---

## 10. Simulación BCI (EEG sintético y control)

### 🎯 Concepto

Simulación de señales EEG sintéticas que permiten explorar patrones de actividad cerebral y su relación con el control visual.

---

### 🧰 Dependencias e instalación

```bash
pip install -r requirements.txt
```
---

```bash
pip install numpy scipy pygame
```

---

### ▶️ Ejecución

```bash
python eeg_sim.py
```

Asegúrate de tener los permisos necesarios para acceder a los dispositivos de entrada si es necesario.

---

### 🧠 Fragmento clave

```python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import butter, lfilter, welch
import pygame, random

# -------- Config EEG --------
FS = 256                   # Hz
WIN = 2.5                  # s por ventana
N  = int(FS*WIN)
ALPHA = (8,12)
BETA  = (13,30)
TH_ALPHA = 2.2             # umbral relativo simple
TH_BETA  = 2.0

# -------- Síntesis ----------
def synth_eeg(n, fs, a_amp=1.0, b_amp=0.8, noise=0.4):
  t = np.arange(n)/fs
  alpha = a_amp*np.sin(2*np.pi*10*t + np.random.rand()*2*np.pi)
  beta  = b_amp*np.sin(2*np.pi*20*t + np.random.rand()*2*np.pi)
  pink  = noise*np.cumsum(np.random.randn(n)); pink /= np.max(np.abs(pink)+1e-6)
  return alpha + beta + 0.4*pink

# ... (resto del código)
```

---

### 💡 Reflexión

La simulación de EEG permite explorar patrones de actividad cerebral y su relación con el control de dispositivos.  
**Aprendizajes:** generación de señales sintéticas y visualización de datos en tiempo real.  
**Retos técnicos:** modelar adecuadamente la variabilidad de las señales EEG reales.  
**Mejoras posibles:** integrar datos reales de EEG y aplicar técnicas de procesamiento de señales para análisis más profundos.

---

### Evidencia gráfica
![Video actividad 10](media/actividad-10/actividad_10.gif)

---

## 11. Espacios proyectivos y matrices de proyección
### 🎯 Concepto
Simulación de proyecciones en 3D utilizando cámaras perspectiva y ortográfica para visualizar la diferencia entre ambas.

---
### ⚙️ Funcionalidades principales
- Alternar entre cámara perspectiva y ortográfica con la tecla `[C]`.  
- Activar/desactivar el mapa de profundidad con la tecla `[D]`.  
- Visualización de un objeto 3D (Torus Knot) en un entorno iluminado.

---
### ▶️ Ejecución
Abre el archivo HTML en un navegador compatible con WebGL.

---
### 🧠 Fragmento clave
```javascript
const persp = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 100);
const ortho = new THREE.OrthographicCamera(-orthoH * innerWidth / innerHeight, orthoH * innerWidth / innerHeight, orthoH, -orthoH, 0.1, 100);
```
---
### 💡 Reflexión
La comparación entre proyecciones perspectiva y ortográfica permite entender cómo afectan la percepción de la profundidad y la escala en entornos 3D.  
**Aprendizajes:** manejo de diferentes tipos de cámaras en Three.js y su impacto visual.  
**Retos técnicos:** optimización del rendimiento al alternar entre cámaras.  
**Mejoras posibles:** agregar más geometrías y efectos visuales para enriquecer la experiencia.

---

### 📸 Evidencia gráfica

![Video actividad 11](media/actividad-11/actividad_11.gif)
