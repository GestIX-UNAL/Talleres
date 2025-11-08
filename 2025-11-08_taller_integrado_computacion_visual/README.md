
# Taller Integral de Computación Visual

## 🌐 5. Visualización de imágenes y video 360°

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

### 📸 Evidencia gráfica (sugerida)

![Video actividad 5](/media/actividad-5/actividad_5.gif)

---

### 💡 Reflexión

**Aprendizajes:** manejo de texturas HDR y videos como `VideoTexture`, control de cámara con OrbitControls.

**Retos técnicos:** sincronización de texturas de video y rendimiento en navegadores.

**Mejoras posibles:** agregar puntos interactivos (hotspots), audio espacial y soporte para visores VR/WebXR.

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

### 📸 Evidencia gráfica (sugerida)

![Video actividad 7](/media/actividad-7/actividad_7.gif)

---

### 💡 Reflexión

Este experimento permitió comprender la **traducción de señales corporales a acciones digitales**.  
**Aprendizajes:** procesamiento de landmarks, suavizado temporal, y calibración de umbrales de detección.  
**Retos técnicos:** la variabilidad de iluminación y la velocidad de procesamiento en tiempo real.  
**Posibles mejoras:** detección de múltiples manos y uso de modelos de aprendizaje profundo para reconocimiento dinámico de gestos.

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

### 📸 Evidencia gráfica (sugerida)

![Video actividad 9](/media/actividad-9/actividad_9.gif)

---

### 💡 Reflexión

Combinar voz y gestos introduce **sinergia cognitiva** en la interacción hombre-máquina.  
**Aprendizajes:** uso de hilos para reconocimiento en paralelo, sincronización de eventos y arquitectura multimodal.  
**Retos técnicos:** latencia en la sincronización voz-gesto y manejo concurrente del micrófono y la cámara.  
**Mejoras futuras:** integrar un módulo de contexto para aprender patrones de interacción del usuario o comandos personalizados.