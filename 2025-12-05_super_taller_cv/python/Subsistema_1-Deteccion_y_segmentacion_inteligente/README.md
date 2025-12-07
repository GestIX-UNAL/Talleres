# Subsistema 1: Detección y Segmentación Inteligente

## 📋 Índice
1. [Descripción General](#descripción-general)
2. [Características](#características)
3. [Tecnologías y Librerías](#tecnologías-y-librerías)
4. [Requisitos del Sistema](#requisitos-del-sistema)
5. [Instalación](#instalación)
6. [Estructura del Proyecto](#estructura-del-proyecto)
7. [Guía de Uso Detallada](#guía-de-uso-detallada)
---

## 🎯 Descripción General

Sistema avanzado de visión por computador que combina:
- **YOLO** para detección de objetos en tiempo real
- **MediaPipe** para seguimiento de manos, pose y rostro
- **CLIP** para embeddings visuales y búsqueda semántica
- **Dashboard interactivo** con métricas en tiempo real

Este subsistema está diseñado para ser **completamente funcional e independiente**, con capacidad de procesamiento de webcam, imágenes y videos.

---

## ✨ Características

### 1. **Detección de Objetos con YOLO**
- Detección en tiempo real con webcam (30+ FPS)
- Soporte para 80+ clases de COCO dataset
- Procesamiento de imágenes y videos
- Exportación de anotaciones en JSON
- Configuración de umbrales de confianza e IoU

### 2. **Seguimiento Multi-Modal con MediaPipe**
- **Manos**: 21 puntos de referencia por mano, detección de gestos
- **Pose**: 33 puntos del cuerpo completo
- **Rostro**: 468 puntos del mesh facial (opcional)
- Reconocimiento de gestos básicos (puño, señalar, palma abierta, paz)
- Cálculo de métricas de postura

### 3. **Embeddings y Búsqueda con CLIP**
- Extracción de vectores de características visuales
- Búsqueda de imágenes por texto en lenguaje natural
- Visualización de espacios de embeddings (PCA y t-SNE)
- Cálculo de similitud coseno entre imágenes y texto

### 4. **Dashboard Interactivo**
- Visualización en tiempo real de detecciones
- Gráficas de rendimiento (FPS, latencia)
- Análisis de distribución de objetos
- Exportación de datos en múltiples formatos

---

## 🔧 Tecnologías y Librerías

### **Visión por Computador**
```
opencv-python >= 4.8.0          # Procesamiento de imágenes y video
opencv-contrib-python >= 4.8.0  # Módulos adicionales de OpenCV
ultralytics >= 8.0.0            # YOLO v8/v9 para detección de objetos
mediapipe >= 0.10.0             # Seguimiento de manos, pose y rostro
```

### **Deep Learning**
```
torch >= 2.0.0                  # PyTorch para modelos de deep learning
torchvision >= 0.15.0           # Utilidades de visión para PyTorch
clip @ git+https://...          # CLIP de OpenAI para embeddings
pillow >= 10.0.0                # Manipulación de imágenes
scikit-image >= 0.21.0          # Procesamiento científico de imágenes
```

### **Ciencia de Datos y Visualización**
```
numpy >= 1.24.0                 # Operaciones numéricas
pandas >= 2.0.0                 # Manipulación de datos tabulares
scikit-learn >= 1.3.0           # PCA, t-SNE y métricas
matplotlib >= 3.7.0             # Visualización de gráficas estáticas
seaborn >= 0.12.0               # Visualización estadística
plotly >= 5.14.0                # Gráficas interactivas
```

### **Dashboard y Web**
```
streamlit >= 1.28.0             # Framework para aplicaciones web
streamlit-webrtc >= 0.45.0      # Soporte de webcam en Streamlit
fastapi >= 0.104.0              # API REST (opcional)
uvicorn >= 0.24.0               # Servidor ASGI
websockets >= 12.0              # Comunicación en tiempo real
```

### **Utilidades**
```
tqdm >= 4.66.0                  # Barras de progreso
python-dotenv >= 1.0.0          # Variables de entorno
pyyaml >= 6.0                   # Configuración en YAML
openpyxl >= 3.1.0               # Exportación a Excel
```

---

## 💻 Requisitos del Sistema

### **Mínimos**
- **Sistema Operativo**: Windows 10/11, Ubuntu 20.04+, macOS 10.15+
- **CPU**: Intel Core i5 o equivalente
- **RAM**: 8 GB
- **Python**: 3.8 o superior
- **Webcam**: Cualquier cámara compatible con OpenCV

### **Recomendados**
- **CPU**: Intel Core i7/i9 o AMD Ryzen 7/9
- **RAM**: 16 GB o más
- **GPU**: NVIDIA con CUDA (GTX 1060 o superior) para mejor rendimiento
- **Espacio en disco**: 5 GB libres

### **Para GPU (Opcional pero Recomendado)**
```bash
# CUDA Toolkit 11.8 o superior
# cuDNN compatible
```

---

## 🚀 Instalación

### **Paso 1: Clonar el Repositorio**
```bash
git clone [URL_DEL_REPOSITORIO]
cd Subsistema_1-Deteccion_y_segmentacion_inteligente
```

### **Paso 2: Crear Entorno Virtual**

**En Linux/macOS:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### **Paso 3: Instalar Dependencias**

**Opción A: Instalación Completa**
```bash
pip install -r requirements.txt
```

**Opción B: Instalación con GPU (NVIDIA CUDA)**
```bash
# Primero instala PyTorch con soporte CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Luego el resto de dependencias
pip install -r requirements.txt
```

### **Paso 4: Descargar Modelos YOLO (Automático)**
Los modelos se descargan automáticamente la primera vez que los usas:
- `yolov8n.pt` - Nano (más rápido, menos preciso) ~6MB
- `yolov8s.pt` - Small (balance) ~22MB
- `yolov8m.pt` - Medium (más preciso) ~52MB
- `yolov8l.pt` - Large (más preciso, más lento) ~83MB

### **Paso 5: Verificar Instalación**
```bash
# Verificar que OpenCV funciona
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# Verificar YOLO
python -c "from ultralytics import YOLO; print('YOLO: OK')"

# Verificar MediaPipe
python -c "import mediapipe as mp; print('MediaPipe: OK')"

# Verificar Streamlit
streamlit --version
```

### **Paso 6: Crear Estructura de Directorios**
```bash
mkdir -p data/input data/output results/images results/videos results/embeddings exports snapshots
```

---

## 📁 Estructura del Proyecto

```
subsystem_1_detection/
│
├── detectors/                      # Módulos de detección
│   ├── __init__.py
│   ├── yolo_detector.py           # Detección con YOLO
│   ├── mediapipe_detector.py      # Seguimiento con MediaPipe
│   └── clip_embeddings.py         # Embeddings con CLIP
│
├── dashboard/                      # Dashboard web
│   └── streamlit_app.py           # Aplicación Streamlit
│
├── data/                           # Datos de entrada
│   ├── input/                     # Imágenes/videos de entrada
│   ├── output/                    # Resultados procesados
│   └── annotations/               # Anotaciones JSON
│
├── results/                        # Resultados generados
│   ├── images/                    # Imágenes anotadas
│   ├── videos/                    # Videos procesados
│   ├── embeddings/                # Embeddings y visualizaciones
│
├── exports/                        # Datos exportados
│   ├── detections_*.json          # Exportaciones JSON
│   ├── detections_*.csv           # Exportaciones CSV
│   └── detections_*.xlsx          # Exportaciones Excel
│
├── snapshots/                      # Capturas de webcam
│
├── demo.py                         # Script de demostración
├── requirements.txt                # Dependencias
├── README.md                       # Este archivo
└── .gitignore
```

---

## 📖 Guía de Uso Detallada

### **A. Dashboard Web (Streamlit) - Uso Completo**

#### **1. Iniciar el Dashboard**
```bash
# Desde la raíz del proyecto
streamlit run dashboard/streamlit_app.py
```

Esto abrirá automáticamente tu navegador en `http://localhost:8501`

#### **2. Interfaz del Dashboard**

##### **Barra Lateral (Configuración)**

**a) Modo de Detección:**
- **YOLO Object Detection**: Detecta 80+ tipos de objetos (personas, vehículos, animales, etc.)
- **MediaPipe Tracking**: Sigue manos, cuerpo y rostro
- **Combined**: Usa ambos simultáneamente

**b) Configuración de Modelos:**

Para **YOLO**:
- **Model Size**: Selecciona el tamaño del modelo
  - `yolov8n.pt` → Más rápido (30+ FPS), menos preciso
  - `yolov8s.pt` → Balance (25 FPS), buena precisión ⭐ **Recomendado**
  - `yolov8m.pt` → Más lento (15 FPS), mejor precisión
  - `yolov8l.pt` → Más lento (10 FPS), máxima precisión
  
- **Confidence Threshold** (0.0 - 1.0): 
  - Valor bajo (0.3): Detecta más objetos, más falsos positivos
  - Valor alto (0.7): Solo detecciones muy seguras, puede perder objetos
  - **Recomendado**: 0.5

- **IoU Threshold** (0.0 - 1.0):
  - Controla la eliminación de cajas duplicadas
  - **Recomendado**: 0.45

Para **MediaPipe**:
- **Track Hands**: ✅ Activa detección de manos y gestos
- **Track Pose**: ✅ Activa seguimiento de pose corporal
- **Track Face**: ☐ Activa mesh facial (468 puntos, consume más CPU)
- **Min Detection Confidence**: Umbral de confianza (recomendado: 0.5)

**c) Fuente de Entrada:**
- **Webcam**: Detección en tiempo real
- **Upload Image**: Procesa una imagen
- **Upload Video**: Procesa un video completo

**d) Configuración de Rendimiento:**
- **Device**: 
  - `cpu` → Compatible con todos, más lento
  - `cuda` → Requiere GPU NVIDIA, 3-5x más rápido

#### **3. Pestaña: 📹 Live Detection**

##### **Usando Webcam:**

**Paso a paso:**

1. **Selecciona el modo**: Ejemplo: "YOLO Object Detection"

2. **Configura parámetros** en la barra lateral:
   ```
   Model Size: yolov8s.pt
   Confidence: 0.5
   IoU: 0.45
   Device: cpu
   ```

3. **Selecciona "Webcam"** en Input Source

4. **Configura Camera ID**: 
   - 0 = Webcam por defecto
   - 1 = Segunda cámara (si existe)

5. **Click en "▶️ Start"**: 
   - Se inicializará el detector (tarda 2-5 segundos la primera vez)
   - Verás el mensaje "✅ YOLO detector initialized!"
   - Comenzará a mostrar el video en tiempo real

6. **Observa la detección**:
   - Cajas de colores alrededor de objetos detectados
   - Etiquetas con nombre y confianza
   - FPS y número de objetos en la esquina

7. **Panel lateral derecho muestra**:
   - **FPS**: Cuadros por segundo en tiempo real
   - **Objects**: Número de objetos en el frame actual
   - **Latency**: Tiempo de procesamiento en milisegundos
   - **Frame**: Número de frame procesado

8. **Tabla de conteo**: 
   - Muestra cuántos objetos de cada clase se han detectado
   - Ejemplo: "person: 45, car: 12, dog: 3"

9. **Botón "📸 Snapshot"**: 
   - Guarda el frame actual en `snapshots/snapshot_[timestamp].jpg`

10. **Click en "⏹️ Stop"** para detener

##### **Subiendo una Imagen:**

1. **Selecciona "Upload Image"** en Input Source

2. **Click en "Browse files"**

3. **Selecciona una imagen** (.jpg, .jpeg, .png)

4. **Procesamiento automático**:
   - Se detectarán objetos/personas/poses
   - Resultado se muestra inmediatamente
   
5. **Expandir "🔍 Detection Details"** para ver:
   ```json
   {
     "frame_id": 0,
     "objects": [
       {
         "label": "person",
         "confidence": 0.95,
         "bbox": [100, 150, 300, 450]
       }
     ],
     "fps": 25.3
   }
   ```

##### **Procesando un Video:**

1. **Selecciona "Upload Video"**

2. **Sube tu video** (.mp4, .avi, .mov)

3. **Click "Process Video"**

4. **Espera el procesamiento**:
   - Verás barra de progreso
   - Cada frame se procesa y anota
   
5. **Resultado**:
   - Video procesado en `results/videos/detected_[nombre].mp4`
   - Anotaciones JSON en `results/videos/detected_[nombre].json`

#### **4. Pestaña: 📊 Metrics**

Aquí verás el análisis completo de rendimiento:

**Métricas Principales** (4 tarjetas superiores):
- **Avg FPS**: FPS promedio de toda la sesión
- **Total Frames**: Cuántos frames se procesaron
- **Total Objects**: Suma de todos los objetos detectados
- **Unique Classes**: Cuántas clases diferentes se detectaron

**Gráfica: FPS Over Time**
- Línea temporal mostrando FPS en cada frame
- Útil para identificar caídas de rendimiento
- Hover sobre la línea para ver valores exactos

**Gráficas: Object Class Distribution**
- **Gráfica de Barras**: Muestra conteo de cada clase
- **Gráfica de Pastel**: Proporción porcentual de cada clase
- Colores dinámicos para cada categoría

#### **5. Pestaña: 🔍 Analysis**

Análisis detallado de las detecciones:

**Detection Timeline:**
- Gráfica de objetos detectados por frame
- Identifica patrones (cuándo hay más/menos objetos)

**Detailed Statistics (2 paneles):**

**Panel Izquierdo - FPS Statistics:**
```json
{
  "Mean": 28.45,      // FPS promedio
  "Median": 29.12,    // FPS mediano
  "Min": 18.23,       // FPS mínimo (peor caso)
  "Max": 32.87,       // FPS máximo (mejor caso)
  "Std Dev": 3.21     // Desviación estándar
}
```

**Panel Derecho - Detection Statistics:**
```json
{
  "Total Detections": 1847,        // Total de objetos
  "Avg per Frame": 3.2,            // Promedio por frame
  "Max per Frame": 8,              // Máximo en un frame
  "Frames with Objects": 573       // Frames con detecciones
}
```

#### **6. Pestaña: 💾 Export**

Exporta todos los datos recopilados:

**Formatos disponibles:**
- **JSON**: Estructura completa con metadatos
- **CSV**: Tabla simple para análisis
- **Excel**: Formato XLSX con columnas organizadas

**Opciones:**
- ☑️ **Include Performance Metrics**: Añade FPS y tiempos
- ☐ **Include Annotated Images**: (próximamente)

**Proceso:**
1. Selecciona formato deseado
2. Marca opciones adicionales
3. Click "📥 Export Data"
4. Espera mensaje "✅ Export completed!"
5. Click "⬇️ Download File" para descargar

**Ubicación de archivos:**
- `exports/detections_[timestamp].json`
- `exports/detections_[timestamp].csv`
- `exports/detections_[timestamp].xlsx`

**Vista previa:**
- Muestra los últimos 10 frames procesados
- Tabla con Frame ID, Objects, FPS, Latency

---

### **B. Script Demo (Terminal) - Uso Detallado**

El script `demo.py` ofrece acceso directo por línea de comandos.

#### **1. Ejecución Rápida (Sin Argumentos)**
```bash
python demo.py
```
- Inicia directamente la **webcam con YOLO**
- Presiona **'q'** para salir
- Presiona **'s'** para guardar snapshot

#### **2. Modos Disponibles**

##### **a) YOLO con Webcam**
```bash
python demo.py yolo-webcam
```

**Con configuración personalizada:**
```bash
python demo.py yolo-webcam --model yolov8s.pt --conf 0.6
```

**Qué hace:**
- Abre tu webcam predeterminada
- Muestra video con detecciones en tiempo real
- Presiona 'q' para salir
- Presiona 's' para guardar frame actual
- Al terminar, imprime métricas:
  ```
  📊 Performance Metrics:
    Average FPS: 28.34
    Total Frames: 1245
    Avg Detections: 2.8
  ```

##### **b) YOLO con Imagen**
```bash
python demo.py yolo-image --input ruta/a/imagen.jpg
```

**Ejemplo completo:**
```bash
python demo.py yolo-image --input data/input/street.jpg --model yolov8m.pt --conf 0.5
```

**Qué hace:**
1. Carga la imagen
2. Ejecuta detección
3. Guarda resultado anotado en `results/images/detected_street.jpg`
4. Guarda JSON en `results/images/detected_street.json`
5. Muestra imagen en ventana
6. Imprime resumen:
   ```
   📋 Detection Summary:
     Objects detected: 8
     FPS: 18.67
     Object counts:
       - person: 4
       - car: 3
       - bicycle: 1
   ```

##### **c) YOLO con Video**
```bash
python demo.py yolo-video --input ruta/a/video.mp4
```

**Ejemplo:**
```bash
python demo.py yolo-video --input data/input/traffic.mp4 --model yolov8s.pt --conf 0.5
```

**Qué hace:**
1. Abre el video
2. Procesa cada frame con detección
3. Muestra progreso: "Progress: 45.2%"
4. Guarda video anotado en `results/videos/detected_traffic.mp4`
5. Guarda JSON con todas las detecciones
6. Presiona 'q' para cancelar
7. Al terminar, imprime métricas completas

**Tiempo estimado:**
- Video de 1 minuto (30 FPS = 1800 frames)
- Con CPU: ~2-3 minutos
- Con GPU: ~30-45 segundos

##### **d) MediaPipe con Webcam**
```bash
python demo.py mediapipe
```

**Qué hace:**
- Detecta **manos** (dibuja 21 puntos por mano)
- Detecta **pose corporal** (33 puntos del cuerpo)
- Reconoce gestos básicos:
  - 👊 Puño cerrado → "fist"
  - ☝️ Un dedo → "pointing"
  - ✌️ Dos dedos → "peace"
  - 🖐️ Mano abierta → "open_palm"
- Muestra etiqueta de mano (Left/Right)
- Calcula métricas de postura
- Presiona 'q' para salir

##### **e) CLIP Embeddings**
```bash
python demo.py clip --input directorio/con/imagenes/
```

**Ejemplo:**
```bash
python demo.py clip --input data/input/
```

**Qué hace:**
1. Carga modelo CLIP (tarda ~10 segundos la primera vez)
2. Procesa todas las imágenes del directorio
3. Genera embeddings (vectores de 512 dimensiones)
4. Crea visualización PCA → `results/embeddings/pca_visualization.png`
5. Crea visualización t-SNE → `results/embeddings/tsne_visualization.png`
6. Guarda embeddings → `results/embeddings/embeddings.npy`
7. Realiza búsqueda demo con queries:
   ```
   Query: 'a person walking'
     1. person_street_01.jpg     (similarity: 0.847)
     2. walking_park.jpg         (similarity: 0.783)
     3. pedestrian.jpg           (similarity: 0.721)
   ```

**Requisitos:**
- Mínimo 3 imágenes en el directorio
- Formatos: .jpg, .jpeg, .png

##### **f) Modo Combinado**
```bash
python demo.py combined
```

**Qué hace:**
- Ejecuta YOLO + MediaPipe simultáneamente
- Presiona:
  - **'1'** → Solo YOLO
  - **'2'** → Solo MediaPipe
  - **'3'** → Ambos (modo combinado)
  - **'q'** → Salir
- Muestra modo actual en pantalla
- Útil para comparar rendimiento

##### **g) Ejecutar Todos los Demos**
```bash
python demo.py all --input data/input/
```

**Qué hace:**
1. Ejecuta demo de YOLO webcam
2. Espera a que cierres (q)
3. Ejecuta demo de MediaPipe
4. Espera a que cierres (q)
5. Ejecuta demo de CLIP (si proporcionaste --input)
6. Muestra todas las métricas

#### **3. Argumentos Disponibles**

```bash
python demo.py [MODO] [OPCIONES]
```

**Modos:**
- `yolo-webcam` - YOLO con webcam
- `yolo-image` - YOLO con imagen
- `yolo-video` - YOLO con video
- `mediapipe` - MediaPipe webcam
- `clip` - Embeddings CLIP
- `combined` - YOLO + MediaPipe
- `all` - Todos los demos

**Opciones:**
- `--input PATH` - Ruta a archivo/directorio
- `--model PATH` - Modelo YOLO (default: yolov8n.pt)
- `--conf FLOAT` - Umbral de confianza (default: 0.5)

