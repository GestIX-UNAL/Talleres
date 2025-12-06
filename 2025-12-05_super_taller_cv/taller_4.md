

# Taller Integral de Computación Visual Avanzada

## Objetivo General

Desarrollar experiencias visuales avanzadas que combinen visión por computador, aprendizaje profundo, visualización 3D e interacción multimodal.
El proyecto puede construirse como un **sistema unificado** o como un **conjunto de subsistemas complementarios**, cada uno enfocado en resolver un aspecto del pipeline visual.

---

## Modalidades de Ejecución

### 1. Sistema Conjunto Integrado

El equipo crea un **único sistema funcional** donde todos los módulos están conectados:

* El servidor de visión (YOLO, MediaPipe).
* La escena 3D responde visualmente a los datos.
* Las interacciones (voz, gestos, EEG) controlan parámetros del entorno.
* El dashboard recopila métricas y el sistema puede publicarse como una aplicación web o simulador interactivo completo.

**Ejemplo:**
Una instalación visual inteligente que detecta personas, reacciona a sus gestos o voz, muestra resultados en 3D y los publica en una web colaborativa.

---

### 2. Subsistemas Especializados

El proyecto se divide en **módulos o experiencias independientes**, pero con el mismo estándar técnico y visual.
Cada subsistema aborda un problema distinto, manteniendo coherencia en estética, documentación y métricas.

**Ejemplos de subsistemas:**

* Subsistema 1: Detección y segmentación inteligente (Python + YOLO + MediaPipe).
* Subsistema 2: Control multimodal (voz + gestos + EEG).
* Subsistema 3: Visualización 3D optimizada (Three.js + AR.js).
* Subsistema 4: Motion design interactivo y cinemática (Unity).
* Subsistema 5: Entrenamiento y comparación de modelos (CNN + Fine-Tuning).

Cada uno se entrega como módulo funcional con documentación, resultados visuales y código ejecutable.

---

## Estructura del Repositorio (para ambas modalidades)

```
yyyy-mm-dd_super_taller_cv/
├── unity/
├── threejs/
├── python/
│   ├── detection/
│   ├── training/
│   ├── mediapipe_voice/
│   ├── websockets_api/
│   ├── dashboards/
│   └── utils/
├── data/
├── web_shared/
├── results/
├── docs/
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── EVIDENCIAS.md
│   ├── METRICAS.md
│   ├── PROMPTS.md
│   └── RUTINAS_DEMO.md
└── .github/ o .gitlab/
```

---

## Módulos Técnicos y Actividades

### A. Percepción y Visión

* Implementar detección en tiempo real con YOLO.
* Integrar segmentación con SAM o DeepLab.
* Visualizar embeddings mediante CLIP + PCA/t-SNE.
* Exportar resultados como imágenes anotadas y JSON.

### B. Interacción Multimodal

* Detección de gestos con MediaPipe (manos, rostro o cuerpo).
* Reconocimiento y síntesis de voz con SpeechRecognition y Pyttsx3.
* Simulación de señales EEG y control mediante umbrales.
* Fusión de entradas (voz + gestos + EEG) para activar acciones visuales.

### C. Visualización 3D

* Escena principal en Three.js / React Three Fiber con overlays dinámicos.
* Implementación de modelos 3D interactivos o animados.
* Integración AR.js con marcadores personalizados.
* En Unity: cinemática directa (FK), inversa (IK), colisiones, partículas y transiciones animadas.

### D. Backend y Comunicación

* Servidor WebSocket para transmitir detecciones, métricas o comandos.
* Serialización en JSON, almacenamiento en CSV o base de datos local.
* Dashboard en Python con métricas de rendimiento (FPS, uso GPU/CPU).
* Visualización de eventos y estados en tiempo real.

### E. Deep Learning

* Entrenamiento de CNN desde cero (Keras o PyTorch).
* Aplicación de validación cruzada y análisis de métricas.
* Fine-tuning con modelos preentrenados (ResNet, MobileNet).
* Comparación entre modelos y presentación de resultados visuales.

### F. Optimización Visual

* Aplicar niveles de detalle (LOD) y compresión de texturas.
* Reducir polígonos y materiales redundantes.
* Controlar sombras e iluminación eficiente.
* Reportar FPS, tamaño total de recursos y latencia general.

### G. Publicación y Evidencias

* Consolidar resultados en una web o dashboard compartido.
* Generar y documentar evidencias visuales (capturas, GIFs, videos).
* Documentar código, dependencias y flujo de ejecución.
* Preparar demo reproducible para presentación final.

---

## Entregables Mínimos

* Detección y segmentación funcional.
* Interacción por voz y gestos.
* CNN entrenada y modelo fine-tuneado.
* Escenas 3D o AR.js funcionales.
* Dashboards con métricas y rendimiento.
* Video (30–60 s) y mínimo 6 GIFs.
* Documentación completa y commits en inglés.

---

## Demostración Final (sugerida)

1. Detección y segmentación activas con webcam.
2. Interacción por voz o gestos para cambiar cámara, luz o materiales.
3. Escena 3D con respuesta visual dinámica o animación de personaje.
4. Panel con métricas en vivo y gráficas de rendimiento.
5. Experiencia AR funcional o animación interactiva en Unity.
6. Visualización comparativa de CNN y modelo fine-tuneado.
7. Video final y repositorio documentado.

---

## Observaciones

* Si el grupo elige la **modalidad integrada**, todas las partes deben conectarse funcionalmente en un flujo único de entrada, procesamiento y visualización.
* Si el grupo elige la **modalidad de subsistemas**, cada equipo debe entregar un módulo autónomo completamente funcional, con interfaces claras y evidencias equivalentes.
* En ambos casos, la documentación, estándares técnicos y formato de entrega son los mismos.

---
