# 🎨 Ejercicio 3: Shaders Personalizados y Efectos

## 📋 Descripción
Este ejercicio demuestra la implementación de **shaders personalizados en GLSL** con **Six.js**, explorando diferentes técnicas de renderizado procedural, efectos visuales y interactividad en tiempo real.

## 🎯 Objetivos Cumplidos

### 1. **Shaders Básicos en GLSL**
- ✅ **Vertex Shaders**: Manipulación de geometría y posiciones de vértices
- ✅ **Fragment Shaders**: Control de colores, texturas y efectos de superficie
- ✅ **Uniforms**: Variables globales compartidas entre CPU y GPU
- ✅ **Varying**: Interpolación de datos entre vertex y fragment shader

### 2. **Color por Posición, Tiempo e Interacción**
- ✅ **Posición**: Gradientes basados en coordenadas UV y posición mundial
- ✅ **Tiempo**: Animaciones procedurales con funciones trigonométricas
- ✅ **Interacción**: Respuesta a movimiento del mouse y teclado

### 3. **Efectos Implementados**

#### **🌊 Water Shader (Shader 1)**
- Ondas complejas con múltiples frecuencias
- Interacción con posición del mouse
- Efectos de espuma y animación temporal
- Colores dinámicos basados en altura

#### **🎭 Toon Shading (Shader 2)**
- Cuantización de iluminación en bandas discretas
- Cálculo de luz difusa con normales
- Efectos de contorno con fresnel
- Iluminación no-fotorrealista

#### **🔲 Wireframe (Shader 3)**
- Efecto de malla procedural usando derivadas de pantalla
- Patrones de grid animados
- Mezcla entre wireframe y color sólido
- Geometría torus para mejor visualización

#### **🌈 Gradient Shader (Shader 4)**
- Múltiples patrones de gradiente superpuestos
- Gradiente radial desde el centro
- Ondas sinusoidales combinadas
- Interacción con posición del mouse

#### **🔥 Procedural Textures (Shader 5)**
- Funciones de ruido procedural implementadas en GLSL
- Múltiples octavas de ruido para complejidad
- Patrones de texturas generadas algorítmicamente
- Combinación de ondas sinusoidales y ruido

#### **🌀 UV Distortion (Shader 6)**
- Distorsión de coordenadas UV en tiempo real
- Patrones de tablero de ajedrez distorsionados
- Efectos de ondas desde la posición del mouse
- Transformaciones de textura dinámicas

### 4. **Sistema de Interactividad**

#### **🎮 Controles de Teclado**
- **Teclas 1-6**: Cambio entre diferentes shaders
- **Flechas ↑↓**: Ajustar intensidad de efectos (0.1 - 2.0)
- **Flechas ←→**: Modificar velocidad de animación (0.1 - 3.0)
- **Espacio**: Generar colores aleatorios en tiempo real

#### **🖱️ Interacción con Mouse**
- **Movimiento**: Influencia en shaders con efectos de proximidad
- **Drag**: Rotación de cámara con OrbitControls
- **Posición**: Uniforms actualizados en coordenadas normalizadas

#### **🎛️ Interfaz de Usuario**
- Panel de control con estilo moderno
- Botones para cada shader con indicador visual
- Sliders para ajuste en tiempo real de parámetros
- Instrucciones de uso integradas

## 🛠️ Implementación Técnica

### **Arquitectura de Shaders**
```javascript
// Uniforms globales compartidos
const globalUniforms = {
  uTime: { value: 0.0 },          // Tiempo para animaciones
  uMouse: { value: Vector2() },    // Posición del mouse
  uResolution: { value: Vector2() }, // Resolución de pantalla
  uIntensity: { value: 1.0 },     // Control de intensidad
  uSpeed: { value: 1.0 },         // Velocidad de animación
  uColor1: { value: Color() },    // Color primario
  uColor2: { value: Color() }     // Color secundario
};
```

### **Sistema de Geometrías Dinámicas**
- **Water**: `PlaneGeometry(4x4, 100x100)` - Alta resolución para ondas suaves
- **Toon**: `SphereGeometry(1.5, 32x32)` - Esfera para iluminación
- **Wireframe**: `TorusGeometry(1.2, 0.4)` - Torus para efectos de malla
- **Gradient/Procedural/Distortion**: `PlaneGeometry(3x3)` - Planes simples

### **Técnicas GLSL Avanzadas**

#### **Funciones de Ruido**
```glsl
// Generación de ruido procedural
float random(vec2 st) {
  return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(vec2 st) {
  // Interpolación bilinear de valores aleatorios
  // Implementación de ruido de Perlin simplificado
}
```

#### **Efectos de Wireframe**
```glsl
// Detección de bordes usando derivadas de pantalla
vec2 grid = abs(fract(vUv * 20.0) - 0.5) / fwidth(vUv * 20.0);
float line = min(grid.x, grid.y);
float wireStrength = 1.0 - step(1.0, line);
```

#### **Toon Shading**
```glsl
// Cuantización de iluminación
float NdotL = dot(vNormal, lightDir);
float toonLevel = floor(NdotL * 4.0) / 4.0;
toonLevel = clamp(toonLevel, 0.2, 1.0);
```

## 🎨 Aspectos Visuales

### **Paleta de Colores Dinámicos**
- **Colores base**: Azul océano (#0080ff) y Magenta (#ff0080)
- **Generación HSL**: `setHSL(random(), 0.8, 0.6)` para variedad cromática
- **Interpolación**: Mezclas suaves con función `mix()` de GLSL

### **Animaciones Procedurales**
- **Ondas sinusoidales**: Múltiples frecuencias para complejidad natural
- **Ruido temporal**: Variación orgánica en texturas procedurales
- **Efectos de pulso**: Respiración visual con `sin(time * frequency)`

## 🔧 Comparativa: Manual vs Procedural

### **Ventajas del Enfoque Procedural**
1. **Rendimiento**: Cálculos en GPU paralela vs CPU secuencial
2. **Memoria**: Sin necesidad de almacenar texturas grandes
3. **Escalabilidad**: Infinita resolución sin pérdida de calidad
4. **Interactividad**: Parámetros modificables en tiempo real
5. **Creatividad**: Efectos imposibles con texturas tradicionales

### **Desventajas**
1. **Complejidad**: Requiere conocimiento profundo de GLSL
2. **Debugging**: Herramientas limitadas para depuración de shaders
3. **Compatibilidad**: Variaciones entre diferentes GPUs
4. **Control artístico**: Menos control directo sobre el resultado final

## 📊 Métricas de Rendimiento
- **FPS objetivo**: 60fps estables
- **Resolución de geometría**: Optimizada según complejidad del shader
- **Uniformes**: Actualizados cada frame (16ms)
- **Memoria GPU**: Uso eficiente con geometrías reutilizables

## 🎯 Logros del Ejercicio

✅ **6 shaders únicos** implementados con técnicas diferentes  
✅ **Sistema de interactividad completo** con mouse y teclado  
✅ **UI moderna y funcional** con controles en tiempo real  
✅ **Efectos procedurales avanzados** usando GLSL puro  
✅ **Optimización de rendimiento** para hardware limitado  
✅ **Documentación técnica completa** con explicaciones detalladas  

## 🔗 Archivos del Proyecto
- **`index.html`**: Estructura HTML con import maps y UI
- **`main.js`**: Lógica principal y sistema de shaders
- **`README.md`**: Documentación completa (este archivo)

## 🚀 Para Ejecutar
1. Abrir terminal en el directorio del proyecto
2. Ejecutar: `python -m http.server 8083`
3. Navegar a: `http://localhost:8083`
4. Disfrutar de los shaders interactivos!

---

**Ejercicio completado como parte del Taller Integrado de Computación Visual 2025-II**  
**Implementación: Three.js + GLSL + JavaScript ES6**