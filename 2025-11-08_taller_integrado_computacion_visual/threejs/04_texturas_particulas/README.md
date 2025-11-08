# 🎨 Ejercicio 4: Texturas Dinámicas y Partículas

## 📋 Descripción
Este ejercicio implementa un sistema avanzado de **texturas dinámicas y partículas sincronizadas** con **Three.js**, demostrando materiales reactivos, mapas animados, sistemas de partículas coordinados y eventos visuales complejos.

## 🎯 Objetivos Cumplidos

### 1. **Materiales Reactivos a Tiempo, Input y Sensores**
- ✅ **Tiempo**: Materiales que cambian propiedades automáticamente
- ✅ **Input Usuario**: Respuesta a mouse tracking y teclado en tiempo real
- ✅ **Sensores**: Reactividad a eventos y parámetros dinámicos

### 2. **Mapas Animados Implementados**

#### **🔥 Emissive Dinámico**
- Textura procedural generada en canvas HTML5
- Patrones de ondas concéntricas animadas
- Intensidad variable basada en parámetros del usuario
- Sincronización con tiempo global del sistema

#### **🌊 Normal Map Animado**
- Ondas sinusoidales para simulación de superficie líquida
- Actualización en tiempo real de coordenadas de normales
- Efectos de relieve dinámicos sobre la geometría

#### **📐 Offset UV Procedural**
- Shader personalizado con deformación de coordenadas UV
- Distorsión basada en funciones trigonométricas
- Interacción con posición del mouse en tiempo real

#### **🎲 Ruido Procedural**
- Implementación de ruido de Perlin en GLSL
- Múltiples octavas para complejidad visual
- Patrones cellulares dinámicos con variación temporal

### 3. **Sistemas de Partículas Sincronizados**

#### **Sistema Multi-Capa**
- **3 sistemas independientes**: 200, 150 y 100 partículas respectivamente
- **Colores sincronizados**: Responden al material activo de la esfera
- **Física básica**: Velocidades, edades, tamaños dinámicos
- **Reseteo automático**: Partículas que salen del rango se regeneran

#### **Sincronización Material-Partícula**
- **Emissive**: Partículas naranjas/rojas con intensidad variable
- **Normal**: Partículas azules con tonos fríos
- **UV**: Partículas verde-cyan con gradiente
- **Noise**: Colores procedurales basados en funciones trigonométricas

### 4. **Eventos Visuales Coordinados Shader + Partículas**

#### **💥 Explosión**
- **Material**: Aumento dramático de intensidad emissive (2.0x)
- **Partículas**: Velocidades aleatorias radiales explosivas
- **Duración**: 300ms con retorno gradual

#### **🌊 Onda de Shock**
- **Material**: Escalado de la esfera principal (1.3x)
- **Partículas**: Ondas sinusoidales basadas en distancia radial
- **Efecto**: Propagación física realista

#### **⚡ Pulso Energético**
- **Material**: Flash de color emissive a blanco puro
- **Partículas**: Movimiento hacia afuera desde centro
- **Sincronización**: 200ms de duración coordinada

#### **⛈️ Tormenta**
- **Material**: Rotación aleatoria de la geometría principal
- **Partículas**: Velocidades caóticas en todas las direcciones
- **Efecto**: Simulación de turbulencia atmosférica

## 🛠️ Implementación Técnica

### **Arquitectura del Sistema**
```javascript
// Uniforms globales para sincronización
const globalUniforms = {
  uTime: { value: 0.0 },        // Tiempo maestro
  uMouse: { value: Vector2() },  // Tracking mouse
  uIntensity: { value: 1.0 },   // Control intensidad
  uSpeed: { value: 1.0 }        // Velocidad global
};
```

### **Texturas Dinámicas Procedurales**
```javascript
// Generación en tiempo real con Canvas API
function updateEmissiveTexture() {
  const time = globalTime * 2;
  for (let x = 0; x < 256; x += 4) {
    for (let y = 0; y < 256; y += 4) {
      const wave = Math.sin((x + y) * 0.02 + time) * 0.5 + 0.5;
      const pulse = Math.sin(time * 3 + distance) * 0.5 + 0.5;
      // Renderizado procedural pixel por pixel
    }
  }
}
```

### **Sistema de Partículas Avanzado**
```javascript
class ParticleSystem {
  // Atributos por partícula
  positions: Float32Array(count * 3)  // XYZ coordinates
  velocities: Float32Array(count * 3) // Movement vectors  
  colors: Float32Array(count * 3)     // RGB values
  sizes: Float32Array(count)          // Individual scaling
  
  // Sincronización con material activo
  updateColors(materialType, globalTime)
}
```

### **Shaders GLSL Personalizados**

#### **UV Offset Vertex Shader**
```glsl
vec3 pos = position;
pos += normal * sin(position.x * 5.0 + uTime * 3.0) * 0.1 * uIntensity;
gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
```

#### **Ruido Procedural Fragment Shader**
```glsl
float noise(vec2 st) {
  // Implementación Perlin noise
  vec2 i = floor(st);
  vec2 f = fract(st);
  // Interpolación bilinear de valores aleatorios
}
```

## 🎮 Sistema de Interactividad

### **Controles de Teclado**
- **Teclas 1-4**: Cambio entre materiales dinámicos
- **E**: Evento explosión coordinado
- **W**: Onda de shock con propagación
- **Q**: Pulso energético sincronizado
- **S**: Tormenta caótica

### **Interacción Mouse**
- **Movimiento**: Uniforms uMouse para shaders UV
- **Tracking**: Coordenadas normalizadas (-1 a 1)
- **Influencia**: Distorsión de patrones en tiempo real

### **Controles UI**
- **Slider Intensidad**: 0.1 - 3.0 (control de efectos)
- **Slider Velocidad**: 0.1 - 5.0 (tempo de animaciones)
- **Botones Material**: Cambio visual con feedback
- **Botones Evento**: Triggers de efectos coordinados

## 📊 Rendimiento y Optimización

### **Métricas del Sistema**
- **Partículas totales**: 450 (optimizado para estabilidad)
- **Textura dinámica**: 256x256 pixels, actualización selectiva
- **Shaders**: 4 materiales únicos con uniforms compartidos
- **FPS objetivo**: 60fps estables

### **Optimizaciones Implementadas**
- **Culling automático**: Partículas fuera de rango se resetean
- **Texture updates**: Solo cuando material está activo
- **Buffer reuse**: Mismos uniforms para múltiples shaders
- **Reduced geometry**: Esferas optimizadas (64x64 subdivisiones)

## 🎨 Aspectos Visuales

### **Paleta de Colores por Material**
- **Emissive**: Naranjas/rojos cálidos (RGB: 1.0, 0.5, 0.2)
- **Normal**: Azules/violetas fríos (RGB: 0.5, 0.3, 1.0)  
- **UV**: Verde-cyan energéticos (RGB: 0.8, 1.0, 0.6)
- **Noise**: Procedurales dinámicos (funciones trigonométricas)

### **Efectos de Iluminación**
- **Ambient Light**: 0.4 intensidad para visibilidad base
- **Point Light**: (10,10,10) posición para highlights
- **Material Response**: PBR metalness/roughness variables

## 🔬 Técnicas Avanzadas

### **Sincronización Temporal**
- **Tiempo maestro**: Único reloj global para coherencia
- **Speed multiplier**: Permite control de velocidad unificado
- **Event timing**: Coordinación precisa entre sistemas

### **Interactividad Multimodal**
- **Mouse + Teclado**: Inputs simultáneos sin conflicto
- **UI + Shortcuts**: Doble método de control
- **Real-time feedback**: Cambios instantáneos visibles

### **Materiales Procedurales**
- **Canvas textures**: Generación HTML5 en tiempo real
- **GLSL noise**: Algoritmos de ruido en GPU
- **UV manipulation**: Distorsión geométrica dinámica

## 🔧 Comparativa: Estático vs Dinámico

### **Ventajas del Texturizado Dinámico**
1. **Memoria eficiente**: Sin almacenar texturas grandes
2. **Infinita variación**: Patrones únicos cada ejecución
3. **Interactividad**: Respuesta inmediata a input usuario
4. **Sincronización**: Coordinación perfecta entre elementos
5. **Escalabilidad**: Parámetros modificables en tiempo real

### **Desventajas vs Texturas Estáticas**
1. **CPU/GPU usage**: Cálculos continuos requeridos
2. **Complejidad código**: Mayor dificultad de implementación
3. **Debugging**: Difícil depuración de efectos procedurales
4. **Predictabilidad**: Menos control artístico directo

## 🎯 Logros del Ejercicio

✅ **4 materiales reactivos** únicos implementados  
✅ **3 sistemas de partículas** sincronizados perfectamente  
✅ **4 eventos coordinados** shader + partículas  
✅ **Texturas procedurales** generadas en tiempo real  
✅ **Interactividad multimodal** completa  
✅ **Performance optimizado** para hardware limitado  

## 🔗 Archivos del Proyecto
- **`index.html`**: HTML con UI completa y import maps
- **`main.js`**: Sistema completo de texturas dinámicas y partículas  
- **`README.md`**: Documentación técnica (este archivo)

## 🚀 Para Ejecutar
1. Abrir terminal en el directorio del proyecto
2. Ejecutar: `python -m http.server 8084`
3. Navegar a: `http://localhost:8084`
4. Experimentar con materiales y eventos coordinados!

---

**Ejercicio completado como parte del Taller Integrado de Computación Visual 2025-II**  
**Implementación**: Three.js + WebGL + Canvas API + GLSL + JavaScript ES6