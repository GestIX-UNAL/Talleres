# Modelado Procedural desde Código - Three.js

## Concepto del proyecto

Este experimento visual demuestra las técnicas fundamentales del modelado procedural mediante algoritmos de generación de geometría en tiempo real. El proyecto explora diferentes enfoques para crear estructuras complejas usando código en lugar de modelado manual, incluyendo patrones geométricos, fractales recursivos y manipulación directa de vértices.

La experiencia permite alternar entre cinco tipos de algoritmos procedurales diferentes, cada uno demostrando una técnica específica de generación geométrica, culminando en una comparación visual directa entre modelado procedural y manual.

## Herramientas y entorno

- **Three.js v0.161**: Motor de renderizado 3D WebGL
- **JavaScript ES6**: Programación algorítmica y recursiva
- **BufferGeometry**: Manipulación directa de vértices
- **HTML5 + CSS3**: Interfaz interactiva de control
- **WebGL Shaders**: Renderizado de geometría procedural

## Descripción de módulos aplicados

### A. Generación de geometría por algoritmos

#### 1. Rejilla Procedural
- **Técnica**: Bucles anidados con variaciones algorítmicas
- **Algoritmo**: Grid 11x11 con alturas aleatorias y colores HSL dinámicos
- **Características**: 121 cubos con propiedades procedurales únicas
```javascript
for(let x=-5; x<=5; x++){
  for(let z=-5; z<=5; z++){
    const height = Math.random() * 2 + 0.5;
    const color = new THREE.Color().setHSL((height / 3), 0.8, 0.6);
  }
}
```

#### 2. Espirales Paramétricas
- **Técnica**: Fórmulas matemáticas para posicionamiento 3D
- **Algoritmos implementados**:
  - Espiral logarítmica con 300 esferas variables
  - Doble hélice DNA con conectores transversales
- **Características**: Rotación sincronizada y colores interpolados

#### 3. Fractales Recursivos
- **Técnica**: Generación recursiva con condiciones de parada
- **Algoritmos implementados**:
  - Árbol fractal 3D con 6 niveles de profundidad
  - Triángulo de Sierpinski con subdivisión recursiva
- **Características**: Auto-similitud y complejidad emergente

#### 4. Manipulación de Vértices
- **Técnica**: BufferGeometry con modificación directa de attributes
- **Algoritmo**: Terreno procedural con ondas combinadas
- **Características**: Vertex colors basados en altura, normales recalculadas

### B. Bucles y recursión para patrones espaciales

#### Bucles Anidados
```javascript
// Rejilla con patrones complejos
for(let x=-5; x<=5; x++){
  for(let z=-5; z<=5; z++){
    // Generación procedural por posición
  }
}
```

#### Recursión Geométrica
```javascript
function createFractalTree(position, direction, length, depth, group) {
  if (depth <= 0 || length < 0.1) return; // Condición de parada
  
  // Crear rama actual
  // Recursión para ramas hijas
  for(let i = 0; i < numBranches; i++) {
    createFractalTree(endPoint, newDirection, newLength, depth - 1, group);
  }
}
```

#### Recursión Fractal (Sierpinski)
```javascript
function sierpinski(p1, p2, p3, depth) {
  if(depth <= 0) return;
  
  // Calcular puntos medios
  const m1 = p1.clone().lerp(p2, 0.5);
  
  // Recursión en triángulos exteriores
  sierpinski(p1, m1, m3, depth - 1);
  sierpinski(m1, p2, m2, depth - 1);
  sierpinski(m3, m2, p3, depth - 1);
}
```

### C. Modificación de vértices y transformaciones dinámicas

#### Terreno Procedural
```javascript
const geometry = new THREE.PlaneGeometry(8, 8, 32, 32);
const positions = geometry.attributes.position;

for(let i = 0; i < positions.count; i++) {
  const x = positions.getX(i);
  const z = positions.getZ(i);
  
  // Función de altura procedural (combinación de ondas)
  const height = 
    Math.sin(x * 0.5) * Math.cos(z * 0.5) * 2 +
    Math.sin(x * 1.2) * 0.5 +
    Math.sin(z * 0.8) * 0.3;
  
  positions.setY(i, height);
}
```

#### Animaciones Dinámicas
- **Deformación senoidal**: Rejilla con ondas temporales
- **Rotaciones procedurales**: Basadas en posición y tiempo
- **Morphing de terreno**: Vertex animation en tiempo real

### D. Comparativa: modelado por código vs modelado manual

#### Modelado Procedural (Izquierda)
- **50 elementos** generados algorítmicamente
- Posiciones basadas en espiral logarítmica
- Rotaciones y colores procedurales
- **Ventajas**: Escalabilidad, variación automática, parametrización

#### Modelado Manual (Derecha)  
- **5 formas básicas** colocadas manualmente
- Posiciones hardcodeadas
- Propiedades fijas definidas en array
- **Ventajas**: Control preciso, predictibilidad

## Algoritmos técnicos implementados

### 1. Generación de Patrones Espaciales
```javascript
// Espiral logarítmica
const angle = (i / 50) * Math.PI * 4;
const radius = i * 0.1;
const position = new THREE.Vector3(
  Math.cos(angle) * radius,
  Math.sin(i * 0.3) * 2,
  Math.sin(angle) * radius
);
```

### 2. Fractales con Control de Profundidad
```javascript
// Control recursivo con probabilidad
const numBranches = 2 + Math.floor(Math.random() * 2);
for(let i = 0; i < numBranches; i++) {
  const newLength = length * (0.6 + Math.random() * 0.2);
  createFractalTree(endPoint, newDirection, newLength, depth - 1, group);
}
```

### 3. Vertex Manipulation con Colors
```javascript
// Asignación de colores por vértice basada en altura
const colors = [];
for(let i = 0; i < positions.count; i++) {
  const height = positions.getY(i);
  const normalizedHeight = (height + 3) / 6;
  const color = new THREE.Color().setHSL(normalizedHeight * 0.6, 0.8, 0.5);
  colors.push(color.r, color.g, color.b);
}
geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
```

## Controles interactivos

### Navegación por Teclado
- **Teclas 1-5**: Modos específicos (Grid, Spiral, Fractal, Vertices, Comparison)
- **Tecla 0**: Vista panorámica de todos los algoritmos
- **Tecla N**: Navegación secuencial entre modos

### Interfaz Gráfica
- **Botones UI**: Selección directa de algoritmos
- **Indicadores**: Modo actual y descripción técnica
- **Mouse Controls**: OrbitControls para navegación 3D libre

## Características técnicas

### Performance
- **Geometría optimizada**: LOD según complejidad algorítmica
- **Culling automático**: Grupos visibles según modo activo
- **Animaciones eficientes**: Transform caching y dirty flags

### Algoritmos Matemáticos
- **Trigonometría**: Funciones sin/cos para patrones circulares
- **Interpolación**: Lerp para transiciones suaves
- **Ruido procedural**: Combinación de ondas para terrenos

## Ventajas del Modelado Procedural

### Escalabilidad
- **Parametrización**: Cambio de parámetros genera variaciones infinitas
- **Automatización**: Generación masiva sin intervención manual
- **Consistencia**: Reglas algorítmicas garantizan coherencia

### Flexibilidad
- **Iteración rápida**: Ajustes de código vs remodeling manual
- **Variación controlada**: Randomness dentro de parámetros definidos
- **Reutilización**: Algoritmos aplicables a diferentes contextos

### Limitaciones
- **Curva de aprendizaje**: Requiere conocimiento matemático/algorítmico
- **Control artístico**: Menos control granular que modelado manual
- **Predictibilidad**: Resultados pueden ser menos orgánicos

## Evidencias técnicas

### Algoritmos Demostrados
1. **Bucles anidados**: Rejilla 11x11 con 121 elementos únicos
2. **Fórmulas paramétricas**: Espirales matemáticamente precisas
3. **Recursión profunda**: Árbol fractal con 6 niveles de subdivisión
4. **Vertex manipulation**: 1024 vértices modificados individualmente
5. **Comparación directa**: Procedural (50 elementos) vs Manual (5 elementos)

### Métricas de Complejidad
- **Rejilla**: O(n²) para grid n×n
- **Fractal**: O(b^d) donde b=ramas, d=profundidad
- **Vértices**: O(n) para n vértices del mesh
- **Total**: ~2000+ elementos generados proceduralmente

## Reflexión técnica

### Aprendizajes clave
- **Algoritmos recursivos**: Implementación práctica de fractales 3D
- **Buffer manipulation**: Acceso directo a vertex attributes en WebGL
- **Pattern generation**: Traducción de fórmulas matemáticas a geometría 3D
- **Performance optimization**: Gestión eficiente de miles de objetos

### Retos superados
- **Memory management**: Disposal correcto de geometrías complejas
- **Recursion depth**: Control de stack overflow en fractales profundos
- **Real-time animation**: Vertex morphing sin degradación de fps
- **UI synchronization**: Estado coherente entre controles y visualización

### Mejoras potenciales
- **GPU Compute**: Traslado de algoritmos a compute shaders
- **Noise libraries**: Implementación de Perlin/Simplex noise
- **L-Systems**: Sistemas de Lindenmayer para fractales más complejos
- **Instanced rendering**: Optimización para miles de elementos similares
- **Parametric UI**: Sliders para tweaking en tiempo real

## Comparación Final: Código vs Manual

### Modelado Procedural
- ✅ **Escalabilidad**: 50→5000 elementos sin esfuerzo adicional
- ✅ **Variación**: Infinitas variaciones con parámetros
- ✅ **Consistencia**: Reglas coherentes aplicadas automáticamente
- ❌ **Control artístico**: Menos precisión en detalles específicos
- ❌ **Complejidad**: Requiere programación y matemáticas

### Modelado Manual
- ✅ **Control preciso**: Cada elemento exactamente como se desea
- ✅ **Intuitividad**: Workflow familiar para artistas
- ✅ **Detalles únicos**: Cada objeto puede ser completamente distinto
- ❌ **Escalabilidad**: Trabajo lineal por cada elemento adicional
- ❌ **Variación**: Cambios requieren trabajo manual repetitivo

## Ejecución

```bash
# Clonar repositorio y navegar
cd threejs/02_procedural

# Servir con servidor HTTP
python -m http.server 8080

# Abrir navegador
http://localhost:8080
```

### Controles de uso
1. Usar teclas 1-5 o botones UI para alternar modos
2. Tecla 0 para vista completa de todos los algoritmos
3. Mouse para navegar en 3D (rotar, zoom, pan)
4. Observar descripciones técnicas en panel UI

**Compatibilidad**: Navegadores con soporte ES6 Modules y WebGL 2.0