# PBR Material & Color Analysis - Three.js

## Concepto del proyecto

Este experimento visual implementa un sistema completo de análisis de materiales PBR (Physically Based Rendering) y modelos cromáticos en tiempo real. El proyecto demuestra la aplicación de técnicas avanzadas de renderizado 3D, iluminación múltiple y análisis científico del color usando el espacio CIELAB para justificación de contraste.

La experiencia permite alternar entre diferentes materiales (madera, ladrillo, concreto, metal) y tipos de cámara (perspectiva/ortográfica) mientras visualiza en tiempo real los valores cromáticos RGB/HSV y el cálculo de contraste ΔE según estándares CIELAB.

## Herramientas y entorno

- **Three.js v0.161**: Motor de renderizado 3D WebGL
- **JavaScript ES6 Modules**: Programación modular
- **HTML5 + CSS3**: Interfaz de usuario
- **HDRI Environment Mapping**: Venice Sunset 1K
- **PBR Textures**: Conjunto completo de mapas (albedo, normal, roughness, metalness)
- **Import Maps**: Resolución de módulos ES6

## Descripción de módulos aplicados

### A. Materiales PBR
- **Albedo maps**: Color base para wood, brick, concrete, metal
- **Normal maps**: Detalles de superficie y microgeometría
- **Roughness maps**: Control de rugosidad superficial
- **Metalness maps**: Diferenciación entre dieléctricos y conductores
- **Displacement maps**: Relieve geométrico (madera)

### B. Iluminación múltiple
- **Key Light**: DirectionalLight principal (blanco, intensidad 2.0)
- **Fill Light**: HemisphereLight ambiente (azul-gris, intensidad 0.4)
- **Rim Light**: DirectionalLight trasero (cálido, intensidad 1.0)
- **HDRI Environment**: Reflexiones ambientales realistas con PMREM

### C. Sistema de cámaras
- **Perspectiva**: FOV 60°, aspect ratio dinámico
- **Ortográfica**: Frustum size 6, proyección paralela
- **Alternancia suave**: Controles OrbitControls independientes
- **Responsive**: Actualización automática en resize

### D. Análisis cromático CIELAB
- **Conversión sRGB → XYZ → Lab**: Algoritmos estándar CIE
- **Cálculo ΔE76**: Distancia euclidiana en espacio CIELAB
- **Justificación de contraste**: Umbrales perceptuales
- **Visualización RGB/HSV**: Paleta en tiempo real

### E. Animaciones procedurales
- **Material dinámico**: Oscilación de roughness y metalness
- **Iluminación variable**: Intensidad animada de key/rim lights
- **Rotación orbital**: Movimiento constante de la esfera
- **UI reactiva**: Actualización sincronizada de valores

## Código relevante

### Carga de texturas PBR
```javascript
const materials = {
  metal: {
    map: texLoader.load('./textures/metal_diff.jpg'),
    normalMap: texLoader.load('./textures/metal_normal.jpg'),
    roughnessMap: texLoader.load('./textures/metal_rough.jpg'),
    metalnessMap: texLoader.load('./textures/metal_metalness.jpg'),
    metalness: 1.0
  }
};
```

### HDRI Environment con PMREM
```javascript
const pmremGenerator = new THREE.PMREMGenerator(renderer);
rgbeLoader.load('./textures/venice_sunset_1k.hdr', (hdr) => {
  const envMap = pmremGenerator.fromEquirectangular(hdr).texture;
  scene.environment = envMap;
});
```

### Cálculo CIELAB ΔE
```javascript
function rgbToLab(r, g, b) { 
  return xyzToLab(...rgbToXyz(r, g, b)); 
}
function deltaE76(lab1, lab2) {
  const dl = lab1[0] - lab2[0], da = lab1[1] - lab2[1], db = lab1[2] - lab2[2];
  return Math.sqrt(dl * dl + da * da + db * db);
}
```

### Animaciones dinámicas
```javascript
const roughnessBase = currentMaterial === 'metal' ? 0.1 : 0.5;
mat.roughness = roughnessBase + 0.3 * (0.5 + 0.5 * Math.sin(t * 0.8));
keyLight.intensity = 1.8 + 0.4 * Math.sin(t * 0.6);
```

## Controles interactivos

- **Tecla C**: Alternar cámara perspectiva ↔ ortográfica
- **Tecla M**: Ciclar materiales (wood → brick → concrete → metal)
- **Mouse**: OrbitControls (rotar, zoom, pan)

## Características técnicas

### Renderizado
- **Tone Mapping**: ACES Filmic para HDR
- **Antialiasing**: MSAA activado
- **Pixel Ratio**: Optimizado para displays HiDPI

### Performance
- **Geometría**: Esfera 64x64 segmentos para detalles PBR
- **Texturas**: 1K resolution para balance calidad/performance
- **Dispose**: Limpieza automática de memoria GPU

## Evidencias requeridas

### Capturas necesarias (6):
1. Material Wood con cámara perspectiva + UI visible
2. Material Brick con análisis CIELAB detallado  
3. Material Concrete mostrando valores RGB/HSV
4. Material Metal con reflexiones HDRI marcadas
5. Vista ortográfica de cualquier material
6. Comparación de contraste ΔE alto vs bajo


### Video demo (30-60s):
- Demostración completa de funcionalidades
- Explicación narrada de características PBR
- Transiciones entre todos los materiales
- Mostrar cálculos CIELAB en acción

## Reflexión

### Aprendizajes clave
- Implementación práctica de workflows PBR en tiempo real
- Aplicación de espacios de color científicos (CIELAB) en graphics
- Integración de múltiples sistemas de iluminación
- Optimización de performance en WebGL con Three.js

### Retos técnicos superados
- **Import Maps**: Resolución de módulos ES6 para CDN
- **PMREM Generation**: Conversión correcta HDRI → environment mapping
- **Tone Mapping**: Balance entre realismo y visibilidad
- **UI Reactiva**: Sincronización frame-perfect entre render y DOM

### Mejoras posibles
- **IBL avanzado**: Implementar diffuse + specular separation
- **Material Editor**: UI para tweaking en vivo de parámetros PBR
- **Post-processing**: Bloom, SSAO, SSR para mayor realismo
- **Texture Streaming**: Carga progresiva de assets de mayor resolución
- **Color Spaces**: Extensión a espacios LCH, Oklab para mejor UX

## Ejecución

1. Clonar repositorio
2. Servir con HTTP server: `python -m http.server 8080`
3. Abrir `http://localhost:8080`
4. Usar controles C/M para interactuar

**Compatibilidad**: Chrome 91+, Firefox 90+, Safari 14+ (Import Maps support)