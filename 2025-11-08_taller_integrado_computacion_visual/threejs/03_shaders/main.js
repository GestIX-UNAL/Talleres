import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Setup básico
const renderer = new THREE.WebGLRenderer({ canvas: document.querySelector("#c"), antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a0a);

const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 100);
camera.position.set(2, 1.6, 3);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Variables de control
let currentShader = 'water';
const shaderTypes = ['water', 'toon', 'wireframe', 'gradient', 'procedural', 'distortion'];
let shaderIndex = 0;

// Mouse tracking para interactividad
const mouse = new THREE.Vector2();
const raycaster = new THREE.Raycaster();

window.addEventListener('mousemove', (event) => {
  mouse.x = (event.clientX / innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / innerHeight) * 2 + 1;
});

// Geometrías para diferentes shaders
const planeGeo = new THREE.PlaneGeometry(4, 4, 100, 100);
// Uniforms globales para todos los shaders
const globalUniforms = {
  uTime: { value: 0.0 },
  uMouse: { value: new THREE.Vector2() },
  uResolution: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uIntensity: { value: 1.0 },
  uSpeed: { value: 1.0 },
  uColor1: { value: new THREE.Color(0x0080ff) },
  uColor2: { value: new THREE.Color(0xff0080) }
};

// Shader 1: Water/Wave (tu shader original mejorado)
const waterShader = {
  vertex: /* glsl */`
    uniform float uTime;
    uniform float uIntensity;
    uniform vec2 uMouse;
    varying vec2 vUv;
    varying float vHeight;
    varying vec3 vPosition;
    
    void main() {
      vUv = uv;
      vPosition = position;
      
      vec3 pos = position;
      // Ondas complejas
      float wave1 = sin(pos.x * 10.0 + uTime * 2.0) * cos(pos.y * 8.0 + uTime * 1.5);
      float wave2 = sin(pos.x * 15.0 - uTime * 3.0) * sin(pos.y * 12.0 + uTime * 2.5);
      
      // Interacción con mouse
      float mouseInfluence = 1.0 - distance(uv, uMouse * 0.5 + 0.5) * 2.0;
      mouseInfluence = max(0.0, mouseInfluence);
      
      pos.z += (wave1 * 0.1 + wave2 * 0.05) * uIntensity + mouseInfluence * 0.2;
      vHeight = pos.z;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `,
  fragment: /* glsl */`
    uniform float uTime;
    uniform vec3 uColor1;
    uniform vec3 uColor2;
    varying vec2 vUv;
    varying float vHeight;
    varying vec3 vPosition;
    
    void main() {
      float height = clamp(vHeight * 5.0 + 0.5, 0.0, 1.0);
      float pulse = sin(uTime * 4.0 + vPosition.x * 10.0) * 0.5 + 0.5;
      
      vec3 color = mix(uColor1, uColor2, height * pulse);
      
      // Foam effect
      float foam = step(0.8, height) * (sin(uTime * 10.0) * 0.5 + 0.5);
      color = mix(color, vec3(1.0), foam * 0.3);
      
      gl_FragColor = vec4(color, 1.0);
    }
  `
};

// Shader 2: Toon Shading
const toonShader = {
  vertex: /* glsl */`
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vPosition;
    
    void main() {
      vUv = uv;
      vNormal = normalize(normalMatrix * normal);
      vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragment: /* glsl */`
    uniform float uTime;
    uniform vec3 uColor1;
    uniform vec3 uColor2;
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vPosition;
    
    void main() {
      // Toon lighting calculation
      vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
      float NdotL = dot(vNormal, lightDir);
      
      // Quantize lighting into bands
      float toonLevel = floor(NdotL * 4.0) / 4.0;
      toonLevel = clamp(toonLevel, 0.2, 1.0);
      
      // Color based on position and time
      vec3 baseColor = mix(uColor1, uColor2, sin(vUv.x * 3.14159 + uTime) * 0.5 + 0.5);
      
      // Apply toon shading
      vec3 color = baseColor * toonLevel;
      
      // Outline effect (simplified)
      float fresnel = 1.0 - abs(dot(vNormal, normalize(-vPosition)));
      if(fresnel > 0.8) {
        color = vec3(0.0);
      }
      
      gl_FragColor = vec4(color, 1.0);
    }
  `
};

// Shader 3: Wireframe procedural
const wireframeShader = {
  vertex: /* glsl */`
    uniform float uTime;
    varying vec2 vUv;
    varying vec3 vBarycentric;
    
    attribute vec3 barycentric;
    
    void main() {
      vUv = uv;
      vBarycentric = barycentric;
      
      vec3 pos = position;
      // Slight animation
      pos += normal * sin(uTime + position.x * 5.0) * 0.05;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `,
  fragment: /* glsl */`
    uniform float uTime;
    uniform vec3 uColor1;
    uniform vec3 uColor2;
    varying vec2 vUv;
    varying vec3 vBarycentric;
    
    void main() {
      // Wireframe effect using screen derivatives
      vec2 grid = abs(fract(vUv * 20.0) - 0.5) / fwidth(vUv * 20.0);
      float line = min(grid.x, grid.y);
      
      // Color based on time and position
      vec3 baseColor = mix(uColor1, uColor2, sin(uTime + vUv.x * 3.14159) * 0.5 + 0.5);
      
      // Mix wireframe with solid color
      float wireStrength = 1.0 - step(1.0, line);
      vec3 color = mix(baseColor * 0.1, baseColor, wireStrength);
      
      gl_FragColor = vec4(color, 1.0);
    }
  `
};

// Shader 4: Gradient complexo
const gradientShader = {
  vertex: /* glsl */`
    uniform float uTime;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    void main() {
      vUv = uv;
      vPosition = position;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragment: /* glsl */`
    uniform float uTime;
    uniform vec2 uMouse;
    uniform vec3 uColor1;
    uniform vec3 uColor2;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    void main() {
      // Multiple gradient patterns
      float gradient1 = length(vUv - 0.5);
      float gradient2 = sin(vUv.x * 10.0 + uTime) * cos(vUv.y * 10.0 + uTime);
      float gradient3 = distance(vUv, uMouse * 0.5 + 0.5);
      
      // Combine gradients
      float pattern = gradient1 + gradient2 * 0.3 + sin(gradient3 * 10.0 + uTime * 3.0) * 0.2;
      pattern = fract(pattern + uTime * 0.5);
      
      // Multi-color gradient
      vec3 color1 = vec3(0.8, 0.2, 0.8);
      vec3 color2 = vec3(0.2, 0.8, 0.8); 
      vec3 color3 = vec3(0.8, 0.8, 0.2);
      
      vec3 color;
      if(pattern < 0.33) {
        color = mix(color1, color2, pattern * 3.0);
      } else if(pattern < 0.66) {
        color = mix(color2, color3, (pattern - 0.33) * 3.0);
      } else {
        color = mix(color3, color1, (pattern - 0.66) * 3.0);
      }
      
      gl_FragColor = vec4(color, 1.0);
    }
  `
};

// Shader 5: Procedural Textures
const proceduralShader = {
  vertex: /* glsl */`
    uniform float uTime;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    void main() {
      vUv = uv;
      vPosition = position;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragment: /* glsl */`
    uniform float uTime;
    uniform float uIntensity;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    // Simple noise function
    float random(vec2 st) {
      return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
    }
    
    float noise(vec2 st) {
      vec2 i = floor(st);
      vec2 f = fract(st);
      
      float a = random(i);
      float b = random(i + vec2(1.0, 0.0));
      float c = random(i + vec2(0.0, 1.0));
      float d = random(i + vec2(1.0, 1.0));
      
      vec2 u = f * f * (3.0 - 2.0 * f);
      
      return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
    }
    
    void main() {
      vec2 uv = vUv * 8.0;
      
      // Layer multiple noise octaves
      float n = 0.0;
      n += noise(uv + uTime * 0.5) * 0.5;
      n += noise(uv * 2.0 + uTime * 0.3) * 0.25;
      n += noise(uv * 4.0 + uTime * 0.1) * 0.125;
      
      // Create patterns
      float pattern1 = sin(vUv.x * 20.0 + n * 5.0 + uTime) * 0.5 + 0.5;
      float pattern2 = cos(vUv.y * 15.0 + n * 3.0 + uTime * 1.5) * 0.5 + 0.5;
      
      // Combine patterns
      float finalPattern = pattern1 * pattern2 * n * uIntensity;
      
      // Color based on pattern
      vec3 color1 = vec3(0.1, 0.1, 0.8);
      vec3 color2 = vec3(0.8, 0.4, 0.1);
      vec3 color3 = vec3(0.1, 0.8, 0.1);
      
      vec3 color = mix(color1, color2, finalPattern);
      color = mix(color, color3, sin(finalPattern * 6.28318 + uTime) * 0.5 + 0.5);
      
      gl_FragColor = vec4(color, 1.0);
    }
  `
};

// Shader 6: UV Distortion
const distortionShader = {
  vertex: /* glsl */`
    uniform float uTime;
    varying vec2 vUv;
    varying vec2 vDistortedUv;
    
    void main() {
      vUv = uv;
      
      // Distort UV coordinates
      vec2 distortion = vec2(
        sin(uv.y * 10.0 + uTime) * 0.1,
        cos(uv.x * 8.0 + uTime * 1.5) * 0.1
      );
      
      vDistortedUv = uv + distortion;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragment: /* glsl */`
    uniform float uTime;
    uniform vec2 uMouse;
    varying vec2 vUv;
    varying vec2 vDistortedUv;
    
    void main() {
      // Create patterns with distorted UVs
      vec2 uv = vDistortedUv;
      
      // Checkerboard pattern
      vec2 grid = floor(uv * 10.0);
      float checker = mod(grid.x + grid.y, 2.0);
      
      // Ripple effect from mouse
      float dist = distance(vUv, uMouse * 0.5 + 0.5);
      float ripple = sin(dist * 20.0 - uTime * 10.0) * 0.5 + 0.5;
      
      // Color mixing
      vec3 color1 = vec3(checker * 0.8 + 0.2, 0.3, 0.8);
      vec3 color2 = vec3(0.8, 0.8, checker * 0.5 + 0.3);
      
      vec3 color = mix(color1, color2, ripple);
      
      gl_FragColor = vec4(color, 1.0);
    }
  `
};

// Colección de todos los shaders
const shaders = {
  water: waterShader,
  toon: toonShader,
  wireframe: wireframeShader,
  gradient: gradientShader,
  procedural: proceduralShader,
  distortion: distortionShader
};

// Estado del sistema
let currentShaderName = 'water';
let currentMesh = null;

// Crear geometrías diferentes para cada shader
const geometries = {
  water: new THREE.PlaneGeometry(4, 4, 100, 100),
  toon: new THREE.SphereGeometry(1.5, 32, 32),
  wireframe: new THREE.TorusGeometry(1.2, 0.4, 16, 32),
  gradient: new THREE.PlaneGeometry(3, 3, 1, 1),
  procedural: new THREE.PlaneGeometry(3, 3, 1, 1),
  distortion: new THREE.PlaneGeometry(3, 3, 1, 1)
};

// Función para crear material shader
function createShaderMaterial(shaderName) {
  const shader = shaders[shaderName];
  return new THREE.ShaderMaterial({
    vertexShader: shader.vertex,
    fragmentShader: shader.fragment,
    uniforms: globalUniforms,
    side: THREE.DoubleSide,
    transparent: false
  });
}

// Función para cambiar shader
function switchShader(shaderName) {
  if (currentMesh) {
    scene.remove(currentMesh);
  }
  
  currentShaderName = shaderName;
  const geometry = geometries[shaderName];
  const material = createShaderMaterial(shaderName);
  
  // Para wireframe, añadir atributo barycentric si es necesario
  if (shaderName === 'wireframe') {
    const positions = geometry.attributes.position;
    const count = positions.count;
    const barycentric = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i += 3) {
      barycentric[i * 3] = 1;
      barycentric[i * 3 + 1] = 0;
      barycentric[i * 3 + 2] = 0;
      
      barycentric[(i + 1) * 3] = 0;
      barycentric[(i + 1) * 3 + 1] = 1;
      barycentric[(i + 1) * 3 + 2] = 0;
      
      barycentric[(i + 2) * 3] = 0;
      barycentric[(i + 2) * 3 + 1] = 0;
      barycentric[(i + 2) * 3 + 2] = 1;
    }
    
    geometry.setAttribute('barycentric', new THREE.BufferAttribute(barycentric, 3));
  }
  
  currentMesh = new THREE.Mesh(geometry, material);
  
  // Posición y rotación específica por shader
  if (shaderName === 'water') {
    currentMesh.rotation.x = -Math.PI / 2;
  }
  
  scene.add(currentMesh);
  
  // Actualizar UI
  document.querySelectorAll('.shader-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  const activeBtn = document.querySelector(`[data-shader="${shaderName}"]`);
  if (activeBtn) activeBtn.classList.add('active');
}

// Manejo de eventos del mouse
let mousePosition = { x: 0, y: 0 };

window.addEventListener('mousemove', (event) => {
  mousePosition.x = (event.clientX / window.innerWidth) * 2 - 1;
  mousePosition.y = -(event.clientY / window.innerHeight) * 2 + 1;
  
  globalUniforms.uMouse.value.x = mousePosition.x;
  globalUniforms.uMouse.value.y = mousePosition.y;
});

// Controles de teclado
window.addEventListener('keydown', (event) => {
  const keys = ['1', '2', '3', '4', '5', '6'];
  const shaderNames = ['water', 'toon', 'wireframe', 'gradient', 'procedural', 'distortion'];
  
  const keyIndex = keys.indexOf(event.key);
  if (keyIndex !== -1) {
    switchShader(shaderNames[keyIndex]);
  }
  
  // Controles de parámetros
  switch(event.key) {
    case 'ArrowUp':
      globalUniforms.uIntensity.value = Math.min(2.0, globalUniforms.uIntensity.value + 0.1);
      updateUI();
      break;
    case 'ArrowDown':
      globalUniforms.uIntensity.value = Math.max(0.1, globalUniforms.uIntensity.value - 0.1);
      updateUI();
      break;
    case 'ArrowRight':
      globalUniforms.uSpeed.value = Math.min(3.0, globalUniforms.uSpeed.value + 0.1);
      updateUI();
      break;
    case 'ArrowLeft':
      globalUniforms.uSpeed.value = Math.max(0.1, globalUniforms.uSpeed.value - 0.1);
      updateUI();
      break;
    case ' ':
      event.preventDefault();
      // Cambiar colores aleatoriamente
      globalUniforms.uColor1.value.setHSL(Math.random(), 0.8, 0.6);
      globalUniforms.uColor2.value.setHSL(Math.random(), 0.8, 0.6);
      break;
  }
});

// Función para actualizar UI
function updateUI() {
  const intensityEl = document.getElementById('intensity-value');
  const speedEl = document.getElementById('speed-value');
  
  if (intensityEl) intensityEl.textContent = globalUniforms.uIntensity.value.toFixed(2);
  if (speedEl) speedEl.textContent = globalUniforms.uSpeed.value.toFixed(2);
}

// Redimensionar
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  globalUniforms.uResolution.value.set(window.innerWidth, window.innerHeight);
});

// Inicializar
renderer.setSize(innerWidth, innerHeight);
camera.position.z = 3;
controls.enableDamping = true;

// Event listeners para controles de UI
document.addEventListener('DOMContentLoaded', () => {
  // Botones de shaders
  document.querySelectorAll('.shader-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const shaderName = btn.dataset.shader;
      switchShader(shaderName);
    });
  });
  
  // Controles deslizantes
  const intensitySlider = document.getElementById('intensity');
  const speedSlider = document.getElementById('speed');
  const randomColorsBtn = document.getElementById('random-colors');
  
  if (intensitySlider) {
    intensitySlider.addEventListener('input', (e) => {
      globalUniforms.uIntensity.value = parseFloat(e.target.value);
      updateUI();
    });
  }
  
  if (speedSlider) {
    speedSlider.addEventListener('input', (e) => {
      globalUniforms.uSpeed.value = parseFloat(e.target.value);
      updateUI();
    });
  }
  
  if (randomColorsBtn) {
    randomColorsBtn.addEventListener('click', () => {
      globalUniforms.uColor1.value.setHSL(Math.random(), 0.8, 0.6);
      globalUniforms.uColor2.value.setHSL(Math.random(), 0.8, 0.6);
    });
  }
});

// Inicializar con el primer shader
switchShader('water');
updateUI();

// Loop de animación
function animate() {
  requestAnimationFrame(animate);
  
  // Actualizar uniformes globales
  globalUniforms.uTime.value += 0.016 * globalUniforms.uSpeed.value;
  
  controls.update();
  renderer.render(scene, camera);
}

animate();
