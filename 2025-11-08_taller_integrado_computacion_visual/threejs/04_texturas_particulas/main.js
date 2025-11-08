import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// ===== CONFIGURACIÓN BÁSICA =====
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('c'), antialias: true });

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
scene.background = new THREE.Color(0x000511);

// Controles de cámara
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Iluminación
const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0xffffff, 1, 100);
pointLight.position.set(10, 10, 10);
scene.add(pointLight);

// ===== VARIABLES GLOBALES =====
let currentMaterial = 'emissive';
let globalTime = 0;
let mousePosition = new THREE.Vector2();
let intensity = 1.0;
let speed = 1.0;
let frameCount = 0;
let fps = 60;

// Uniforms globales para materiales dinámicos
const globalUniforms = {
  uTime: { value: 0.0 },
  uMouse: { value: new THREE.Vector2() },
  uIntensity: { value: 1.0 },
  uSpeed: { value: 1.0 },
  uResolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
};

// ===== TEXTURAS DINÁMICAS =====
function createDynamicTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 256;
  const ctx = canvas.getContext('2d');
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return { canvas, ctx, texture };
}

// Texturas dinámicas
const emissiveTexture = createDynamicTexture();
const normalTexture = createDynamicTexture();

// ===== MATERIALES DINÁMICOS =====

// Material 1: Emissive Dinámico con textura procedural
const emissiveMaterial = new THREE.MeshStandardMaterial({
  color: 0x2244ff,
  emissive: 0x001155,
  emissiveIntensity: 0.0,
  emissiveMap: emissiveTexture.texture,
  metalness: 0.7,
  roughness: 0.2
});

// Material 2: Normal Map Animado
const normalMaterial = new THREE.MeshStandardMaterial({
  color: 0x8844ff,
  normalMap: normalTexture.texture,
  normalScale: new THREE.Vector2(2.0, 2.0),
  metalness: 0.3,
  roughness: 0.6
});

// Material 3: UV Offset Shader
const uvMaterial = new THREE.ShaderMaterial({
  uniforms: globalUniforms,
  vertexShader: `
    uniform float uTime;
    uniform float uIntensity;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    void main() {
      vUv = uv;
      vPosition = position;
      
      vec3 pos = position;
      // Deformación UV basada en tiempo
      pos += normal * sin(position.x * 5.0 + uTime * 3.0) * 0.1 * uIntensity;
      
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
  `,
  fragmentShader: `
    uniform float uTime;
    uniform vec2 uMouse;
    uniform float uIntensity;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    void main() {
      // UV offset animado
      vec2 uv = vUv + sin(vUv * 10.0 + uTime * 2.0) * 0.05;
      uv += cos(vUv.yx * 15.0 + uTime * 1.5) * 0.03;
      
      // Patrón dinámico
      float pattern = sin(uv.x * 20.0 + uTime) * cos(uv.y * 20.0 + uTime * 1.3);
      pattern += sin(length(uv - 0.5) * 15.0 - uTime * 5.0) * 0.5;
      
      // Color base con gradiente
      vec3 color1 = vec3(0.2, 0.4, 1.0);
      vec3 color2 = vec3(1.0, 0.3, 0.8);
      vec3 color = mix(color1, color2, pattern * 0.5 + 0.5);
      
      // Influencia del mouse
      float mouseInfluence = 1.0 - distance(vUv, uMouse * 0.5 + 0.5);
      color *= (1.0 + mouseInfluence * uIntensity);
      
      gl_FragColor = vec4(color, 1.0);
    }
  `
});

// Material 4: Ruido Procedural
const noiseMaterial = new THREE.ShaderMaterial({
  uniforms: globalUniforms,
  vertexShader: `
    varying vec2 vUv;
    varying vec3 vPosition;
    
    void main() {
      vUv = uv;
      vPosition = position;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform float uTime;
    uniform float uIntensity;
    uniform vec2 uMouse;
    varying vec2 vUv;
    varying vec3 vPosition;
    
    // Función de ruido simple
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
      vec2 st = vUv * 8.0 + uTime * 0.5;
      
      // Múltiples octavas de ruido
      float n = 0.0;
      n += noise(st) * 0.5;
      n += noise(st * 2.0 + uTime) * 0.25;
      n += noise(st * 4.0 + uTime * 1.5) * 0.125;
      n += noise(st * 8.0 + uTime * 2.0) * 0.0625;
      
      // Patrón de células
      vec2 cell = floor(vUv * 6.0);
      float cellNoise = random(cell + floor(uTime * 2.0));
      
      // Mezcla de patrones
      float finalPattern = n * cellNoise * uIntensity;
      
      // Colores dinámicos
      vec3 color1 = vec3(0.8, 0.2, 0.2);
      vec3 color2 = vec3(0.2, 0.8, 0.4);
      vec3 color3 = vec3(0.2, 0.2, 0.8);
      
      vec3 color = mix(color1, color2, finalPattern);
      color = mix(color, color3, sin(finalPattern * 6.28 + uTime * 3.0) * 0.5 + 0.5);
      
      gl_FragColor = vec4(color, 1.0);
    }
  `
});

// Colección de materiales
const materials = {
  emissive: emissiveMaterial,
  normal: normalMaterial,
  uv: uvMaterial,
  noise: noiseMaterial
};

const materialNames = {
  emissive: 'Emissive Dinámico',
  normal: 'Normal Map Animado', 
  uv: 'UV Offset Shader',
  noise: 'Ruido Procedural'
};

// ===== GEOMETRÍA PRINCIPAL =====
const sphereGeometry = new THREE.SphereGeometry(1.5, 64, 64);
const mainSphere = new THREE.Mesh(sphereGeometry, materials.emissive);
scene.add(mainSphere);

// ===== SISTEMA DE PARTÍCULAS =====
class ParticleSystem {
  constructor(count, type) {
    this.count = count;
    this.type = type;
    this.geometry = new THREE.BufferGeometry();
    
    // Arrays de datos de partículas
    this.positions = new Float32Array(count * 3);
    this.velocities = new Float32Array(count * 3);
    this.ages = new Float32Array(count);
    this.sizes = new Float32Array(count);
    this.colors = new Float32Array(count * 3);
    
    this.resetParticles();
    
    // Configurar geometría
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('size', new THREE.BufferAttribute(this.sizes, 1));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));
    
    // Material de partículas
    this.material = new THREE.PointsMaterial({
      size: 0.05,
      transparent: true,
      opacity: 0.8,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true
    });
    
    this.points = new THREE.Points(this.geometry, this.material);
    scene.add(this.points);
  }
  
  resetParticles() {
    for (let i = 0; i < this.count; i++) {
      const i3 = i * 3;
      
      // Posición inicial aleatoria
      this.positions[i3] = (Math.random() - 0.5) * 8;
      this.positions[i3 + 1] = (Math.random() - 0.5) * 8;
      this.positions[i3 + 2] = (Math.random() - 0.5) * 8;
      
      // Velocidad inicial
      this.velocities[i3] = (Math.random() - 0.5) * 0.02;
      this.velocities[i3 + 1] = (Math.random() - 0.5) * 0.02;
      this.velocities[i3 + 2] = (Math.random() - 0.5) * 0.02;
      
      this.ages[i] = Math.random();
      this.sizes[i] = Math.random() * 0.1 + 0.02;
    }
  }
  
  update() {
    const spherePos = mainSphere.position;
    
    for (let i = 0; i < this.count; i++) {
      const i3 = i * 3;
      
      // Actualizar posición
      this.positions[i3] += this.velocities[i3] * speed;
      this.positions[i3 + 1] += this.velocities[i3 + 1] * speed;
      this.positions[i3 + 2] += this.velocities[i3 + 2] * speed;
      
      // Atracción hacia la esfera principal
      const dx = spherePos.x - this.positions[i3];
      const dy = spherePos.y - this.positions[i3 + 1];
      const dz = spherePos.z - this.positions[i3 + 2];
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      
      if (dist > 0) {
        const force = 0.0001 * intensity;
        this.velocities[i3] += dx / dist * force;
        this.velocities[i3 + 1] += dy / dist * force;
        this.velocities[i3 + 2] += dz / dist * force;
      }
      
      // Envejecer partícula
      this.ages[i] += 0.005 * speed;
      
      // Resetear si es muy vieja o muy lejos
      if (this.ages[i] > 1.0 || dist > 15) {
        this.ages[i] = 0.0;
        this.positions[i3] = (Math.random() - 0.5) * 8;
        this.positions[i3 + 1] = (Math.random() - 0.5) * 8;
        this.positions[i3 + 2] = (Math.random() - 0.5) * 8;
      }
      
      // Color basado en material actual y edad
      const age = this.ages[i];
      let r, g, b;
      
      switch (currentMaterial) {
        case 'emissive':
          r = 1.0;
          g = 0.5 + age * 0.5;
          b = age;
          break;
        case 'normal':
          r = 0.5 + age * 0.5;
          g = 0.3;
          b = 1.0;
          break;
        case 'uv':
          r = age;
          g = 1.0 - age;
          b = 0.8;
          break;
        case 'noise':
          r = Math.sin(age * Math.PI + globalTime) * 0.5 + 0.5;
          g = Math.cos(age * Math.PI * 2 + globalTime) * 0.5 + 0.5;
          b = age;
          break;
        default:
          r = g = b = 1.0;
      }
      
      this.colors[i3] = r;
      this.colors[i3 + 1] = g;
      this.colors[i3 + 2] = b;
      
      // Tamaño basado en edad
      this.sizes[i] = (1.0 - age) * 0.1 + 0.02;
    }
    
    // Actualizar atributos
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
    this.geometry.attributes.size.needsUpdate = true;
  }
  
  triggerEvent(eventType) {
    const forceMultiplier = intensity * 0.1;
    
    switch (eventType) {
      case 'explosion':
        for (let i = 0; i < this.count; i++) {
          const i3 = i * 3;
          const force = (Math.random() + 0.5) * forceMultiplier;
          this.velocities[i3] = (Math.random() - 0.5) * force;
          this.velocities[i3 + 1] = (Math.random() - 0.5) * force;
          this.velocities[i3 + 2] = (Math.random() - 0.5) * force;
        }
        break;
        
      case 'wave':
        const waveTime = globalTime * 10;
        for (let i = 0; i < this.count; i++) {
          const i3 = i * 3;
          const dist = Math.sqrt(
            this.positions[i3] * this.positions[i3] + 
            this.positions[i3 + 2] * this.positions[i3 + 2]
          );
          const wave = Math.sin(dist * 0.5 - waveTime) * forceMultiplier;
          this.velocities[i3 + 1] += wave;
        }
        break;
        
      case 'pulse':
        for (let i = 0; i < this.count; i++) {
          const i3 = i * 3;
          const centerForce = forceMultiplier * 0.5;
          this.velocities[i3] += (this.positions[i3] > 0 ? 1 : -1) * centerForce;
          this.velocities[i3 + 1] += (this.positions[i3 + 1] > 0 ? 1 : -1) * centerForce;
          this.velocities[i3 + 2] += (this.positions[i3 + 2] > 0 ? 1 : -1) * centerForce;
        }
        break;
        
      case 'storm':
        for (let i = 0; i < this.count; i++) {
          const i3 = i * 3;
          this.velocities[i3] += (Math.random() - 0.5) * forceMultiplier * 0.5;
          this.velocities[i3 + 1] += (Math.random() - 0.5) * forceMultiplier * 0.5;
          this.velocities[i3 + 2] += (Math.random() - 0.5) * forceMultiplier * 0.5;
        }
        break;
    }
  }
}

// Crear múltiples sistemas de partículas
const particleSystems = [
  new ParticleSystem(300, 'primary'),   // Reducido para performance
  new ParticleSystem(200, 'secondary'),
  new ParticleSystem(100, 'ambient')
];

// ===== FUNCIONES DE TEXTURAS DINÁMICAS =====
function updateEmissiveTexture() {
  const ctx = emissiveTexture.ctx;
  const size = 256;
  
  // Limpiar canvas
  ctx.fillStyle = '#000011';
  ctx.fillRect(0, 0, size, size);
  
  // Crear patrón dinámico
  const time = globalTime * 2;
  for (let x = 0; x < size; x += 4) {
    for (let y = 0; y < size; y += 4) {
      const wave = Math.sin((x + y) * 0.02 + time) * 0.5 + 0.5;
      const pulse = Math.sin(time * 3 + Math.sqrt((x-128)*(x-128) + (y-128)*(y-128)) * 0.01) * 0.5 + 0.5;
      
      const brightness = Math.floor((wave * pulse) * 255 * intensity);
      const red = brightness;
      const green = Math.floor(brightness * 0.5);
      const blue = Math.floor(brightness * 0.2);
      
      ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      ctx.fillRect(x, y, 4, 4);
    }
  }
  
  emissiveTexture.texture.needsUpdate = true;
}

function updateNormalTexture() {
  const ctx = normalTexture.ctx;
  const size = 256;
  
  // Color base para normal map (azul)
  ctx.fillStyle = '#8080FF';
  ctx.fillRect(0, 0, size, size);
  
  // Crear ondas para normal map
  const time = globalTime * speed;
  for (let x = 0; x < size; x += 2) {
    for (let y = 0; y < size; y += 2) {
      const wave1 = Math.sin(x * 0.05 + time * 2) * 0.3;
      const wave2 = Math.cos(y * 0.05 + time * 1.5) * 0.3;
      
      const normalX = Math.floor((wave1 + 1) * 127.5);
      const normalY = Math.floor((wave2 + 1) * 127.5);
      
      ctx.fillStyle = `rgb(${normalX}, ${normalY}, 255)`;
      ctx.fillRect(x, y, 2, 2);
    }
  }
  
  normalTexture.texture.needsUpdate = true;
}

// ===== FUNCIONES DE CONTROL =====
function switchMaterial(materialKey) {
  if (materials[materialKey]) {
    currentMaterial = materialKey;
    mainSphere.material = materials[materialKey];
    
    // Actualizar UI
    document.querySelectorAll('.btn').forEach(btn => {
      btn.classList.remove('active');
    });
    const activeBtn = document.querySelector(`[data-material="${materialKey}"]`);
    if (activeBtn) activeBtn.classList.add('active');
    
    // Actualizar texto de material actual
    const currentMaterialEl = document.getElementById('current-material');
    if (currentMaterialEl) {
      currentMaterialEl.textContent = materialNames[materialKey] || materialKey;
    }
  }
}

function triggerCoordinatedEvent(eventType) {
  // Efectos en el material principal
  switch (eventType) {
    case 'explosion':
      if (materials[currentMaterial].emissiveIntensity !== undefined) {
        materials[currentMaterial].emissiveIntensity = 2.0;
        setTimeout(() => materials[currentMaterial].emissiveIntensity = 0.5, 300);
      }
      break;
      
    case 'wave':
      mainSphere.scale.setScalar(1.3);
      setTimeout(() => mainSphere.scale.setScalar(1.0), 500);
      break;
      
    case 'pulse':
      if (materials[currentMaterial].emissive) {
        materials[currentMaterial].emissive.setHex(0xffffff);
        setTimeout(() => materials[currentMaterial].emissive.setHex(0x001155), 200);
      }
      break;
      
    case 'storm':
      // Rotación aleatoria
      mainSphere.rotation.x += Math.random() * 0.5;
      mainSphere.rotation.y += Math.random() * 0.5;
      break;
  }
  
  // Triggear evento en todas las partículas
  particleSystems.forEach(system => {
    system.triggerEvent(eventType);
  });
}

// ===== EVENT LISTENERS =====
document.addEventListener('DOMContentLoaded', () => {
  // Botones de materiales
  document.querySelectorAll('[data-material]').forEach(btn => {
    btn.addEventListener('click', () => {
      const material = btn.dataset.material;
      switchMaterial(material);
    });
  });
  
  // Botones de eventos
  document.getElementById('explosion').addEventListener('click', () => triggerCoordinatedEvent('explosion'));
  document.getElementById('wave').addEventListener('click', () => triggerCoordinatedEvent('wave'));
  document.getElementById('pulse').addEventListener('click', () => triggerCoordinatedEvent('pulse'));
  document.getElementById('storm').addEventListener('click', () => triggerCoordinatedEvent('storm'));

  // Sliders
  const intensitySlider = document.getElementById('intensity');
  const speedSlider = document.getElementById('speed');
  
  if (intensitySlider) {
    intensitySlider.addEventListener('input', (e) => {
      intensity = parseFloat(e.target.value);
      globalUniforms.uIntensity.value = intensity;
      document.getElementById('intensity-value').textContent = intensity.toFixed(1);
    });
  }
  
  if (speedSlider) {
    speedSlider.addEventListener('input', (e) => {
      speed = parseFloat(e.target.value);
      globalUniforms.uSpeed.value = speed;
      document.getElementById('speed-value').textContent = speed.toFixed(1);
    });
  }
});

// Controles de teclado
window.addEventListener('keydown', (event) => {
  switch(event.key) {
    case '1':
      switchMaterial('emissive');
      break;
    case '2':
      switchMaterial('normal');
      break;
    case '3':
      switchMaterial('uv');
      break;
    case '4':
      switchMaterial('noise');
      break;
    case 'e':
    case 'E':
      triggerCoordinatedEvent('explosion');
      break;
    case 'w':
    case 'W':
      triggerCoordinatedEvent('wave');
      break;
    case 'q':
    case 'Q':
      triggerCoordinatedEvent('pulse');
      break;
    case 's':
    case 'S':
      triggerCoordinatedEvent('storm');
      break;
  }
});

// Mouse tracking
window.addEventListener('mousemove', (event) => {
  mousePosition.x = (event.clientX / window.innerWidth) * 2 - 1;
  mousePosition.y = -(event.clientY / window.innerHeight) * 2 + 1;
  
  globalUniforms.uMouse.value.x = mousePosition.x;
  globalUniforms.uMouse.value.y = mousePosition.y;
});

// Redimensionar
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  globalUniforms.uResolution.value.set(window.innerWidth, window.innerHeight);
});

// ===== INICIALIZACIÓN =====
camera.position.z = 5;
controls.enableDamping = true;

// Variables para FPS
let lastTime = performance.now();
let frameCountForFPS = 0;

// ===== LOOP PRINCIPAL =====
function animate() {
  requestAnimationFrame(animate);
  
  // Actualizar tiempo global
  globalTime += 0.016 * speed;
  globalUniforms.uTime.value = globalTime;
  
  // Calcular FPS
  frameCountForFPS++;
  const currentTime = performance.now();
  if (currentTime - lastTime >= 1000) {
    fps = frameCountForFPS;
    frameCountForFPS = 0;
    lastTime = currentTime;
    
    // Actualizar UI
    const fpsEl = document.getElementById('fps');
    if (fpsEl) fpsEl.textContent = fps;
  }
  
  // Actualizar texturas dinámicas según material actual
  if (currentMaterial === 'emissive') {
    updateEmissiveTexture();
    emissiveMaterial.emissiveIntensity = 0.5 + Math.sin(globalTime * 3) * 0.3;
  }
  
  if (currentMaterial === 'normal') {
    updateNormalTexture();
  }
  
  // Actualizar sistemas de partículas
  let totalParticles = 0;
  particleSystems.forEach(system => {
    system.update();
    totalParticles += system.count;
  });
  
  // Actualizar contador de partículas
  const particleCountEl = document.getElementById('particle-count');
  if (particleCountEl) particleCountEl.textContent = totalParticles;
  
  // Rotación automática de la esfera
  mainSphere.rotation.y += 0.005;
  mainSphere.rotation.x += 0.002;
  
  // Actualizar controles y renderizar
  controls.update();
  renderer.render(scene, camera);
}

animate();