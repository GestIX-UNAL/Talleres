import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// ===== CONFIGURACIÓN BÁSICA =====
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('c'), antialias: true });

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
scene.background = new THREE.Color(0x000511);

// Controles de cámara
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// ===== VARIABLES GLOBALES =====
let globalTime = 0;
let mousePosition = new THREE.Vector2();
let touchActive = false;
let cameraAutoRotate = false;
let cameraSpeed = 1.0;
let collisionSensitivity = 1.0;
let collisionMode = 'click'; // 'click', 'hover', 'proximity'
let collisionCount = 0;
let lastKeyPressed = '-';

// Arrays de objetos interactivos
const interactiveObjects = [];
const particles = [];

// Estados de teclado
const keyStates = {
  w: false, a: false, s: false, d: false,
  q: false, e: false, space: false, ctrl: false
};

// ===== ILUMINACIÓN =====
const ambientLight = new THREE.AmbientLight(0x404040, 0.3);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
directionalLight.position.set(5, 5, 5);
directionalLight.castShadow = true;
directionalLight.shadow.mapSize.width = 2048;
directionalLight.shadow.mapSize.height = 2048;
scene.add(directionalLight);

const pointLight = new THREE.PointLight(0xffffff, 0.5, 100);
pointLight.position.set(-5, 3, -5);
scene.add(pointLight);

// ===== OBJETOS INICIALES =====
// Suelo para proyectar sombras
const floorGeometry = new THREE.PlaneGeometry(20, 20);
const floorMaterial = new THREE.MeshStandardMaterial({ 
  color: 0x222222,
  roughness: 0.8,
  metalness: 0.1
});
const floor = new THREE.Mesh(floorGeometry, floorMaterial);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -2;
floor.receiveShadow = true;
scene.add(floor);

// Función para crear objetos interactivos
function createInteractiveObject(x = 0, y = 0, z = 0) {
  const geometries = [
    new THREE.BoxGeometry(1, 1, 1),
    new THREE.SphereGeometry(0.6, 16, 16),
    new THREE.ConeGeometry(0.6, 1.2, 8),
    new THREE.CylinderGeometry(0.4, 0.4, 1, 8)
  ];
  
  const geometry = geometries[Math.floor(Math.random() * geometries.length)];
  const material = new THREE.MeshStandardMaterial({ 
    color: new THREE.Color().setHSL(Math.random(), 0.7, 0.6),
    roughness: 0.3,
    metalness: 0.4
  });
  
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  
  // Propiedades para física básica
  mesh.userData = {
    velocity: new THREE.Vector3(),
    originalScale: mesh.scale.clone(),
    originalColor: material.color.clone(),
    isAnimating: false,
    animationTime: 0
  };
  
  scene.add(mesh);
  interactiveObjects.push(mesh);
  return mesh;
}

// Crear objetos iniciales
createInteractiveObject(0, 0, 0);
createInteractiveObject(-2, 0, 1);
createInteractiveObject(2, 0, -1);
createInteractiveObject(0, 2, 0);

// ===== SISTEMA DE PARTÍCULAS =====
class ParticleEffect {
  constructor(position, color = 0xffffff, count = 50) {
    this.geometry = new THREE.BufferGeometry();
    this.count = count;
    
    const positions = new Float32Array(count * 3);
    const velocities = new Float32Array(count * 3);
    const lifetimes = new Float32Array(count);
    
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      
      // Posición inicial
      positions[i3] = position.x + (Math.random() - 0.5) * 0.5;
      positions[i3 + 1] = position.y + (Math.random() - 0.5) * 0.5;
      positions[i3 + 2] = position.z + (Math.random() - 0.5) * 0.5;
      
      // Velocidad aleatoria
      velocities[i3] = (Math.random() - 0.5) * 5;
      velocities[i3 + 1] = Math.random() * 3 + 1;
      velocities[i3 + 2] = (Math.random() - 0.5) * 5;
      
      lifetimes[i] = 1.0;
    }
    
    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.velocities = velocities;
    this.lifetimes = lifetimes;
    
    const material = new THREE.PointsMaterial({
      color: color,
      size: 0.1,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending
    });
    
    this.points = new THREE.Points(this.geometry, material);
    scene.add(this.points);
    
    this.alive = true;
  }
  
  update(deltaTime) {
    if (!this.alive) return false;
    
    const positions = this.geometry.attributes.position.array;
    let allDead = true;
    
    for (let i = 0; i < this.count; i++) {
      const i3 = i * 3;
      
      if (this.lifetimes[i] > 0) {
        allDead = false;
        
        // Actualizar posición
        positions[i3] += this.velocities[i3] * deltaTime;
        positions[i3 + 1] += this.velocities[i3 + 1] * deltaTime;
        positions[i3 + 2] += this.velocities[i3 + 2] * deltaTime;
        
        // Aplicar gravedad
        this.velocities[i3 + 1] -= 9.8 * deltaTime;
        
        // Reducir tiempo de vida
        this.lifetimes[i] -= deltaTime;
      }
    }
    
    this.geometry.attributes.position.needsUpdate = true;
    this.points.material.opacity = Math.max(0, Math.min(...this.lifetimes));
    
    if (allDead) {
      scene.remove(this.points);
      this.alive = false;
    }
    
    return this.alive;
  }
}

// ===== RAYCASTING Y COLISIONES =====
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function updateMousePosition(clientX, clientY) {
  mouse.x = (clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(clientY / window.innerHeight) * 2 + 1;
  
  mousePosition.copy(mouse);
  
  // Actualizar UI
  document.getElementById('mouse-x').textContent = mouse.x.toFixed(2);
  document.getElementById('mouse-y').textContent = mouse.y.toFixed(2);
}

function performRaycast() {
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(interactiveObjects);
  return intersects;
}

function triggerCollisionEffect(object, intersect) {
  collisionCount++;
  document.getElementById('collision-count').textContent = collisionCount;
  
  const mesh = object;
  const userData = mesh.userData;
  
  if (!userData.isAnimating) {
    userData.isAnimating = true;
    userData.animationTime = 0;
    
    // Efectos visuales
    mesh.material.color.setHSL(Math.random(), 0.8, 0.7);
    
    // Física básica
    if (intersect) {
      const direction = new THREE.Vector3()
        .subVectors(mesh.position, intersect.point)
        .normalize()
        .multiplyScalar(5 * collisionSensitivity);
      userData.velocity.add(direction);
    }
    
    // Crear partículas
    const effect = new ParticleEffect(mesh.position, mesh.material.color.getHex());
    particles.push(effect);
    
    // Mostrar indicador
    const indicator = document.getElementById('collision-indicator');
    indicator.classList.add('active');
    setTimeout(() => indicator.classList.remove('active'), 500);
  }
}

// ===== EVENTOS DE MOUSE =====
window.addEventListener('mousemove', (event) => {
  updateMousePosition(event.clientX, event.clientY);
  
  if (collisionMode === 'hover') {
    const intersects = performRaycast();
    if (intersects.length > 0) {
      triggerCollisionEffect(intersects[0].object, intersects[0]);
    }
  }
});

window.addEventListener('click', (event) => {
  if (collisionMode === 'click') {
    updateMousePosition(event.clientX, event.clientY);
    const intersects = performRaycast();
    if (intersects.length > 0) {
      triggerCollisionEffect(intersects[0].object, intersects[0]);
    }
  }
});

window.addEventListener('wheel', (event) => {
  event.preventDefault();
  
  // Zoom de cámara con wheel
  const zoomSpeed = 0.1;
  const direction = event.deltaY > 0 ? 1 : -1;
  
  camera.position.multiplyScalar(1 + direction * zoomSpeed);
  camera.position.clampLength(2, 50);
});

// ===== EVENTOS DE TECLADO =====
window.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  lastKeyPressed = key.toUpperCase();
  document.getElementById('last-key').textContent = lastKeyPressed;
  
  // Actualizar estados
  if (key in keyStates) {
    keyStates[key] = true;
  }
  
  // Comandos especiales
  switch (key) {
    case 'r':
      // Reset cámara
      camera.position.set(5, 5, 5);
      camera.lookAt(0, 0, 0);
      controls.reset();
      break;
      
    case 'f':
      // Pantalla completa
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
      break;
      
    case 'tab':
      event.preventDefault();
      // Cambiar modo de colisión
      const modes = ['click', 'hover', 'proximity'];
      const currentIndex = modes.indexOf(collisionMode);
      collisionMode = modes[(currentIndex + 1) % modes.length];
      document.getElementById('collision-mode').textContent = `Modo: ${collisionMode}`;
      break;
      
    case '1':
      triggerEffect('explosion');
      break;
    case '2':
      triggerEffect('wave');
      break;
    case '3':
      triggerEffect('rainbow');
      break;
    case '4':
      triggerEffect('gravity');
      break;
  }
});

window.addEventListener('keyup', (event) => {
  const key = event.key.toLowerCase();
  if (key in keyStates) {
    keyStates[key] = false;
  }
});

// ===== EVENTOS DE TOUCH =====
const touchZone = document.getElementById('touch-zone');
let touchStartPos = null;

touchZone.addEventListener('touchstart', (event) => {
  event.preventDefault();
  touchActive = true;
  document.getElementById('touch-active').textContent = 'Sí';
  
  const touch = event.touches[0];
  touchStartPos = { x: touch.clientX, y: touch.clientY };
  
  touchZone.style.background = 'rgba(0, 209, 178, 0.3)';
});

touchZone.addEventListener('touchmove', (event) => {
  event.preventDefault();
  
  if (touchStartPos) {
    const touch = event.touches[0];
    const deltaX = touch.clientX - touchStartPos.x;
    const deltaY = touch.clientY - touchStartPos.y;
    
    // Rotar cámara basado en gesture
    controls.azimuthalAngle += deltaX * 0.01;
    controls.polarAngle += deltaY * 0.01;
    
    touchStartPos = { x: touch.clientX, y: touch.clientY };
  }
});

touchZone.addEventListener('touchend', (event) => {
  event.preventDefault();
  touchActive = false;
  document.getElementById('touch-active').textContent = 'No';
  touchStartPos = null;
  
  touchZone.style.background = 'rgba(0, 209, 178, 0.1)';
  
  // Trigger effect en touch end
  triggerEffect('explosion');
});

// Soporte para multi-touch
window.addEventListener('touchstart', (event) => {
  if (event.touches.length === 2) {
    // Pinch to zoom gesture
    event.preventDefault();
  }
});

// ===== FUNCIONES DE EFECTOS =====
function triggerEffect(type) {
  interactiveObjects.forEach((object, index) => {
    const userData = object.userData;
    
    switch (type) {
      case 'explosion':
        const explosionForce = new THREE.Vector3(
          (Math.random() - 0.5) * 10,
          Math.random() * 5 + 2,
          (Math.random() - 0.5) * 10
        );
        userData.velocity.add(explosionForce);
        break;
        
      case 'wave':
        const distance = object.position.distanceTo(new THREE.Vector3(0, 0, 0));
        const waveForce = Math.sin(globalTime * 5 - distance) * 2;
        userData.velocity.y += waveForce;
        break;
        
      case 'rainbow':
        const hue = (index / interactiveObjects.length + globalTime * 0.5) % 1;
        object.material.color.setHSL(hue, 0.8, 0.6);
        break;
        
      case 'gravity':
        userData.velocity.y += 5;
        break;
    }
    
    // Crear partículas
    if (type !== 'rainbow') {
      const effect = new ParticleEffect(object.position, object.material.color.getHex(), 30);
      particles.push(effect);
    }
  });
}

function addNewObject() {
  const x = (Math.random() - 0.5) * 10;
  const y = Math.random() * 3 + 1;
  const z = (Math.random() - 0.5) * 10;
  createInteractiveObject(x, y, z);
  
  document.getElementById('object-count').textContent = interactiveObjects.length;
}

function clearAllObjects() {
  interactiveObjects.forEach(obj => scene.remove(obj));
  interactiveObjects.length = 0;
  document.getElementById('object-count').textContent = 0;
}

// ===== EVENT LISTENERS UI =====
document.addEventListener('DOMContentLoaded', () => {
  // Sliders de iluminación
  document.getElementById('light-intensity').addEventListener('input', (e) => {
    const value = parseFloat(e.target.value);
    directionalLight.intensity = value;
    document.getElementById('light-intensity-value').textContent = value.toFixed(1);
  });
  
  document.getElementById('light-hue').addEventListener('input', (e) => {
    const hue = parseInt(e.target.value);
    directionalLight.color.setHSL(hue / 360, 1, 0.5);
    document.getElementById('light-hue-value').textContent = hue;
  });
  
  // Controles de cámara
  document.getElementById('camera-reset').addEventListener('click', () => {
    camera.position.set(5, 5, 5);
    camera.lookAt(0, 0, 0);
    controls.reset();
  });
  
  document.getElementById('camera-auto').addEventListener('click', (e) => {
    cameraAutoRotate = !cameraAutoRotate;
    controls.autoRotate = cameraAutoRotate;
    e.target.classList.toggle('active', cameraAutoRotate);
  });
  
  document.getElementById('camera-speed').addEventListener('input', (e) => {
    cameraSpeed = parseFloat(e.target.value);
    controls.autoRotateSpeed = cameraSpeed * 2;
    document.getElementById('camera-speed-value').textContent = cameraSpeed.toFixed(1);
  });
  
  // Botones de efectos
  document.getElementById('effect-explosion').addEventListener('click', () => triggerEffect('explosion'));
  document.getElementById('effect-wave').addEventListener('click', () => triggerEffect('wave'));
  document.getElementById('effect-rainbow').addEventListener('click', () => triggerEffect('rainbow'));
  document.getElementById('effect-gravity').addEventListener('click', () => triggerEffect('gravity'));
  
  // Controles de colisiones
  document.getElementById('collision-mode').addEventListener('click', (e) => {
    const modes = ['click', 'hover', 'proximity'];
    const currentIndex = modes.indexOf(collisionMode);
    collisionMode = modes[(currentIndex + 1) % modes.length];
    e.target.textContent = `Modo: ${collisionMode}`;
  });
  
  document.getElementById('add-object').addEventListener('click', addNewObject);
  document.getElementById('clear-objects').addEventListener('click', clearAllObjects);
  
  document.getElementById('collision-sensitivity').addEventListener('input', (e) => {
    collisionSensitivity = parseFloat(e.target.value);
    document.getElementById('collision-sensitivity-value').textContent = collisionSensitivity.toFixed(1);
  });
  
  // Inicializar contador de objetos
  document.getElementById('object-count').textContent = interactiveObjects.length;
});

// ===== RESIZE =====
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ===== INICIALIZACIÓN =====
camera.position.set(5, 5, 5);
camera.lookAt(0, 0, 0);

// Variables para FPS
let lastTime = performance.now();
let frameCount = 0;

// ===== LOOP PRINCIPAL =====
function animate() {
  requestAnimationFrame(animate);
  
  const currentTime = performance.now();
  const deltaTime = (currentTime - lastTime) / 1000;
  lastTime = currentTime;
  
  globalTime += deltaTime;
  
  // Calcular FPS
  frameCount++;
  if (frameCount % 60 === 0) {
    const fps = Math.round(1 / deltaTime);
    document.getElementById('fps').textContent = fps;
  }
  
  // Movimiento de cámara con teclado
  const moveSpeed = cameraSpeed * deltaTime * 10;
  const cameraDirection = new THREE.Vector3();
  camera.getWorldDirection(cameraDirection);
  
  if (keyStates.w) camera.position.addScaledVector(cameraDirection, moveSpeed);
  if (keyStates.s) camera.position.addScaledVector(cameraDirection, -moveSpeed);
  if (keyStates.a) {
    const leftVector = new THREE.Vector3().crossVectors(camera.up, cameraDirection);
    camera.position.addScaledVector(leftVector, moveSpeed);
  }
  if (keyStates.d) {
    const rightVector = new THREE.Vector3().crossVectors(cameraDirection, camera.up);
    camera.position.addScaledVector(rightVector, moveSpeed);
  }
  if (keyStates.q) camera.position.y -= moveSpeed;
  if (keyStates.e) camera.position.y += moveSpeed;
  
  // Actualizar objetos interactivos
  interactiveObjects.forEach((object) => {
    const userData = object.userData;
    
    // Aplicar física básica
    object.position.add(userData.velocity.clone().multiplyScalar(deltaTime));
    userData.velocity.multiplyScalar(0.95); // Fricción
    userData.velocity.y -= 9.8 * deltaTime; // Gravedad
    
    // Colisión con suelo
    if (object.position.y < -1) {
      object.position.y = -1;
      userData.velocity.y = Math.abs(userData.velocity.y) * 0.7; // Rebote
    }
    
    // Animaciones
    if (userData.isAnimating) {
      userData.animationTime += deltaTime;
      const scale = 1 + Math.sin(userData.animationTime * 10) * 0.1;
      object.scale.setScalar(scale);
      
      if (userData.animationTime > 2) {
        userData.isAnimating = false;
        object.scale.copy(userData.originalScale);
        object.material.color.copy(userData.originalColor);
      }
    }
  });
  
  // Modo proximidad
  if (collisionMode === 'proximity') {
    interactiveObjects.forEach((object) => {
      const distance = camera.position.distanceTo(object.position);
      if (distance < 3 * collisionSensitivity && !object.userData.isAnimating) {
        triggerCollisionEffect(object);
      }
    });
  }
  
  // Actualizar partículas
  for (let i = particles.length - 1; i >= 0; i--) {
    if (!particles[i].update(deltaTime)) {
      particles.splice(i, 1);
    }
  }
  
  // Actualizar controles
  controls.update();
  
  // Renderizar
  renderer.render(scene, camera);
}

animate();

