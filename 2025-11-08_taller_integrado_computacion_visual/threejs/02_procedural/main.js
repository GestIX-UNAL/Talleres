import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Setup básico OPTIMIZADO
const canvas = document.querySelector("#c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false }); // Desactivar antialiasing
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(1); // Forzar pixel ratio a 1
// Desactivar sombras para mejor performance
// renderer.shadowMap.enabled = true;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
scene.fog = new THREE.Fog(0x1a1a2e, 10, 50);

// Cámara
const camera = new THREE.PerspectiveCamera(75, innerWidth/innerHeight, 0.1, 100);
camera.position.set(8, 6, 8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Iluminación OPTIMIZADA
const ambientLight = new THREE.AmbientLight(0x404040, 0.6); // Más luz ambiente
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(5, 10, 5);
// Sin sombras para mejor performance
scene.add(directionalLight);

// Variables de control
let currentMode = 'grid'; // 'grid', 'spiral', 'fractal', 'vertices', 'comparison'
const modes = ['grid', 'spiral', 'fractal', 'vertices', 'comparison'];
let modeIndex = 0;

// Grupos para diferentes algoritmos
const gridGroup = new THREE.Group();
const spiralGroup = new THREE.Group();
const fractalGroup = new THREE.Group();
const vertexGroup = new THREE.Group();
const comparisonGroup = new THREE.Group();

scene.add(gridGroup);
scene.add(spiralGroup);
scene.add(fractalGroup);
scene.add(vertexGroup);
scene.add(comparisonGroup);

// Rejilla de cubos (bucles)
const cubes = [];
// OPTIMIZADO: Rejilla más pequeña y eficiente
function createGrid() {
  gridGroup.clear();
  cubes.length = 0; // Limpiar array
  
  // Reducir de 11x11 a 7x7 para mejor performance
  for(let x=-3; x<=3; x++){
    for(let z=-3; z<=3; z++){
      const height = Math.random() * 1.5 + 0.3;
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, height, 0.6),
        new THREE.MeshStandardMaterial({ 
          color: new THREE.Color().setHSL((height / 2), 0.8, 0.6),
          metalness: 0.2, 
          roughness: 0.7 
        })
      );
      m.position.set(x, height/2, z);
      // Remover shadows para mejor performance
      gridGroup.add(m);
      cubes.push(m);
    }
  }
}

// OPTIMIZADO: Espirales más simples
function createSpiral() {
  spiralGroup.clear();
  
  // Espiral logarítmica reducida
  for(let i=0; i<50; i++){ // Reducido de 300 a 50
    const t = i * 0.2;
    const r = 0.05 * i;
    const s = new THREE.Mesh(
      new THREE.SphereGeometry(0.08, 6, 6), // Menos segmentos
      new THREE.MeshBasicMaterial({ // Cambiar a Basic para mejor performance
        color: new THREE.Color().setHSL((i / 50) * 2, 0.8, 0.6)
      })
    );
    s.position.set(
      r * Math.cos(t), 
      0.05 * i - 2, 
      r * Math.sin(t)
    );
    spiralGroup.add(s);
  }
  
  // Helix simple
  for(let i=0; i<20; i++){ // Reducido de 100 a 20
    const t = i * 0.5;
    const radius = 1.5;
    
    const h1 = new THREE.Mesh(
      new THREE.SphereGeometry(0.1, 6, 6),
      new THREE.MeshBasicMaterial({ color: 0x00ff88 })
    );
    h1.position.set(
      radius * Math.cos(t), 
      i * 0.15 - 3,
      radius * Math.sin(t)
    );
    spiralGroup.add(h1);
  }
  
  spiralGroup.position.set(-6, 0, 0);
}

// OPTIMIZADO: Fractal más simple
function createFractalTree(position, direction, length, depth, group) {
  if (depth <= 0 || length < 0.2) return; // Mayor umbral para parar antes
  
  // Crear rama más simple
  const geometry = new THREE.CylinderGeometry(
    length * 0.05, 
    length * 0.08, 
    length, 
    4 // Menos segmentos
  );
  const material = new THREE.MeshBasicMaterial({ // Cambiar a Basic
    color: new THREE.Color().setHSL(0.12 - depth * 0.05, 0.8, 0.4 + depth * 0.1)
  });
  
  const branch = new THREE.Mesh(geometry, material);
  branch.position.copy(position);
  branch.position.add(direction.clone().multiplyScalar(length / 2));
  
  branch.lookAt(position.clone().add(direction));
  branch.rotateX(Math.PI / 2);
  
  group.add(branch);
  
  // Punto final de la rama
  const endPoint = position.clone().add(direction.clone().multiplyScalar(length));
  
  // Menos ramas para reducir complejidad
  const numBranches = 2; // Fijo en 2 en lugar de random
  for(let i = 0; i < numBranches; i++) {
    const angle = (i / numBranches) * Math.PI * 2;
    const newDirection = new THREE.Vector3(
      Math.cos(angle) * 0.7 + direction.x * 0.3,
      direction.y * 0.8,
      Math.sin(angle) * 0.7 + direction.z * 0.3
    ).normalize();
    
    createFractalTree(
      endPoint,
      newDirection,
      length * 0.7, // Factor fijo
      depth - 1,
      group
    );
  }
}

function createFractals() {
  fractalGroup.clear();
  
  // Árbol fractal más pequeño
  createFractalTree(
    new THREE.Vector3(0, -2, 0),
    new THREE.Vector3(0, 1, 0),
    1.5,
    4, // Reducido de 6 a 4 niveles
    fractalGroup
  );
  
  // Sierpinski simplificado
  function sierpinski(p1, p2, p3, depth) {
    if(depth <= 0) return;
    
    const m1 = p1.clone().lerp(p2, 0.5);
    const m2 = p2.clone().lerp(p3, 0.5);
    const m3 = p3.clone().lerp(p1, 0.5);
    
    const geometry = new THREE.BufferGeometry();
    const vertices = new Float32Array([
      m1.x, m1.y, m1.z,
      m2.x, m2.y, m2.z,
      m3.x, m3.y, m3.z
    ]);
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    
    const material = new THREE.MeshBasicMaterial({ 
      color: new THREE.Color().setHSL(depth * 0.2, 0.8, 0.6),
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.6
    });
    
    const triangle = new THREE.Mesh(geometry, material);
    fractalGroup.add(triangle);
    
    // Solo recursión si depth > 1 para evitar demasiados triángulos
    if(depth > 1) {
      sierpinski(p1, m1, m3, depth - 1);
      sierpinski(m1, p2, m2, depth - 1);
      sierpinski(m3, m2, p3, depth - 1);
    }
  }
  
  // Sierpinski más pequeño
  sierpinski(
    new THREE.Vector3(-2, 1, 1),
    new THREE.Vector3(2, 1, 1), 
    new THREE.Vector3(0, 3, 1),
    3 // Reducido de 5 a 3 niveles
  );
  
  fractalGroup.position.set(5, 0, 0);
}

// OPTIMIZADO: Terreno más simple
function createCustomVertexMesh() {
  vertexGroup.clear();
  
  // Geometría más pequeña
  const geometry = new THREE.PlaneGeometry(6, 6, 16, 16); // Reducido de 32x32 a 16x16
  const positions = geometry.attributes.position;
  
  // Función de altura más simple
  for(let i = 0; i < positions.count; i++) {
    const x = positions.getX(i);
    const z = positions.getZ(i);
    
    // Ondas más simples
    const height = Math.sin(x * 0.3) * Math.cos(z * 0.3) * 1;
    positions.setY(i, height);
  }
  
  geometry.computeVertexNormals();
  
  // Material más simple
  const material = new THREE.MeshBasicMaterial({
    vertexColors: true,
    wireframe: false
  });
  
  // Colores más simples
  const colors = [];
  for(let i = 0; i < positions.count; i++) {
    const height = positions.getY(i);
    const normalizedHeight = (height + 2) / 4;
    const color = new THREE.Color().setHSL(normalizedHeight * 0.4, 0.7, 0.5);
    colors.push(color.r, color.g, color.b);
  }
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = -Math.PI / 2;
  vertexGroup.add(mesh);
  
  vertexGroup.position.set(0, -2, -5);
}

// OPTIMIZADO: Comparación más simple
function createComparison() {
  comparisonGroup.clear();
  
  // Lado izquierdo: Modelado procedural reducido
  const proceduralGroup = new THREE.Group();
  
  // Solo 15 elementos en lugar de 50
  for(let i = 0; i < 15; i++) {
    const angle = (i / 15) * Math.PI * 3;
    const radius = i * 0.15;
    const height = Math.sin(i * 0.5) * 1;
    
    const geometry = new THREE.BoxGeometry(0.3, 0.3, 0.3);
    const material = new THREE.MeshBasicMaterial({ // Cambiar a Basic
      color: new THREE.Color().setHSL((i / 15), 0.8, 0.6)
    });
    
    const cube = new THREE.Mesh(geometry, material);
    cube.position.set(
      Math.cos(angle) * radius,
      height,
      Math.sin(angle) * radius
    );
    cube.rotation.y = angle;
    proceduralGroup.add(cube);
  }
  proceduralGroup.position.set(-3, 0, 0);
  comparisonGroup.add(proceduralGroup);
  
  // Lado derecho: Modelado manual simplificado
  const manualGroup = new THREE.Group();
  
  const shapes = [
    { geo: new THREE.BoxGeometry(0.8, 0.8, 0.8), pos: [0, 0, 0], color: 0xff6b6b },
    { geo: new THREE.SphereGeometry(0.5, 8, 8), pos: [1.5, 0.5, 0], color: 0x4ecdc4 },
    { geo: new THREE.ConeGeometry(0.4, 1), pos: [-1.5, 0.5, 0], color: 0x45b7d1 }
  ];
  
  shapes.forEach(shape => {
    const material = new THREE.MeshBasicMaterial({ color: shape.color });
    const mesh = new THREE.Mesh(shape.geo, material);
    mesh.position.set(...shape.pos);
    manualGroup.add(mesh);
  });
  
  manualGroup.position.set(3, 0, 0);
  comparisonGroup.add(manualGroup);
  
  comparisonGroup.position.set(0, 0, 5);
}

// Deformación senoidal de la rejilla (modificación dinámica)
const clock = new THREE.Clock();
// Inicialización
createGrid();
createSpiral();
createFractals();
createCustomVertexMesh();
createComparison();

// Controles de visibilidad
function showMode(mode) {
  // Ocultar todos los grupos
  [gridGroup, spiralGroup, fractalGroup, vertexGroup, comparisonGroup].forEach(group => {
    group.visible = false;
  });
  
  // Mostrar el grupo seleccionado
  switch(mode) {
    case 'grid':
      gridGroup.visible = true;
      camera.position.set(8, 6, 8);
      break;
    case 'spiral':
      spiralGroup.visible = true;
      camera.position.set(-2, 4, 8);
      break;
    case 'fractal':
      fractalGroup.visible = true;
      camera.position.set(12, 6, 8);
      break;
    case 'vertices':
      vertexGroup.visible = true;
      camera.position.set(0, 4, 4);
      break;
    case 'comparison':
      comparisonGroup.visible = true;
      camera.position.set(0, 6, 15);
      break;
    case 'all':
      [gridGroup, spiralGroup, fractalGroup, vertexGroup, comparisonGroup].forEach(group => {
        group.visible = true;
      });
      camera.position.set(0, 12, 20);
      break;
  }
  controls.update();
}

// Controles de teclado
window.addEventListener('keydown', (event) => {
  const key = event.key.toLowerCase();
  
  switch(key) {
    case '1':
      currentMode = 'grid';
      modeIndex = 0;
      break;
    case '2':
      currentMode = 'spiral';
      modeIndex = 1;
      break;
    case '3':
      currentMode = 'fractal';
      modeIndex = 2;
      break;
    case '4':
      currentMode = 'vertices';
      modeIndex = 3;
      break;
    case '5':
      currentMode = 'comparison';
      modeIndex = 4;
      break;
    case '0':
      currentMode = 'all';
      modeIndex = 5;
      break;
    case 'n': // Next mode
      modeIndex = (modeIndex + 1) % (modes.length + 1);
      currentMode = modeIndex < modes.length ? modes[modeIndex] : 'all';
      break;
  }
  
  showMode(currentMode);
  updateUI();
});

// Crear UI
function createUI() {
  const ui = document.getElementById('ui');
  ui.innerHTML = `
    <h3>Modelado Procedural</h3>
    <div class="controls">
      <strong>Controles:</strong><br>
      1-5 = Cambiar modo | 0 = Mostrar todo | N = Siguiente<br>
      Mouse = Rotar/Zoom
    </div>
    <div>
      <strong>Modo actual:</strong> <span id="current-mode">Rejilla</span><br>
      <strong>Algoritmo:</strong> <span id="algorithm-info">Bucles anidados</span>
    </div>
    <div class="controls">
      <button class="mode-btn" onclick="switchMode('grid')">1. Rejilla</button>
      <button class="mode-btn" onclick="switchMode('spiral')">2. Espiral</button>
      <button class="mode-btn" onclick="switchMode('fractal')">3. Fractal</button>
      <button class="mode-btn" onclick="switchMode('vertices')">4. Vértices</button>
      <button class="mode-btn" onclick="switchMode('comparison')">5. Comparación</button>
      <button class="mode-btn" onclick="switchMode('all')">0. Todo</button>
    </div>
    <div id="mode-description">
      <strong>Descripción:</strong><br>
      <span id="description-text">Rejilla procedural con alturas aleatorias y colores HSL dinámicos.</span>
    </div>
  `;
}

// Función global para botones
window.switchMode = function(mode) {
  currentMode = mode;
  showMode(mode);
  updateUI();
};

function updateUI() {
  const modeNames = {
    'grid': 'Rejilla Procedural',
    'spiral': 'Espirales Paramétricas', 
    'fractal': 'Fractales Recursivos',
    'vertices': 'Manipulación de Vértices',
    'comparison': 'Código vs Manual',
    'all': 'Vista Completa'
  };
  
  const algorithms = {
    'grid': 'Bucles anidados + Random + HSL',
    'spiral': 'Fórmulas paramétricas + Helix DNA',
    'fractal': 'Recursión + Sierpinski Triangle',
    'vertices': 'BufferGeometry + Vertex Shading',
    'comparison': 'Procedural vs Hardcoded',
    'all': 'Todos los algoritmos'
  };
  
  const descriptions = {
    'grid': 'Rejilla de cubos con alturas aleatorias y colores HSL basados en altura. Animación senoidal aplicada dinámicamente.',
    'spiral': 'Espiral logarítmica con esferas variables y doble hélice DNA con conectores. Rotación y colores interpolados.',
    'fractal': 'Árbol fractal 3D generado recursivamente con ramas aleatorias y hojas. Triángulo de Sierpinski con profundidad controlada.',
    'vertices': 'Terreno procedural mediante modificación directa de vértices. Colores por vertex basados en altura del terreno.',
    'comparison': 'Comparación lado a lado: estructura procedural compleja vs modelado manual con formas básicas.',
    'all': 'Visualización simultánea de todos los algoritmos procedurales implementados en diferentes posiciones.'
  };
  
  document.getElementById('current-mode').textContent = modeNames[currentMode];
  document.getElementById('algorithm-info').textContent = algorithms[currentMode];
  document.getElementById('description-text').textContent = descriptions[currentMode];
  
  // Actualizar botones activos
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  
  const activeBtn = document.querySelector(`button[onclick="switchMode('${currentMode}')"]`);
  if(activeBtn) activeBtn.classList.add('active');
}

// Resize handler
window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// Loop de renderizado principal
function render(){
  const t = clock.getElapsedTime();
  
  // Animación de rejilla (deformación senoidal)
  if(gridGroup.visible) {
    cubes.forEach((m, idx) => {
      const y = Math.sin(0.5 * t + m.position.x * 0.7 + m.position.z * 0.7);
      m.position.y = m.geometry.parameters.height / 2 + 0.6 * y;
    });
  }
  
  // Animaciones de espirales
  if(spiralGroup.visible) {
    spiralGroup.rotation.y = t * 0.2;
    spiralGroup.children.forEach((child, idx) => {
      if(child.geometry.type === 'SphereGeometry') {
        child.rotation.x = t + idx * 0.1;
      }
    });
  }
  
  // Animación de fractales
  if(fractalGroup.visible) {
    fractalGroup.rotation.y = Math.sin(t * 0.3) * 0.1;
    fractalGroup.traverse((child) => {
      if(child.material && child.material.color) {
        child.material.color.setHSL(
          (Math.sin(t * 0.5) + 1) * 0.5 * 0.6,
          0.8,
          0.5
        );
      }
    });
  }
  
  // Animación de terreno de vértices
  if(vertexGroup.visible) {
    const terrain = vertexGroup.children[0];
    if(terrain && terrain.geometry) {
      const positions = terrain.geometry.attributes.position;
      const originalPositions = terrain.geometry.userData.originalPositions;
      
      if(!originalPositions) {
        terrain.geometry.userData.originalPositions = positions.array.slice();
      } else {
        for(let i = 0; i < positions.count; i++) {
          const x = positions.getX(i);
          const z = positions.getZ(i);
          const originalY = originalPositions[i * 3 + 1];
          
          const wave = Math.sin(t * 2 + x * 0.5) * Math.cos(t * 1.5 + z * 0.3) * 0.5;
          positions.setY(i, originalY + wave);
        }
        positions.needsUpdate = true;
        terrain.geometry.computeVertexNormals();
      }
    }
  }
  
  // Animación de comparación
  if(comparisonGroup.visible) {
    comparisonGroup.children[0].rotation.y = t * 0.3; // Procedural side
    comparisonGroup.children[1].children.forEach((child, idx) => {
      child.rotation.y = t * 0.5 + idx;
      child.position.y = Math.sin(t + idx) * 0.2;
    });
  }
  
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(render);
}

// Inicialización final
createUI();
showMode(currentMode);
updateUI();
render();
