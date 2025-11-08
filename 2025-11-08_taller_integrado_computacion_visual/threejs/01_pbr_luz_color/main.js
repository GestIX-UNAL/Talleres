import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RGBELoader } from "three/examples/jsm/loaders/RGBELoader.js";



const canvas = document.querySelector("#c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1115);

// Loaders
const texLoader = new THREE.TextureLoader();
const rgbeLoader = new RGBELoader();

// Material Sets
const materials = {
  wood: {
    map: texLoader.load('./textures/wood_diff.jpg'),
    normalMap: texLoader.load('./textures/wood_normal.jpg'),
    roughnessMap: texLoader.load('./textures/wood_rough.jpg'),
    displacementMap: texLoader.load('./textures/wood_floor_disp_1k.jpg'),
    metalness: 0.0
  },
  brick: {
    map: texLoader.load('./textures/brick_diff.jpg'),
    normalMap: texLoader.load('./textures/brick_normal.jpg'),
    roughnessMap: texLoader.load('./textures/brick_rough.jpg'),
    metalness: 0.0
  },
  concrete: {
    map: texLoader.load('./textures/concrete_diff.jpg'),
    normalMap: texLoader.load('./textures/concrete_normal.jpg'),
    roughnessMap: texLoader.load('./textures/concrete_rough.jpg'),
    metalness: 0.0
  },
  metal: {
    map: texLoader.load('./textures/metal_diff.jpg'),
    normalMap: texLoader.load('./textures/metal_normal.jpg'),
    roughnessMap: texLoader.load('./textures/metal_rough.jpg'),
    metalnessMap: texLoader.load('./textures/metal_metalness.jpg'),
    metalness: 1.0
  }
};

// Load HDRI Environment
const pmremGenerator = new THREE.PMREMGenerator(renderer);
pmremGenerator.compileEquirectangularShader();
rgbeLoader.load('./textures/venice_sunset_1k.hdr', (hdr) => {
  const envMap = pmremGenerator.fromEquirectangular(hdr).texture;
  scene.environment = envMap;
  // scene.background = envMap; // Uncomment to use HDRI as background
  hdr.dispose();
  pmremGenerator.dispose();
});

// Cámaras
const frustumSize = 6;
const persp = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 100);
persp.position.set(3, 2, 6);

let aspect = innerWidth / innerHeight;
const ortho = new THREE.OrthographicCamera(
  -frustumSize * aspect / 2, frustumSize * aspect / 2,
  frustumSize / 2, -frustumSize / 2,
  0.1, 100
);
ortho.position.copy(persp.position);

const controlsPersp = new OrbitControls(persp, renderer.domElement);
const controlsOrtho = new OrbitControls(ortho, renderer.domElement);
controlsOrtho.enabled = false;

// Iluminación múltiple: key, fill, rim
const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
keyLight.position.set(5, 4, 3);
scene.add(keyLight);

const fillLight = new THREE.HemisphereLight(0x444466, 0x222222, 0.4);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffe0c0, 1.0);
rimLight.position.set(-5, 2, -3);
scene.add(rimLight);

// Esfera PBR
const geo = new THREE.SphereGeometry(1, 64, 64);
const mat = new THREE.MeshStandardMaterial({
  map: materials.wood.map,
  normalMap: materials.wood.normalMap,
  roughnessMap: materials.wood.roughnessMap,
  displacementMap: materials.wood.displacementMap,
  displacementScale: 0.02,
  metalness: materials.wood.metalness,
  roughness: 1.0,
  normalScale: new THREE.Vector2(1, 1)
});
const sphere = new THREE.Mesh(geo, mat);
scene.add(sphere);

// Current material type
let currentMaterial = 'wood';
const materialTypes = ['wood', 'brick', 'concrete', 'metal'];

// Controles de teclado
let usingOrtho = false;
window.addEventListener("keydown", e => {
  const key = e.key.toLowerCase();
  
  // Alternar cámara con "C"
  if (key === "c") {
    usingOrtho = !usingOrtho;
    controlsPersp.enabled = !usingOrtho;
    controlsOrtho.enabled = usingOrtho;
  }
  
  // Alternar material con "M"
  if (key === "m") {
    const currentIndex = materialTypes.indexOf(currentMaterial);
    const nextIndex = (currentIndex + 1) % materialTypes.length;
    currentMaterial = materialTypes[nextIndex];
    applyMaterial(currentMaterial);
  }
});

// Función para aplicar material
function applyMaterial(type) {
  const material = materials[type];
  mat.map = material.map;
  mat.normalMap = material.normalMap;
  mat.roughnessMap = material.roughnessMap;
  mat.metalnessMap = material.metalnessMap || null;
  mat.displacementMap = material.displacementMap || null;
  mat.metalness = material.metalness;
  mat.displacementScale = material.displacementMap ? 0.02 : 0;
  mat.needsUpdate = true;
}

// Resize
window.addEventListener("resize", () => {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  
  persp.aspect = w / h;
  persp.updateProjectionMatrix();
  
  const aspect = w / h;
  ortho.left = -frustumSize * aspect / 2;
  ortho.right = frustumSize * aspect / 2;
  ortho.top = frustumSize / 2;
  ortho.bottom = -frustumSize / 2;
  ortho.updateProjectionMatrix();
});

// Funciones para cálculos CIELAB
function srgbToLinear(c) { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
function rgbToXyz(r, g, b) {
  r = srgbToLinear(r); g = srgbToLinear(g); b = srgbToLinear(b);
  return [
    r * 0.4124564 + g * 0.3575761 + b * 0.1804375,
    r * 0.2126729 + g * 0.7151522 + b * 0.0721750,
    r * 0.0193339 + g * 0.1191920 + b * 0.9503041
  ];
}
function xyzToLab(x, y, z) {
  const Xn = 0.95047, Yn = 1.00000, Zn = 1.08883;
  const eps = 216/24389, k = 24389/27;
  function f(t) { return t > eps ? Math.cbrt(t) : (k * t + 16) / 116; }
  const fx = f(x / Xn), fy = f(y / Yn), fz = f(z / Zn);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
function rgbToLab(r, g, b) { return xyzToLab(...rgbToXyz(r, g, b)); }
function deltaE76(lab1, lab2) {
  const dl = lab1[0] - lab2[0], da = lab1[1] - lab2[1], db = lab1[2] - lab2[2];
  return Math.sqrt(dl * dl + da * da + db * db);
}
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const diff = max - min;
  const h = diff === 0 ? 0 : max === r ? ((g - b) / diff + 6) % 6 : max === g ? (b - r) / diff + 2 : (r - g) / diff + 4;
  return [Math.round(h * 60), Math.round(diff === 0 ? 0 : diff / max * 100), Math.round(max * 100)];
}

// Crear UI
const ui = document.getElementById('ui');
ui.innerHTML = `
  <div style="background: rgba(0,0,0,0.8); padding: 15px; border-radius: 8px; color: white; font-family: monospace; font-size: 12px;">
    <h3 style="margin: 0 0 10px 0;">PBR Material & Color Analysis</h3>
    <p><strong>Controls:</strong> C = Camera | M = Material</p>
    <p><strong>Current:</strong> <span id="current-material">Wood</span> | <span id="current-camera">Perspective</span></p>
    
    <div style="margin: 10px 0;">
      <h4>Color Palette</h4>
      <div style="display: flex; gap: 10px; margin: 5px 0;">
        <div style="display: flex; flex-direction: column; align-items: center;">
          <div id="color1" style="width: 30px; height: 30px; border: 1px solid white; background: #8B4513;"></div>
          <small>Material</small>
        </div>
        <div style="display: flex; flex-direction: column; align-items: center;">
          <div id="color2" style="width: 30px; height: 30px; border: 1px solid white; background: #FFE0C0;"></div>
          <small>Rim Light</small>
        </div>
      </div>
      
      <div id="color-info" style="margin: 10px 0; font-size: 11px;">
        <div><strong>Material RGB:</strong> <span id="rgb1">139, 69, 19</span></div>
        <div><strong>Material HSV:</strong> <span id="hsv1">30°, 86%, 55%</span></div>
        <div><strong>Rim Light RGB:</strong> <span id="rgb2">255, 224, 192</span></div>
        <div><strong>Rim Light HSV:</strong> <span id="hsv2">30°, 25%, 100%</span></div>
        <div><strong>CIELAB ΔE:</strong> <span id="delta-e">45.2</span> (High Contrast ✓)</div>
      </div>
    </div>
  </div>
`;

// Colores de referencia para cada material
const materialColors = {
  wood: { r: 139, g: 69, b: 19 },
  brick: { r: 150, g: 75, b: 50 },
  concrete: { r: 120, g: 120, b: 120 },
  metal: { r: 180, g: 180, b: 180 }
};
const rimColor = { r: 255, g: 224, b: 192 };

function updateColorInfo() {
  const matColor = materialColors[currentMaterial];
  const lab1 = rgbToLab(matColor.r, matColor.g, matColor.b);
  const lab2 = rgbToLab(rimColor.r, rimColor.g, rimColor.b);
  const hsv1 = rgbToHsv(matColor.r, matColor.g, matColor.b);
  const hsv2 = rgbToHsv(rimColor.r, rimColor.g, rimColor.b);
  const deltaE = deltaE76(lab1, lab2);
  
  document.getElementById('current-material').textContent = currentMaterial.charAt(0).toUpperCase() + currentMaterial.slice(1);
  document.getElementById('current-camera').textContent = usingOrtho ? 'Orthographic' : 'Perspective';
  document.getElementById('color1').style.background = `rgb(${matColor.r}, ${matColor.g}, ${matColor.b})`;
  document.getElementById('rgb1').textContent = `${matColor.r}, ${matColor.g}, ${matColor.b}`;
  document.getElementById('hsv1').textContent = `${hsv1[0]}°, ${hsv1[1]}%, ${hsv1[2]}%`;
  document.getElementById('rgb2').textContent = `${rimColor.r}, ${rimColor.g}, ${rimColor.b}`;
  document.getElementById('hsv2').textContent = `${hsv2[0]}°, ${hsv2[1]}%, ${hsv2[2]}%`;
  document.getElementById('delta-e').textContent = `${deltaE.toFixed(1)} ${deltaE > 20 ? '(High Contrast ✓)' : deltaE > 2 ? '(Perceptible)' : '(Low Contrast)'}`;
}

// Loop de renderizado
const clock = new THREE.Clock();
(function render() {
  const t = clock.getElapsedTime();
  
  // Animaciones de material y luz
  const roughnessBase = currentMaterial === 'metal' ? 0.1 : 0.5;
  mat.roughness = roughnessBase + 0.3 * (0.5 + 0.5 * Math.sin(t * 0.8));
  
  if (currentMaterial === 'metal') {
    mat.metalness = 0.8 + 0.2 * (0.5 + 0.5 * Math.sin(t * 0.6 + 1.5));
  }
  
  // Animar luces para mostrar variaciones
  keyLight.intensity = 1.8 + 0.4 * Math.sin(t * 0.6);
  rimLight.intensity = 0.8 + 0.3 * Math.sin(t * 0.9 + 1.0);
  
  sphere.rotation.y += 0.01;
  
  // Actualizar controles
  if (usingOrtho) controlsOrtho.update();
  else controlsPersp.update();
  
  // Actualizar UI
  updateColorInfo();
  
  renderer.render(scene, usingOrtho ? ortho : persp);
  requestAnimationFrame(render);
})();
