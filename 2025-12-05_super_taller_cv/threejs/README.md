# Subsistema 3: Visualización 3D optimizada (Three.js + AR.js)

* Escena principal en Three.js / React Three Fiber con overlays dinámicos.
* Implementación de modelos 3D interactivos o animados.
* Integración AR.js con marcadores personalizados.

## Cómo ejecutar la demo AR (sin bundlers)

No usamos herramientas de empaquetado; todo se carga por CDN en `index.html`. Para evitar problemas de permisos de cámara, sirve la carpeta con un servidor simple:

```bash
cd threejs
python3 -m http.server 8080
# luego abre http://localhost:8080 en el navegador y permite la cámara
```

## Arquitectura mínima

* HTML estático (`index.html`) + scripts CDN (`three@0.122.0`, `ar.js@3.4.4`).
* Sin build steps ni bundlers; cualquier cambio se refleja al recargar.
* El feed de la cámara llena la pantalla y el canvas se dibuja encima.
* Los assets viven en `threejs/assets/` (marcadores, modelos, texturas).

## Marcadores personalizados

* Para marcador patrón (`.patt`), usa el generador web:  
  https://jeromeetienne.github.io/AR.js/three.js/examples/marker-training/examples/generator.html
* Para image/NFT (`.mind`), genera el descriptor con `@ar-js-org/artoolkit5-nft-generator` o el creador web de AR.js y pon el archivo en `assets/`.
* En `index.html`, ajusta `descriptorsUrl` (NFT) o `patternUrl` (patt) a tu archivo en `assets/`.

## Modelos animados

Utilizamos un modelo GLB de una araña de https://poly.pizza/m/yRYJiAJyiM para obtener `Spider.glb` con animaciones.
