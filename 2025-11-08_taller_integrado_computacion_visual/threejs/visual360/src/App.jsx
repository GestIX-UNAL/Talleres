import { Canvas } from "@react-three/fiber"
import { OrbitControls, Environment } from "@react-three/drei"
import * as THREE from "three"
import React, { useState, useEffect, useRef } from "react"

function PanoramaImage() {
  return (
    <>
      <OrbitControls />
      <Environment files="/bloem_field_sunrise_4k.hdr" background />
    </>
  )
}

function PanoramaVideo() {
  const meshRef = useRef()

  useEffect(() => {
    const video = document.createElement("video")
    video.src = "/20257855-hd_1920_1080_60fps.mp4"
    video.crossOrigin = "anonymous"
    video.loop = true
    video.muted = true
    video.play()

    const texture = new THREE.VideoTexture(video)
    texture.minFilter = THREE.LinearFilter
    texture.magFilter = THREE.LinearFilter
    texture.format = THREE.RGBFormat

    meshRef.current.material.map = texture
    meshRef.current.material.needsUpdate = true
  }, [])

  const togglePlay = () => {
    if (!videoRef.current) return;

    if (videoRef.current.paused) {
      videoRef.current.play();
      setIsPlaying(true);
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  return (
    <mesh ref={meshRef} scale={[-1, 1, 1]}>
      <sphereGeometry args={[500, 60, 40]} />
      <meshBasicMaterial side={THREE.BackSide} />
    </mesh>
  )
}

export default function App() {
  const [mode, setMode] = useState("image") // "image" o "video"

  return (
    <>
      <Canvas>
        <OrbitControls enableZoom={false} />
        {mode === "image" ? <PanoramaImage /> : <PanoramaVideo />}
      </Canvas>

      {/* Controles UI */}
      <div style={{ position: "absolute", top: 20, left: 20 }}>
        <button
          onClick={() => setMode("image")}
          style={{
            marginRight: 10,
            padding: "8px 14px",
            background: mode === "image" ? "#2c7" : "#555",
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
          }}
        >
          Imagen 360°
        </button>
        <button
          onClick={() => setMode("video")}
          style={{
            padding: "8px 14px",
            background: mode === "video" ? "#2c7" : "#555",
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: "pointer",
          }}
        >
          Video 360°
        </button>
      </div>
    </>
  )
}
