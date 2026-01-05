"use client"

import { useEffect, useRef, useState } from "react"
import { Canvas, useLoader } from "@react-three/fiber"
import { OrbitControls, PerspectiveCamera, Center } from "@react-three/drei"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Download } from "lucide-react"
import * as THREE from "three"
import { OBJECT_TEMPLATES, OBJECT_TYPES, removeBackground, extractDominantColor } from "@/lib/object-detector"

interface ModelViewerProps {
  imageUrl: string | null
}

// Create 3D geometry based on object type
function createObjectGeometry(objectType: string): THREE.BufferGeometry {
  const template = OBJECT_TEMPLATES[objectType]

  switch (template.type) {
    case 'sphere': {
      const geometry = new THREE.SphereGeometry(
        template.params.radius,
        template.params.widthSegments,
        template.params.heightSegments
      )

      // Apply modifications
      if (template.params.indentTop) {
        // Create indent at top for apple
        const positions = geometry.attributes.position
        for (let i = 0; i < positions.count; i++) {
          const y = positions.getY(i)
          if (y > 0.7) {
            const factor = (y - 0.7) / 0.3
            positions.setY(i, y - factor * 0.15)
          }
        }
        positions.needsUpdate = true
      }

      if (template.params.flattenY) {
        // Flatten sphere for tomato
        const positions = geometry.attributes.position
        for (let i = 0; i < positions.count; i++) {
          positions.setY(i, positions.getY(i) * template.params.flattenY)
        }
        positions.needsUpdate = true
      }

      geometry.computeVertexNormals()
      return geometry
    }

    case 'cylinder': {
      const geometry = new THREE.CylinderGeometry(
        template.params.radiusTop,
        template.params.radiusBottom,
        template.params.height,
        template.params.radialSegments
      )
      geometry.computeVertexNormals()
      return geometry
    }

    case 'box': {
      const geometry = new THREE.BoxGeometry(
        template.params.width,
        template.params.height,
        template.params.depth,
        32, 32, 32
      )

      if (template.params.rounded) {
        // Smooth corners
        const positions = geometry.attributes.position
        for (let i = 0; i < positions.count; i++) {
          const x = positions.getX(i)
          const y = positions.getY(i)
          const z = positions.getZ(i)

          const maxX = template.params.width / 2
          const maxY = template.params.height / 2
          const maxZ = template.params.depth / 2

          if (Math.abs(x) > maxX * 0.9 && Math.abs(y) > maxY * 0.9) {
            const factor = 0.95
            positions.setX(i, x * factor)
            positions.setY(i, y * factor)
          }
          if (Math.abs(x) > maxX * 0.9 && Math.abs(z) > maxZ * 0.9) {
            const factor = 0.95
            positions.setX(i, x * factor)
            positions.setZ(i, z * factor)
          }
          if (Math.abs(y) > maxY * 0.9 && Math.abs(z) > maxZ * 0.9) {
            const factor = 0.95
            positions.setY(i, y * factor)
            positions.setZ(i, z * factor)
          }
        }
        positions.needsUpdate = true
      }

      geometry.computeVertexNormals()
      return geometry
    }

    case 'cone': {
      const geometry = new THREE.ConeGeometry(
        template.params.radius,
        template.params.height,
        template.params.radialSegments
      )
      geometry.computeVertexNormals()
      return geometry
    }

    case 'torus': {
      const geometry = new THREE.TorusGeometry(
        template.params.radius,
        template.params.tube,
        template.params.radialSegments,
        template.params.tubularSegments
      )
      geometry.computeVertexNormals()
      return geometry
    }

    case 'ellipsoid': {
      const geometry = new THREE.SphereGeometry(1, template.params.segments, template.params.segments)
      const positions = geometry.attributes.position
      for (let i = 0; i < positions.count; i++) {
        positions.setX(i, positions.getX(i) * template.params.radiusX)
        positions.setY(i, positions.getY(i) * template.params.radiusY)
        positions.setZ(i, positions.getZ(i) * template.params.radiusZ)
      }
      positions.needsUpdate = true
      geometry.computeVertexNormals()
      return geometry
    }

    case 'hemisphere': {
      const geometry = new THREE.SphereGeometry(
        template.params.radius,
        template.params.widthSegments,
        template.params.heightSegments,
        0,
        Math.PI * 2,
        0,
        Math.PI / 2
      )
      geometry.computeVertexNormals()
      return geometry
    }

    default:
      return new THREE.SphereGeometry(1, 64, 64)
  }
}

// Object-based 3D Model Component
function ObjectModel({ imageUrl, objectType }: { imageUrl: string; objectType: string }) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [texture, setTexture] = useState<THREE.Texture | null>(null)
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null)

  useEffect(() => {
    if (!imageUrl || !objectType) return

    // Create geometry for selected object type
    const geom = createObjectGeometry(objectType)
    setGeometry(geom)

    // Load and process texture
    const img = new Image()
    img.crossOrigin = "anonymous"
    img.onload = () => {
      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("2d")
      if (!ctx) return

      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

      // Remove background
      const { processedData } = removeBackground(imageData, canvas.width, canvas.height)

      // Create texture from processed image
      ctx.putImageData(processedData, 0, 0)

      const tex = new THREE.CanvasTexture(canvas)
      tex.needsUpdate = true
      setTexture(tex)
    }

    img.src = imageUrl
  }, [imageUrl, objectType])

  if (!texture || !geometry) return null

  const template = OBJECT_TEMPLATES[objectType]
  const materialHints = template.materialHints

  return (
    <Center>
      <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
        <meshStandardMaterial
          map={texture}
          roughness={materialHints.roughness}
          metalness={materialHints.metalness}
          envMapIntensity={0.8}
          transparent={true}
        />
      </mesh>
    </Center>
  )
}

export default function ModelViewer({ imageUrl }: ModelViewerProps) {
  const [objectType, setObjectType] = useState<string>("apple")

  const handleDownload = () => {
    alert("In a real implementation, this would download the 3D model file (GLB/OBJ).")
  }

  return (
    <div className="relative h-full flex flex-col">
      <div className="absolute top-0 left-0 right-0 z-10 p-2 flex justify-between items-center">
        <Select value={objectType} onValueChange={setObjectType}>
          <SelectTrigger className="w-[200px] bg-white">
            <SelectValue placeholder="Select object type" />
          </SelectTrigger>
          <SelectContent>
            {OBJECT_TYPES.map((type) => (
              <SelectItem key={type} value={type}>
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button variant="outline" size="sm" className="bg-white" onClick={handleDownload}>
          <Download className="h-4 w-4 mr-1" />
          Download Model
        </Button>
      </div>

      <div className="flex-1 min-h-[400px] bg-gradient-to-b from-gray-50 to-gray-200 rounded-md overflow-hidden mt-12">
        {imageUrl ? (
          <Canvas shadows camera={{ position: [0, 0, 3], fov: 50 }}>
            <PerspectiveCamera makeDefault position={[0, 0, 3]} />

            {/* Three-point lighting setup for realistic illumination */}
            <hemisphereLight intensity={0.6} color="#ffffff" groundColor="#444444" />

            {/* Key light - main light source */}
            <directionalLight
              position={[5, 5, 5]}
              intensity={1.2}
              castShadow
              shadow-mapSize-width={2048}
              shadow-mapSize-height={2048}
              shadow-camera-far={50}
              shadow-camera-left={-10}
              shadow-camera-right={10}
              shadow-camera-top={10}
              shadow-camera-bottom={-10}
            />

            {/* Fill light - soften shadows */}
            <directionalLight position={[-3, 2, -2]} intensity={0.4} color="#b0c4de" />

            {/* Rim light - highlight edges */}
            <pointLight position={[0, -3, -3]} intensity={0.6} color="#ffeedd" />

            {/* Additional ambient for overall brightness */}
            <ambientLight intensity={0.3} />

            {/* Ground plane for shadows */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1.5, 0]} receiveShadow>
              <planeGeometry args={[10, 10]} />
              <shadowMaterial opacity={0.3} />
            </mesh>

            <ObjectModel imageUrl={imageUrl} objectType={objectType} />

            <OrbitControls autoRotate autoRotateSpeed={1} enableDamping dampingFactor={0.05} />
          </Canvas>
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <p className="text-gray-500">Upload an image to get started</p>
          </div>
        )}
      </div>

      <div className="mt-4 text-center text-sm text-gray-600">
        <p>Select the object type that best matches your image for realistic 3D conversion</p>
      </div>
    </div>
  )
}
