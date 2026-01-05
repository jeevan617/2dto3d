"use client"

import { useEffect, useRef, useState } from "react"
import { Canvas } from "@react-three/fiber"
import { OrbitControls, PerspectiveCamera, Center, useGLTF } from "@react-three/drei"
import { Button } from "@/components/ui/button"
import { Download } from "lucide-react"
import * as THREE from "three"
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js"
import type { ModelInfo } from "@/lib/model-registry"

interface ModelLoaderProps {
    modelInfo: ModelInfo | null
}

// Component to load and display GLB models
function GLBModel({ modelPath }: { modelPath: string }) {
    const { scene } = useGLTF(modelPath)

    // Clone scene to avoid instances sharing state if needed
    const sceneClone = useRef<THREE.Group>(null)
    const [clonedScene, setClonedScene] = useState<THREE.Group | null>(null)

    useEffect(() => {
        if (scene) {
            setClonedScene(scene.clone())
        }
    }, [scene])

    if (!clonedScene) return null

    return (
        <Center>
            <primitive object={clonedScene} ref={sceneClone} />
        </Center>
    )
}

// Component to load and display PLY models
function PLYModel({ modelPath }: { modelPath: string }) {
    const meshRef = useRef<THREE.Mesh>(null)
    const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null)

    useEffect(() => {
        const loader = new PLYLoader()
        loader.load(
            modelPath,
            (geometry) => {
                geometry.computeVertexNormals()
                setGeometry(geometry)
            },
            undefined,
            (error) => {
                console.error("Error loading PLY model:", error)
            }
        )
    }, [modelPath])

    if (!geometry) return null

    return (
        <Center>
            <mesh ref={meshRef} geometry={geometry}>
                <meshStandardMaterial
                    color="#cccccc"
                    roughness={0.5}
                    metalness={0.2}
                />
            </mesh>
        </Center>
    )
}

export default function ModelLoader({ modelInfo }: ModelLoaderProps) {
    const handleDownload = () => {
        if (modelInfo) {
            // Initiate download
            const link = document.createElement('a')
            link.href = modelInfo.modelPath
            link.download = modelInfo.name + (modelInfo.fileType === 'glb' ? '.glb' : '.ply')
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
        }
    }

    if (!modelInfo) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-gradient-to-b from-gray-50 to-gray-200 rounded-md min-h-[400px]">
                <p className="text-gray-500">Draw a sketch to generate 3D model</p>
            </div>
        )
    }

    return (
        <div className="relative h-full flex flex-col">
            <div className="absolute top-0 left-0 right-0 z-10 p-2 flex justify-between items-center">
                <div className="bg-white/90 backdrop-blur-sm px-3 py-1.5 rounded-md shadow-sm">
                    <p className="text-sm font-medium">{modelInfo.name}</p>
                    <p className="text-xs text-gray-500 capitalize">{modelInfo.category.replace('_', ' ')}</p>
                </div>

                <Button variant="outline" size="sm" className="bg-white/90 backdrop-blur-sm" onClick={handleDownload}>
                    <Download className="h-4 w-4 mr-1" />
                    Download
                </Button>
            </div>

            <div className="flex-1 min-h-[400px] bg-gradient-to-b from-gray-50 to-gray-200 rounded-md overflow-hidden mt-12">
                <Canvas shadows camera={{ position: [0, 0, 5], fov: 50 }}>
                    <PerspectiveCamera makeDefault position={[0, 0, 5]} />

                    {/* Lighting setup */}
                    <hemisphereLight intensity={0.6} color="#ffffff" groundColor="#444444" />
                    <directionalLight
                        position={[5, 5, 5]}
                        intensity={1.2}
                        castShadow
                        shadow-mapSize-width={2048}
                        shadow-mapSize-height={2048}
                    />
                    <directionalLight position={[-3, 2, -2]} intensity={0.4} color="#b0c4de" />
                    <pointLight position={[0, -3, -3]} intensity={0.6} color="#ffeedd" />
                    <ambientLight intensity={0.3} />

                    {/* Ground plane for shadows */}
                    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2, 0]} receiveShadow>
                        <planeGeometry args={[20, 20]} />
                        <shadowMaterial opacity={0.3} />
                    </mesh>

                    {/* Load the appropriate model type */}
                    {modelInfo.fileType === 'glb' ? (
                        <GLBModel modelPath={modelInfo.modelPath} />
                    ) : (
                        <PLYModel modelPath={modelInfo.modelPath} />
                    )}

                    <OrbitControls autoRotate autoRotateSpeed={1} enableDamping dampingFactor={0.05} />
                </Canvas>
            </div>
        </div>
    )
}
