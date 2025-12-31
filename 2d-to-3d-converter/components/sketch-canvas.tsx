"use client"

import { useRef, useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Eraser, Trash2 } from "lucide-react"

interface SketchCanvasProps {
    onSketchComplete?: (canvas: HTMLCanvasElement) => void
    width?: number
    height?: number
}

export default function SketchCanvas({ onSketchComplete, width = 400, height = 400 }: SketchCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const [isDrawing, setIsDrawing] = useState(false)
    const [hasDrawn, setHasDrawn] = useState(false)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext("2d")
        if (!ctx) return

        // Set canvas background to white
        ctx.fillStyle = "white"
        ctx.fillRect(0, 0, width, height)

        // Set drawing style
        ctx.strokeStyle = "#000000"
        ctx.lineWidth = 3
        ctx.lineCap = "round"
        ctx.lineJoin = "round"
    }, [width, height])

    const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext("2d")
        if (!ctx) return

        setIsDrawing(true)
        setHasDrawn(true)

        const rect = canvas.getBoundingClientRect()
        const x = "touches" in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left
        const y = "touches" in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top

        ctx.beginPath()
        ctx.moveTo(x, y)
    }

    const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
        if (!isDrawing) return

        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext("2d")
        if (!ctx) return

        const rect = canvas.getBoundingClientRect()
        const x = "touches" in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left
        const y = "touches" in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top

        ctx.lineTo(x, y)
        ctx.stroke()
    }

    const stopDrawing = () => {
        setIsDrawing(false)
    }

    const clearCanvas = () => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext("2d")
        if (!ctx) return

        ctx.fillStyle = "white"
        ctx.fillRect(0, 0, width, height)
        setHasDrawn(false)
    }

    const handleRecognize = () => {
        const canvas = canvasRef.current
        if (canvas && onSketchComplete) {
            onSketchComplete(canvas)
        }
    }

    return (
        <div className="flex flex-col gap-4">
            <div className="border-2 border-gray-300 rounded-lg overflow-hidden bg-white shadow-sm">
                <canvas
                    ref={canvasRef}
                    width={width}
                    height={height}
                    className="cursor-crosshair touch-none"
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    onTouchStart={startDrawing}
                    onTouchMove={draw}
                    onTouchEnd={stopDrawing}
                />
            </div>

            <div className="flex gap-2">
                <Button
                    variant="outline"
                    size="sm"
                    onClick={clearCanvas}
                    className="flex-1"
                >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Clear
                </Button>
                <Button
                    onClick={handleRecognize}
                    disabled={!hasDrawn}
                    className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white border-0"
                >
                    Generate Model
                </Button>
            </div>

            <p className="text-sm text-gray-500 text-center">
                Draw your vision above to generate a unique 3D model
            </p>
        </div>
    )
}
