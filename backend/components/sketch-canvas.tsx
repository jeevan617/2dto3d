"use client"

import { useRef, useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Eraser, Trash2 } from "lucide-react"
import { drawGuide } from "@/lib/guide-drawer"

interface SketchCanvasProps {
    onSketchComplete?: (canvas: HTMLCanvasElement) => void
    width?: number
    height?: number
    guideObject?: string | null
}

export default function SketchCanvas({ onSketchComplete, width = 400, height = 400, guideObject }: SketchCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const guideCanvasRef = useRef<HTMLCanvasElement>(null)
    const [isDrawing, setIsDrawing] = useState(false)
    const [hasDrawn, setHasDrawn] = useState(false)

    // Draw Guide Effect
    useEffect(() => {
        const guideCanvas = guideCanvasRef.current
        if (!guideCanvas || !guideObject) return

        const ctx = guideCanvas.getContext("2d")
        if (!ctx) return

        // Clear previous guide
        ctx.clearRect(0, 0, width, height)

        // Draw new guide
        drawGuide(ctx, guideObject, width, height)

    }, [guideObject, width, height])

    // Clear guide if none selected
    useEffect(() => {
        const guideCanvas = guideCanvasRef.current
        if (!guideCanvas) return
        const ctx = guideCanvas.getContext("2d")

        if (!guideObject && ctx) {
            ctx.clearRect(0, 0, width, height)
        }
    }, [guideObject, width, height])

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext("2d")
        if (!ctx) return

        // Initialize transparent canvas
        ctx.clearRect(0, 0, width, height)

        // Set drawing style
        ctx.strokeStyle = "#000000"
        ctx.lineWidth = 4
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

        ctx.clearRect(0, 0, width, height)
        setHasDrawn(false)
    }

    const handleRecognize = () => {
        const canvas = canvasRef.current
        if (canvas && onSketchComplete) {
            // Create a temp canvas with white background for the recognizer
            // (Recognizers usually expect white bg, not transparent)
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = width;
            tempCanvas.height = height;
            const tCtx = tempCanvas.getContext('2d');
            if (tCtx) {
                tCtx.fillStyle = 'white';
                tCtx.fillRect(0, 0, width, height);
                tCtx.drawImage(canvas, 0, 0);
                onSketchComplete(tempCanvas);
            } else {
                onSketchComplete(canvas);
            }
        }
    }

    return (
        <div className="flex flex-col gap-4">
            <div className="relative border-2 border-gray-300 rounded-lg overflow-hidden bg-white shadow-sm" style={{ width, height }}>
                {/* Procedural Guide Canvas Layer */}
                <canvas
                    ref={guideCanvasRef}
                    width={width}
                    height={height}
                    className="absolute inset-0 pointer-events-none z-0 opacity-40 mix-blend-multiply"
                />

                {/* User Drawing Layer */}
                <canvas
                    ref={canvasRef}
                    width={width}
                    height={height}
                    className="absolute inset-0 cursor-crosshair touch-none z-10"
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
