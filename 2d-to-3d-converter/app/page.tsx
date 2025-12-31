"use client"

import { useState, useRef } from "react"
import { Card } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import SketchCanvas from "@/components/sketch-canvas"
import ModelLoader from "@/components/model-loader"
import { getBestMatch } from "@/lib/sketch-recognizer"
import { getModelByName, getModelNamesForCategory, CATEGORIES, CATEGORY_LABELS, type ModelInfo, type Category } from "@/lib/model-registry"
import { Sparkles, Box, Layers, Wand2 } from "lucide-react"

import { Progress } from "@/components/ui/progress"
import { AlertCircle, WifiOff } from "lucide-react"

export default function Home() {
  const [selectedCategory, setSelectedCategory] = useState<Category | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [generated2D, setGenerated2D] = useState<string | null>(null)
  const [sketchTexture, setSketchTexture] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingStep, setProcessingStep] = useState("")

  // Loading animation state
  const [progress2D, setProgress2D] = useState(0)
  const [progress3D, setProgress3D] = useState(0)
  const [generationStage, setGenerationStage] = useState<'idle' | 'generating-2d' | 'generating-3d' | 'complete'>('idle')

  // Network simulation state
  const [networkStatus, setNetworkStatus] = useState<string | null>(null)
  const [showNetworkWarning, setShowNetworkWarning] = useState(false)

  const handleCategoryChange = (category: string) => {
    setSelectedCategory(category as Category)
    // Reset when category changes
    setModelInfo(null)
    setGenerated2D(null)
    setSketchTexture(null)
    setGenerationStage('idle')
    setProgress2D(0)
    setProgress3D(0)
    setNetworkStatus(null)
    setShowNetworkWarning(false)
  }

  // Helper to simulate progress bar with network interruptions
  const simulateProgress = (
    setProgress: (value: number) => void,
    minTimeMs: number,
    maxTimeMs: number
  ): Promise<void> => {
    return new Promise((resolve) => {
      const startTime = Date.now()
      const duration = Math.floor(Math.random() * (maxTimeMs - minTimeMs + 1)) + minTimeMs
      const intervalTime = 100 // Update every 100ms

      const timer = setInterval(() => {
        const elapsed = Date.now() - startTime
        const percentage = Math.min((elapsed / duration) * 100, 99) // Cap at 99 until done

        setProgress(percentage)

        // Simulating network issues at specific percentages
        if (percentage > 40 && percentage < 45) {
          setNetworkStatus("Network unstable... rerouting packets")
        } else if (percentage > 75 && percentage < 80) {
          setNetworkStatus("Bandwidth low... optimizing download")
        } else {
          setNetworkStatus(null)
        }

        if (elapsed >= duration) {
          clearInterval(timer)
          setProgress(100)
          setNetworkStatus(null)
          resolve()
        }
      }, intervalTime)
    })
  }

  const handleSketchComplete = async (canvas: HTMLCanvasElement) => {
    if (!selectedCategory) {
      alert("Please select a category first!")
      return
    }

    // Clear previous results
    setModelInfo(null)
    setGenerated2D(null)
    setSketchTexture(null)
    setProgress2D(0)
    setProgress3D(0)
    setGenerationStage('idle')
    setNetworkStatus(null)
    setShowNetworkWarning(false)

    setIsProcessing(true)
    setProcessingStep("Analyzing sketch...")

    // 1. Analyze and Recognize
    await new Promise(r => setTimeout(r, 1000))

    // Recognize matching object (relaxed matching)
    const result = getBestMatch(canvas, selectedCategory)

    // Fallback logic
    let matchedModel: ModelInfo | undefined
    // Check confidence - if low (or fallback used), treat as "wrong/unstable" result
    const lowConfidence = !result || result.confidence < 0.4

    if (result) {
      matchedModel = getModelByName(result.objectName)
    } else {
      const available = getModelNamesForCategory(selectedCategory)
      if (available.length > 0) {
        matchedModel = getModelByName(available[0])
      }
    }

    if (!matchedModel) {
      setIsProcessing(false)
      return
    }

    // 2. Generate 2D (20-30 seconds)
    setGenerationStage('generating-2d')
    setProcessingStep("Generating high-fidelity 2D image...")
    await simulateProgress(setProgress2D, 20000, 30000)

    setGenerated2D(matchedModel.previewPath)

    // 3. Generate 3D (80-95 seconds)
    setGenerationStage('generating-3d')
    setProcessingStep("Constructing 3D geometry from 2D output...")
    await simulateProgress(setProgress3D, 80000, 95000)

    setModelInfo(matchedModel)

    // If it was a low confidence match, show the network warning
    if (lowConfidence) {
      setShowNetworkWarning(true)
    }

    setGenerationStage('complete')
    setIsProcessing(false)
    setProcessingStep("")
  }

  const availableObjects = selectedCategory ? getModelNamesForCategory(selectedCategory) : []

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8 bg-gradient-to-br from-gray-50 via-white to-gray-100">
      <div className="w-full max-w-7xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Sketch to 2D & 3D GenAI
          </h1>
          <p className="text-gray-600">Select a category, draw your vision, and generate a 3D model instantly</p>
        </div>

        {/* Category Selection */}
        <Card className="p-6 mb-6 bg-white shadow-lg border-2 border-blue-100">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Layers className="h-6 w-6 text-blue-600" />
              <h2 className="text-xl font-semibold">Step 1: Select Category</h2>
            </div>
            <Select value={selectedCategory || ""} onValueChange={handleCategoryChange}>
              <SelectTrigger className="w-[280px] border-2 border-blue-200">
                <SelectValue placeholder="Choose a category..." />
              </SelectTrigger>
              <SelectContent>
                {CATEGORIES.map((category) => (
                  <SelectItem key={category} value={category}>
                    {CATEGORY_LABELS[category]}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {selectedCategory && (
              <div className="ml-auto text-sm text-gray-600">
                {availableObjects.length} base models available
              </div>
            )}
          </div>

          {selectedCategory && (
            <div className="mt-4 p-3 bg-blue-50 rounded-md">
              <p className="text-sm font-medium text-blue-800 mb-2">
                Available base shapes in {CATEGORY_LABELS[selectedCategory]}:
              </p>
              <div className="flex flex-wrap gap-2">
                {availableObjects.map((obj) => (
                  <span
                    key={obj}
                    className="px-2 py-1 bg-white border border-blue-200 rounded text-xs capitalize"
                  >
                    {obj}
                  </span>
                ))}
              </div>
            </div>
          )}
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Sketch Input Section */}
          <Card className={`p-6 bg-white shadow-lg ${!selectedCategory ? 'opacity-50' : ''}`}>
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                <span className="text-blue-600 font-semibold text-sm">2</span>
              </div>
              <h2 className="text-xl font-semibold">Draw Your Vision</h2>
            </div>

            {selectedCategory ? (
              <>
                <SketchCanvas onSketchComplete={handleSketchComplete} width={350} height={350} />

                {isProcessing && (
                  <div className="mt-4 p-3 bg-blue-50 rounded-md flex items-center justify-center gap-2">
                    <Wand2 className="h-4 w-4 text-blue-700 animate-pulse" />
                    <p className="text-sm text-blue-700 font-medium">{processingStep}</p>
                  </div>
                )}
              </>
            ) : (
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center aspect-square flex flex-col items-center justify-center">
                <Layers className="h-16 w-16 text-gray-300 mb-4" />
                <p className="text-gray-500 font-medium">Please select a category first</p>
              </div>
            )}
          </Card>

          {/* 2D Output Section */}
          <Card className="p-6 bg-white shadow-lg">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center">
                <span className="text-purple-600 font-semibold text-sm">3</span>
              </div>
              <h2 className="text-xl font-semibold">2D Output</h2>
              <Sparkles className="h-5 w-5 text-purple-500 ml-auto" />
            </div>

            <div className="border-2 border-dashed border-gray-300 rounded-lg overflow-hidden bg-gray-50 aspect-square flex items-center justify-center relative p-4">
              {generationStage === 'generating-2d' ? (
                <div className="w-full text-center">
                  <Sparkles className="h-10 w-10 text-purple-500 animate-pulse mx-auto mb-4" />
                  <p className="text-sm font-medium text-purple-700 mb-2">Generating 2D Image...</p>
                  <Progress value={progress2D} className="h-2 w-full" />
                  <p className="text-xs text-gray-500 mt-2">{Math.round(progress2D)}%</p>
                  {networkStatus && (
                    <div className="mt-2 flex items-center justify-center gap-1 text-amber-600 animate-pulse">
                      <WifiOff className="h-3 w-3" />
                      <p className="text-xs font-medium">{networkStatus}</p>
                    </div>
                  )}
                </div>
              ) : generated2D ? (
                <>
                  <img
                    src={generated2D}
                    alt="Generated 2D"
                    className="w-full h-full object-contain"
                  />
                  <div className="absolute inset-0 bg-purple-500/10 mix-blend-overlay pointer-events-none" />
                </>
              ) : (
                <div className="text-center p-8">
                  <Sparkles className="h-12 w-12 text-gray-300 mx-auto mb-2" />
                  <p className="text-gray-400 text-sm">2D view will appear here</p>
                </div>
              )}
            </div>
          </Card>

          {/* 3D Model Section */}
          <Card className="p-6 bg-white shadow-lg">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
                <span className="text-green-600 font-semibold text-sm">4</span>
              </div>
              <h2 className="text-xl font-semibold">Generated 3D Model</h2>
              <Box className="h-5 w-5 text-green-500 ml-auto" />
            </div>

            <div className="h-[400px]">
              {generationStage === 'generating-3d' ? (
                <div className="w-full h-full flex flex-col items-center justify-center bg-gray-50 rounded-md p-8">
                  <Box className="h-12 w-12 text-green-600 animate-bounce mb-4" />
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Constructing 3D Model</h3>
                  <Progress value={progress3D} className="h-3 w-64 mb-2" />
                  <p className="text-sm text-gray-500">Processing geometry... {Math.round(progress3D)}%</p>
                  {networkStatus && (
                    <div className="mt-4 p-2 bg-amber-50 border border-amber-200 rounded text-center">
                      <p className="text-xs font-bold text-amber-700 flex items-center justify-center gap-2">
                        <WifiOff className="h-3 w-3" />
                        {networkStatus}
                      </p>
                    </div>
                  )}
                </div>
              ) : (
                <ModelLoader modelInfo={modelInfo} />
              )}
            </div>

            {modelInfo && generationStage === 'complete' && (
              <div className="mt-4 p-3 bg-green-50 rounded-md">
                <p className="text-xs text-green-600 flex items-center gap-1">
                  <Wand2 className="h-3 w-3" />
                  Model generated from sketch input
                </p>
                {showNetworkWarning && (
                  <div className="mt-2 p-2 bg-red-50 border border-red-100 rounded">
                    <p className="text-xs text-red-600 flex items-start gap-1">
                      <AlertCircle className="h-4 w-4 shrink-0" />
                      Generation affected by highly fluctuating internet speed. Result may vary.
                    </p>
                  </div>
                )}
              </div>
            )}
          </Card>
        </div>
      </div>
    </main>
  )
}
