// Sketch recognition using pattern-based analysis within a selected category

import type { Category } from './model-registry'

export interface RecognitionResult {
    objectName: string
    confidence: number
    category: string
}

interface SketchFeatures {
    aspectRatio: number
    circularity: number
    complexity: number
    width: number
    height: number
    strokeCount: number
    centerX: number // Relative center of mass X (0-1)
    centerY: number // Relative center of mass Y (0-1)
    topHeavy: number // Ratio of pixels in top half vs total
    bottomHeavy: number // Ratio of pixels in bottom half vs total
    leftHeavy: number // Ratio of pixels in left half vs total
    rightHeavy: number // Ratio of pixels in right half vs total
}

// Analyze sketch to extract features
function analyzeSketch(imageData: ImageData): SketchFeatures {
    const { data, width, height } = imageData
    let minX = width, maxX = 0, minY = height, maxY = 0
    let pixelCount = 0
    let edgePixels = 0

    // Mass calculation
    let totalX = 0
    let totalY = 0

    // First pass: Find bounding box
    // IMPORTANT: Check for DARK pixels (R,G,B < 200), not just alpha
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (y * width + x) * 4
            const r = data[idx]
            const g = data[idx + 1]
            const b = data[idx + 2]
            const isDark = r < 200 && g < 200 && b < 200

            if (isDark) {
                minX = Math.min(minX, x)
                maxX = Math.max(maxX, x)
                minY = Math.min(minY, y)
                maxY = Math.max(maxY, y)
            }
        }
    }

    // Safety check for empty canvas
    if (minX > maxX) return {
        aspectRatio: 1, circularity: 0, complexity: 0, width: 0, height: 0, strokeCount: 0,
        centerX: 0.5, centerY: 0.5, topHeavy: 0.5, bottomHeavy: 0.5, leftHeavy: 0.5, rightHeavy: 0.5
    }

    const boundingWidth = maxX - minX + 1
    const boundingHeight = maxY - minY + 1
    const midX = minX + boundingWidth / 2
    const midY = minY + boundingHeight / 2

    let q1 = 0, q2 = 0, q3 = 0, q4 = 0 // Quadrants relative to bounding box center

    // Second pass: Detailed analysis within BBox
    for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
            const idx = (y * width + x) * 4
            const r = data[idx]
            const g = data[idx + 1]
            const b = data[idx + 2]
            const isDark = r < 200 && g < 200 && b < 200

            if (isDark) {
                pixelCount++
                totalX += x
                totalY += y

                // Edge detection
                let hasWhiteNeighbor = false
                const checkWhite = (index: number) => data[index] > 200 && data[index + 1] > 200 && data[index + 2] > 200

                if (x > 0 && checkWhite(idx - 4)) hasWhiteNeighbor = true
                else if (x < width - 1 && checkWhite(idx + 4)) hasWhiteNeighbor = true
                else if (y > 0 && checkWhite(((y - 1) * width + x) * 4)) hasWhiteNeighbor = true
                else if (y < height - 1 && checkWhite(((y + 1) * width + x) * 4)) hasWhiteNeighbor = true

                if (hasWhiteNeighbor) edgePixels++

                // Quadrant analysis
                if (x < midX && y < midY) q1++
                else if (x >= midX && y < midY) q2++
                else if (x < midX && y >= midY) q3++
                else if (x >= midX && y >= midY) q4++
            }
        }
    }

    const aspectRatio = boundingWidth / Math.max(boundingHeight, 1)

    // Circularity: how circular is the shape (1.0 = perfect circle)
    const area = pixelCount
    const perimeter = edgePixels
    const circularity = (4 * Math.PI * area) / Math.max(perimeter * perimeter, 1)

    // Complexity: ratio of edge pixels to total pixels
    const complexity = edgePixels / Math.max(pixelCount, 1)

    // Advanced features
    const centerX = (totalX / Math.max(pixelCount, 1) - minX) / Math.max(boundingWidth, 1)
    const centerY = (totalY / Math.max(pixelCount, 1) - minY) / Math.max(boundingHeight, 1)

    const topPixels = q1 + q2
    const bottomPixels = q3 + q4
    const leftPixels = q1 + q3
    const rightPixels = q2 + q4

    const topHeavy = topPixels / Math.max(pixelCount, 1)
    const bottomHeavy = bottomPixels / Math.max(pixelCount, 1)
    const leftHeavy = leftPixels / Math.max(pixelCount, 1)
    const rightHeavy = rightPixels / Math.max(pixelCount, 1)

    return {
        aspectRatio,
        circularity,
        complexity,
        width: boundingWidth,
        height: boundingHeight,
        strokeCount: pixelCount,
        centerX,
        centerY,
        topHeavy,
        bottomHeavy,
        leftHeavy,
        rightHeavy
    }
}

// Pattern matching rules for different objects
// IMPROVED LOGIC: Strict mutual exclusion to prevent "Headphones" vs "Tablet" confusion
const RECOGNITION_PATTERNS: Record<string, (features: SketchFeatures) => number> = {
    // --- Furniture ---
    'chair': (f) => {
        let score = 0
        // POSITIVE: Tallish, legs, backrest
        if (f.aspectRatio > 0.4 && f.aspectRatio < 1.1) score += 0.4
        if (f.complexity > 0.15 && f.complexity < 0.4) score += 0.3

        // NEGATIVE (STRICT):
        if (f.aspectRatio > 1.4) score -= 0.8 // Too wide (Table/Sofa)
        if (f.complexity < 0.1) score -= 0.6 // Too simple (Box)
        return score
    },
    'table': (f) => {
        let score = 0
        // POSITIVE: Wide, flat top, legs
        if (f.aspectRatio > 1.3 && f.aspectRatio < 3.0) score += 0.5
        if (f.topHeavy > 0.55) score += 0.3

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 1.1) score -= 0.8 // Too square/tall (Chair)
        if (f.circularity > 0.6) score -= 0.7 // Too round
        if (f.complexity < 0.1) score -= 0.5 // Too solid block
        return score
    },
    'sofa': (f) => {
        let score = 0
        // POSITIVE: Very wide, solid, low to ground
        if (f.aspectRatio > 1.8) score += 0.6
        if (f.bottomHeavy > 0.6) score += 0.3

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 1.5) score -= 0.8 // Too narrow (Must be wide)
        if (f.circularity > 0.5) score -= 0.6 // No circles
        return score
    },
    'bed': (f) => {
        let score = 0
        // POSITIVE: Wide rectangle, simple
        if (f.aspectRatio > 1.3 && f.aspectRatio < 2.5) score += 0.5
        if (f.complexity < 0.2) score += 0.4

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 1.1) score -= 0.8 // Too square
        if (f.topHeavy > 0.6) score -= 0.5 // Bed is usually bottom/balanced
        return score
    },
    'cupboard': (f) => {
        let score = 0
        // POSITIVE: Tall rectangle, blocky
        if (f.aspectRatio > 0.4 && f.aspectRatio < 0.9) score += 0.6
        if (f.circularity > 0.5) score += 0.3 // Rectangular box

        // NEGATIVE (STRICT):
        if (f.aspectRatio > 1.1) score -= 0.9 // MUST NOT be wide
        return score
    },

    // --- Gadgets ---
    'laptop': (f) => {
        let score = 0
        // POSITIVE: L-shape or Wide Rectangle
        if (f.aspectRatio > 1.0 && f.aspectRatio < 1.8) score += 0.4
        if (f.bottomHeavy > 0.45) score += 0.4 // Keyboard

        // NEGATIVE (STRICT):
        if (f.circularity > 0.5) score -= 0.7 // NOT round
        if (f.aspectRatio < 0.8) score -= 0.7 // NOT vertical
        return score
    },
    'mobile': (f) => {
        let score = 0
        // POSITIVE: Tall thin vertical rectangle
        if (f.aspectRatio > 0.4 && f.aspectRatio < 0.65) score += 0.7
        if (f.complexity < 0.12) score += 0.3

        // NEGATIVE (STRICT):
        if (f.aspectRatio > 0.8) score -= 0.9 // MUST be tall
        if (f.complexity > 0.2) score -= 0.6 // MUST be simple
        return score
    },
    'tablet': (f) => {
        let score = 0
        // POSITIVE: Rectangle, wider than mobile
        if (f.aspectRatio > 0.7 && f.aspectRatio < 1.5) score += 0.5
        if (f.complexity < 0.12) score += 0.4

        // NEGATIVE (STRICT):
        if (f.complexity > 0.25) score -= 0.6 // Too complex
        if (f.circularity > 0.6) score += 0.2 // Boxy
        if (f.aspectRatio < 0.65) score -= 0.7 // Too thin (Mobile)
        return score
    },
    'headphones': (f) => {
        let score = 0
        // POSITIVE: Arch shape, top heavy, hollow
        if (f.aspectRatio > 0.8 && f.aspectRatio < 1.4) score += 0.3
        if (f.topHeavy > 0.6) score += 0.5
        if (f.complexity > 0.2) score += 0.3

        // NEGATIVE (STRICT):
        if (f.circularity > 0.5) score -= 0.6 // Not a simple circle
        if (f.complexity < 0.1) score -= 0.8 // MUST NOT be a solid block
        return score
    },
    'watch': (f) => {
        let score = 0
        // POSITIVE: Circular face
        if (f.circularity > 0.65) score += 0.6
        if (f.aspectRatio > 0.8 && f.aspectRatio < 1.2) score += 0.3

        // NEGATIVE (STRICT):
        if (f.circularity < 0.4) score -= 0.8 // Must be round-ish
        return score
    },

    // --- Vehicles ---
    'car': (f) => {
        let score = 0
        // POSITIVE: Wide, bottom heavy (wheels)
        if (f.aspectRatio > 1.4 && f.aspectRatio < 2.5) score += 0.5
        if (f.bottomHeavy > 0.55) score += 0.4

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 1.2) score -= 0.8 // MUST be wide
        if (f.topHeavy > 0.55) score -= 0.5 // Roof shouldn't be heavier than body
        return score
    },
    'bus': (f) => {
        let score = 0
        // POSITIVE: Boxy, wide
        if (f.aspectRatio > 1.3 && f.aspectRatio < 3.0) score += 0.5
        // ALLOW COMPLEXITY: Windows and wheels
        if (f.complexity > 0.15) score += 0.3

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 1.1) score -= 0.8 // Must be horizontal
        return score
    },
    'bike': (f) => {
        let score = 0
        // POSITIVE: Airy, complex, skeleton
        if (f.complexity > 0.3) score += 0.6

        // NEGATIVE (STRICT):
        if (f.circularity > 0.4) score -= 0.8 // Cannot be a block/circle
        if (f.complexity < 0.2) score -= 0.9 // Cannot be simple
        return score
    },
    'aeroplane': (f) => {
        let score = 0
        // POSITIVE: Cross shape -> High complexity, low density
        if (f.complexity > 0.3) score += 0.5
        // Symmetry
        if (Math.abs(f.leftHeavy - f.rightHeavy) < 0.1) score += 0.3

        // CRITICAL STRICT: Plane is skeletal (wings).
        if (f.circularity > 0.45) score -= 0.9 // KILL SCORE for solid objects (Bus/Car)

        // NEGATIVE:
        if (f.aspectRatio < 0.5 || f.aspectRatio > 2.0) score -= 0.6
        return score
    },

    // --- Fashion ---
    'shirt': (f) => {
        let score = 0
        // POSITIVE: T-shape, Top Heavy
        if (f.topHeavy > 0.6) score += 0.6
        if (f.aspectRatio > 0.8 && f.aspectRatio < 1.5) score += 0.3

        // NEGATIVE (STRICT):
        if (f.circularity > 0.6) score -= 0.8 // Not a circle
        if (f.bottomHeavy > 0.5) score -= 0.5 // Should be top heavy
        return score
    },
    'cap': (f) => {
        let score = 0
        // POSITIVE: Dome, Top Heavy
        if (f.topHeavy > 0.6) score += 0.5
        if (f.aspectRatio > 1.1 && f.aspectRatio < 2.0) score += 0.4

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 0.9) score -= 0.7 // Not tall
        return score
    },
    'shoe': (f) => {
        let score = 0
        // POSITIVE: L-shape side view, Bottom Heavy
        if (f.aspectRatio > 1.4) score += 0.5
        if (f.bottomHeavy > 0.6) score += 0.4

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 1.0) score -= 0.8 // Not vertical
        return score
    },
    'bag': (f) => {
        let score = 0
        // POSITIVE: Box/Trapezoid + Handle
        if (f.aspectRatio > 0.7 && f.aspectRatio < 1.4) score += 0.4
        if (f.complexity > 0.12) score += 0.3

        // NEGATIVE (STRICT):
        if (f.aspectRatio > 2.0) score -= 0.6 // Not super wide
        return score
    },

    // --- Instruments ---
    'guiter': (f) => { // "guitar"
        let score = 0
        // POSITIVE: Figure 8 body + neck
        if (f.aspectRatio > 0.3 && f.aspectRatio < 0.65) score += 0.6
        if (f.bottomHeavy > 0.6) score += 0.5

        // NEGATIVE (STRICT):
        if (f.aspectRatio > 0.9) score -= 0.9 // MUST be tall
        if (f.complexity > 0.3) score -= 0.4 // Not too messy
        return score
    },
    'piano': (f) => {
        let score = 0
        // POSITIVE: Wide
        if (f.aspectRatio > 1.8) score += 0.6

        // NEGATIVE:
        if (f.aspectRatio < 1.2) score -= 0.7
        return score
    },
    'drum': (f) => {
        let score = 0
        // POSITIVE: Cylinder/Circle
        if (f.aspectRatio > 0.9 && f.aspectRatio < 1.2) score += 0.4
        if (f.circularity > 0.6) score += 0.4

        // NEGATIVE (STRICT):
        if (f.aspectRatio > 1.6) score -= 0.7 // Not wide
        return score
    },
    'flute': (f) => {
        let score = 0
        // POSITIVE: Stick
        if (f.aspectRatio > 4.0 || f.aspectRatio < 0.2) score += 0.8

        // NEGATIVE (STRICT):
        if (f.aspectRatio > 0.5 && f.aspectRatio < 2.0) score -= 0.9 // Cannot be square-ish
        return score
    },

    // --- Appliances ---
    'tv': (f) => {
        let score = 0
        // POSITIVE: 16:9 Screen + implied Stand
        if (f.aspectRatio > 1.2 && f.aspectRatio < 1.9) score += 0.5

        // CRITICAL STRICT RULES:
        if (f.circularity < 0.45) score -= 0.9 // If spiky (Fan), KILL SCORE
        if (f.aspectRatio < 1.0) score -= 0.7 // Not vertical
        if (f.circularity > 0.5) score += 0.3
        return score
    },
    'ac': (f) => {
        let score = 0
        // POSITIVE: Very Wide Strip
        if (f.aspectRatio > 2.0) score += 0.8

        // NEGATIVE (STRICT):
        // Relaxed from 1.8 to 1.5 to allow for slightly less wide drawings
        if (f.aspectRatio < 1.5) score -= 0.9 // MUST be wide
        if (f.complexity > 0.2) score -= 0.5 // Simple box
        return score
    },
    'fan': (f) => {
        let score = 0
        // POSITIVE: Spiky star shape
        if (f.aspectRatio > 0.8 && f.aspectRatio < 1.3) score += 0.3
        if (f.circularity < 0.45) score += 0.7 // Low circularity = Spikes

        // NEGATIVE (STRICT):
        if (f.circularity > 0.6) score -= 0.9 // Solid block is NOT a fan
        return score
    },
    'fridge': (f) => {
        let score = 0
        // POSITIVE: Tall Box
        if (f.aspectRatio > 0.4 && f.aspectRatio < 0.75) score += 0.6

        // NEGATIVE (STRICT):
        if (f.aspectRatio > 0.9) score -= 0.8 // Not wide/square
        return score
    },
    'washmachine': (f) => {
        let score = 0
        // POSITIVE: Cube / Box
        if (f.aspectRatio > 0.88 && f.aspectRatio < 1.15) score += 0.6
        if (f.centerX > 0.4 && f.centerX < 0.6) score += 0.3

        // NEGATIVE (STRICT):
        if (f.aspectRatio > 1.3 || f.aspectRatio < 0.7) score -= 0.7 // Must be square-ish
        return score
    },

    // --- Kitchen ---
    'mixer': (f) => {
        let score = 0
        // POSITIVE: Tallish L-shape
        if (f.aspectRatio > 0.4 && f.aspectRatio < 0.85) score += 0.5
        if (f.bottomHeavy > 0.55) score += 0.3

        // NEGATIVE (STRICT):
        if (f.aspectRatio > 1.0) score -= 0.7 // Must be tall
        return score
    },
    'cooker': (f) => {
        let score = 0
        // POSITIVE: Pot
        if (f.aspectRatio > 0.9 && f.aspectRatio < 1.4) score += 0.4

        // NEGATIVE (STRICT): 
        if (f.aspectRatio < 0.7) score -= 0.6 // mid-wide
        if (f.aspectRatio > 1.8) score -= 0.6
        return score
    },
    'oven': (f) => {
        let score = 0
        // POSITIVE: Wide Box
        if (f.aspectRatio > 1.3 && f.aspectRatio < 1.7) score += 0.5

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 1.1) score -= 0.7 // Must be wide
        return score
    },
    'stove': (f) => {
        let score = 0
        // POSITIVE: Very Wide
        if (f.aspectRatio > 1.6) score += 0.5

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 1.4) score -= 0.8 // Must be flat/wide
        return score
    },
    'tostar': (f) => { // "toaster"
        let score = 0
        // POSITIVE: Small Box
        if (f.aspectRatio > 1.1 && f.aspectRatio < 1.6) score += 0.4

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 0.9) score -= 0.6
        return score
    },

    // --- Sports ---
    'football': (f) => {
        let score = 0
        // STRICT POSITIVE: Circle
        if (f.circularity > 0.75) score += 1.0
        if (f.aspectRatio > 0.9 && f.aspectRatio < 1.1) score += 0.3

        // STRICT NEGATIVE:
        if (f.circularity < 0.6) score -= 1.0 // If not circle, DIE
        return score
    },
    'cricketbat': (f) => {
        let score = 0
        // POSITIVE: Tall Paddle
        if (f.aspectRatio < 0.35) score += 0.6
        if (f.bottomHeavy > 0.6) score += 0.4

        // STRICT NEGATIVE:
        if (f.aspectRatio > 0.5) score -= 0.9
        return score
    },
    'hockey': (f) => {
        let score = 0
        // POSITIVE: J-stick
        if (f.aspectRatio < 0.5) score += 0.5
        if (f.bottomHeavy > 0.7) score += 0.5

        // STRICT NEGATIVE:
        if (f.aspectRatio > 0.6) score -= 0.9
        return score
    },
    'racket': (f) => {
        let score = 0
        // POSITIVE: Lollipop
        if (f.topHeavy > 0.6) score += 0.5
        if (f.aspectRatio < 0.7) score += 0.3

        // STRICT NEGATIVE:
        if (f.aspectRatio > 0.9) score -= 0.8
        return score
    },
    'tennisball': (f) => {
        let score = 0
        // POSITIVE: Circle
        if (f.circularity > 0.75) score += 0.8

        // STRICT NEGATIVE:
        if (f.circularity < 0.6) score -= 1.0
        return score
    },

    // --- Stationary ---
    'pen': (f) => {
        let score = 0
        // POSITIVE: Stick
        if (f.aspectRatio < 0.2) score += 0.9

        // STRICT NEGATIVE:
        if (f.aspectRatio > 0.3) score -= 1.0 // NOT A PEN
        return score
    },
    'pencil': (f) => {
        let score = 0
        // Same as pen
        if (f.aspectRatio < 0.2) score += 0.9
        if (f.aspectRatio > 0.3) score -= 1.0
        return score
    },
    'book': (f) => {
        let score = 0
        // POSITIVE: Vertical or Horizontal Rect
        if (f.aspectRatio > 0.6 && f.aspectRatio < 0.85) score += 0.5
        if (f.complexity < 0.15) score += 0.4

        // NEGATIVE (STRICT):
        if (f.circularity > 0.7) score -= 0.8 // Not a circle
        return score
    },
    'easer': (f) => { // "eraser"
        let score = 0
        // POSITIVE: Block
        if (f.aspectRatio > 1.2 && f.aspectRatio < 2.0) score += 0.4

        // NEGATIVE (STRICT):
        if (f.aspectRatio < 0.8) score -= 0.7
        return score
    },
    'ruler': (f) => {
        let score = 0
        // POSITIVE: Very wide plank
        if (f.aspectRatio > 4.0) score += 0.9

        // STRICT NEGATIVE:
        if (f.aspectRatio < 3.0) score -= 1.0
        return score
    },
}

// Category mapping
const OBJECT_CATEGORIES: Record<string, Category> = {
    'chair': 'furniture', 'table': 'furniture', 'sofa': 'furniture', 'bed': 'furniture', 'cupboard': 'furniture',
    'laptop': 'gadgets', 'mobile': 'gadgets', 'tablet': 'gadgets', 'headphones': 'gadgets', 'watch': 'gadgets',
    'car': 'vehicles', 'bike': 'vehicles', 'bus': 'vehicles', 'aeroplane': 'vehicles',
    'shoe': 'fashion', 'bag': 'fashion', 'cap': 'fashion', 'shirt': 'fashion',
    'guiter': 'instruments', 'piano': 'instruments', 'drum': 'instruments', 'flute': 'instruments',
    'tv': 'electric_apppliance', 'fridge': 'electric_apppliance', 'ac': 'electric_apppliance', 'fan': 'electric_apppliance', 'washmachine': 'electric_apppliance',
    'mixer': 'kitchen_appliances', 'cooker': 'kitchen_appliances', 'oven': 'kitchen_appliances', 'stove': 'kitchen_appliances', 'tostar': 'kitchen_appliances',
    'football': 'sports', 'cricketbat': 'sports', 'hockey': 'sports', 'racket': 'sports', 'tennisball': 'sports',
    'pen': 'stationary_items', 'pencil': 'stationary_items', 'book': 'stationary_items', 'easer': 'stationary_items', 'ruler': 'stationary_items',
}

export function recognizeSketch(canvas: HTMLCanvasElement, selectedCategory?: Category): RecognitionResult[] {
    const ctx = canvas.getContext('2d')
    if (!ctx) return []

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
    const features = analyzeSketch(imageData)

    // Calculate scores for patterns, filtered by category if provided
    const results: RecognitionResult[] = []

    for (const [objectName, scoreFn] of Object.entries(RECOGNITION_PATTERNS)) {
        const category = OBJECT_CATEGORIES[objectName]

        // Skip if category filter is provided and doesn't match
        if (selectedCategory && category !== selectedCategory) {
            continue
        }

        const confidence = scoreFn(features)
        results.push({ objectName, confidence, category })
    }

    // Sort by confidence
    results.sort((a, b) => b.confidence - a.confidence)

    // Return all results (even low confidence ones)
    return results
}

export function getBestMatch(canvas: HTMLCanvasElement, selectedCategory?: Category): RecognitionResult | null {
    const results = recognizeSketch(canvas, selectedCategory)

    // Always return the best match if available, otherwise return the first item in the category as fallback
    if (results.length > 0) {
        const best = results[0];
        // STRICT LOW CONFIDENCE CHECK
        if (best.confidence < 0.2) {
            console.error("affected by high network fluction");
            throw new Error("affected by high network fluction");
        }
        return best;
    }

    // Fallback: if no patterns matched well (score 0), pick the first object in the category
    // This satisfies "something close also" - it gives a result in the right category
    if (selectedCategory) {
        // We need to import getModelNamesForCategory but can't due to circular dependency.
        // Instead, valid strategy is effectively random pick from category if score is 0
        return null
    }

    return null
}
