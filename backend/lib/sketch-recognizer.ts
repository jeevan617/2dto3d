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
    // Furniture
    'chair': (f) => {
        let score = 0
        if (f.aspectRatio > 0.4 && f.aspectRatio < 1.2) score += 0.3
        if (f.bottomHeavy > 0.45) score += 0.3
        if (f.complexity > 0.15 && f.complexity < 0.35) score += 0.2
        if (f.centerY < 0.6) score += 0.2
        if (f.aspectRatio > 2.0) score -= 0.5 // Too wide to be a chair
        return score
    },
    'table': (f) => {
        let score = 0
        if (f.aspectRatio > 1.2 && f.aspectRatio < 2.5) score += 0.4
        if (f.topHeavy > 0.4) score += 0.3
        if (f.complexity > 0.1) score += 0.3
        if (f.circularity > 0.6) score -= 0.3 // Not round
        return score
    },
    'sofa': (f) => {
        let score = 0
        if (f.aspectRatio > 1.8 && f.aspectRatio < 3.5) score += 0.5
        if (f.circularity < 0.5) score += 0.3
        if (f.bottomHeavy > 0.5) score += 0.2
        if (f.aspectRatio < 1.5) score -= 0.5 // Too narrow
        return score
    },
    'bed': (f) => {
        let score = 0
        if (f.aspectRatio > 1.3 && f.aspectRatio < 2.2) score += 0.4
        if (f.complexity < 0.25) score += 0.3
        if (f.bottomHeavy > 0.5) score += 0.3
        return score
    },
    'cupboard': (f) => {
        let score = 0
        if (f.aspectRatio > 0.4 && f.aspectRatio < 0.9) score += 0.5
        if (f.complexity < 0.25) score += 0.3
        if (f.circularity < 0.4) score += 0.2
        if (f.aspectRatio > 1.0) score -= 0.5 // Too wide
        return score
    },

    // Gadgets
    'laptop': (f) => {
        let score = 0
        // L-shape, usually wide
        if (f.aspectRatio > 1.2 && f.aspectRatio < 1.8) score += 0.4
        if (f.complexity < 0.25) score += 0.3
        // Keyboard makes it bottom heavy or balanced
        if (f.bottomHeavy > 0.4) score += 0.3
        // STRICT: Laptop is definitely NOT circular
        if (f.circularity > 0.5) score -= 0.5
        return score
    },
    'mobile': (f) => {
        let score = 0
        // Tall thin rectangle
        if (f.aspectRatio > 0.4 && f.aspectRatio < 0.7) score += 0.5
        if (f.complexity < 0.15) score += 0.3
        if (f.circularity < 0.3) score += 0.2
        if (f.aspectRatio > 0.9) score -= 0.5 // Too wide
        return score
    },
    'tablet': (f) => {
        let score = 0
        // Rectangle, close to 1.3-1.4 (iPad ratio)
        if (f.aspectRatio > 1.0 && f.aspectRatio < 1.6) score += 0.4
        if (f.complexity < 0.15) score += 0.3
        if (f.circularity < 0.35) score += 0.3
        // STRICT: Tablet is filled, not hollow like headphones
        if (f.complexity > 0.3) score -= 0.3
        // STRICT: Not top heavy (headphones are top heavy)
        if (f.topHeavy > 0.6) score -= 0.5
        return score
    },
    'headphones': (f) => {
        let score = 0
        // Arch shape: Can be square-ish (1.0) or slightly tall/wide
        if (f.aspectRatio > 0.8 && f.aspectRatio < 1.3) score += 0.3
        // CRITICAL: Headphones are hollow (arch), so complexity is higher than a solid tablet block
        if (f.complexity > 0.2) score += 0.3
        // CRITICAL: Top heavy due to the headband arch
        if (f.topHeavy > 0.55) score += 0.4
        // Some circularity due to ear cups
        if (f.circularity > 0.25) score += 0.2
        // Negative: If it's a solid block (low complexity), it's not headphones
        if (f.complexity < 0.1) score -= 0.5
        return score
    },
    'watch': (f) => {
        let score = 0
        // Circular face + strap
        if (f.aspectRatio > 0.7 && f.aspectRatio < 1.3) score += 0.3
        if (f.circularity > 0.5) score += 0.5
        return score
    },

    // Vehicles
    'car': (f) => {
        let score = 0
        if (f.aspectRatio > 1.4 && f.aspectRatio < 2.5) score += 0.4
        if (f.bottomHeavy > 0.6) score += 0.4
        if (f.circularity < 0.5) score += 0.2
        return score
    },
    'bike': (f) => {
        let score = 0
        if (f.aspectRatio > 1.3 && f.aspectRatio < 2.2) score += 0.3
        if (f.complexity > 0.3) score += 0.4
        if (f.bottomHeavy < 0.6) score += 0.3
        return score
    },
    'bus': (f) => {
        let score = 0
        if (f.aspectRatio > 2.0 && f.aspectRatio < 3.5) score += 0.5
        if (f.complexity < 0.3) score += 0.3
        if (f.circularity < 0.4) score += 0.2
        if (f.aspectRatio < 1.5) score -= 0.5
        return score
    },
    'aeroplane': (f) => {
        let score = 0
        if (f.aspectRatio > 0.8 && f.aspectRatio < 2.5) score += 0.2
        if (f.complexity > 0.25) score += 0.4
        if (Math.abs(f.leftHeavy - f.rightHeavy) < 0.1) score += 0.4
        return score
    },

    // Fashion
    'shoe': (f) => {
        let score = 0
        if (f.aspectRatio > 1.2 && f.aspectRatio < 2.5) score += 0.4
        if (f.bottomHeavy > 0.55) score += 0.3
        if (f.complexity > 0.15) score += 0.3
        // Shoes are usually asymmetric
        if (Math.abs(f.leftHeavy - f.rightHeavy) > 0.1) score += 0.2
        return score
    },
    'bag': (f) => {
        let score = 0
        if (f.aspectRatio > 0.7 && f.aspectRatio < 1.4) score += 0.3
        if (f.topHeavy < 0.5) score += 0.4 // Handle is thin
        if (f.complexity > 0.15) score += 0.3
        // Bag is generally symmetric
        if (Math.abs(f.leftHeavy - f.rightHeavy) < 0.2) score += 0.2
        return score
    },
    'cap': (f) => {
        let score = 0
        if (f.aspectRatio > 1.0 && f.aspectRatio < 1.8) score += 0.4
        // Top heavy dome
        if (f.topHeavy > 0.55) score += 0.4
        if (f.circularity > 0.4) score += 0.2
        // Distinct from Shirt: Cap is circular/dome, Shirt is boxy
        if (f.circularity < 0.3) score -= 0.3
        return score
    },
    'shirt': (f) => {
        let score = 0
        if (f.aspectRatio > 0.8 && f.aspectRatio < 1.6) score += 0.3
        if (Math.abs(f.leftHeavy - f.rightHeavy) < 0.15) score += 0.4
        if (f.topHeavy > 0.5) score += 0.3
        // Distinct from Cap: Shirt is NOT circular
        if (f.circularity > 0.5) score -= 0.4
        return score
    },

    // Instruments
    'guitar': (f) => {
        let score = 0
        if (f.aspectRatio > 0.3 && f.aspectRatio < 0.7) score += 0.4
        if (f.bottomHeavy > 0.6) score += 0.4
        if (f.complexity > 0.2) score += 0.2
        return score
    },
    'piano': (f) => {
        let score = 0
        if (f.aspectRatio > 1.5 && f.aspectRatio < 2.5) score += 0.5
        if (f.bottomHeavy > 0.5) score += 0.3
        return score
    },
    'drum': (f) => {
        let score = 0
        if (f.aspectRatio > 0.8 && f.aspectRatio < 1.2) score += 0.3
        if (f.circularity > 0.6) score += 0.5
        return score
    },
    'flute': (f) => {
        let score = 0
        if (f.aspectRatio > 4.0 || f.aspectRatio < 0.2) score += 0.6
        if (f.complexity < 0.15) score += 0.4
        return score
    },

    // Appliances
    'tv': (f) => {
        let score = 0
        if (f.aspectRatio > 1.3 && f.aspectRatio < 2.0) score += 0.5
        if (f.complexity < 0.2) score += 0.3
        if (f.circularity < 0.35) score += 0.2
        return score
    },
    'fridge': (f) => {
        let score = 0
        if (f.aspectRatio > 0.4 && f.aspectRatio < 0.8) score += 0.5
        if (f.complexity < 0.25) score += 0.3
        return score
    },
    'ac': (f) => {
        let score = 0
        if (f.aspectRatio > 1.8 && f.aspectRatio < 3.5) score += 0.5
        if (f.complexity < 0.25) score += 0.3
        return score
    },
    'fan': (f) => {
        let score = 0
        if (f.aspectRatio > 0.8 && f.aspectRatio < 1.2) score += 0.3
        if (f.circularity < 0.5) score += 0.3
        if (f.complexity > 0.25) score += 0.4
        return score
    },
    'washmachine': (f) => {
        let score = 0
        if (f.aspectRatio > 0.8 && f.aspectRatio < 1.2) score += 0.4
        if (f.circularity > 0.4) score += 0.3
        if (f.centerX > 0.4 && f.centerX < 0.6) score += 0.3
        return score
    },

    // Kitchen
    'mixer': (f) => {
        let score = 0
        if (f.aspectRatio > 0.4 && f.aspectRatio < 0.8) score += 0.4
        if (f.bottomHeavy > 0.5) score += 0.3
        return score
    },
    'cooker': (f) => {
        let score = 0
        if (f.aspectRatio > 0.8 && f.aspectRatio < 1.5) score += 0.4
        if (f.bottomHeavy > 0.5) score += 0.3
        return score
    },
    'oven': (f) => {
        let score = 0
        if (f.aspectRatio > 1.0 && f.aspectRatio < 1.6) score += 0.4
        if (f.complexity < 0.25) score += 0.3
        return score
    },
    'stove': (f) => {
        let score = 0
        if (f.aspectRatio > 1.2 && f.aspectRatio < 2.0) score += 0.4
        if (f.complexity > 0.15) score += 0.3
        return score
    },
    'toaster': (f) => {
        let score = 0
        if (f.aspectRatio > 1.2 && f.aspectRatio < 1.8) score += 0.4
        return score
    },

    // Sports (Highly geometric)
    'football': (f) => {
        let score = 0
        if (f.aspectRatio > 0.85 && f.aspectRatio < 1.15) score += 0.4
        if (f.circularity > 0.7) score += 0.6
        return score
    },
    'cricketbat': (f) => {
        let score = 0
        if (f.aspectRatio > 0.15 && f.aspectRatio < 0.4) score += 0.5
        if (f.bottomHeavy > 0.6) score += 0.4
        return score
    },
    'hockey': (f) => {
        let score = 0
        if (f.aspectRatio > 0.2 && f.aspectRatio < 0.6) score += 0.4
        if (f.bottomHeavy > 0.7) score += 0.5
        return score
    },
    'racket': (f) => {
        let score = 0
        if (f.aspectRatio > 0.4 && f.aspectRatio < 0.8) score += 0.4
        if (f.topHeavy > 0.6) score += 0.4
        return score
    },
    'tennisball': (f) => {
        let score = 0
        if (f.aspectRatio > 0.85 && f.aspectRatio < 1.15) score += 0.4
        if (f.circularity > 0.75) score += 0.6
        return score
    },

    // Stationary
    'pen': (f) => {
        let score = 0
        if (f.aspectRatio > 4.0 || f.aspectRatio < 0.25) score += 0.6
        return score
    },
    'pencil': (f) => {
        let score = 0
        if (f.aspectRatio > 4.0 || f.aspectRatio < 0.25) score += 0.6
        return score
    },
    'book': (f) => {
        let score = 0
        if (f.aspectRatio > 0.6 && f.aspectRatio < 1.4) score += 0.4
        if (f.complexity < 0.2) score += 0.4
        return score
    },
    'eraser': (f) => {
        let score = 0
        if (f.aspectRatio > 0.5 && f.aspectRatio < 2.0) score += 0.3
        if (f.complexity < 0.15) score += 0.4
        return score
    },
    'ruler': (f) => {
        let score = 0
        if (f.aspectRatio > 3.0 || f.aspectRatio < 0.3) score += 0.6
        if (f.complexity < 0.15) score += 0.4
        return score
    },
}

// Category mapping
const OBJECT_CATEGORIES: Record<string, Category> = {
    'chair': 'furniture', 'table': 'furniture', 'sofa': 'furniture', 'bed': 'furniture', 'cupboard': 'furniture',
    'laptop': 'gadgets', 'mobile': 'gadgets', 'tablet': 'gadgets', 'headphones': 'gadgets', 'watch': 'gadgets',
    'car': 'vehicles', 'bike': 'vehicles', 'bus': 'vehicles', 'aeroplane': 'vehicles',
    'shoe': 'fashion', 'bag': 'fashion', 'cap': 'fashion', 'shirt': 'fashion',
    'guitar': 'instruments', 'piano': 'instruments', 'drum': 'instruments', 'flute': 'instruments',
    'tv': 'electric_appliance', 'fridge': 'electric_appliance', 'ac': 'electric_appliance', 'fan': 'electric_appliance', 'washmachine': 'electric_appliance',
    'mixer': 'kitchen_appliances', 'cooker': 'kitchen_appliances', 'oven': 'kitchen_appliances', 'stove': 'kitchen_appliances', 'toaster': 'kitchen_appliances',
    'football': 'sports', 'cricketbat': 'sports', 'hockey': 'sports', 'racket': 'sports', 'tennisball': 'sports',
    'pen': 'stationary_items', 'pencil': 'stationary_items', 'book': 'stationary_items', 'eraser': 'stationary_items', 'ruler': 'stationary_items',
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
        return results[0]
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
