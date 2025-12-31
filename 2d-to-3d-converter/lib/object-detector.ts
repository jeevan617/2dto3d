// Object detection and background removal utilities

export interface ObjectShape {
    type: 'sphere' | 'cylinder' | 'box' | 'cone' | 'torus' | 'ellipsoid' | 'hemisphere'
    params: any
    materialHints: {
        roughness: number
        metalness: number
        subsurface?: number
    }
}

// Define 25 object types with their 3D shape templates
export const OBJECT_TEMPLATES: Record<string, ObjectShape> = {
    // Round/Spherical Objects
    apple: {
        type: 'sphere',
        params: { radius: 1, widthSegments: 64, heightSegments: 64, indentTop: true },
        materialHints: { roughness: 0.3, metalness: 0.0, subsurface: 0.2 },
    },
    orange: {
        type: 'sphere',
        params: { radius: 1, widthSegments: 64, heightSegments: 64 },
        materialHints: { roughness: 0.4, metalness: 0.0, subsurface: 0.3 },
    },
    ball: {
        type: 'sphere',
        params: { radius: 1, widthSegments: 64, heightSegments: 64 },
        materialHints: { roughness: 0.2, metalness: 0.1 },
    },
    globe: {
        type: 'sphere',
        params: { radius: 1, widthSegments: 64, heightSegments: 64 },
        materialHints: { roughness: 0.5, metalness: 0.0 },
    },
    tomato: {
        type: 'sphere',
        params: { radius: 1, widthSegments: 64, heightSegments: 64, flattenY: 0.9 },
        materialHints: { roughness: 0.25, metalness: 0.0, subsurface: 0.2 },
    },

    // Cylindrical Objects
    can: {
        type: 'cylinder',
        params: { radiusTop: 0.4, radiusBottom: 0.4, height: 1.5, radialSegments: 64 },
        materialHints: { roughness: 0.15, metalness: 0.8 },
    },
    cup: {
        type: 'cylinder',
        params: { radiusTop: 0.5, radiusBottom: 0.4, height: 1.2, radialSegments: 64 },
        materialHints: { roughness: 0.3, metalness: 0.0 },
    },
    bottle: {
        type: 'cylinder',
        params: { radiusTop: 0.3, radiusBottom: 0.45, height: 1.8, radialSegments: 64, hasNeck: true },
        materialHints: { roughness: 0.1, metalness: 0.0 },
    },
    vase: {
        type: 'cylinder',
        params: { radiusTop: 0.6, radiusBottom: 0.4, height: 1.5, radialSegments: 64 },
        materialHints: { roughness: 0.2, metalness: 0.0 },
    },
    candle: {
        type: 'cylinder',
        params: { radiusTop: 0.35, radiusBottom: 0.35, height: 1.2, radialSegments: 64 },
        materialHints: { roughness: 0.6, metalness: 0.0 },
    },

    // Box/Rectangular Objects
    book: {
        type: 'box',
        params: { width: 1, height: 1.4, depth: 0.15 },
        materialHints: { roughness: 0.8, metalness: 0.0 },
    },
    phone: {
        type: 'box',
        params: { width: 0.7, height: 1.4, depth: 0.08, rounded: true },
        materialHints: { roughness: 0.2, metalness: 0.3 },
    },
    box: {
        type: 'box',
        params: { width: 1, height: 1, depth: 1 },
        materialHints: { roughness: 0.7, metalness: 0.0 },
    },
    tablet: {
        type: 'box',
        params: { width: 1.2, height: 0.9, depth: 0.06, rounded: true },
        materialHints: { roughness: 0.2, metalness: 0.4 },
    },
    monitor: {
        type: 'box',
        params: { width: 1.6, height: 1, depth: 0.05 },
        materialHints: { roughness: 0.3, metalness: 0.5 },
    },

    // Food Items
    banana: {
        type: 'cylinder',
        params: { radiusTop: 0.25, radiusBottom: 0.2, height: 1.5, radialSegments: 32, curved: true },
        materialHints: { roughness: 0.4, metalness: 0.0 },
    },
    carrot: {
        type: 'cone',
        params: { radius: 0.3, height: 1.5, radialSegments: 32 },
        materialHints: { roughness: 0.5, metalness: 0.0 },
    },
    egg: {
        type: 'ellipsoid',
        params: { radiusX: 0.5, radiusY: 0.7, radiusZ: 0.5, segments: 64 },
        materialHints: { roughness: 0.3, metalness: 0.0 },
    },
    bread: {
        type: 'box',
        params: { width: 1.2, height: 0.8, depth: 0.8, rounded: true },
        materialHints: { roughness: 0.9, metalness: 0.0 },
    },
    donut: {
        type: 'torus',
        params: { radius: 0.8, tube: 0.3, radialSegments: 64, tubularSegments: 64 },
        materialHints: { roughness: 0.4, metalness: 0.0 },
    },

    // Household Items
    lamp: {
        type: 'cone',
        params: { radius: 0.8, height: 1.2, radialSegments: 64 },
        materialHints: { roughness: 0.4, metalness: 0.0 },
    },
    clock: {
        type: 'cylinder',
        params: { radiusTop: 0.8, radiusBottom: 0.8, height: 0.15, radialSegments: 64 },
        materialHints: { roughness: 0.3, metalness: 0.2 },
    },
    pillow: {
        type: 'box',
        params: { width: 1.2, height: 0.4, depth: 0.8, rounded: true, soft: true },
        materialHints: { roughness: 0.7, metalness: 0.0 },
    },
    hat: {
        type: 'cylinder',
        params: { radiusTop: 0.7, radiusBottom: 0.5, height: 0.8, radialSegments: 64, hasTop: true },
        materialHints: { roughness: 0.6, metalness: 0.0 },
    },
    bowl: {
        type: 'hemisphere',
        params: { radius: 1, widthSegments: 64, heightSegments: 32 },
        materialHints: { roughness: 0.3, metalness: 0.0 },
    },
}

// Get list of all object types for UI
export const OBJECT_TYPES = Object.keys(OBJECT_TEMPLATES).sort()

// Remove background from image using color-based segmentation
export function removeBackground(
    imageData: ImageData,
    width: number,
    height: number
): { processedData: ImageData; mask: Uint8Array } {
    const data = imageData.data
    const mask = new Uint8Array(width * height)
    const processedData = new ImageData(width, height)

    // Sample corner pixels to determine background color
    const cornerSamples: number[][] = []
    const sampleSize = 10

    // Sample from corners
    for (let y = 0; y < sampleSize; y++) {
        for (let x = 0; x < sampleSize; x++) {
            // Top-left
            cornerSamples.push([data[(y * width + x) * 4], data[(y * width + x) * 4 + 1], data[(y * width + x) * 4 + 2]])
            // Top-right
            const xr = width - 1 - x
            cornerSamples.push([data[(y * width + xr) * 4], data[(y * width + xr) * 4 + 1], data[(y * width + xr) * 4 + 2]])
            // Bottom-left
            const yb = height - 1 - y
            cornerSamples.push([data[(yb * width + x) * 4], data[(yb * width + x) * 4 + 1], data[(yb * width + x) * 4 + 2]])
            // Bottom-right
            cornerSamples.push([
                data[(yb * width + xr) * 4],
                data[(yb * width + xr) * 4 + 1],
                data[(yb * width + xr) * 4 + 2],
            ])
        }
    }

    // Calculate average background color
    let avgR = 0,
        avgG = 0,
        avgB = 0
    cornerSamples.forEach(([r, g, b]) => {
        avgR += r
        avgG += g
        avgB += b
    })
    avgR /= cornerSamples.length
    avgG /= cornerSamples.length
    avgB /= cornerSamples.length

    // Threshold for background detection
    const threshold = 40

    // Process each pixel
    for (let i = 0; i < width * height; i++) {
        const idx = i * 4
        const r = data[idx]
        const g = data[idx + 1]
        const b = data[idx + 2]

        // Calculate color distance from background
        const dist = Math.sqrt((r - avgR) ** 2 + (g - avgG) ** 2 + (b - avgB) ** 2)

        // If pixel is similar to background, mark as background
        if (dist < threshold) {
            mask[i] = 0 // Background
            processedData.data[idx] = r
            processedData.data[idx + 1] = g
            processedData.data[idx + 2] = b
            processedData.data[idx + 3] = 0 // Transparent
        } else {
            mask[i] = 255 // Foreground
            processedData.data[idx] = r
            processedData.data[idx + 1] = g
            processedData.data[idx + 2] = b
            processedData.data[idx + 3] = 255 // Opaque
        }
    }

    return { processedData, mask }
}

// Extract dominant color from image (for material hints)
export function extractDominantColor(imageData: ImageData, mask: Uint8Array): [number, number, number] {
    const data = imageData.data
    let r = 0,
        g = 0,
        b = 0
    let count = 0

    for (let i = 0; i < mask.length; i++) {
        if (mask[i] > 0) {
            // Only count foreground pixels
            const idx = i * 4
            r += data[idx]
            g += data[idx + 1]
            b += data[idx + 2]
            count++
        }
    }

    if (count === 0) return [128, 128, 128]

    return [r / count, g / count, b / count]
}
