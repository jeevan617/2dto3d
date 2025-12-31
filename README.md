# 2D to 3D Image Converter

A powerful web application that transforms hand-drawn sketches into detailed 3D models using a cutting-edge AI pipeline powered by **ControlNet** and **TripoSR**.

![Demo Image](https://github.com/user-attachments/assets/9ebd4285-8192-4bcb-9130-520d8eacf8c0)

## Features

- **Sketch-to-3D Generation**: Convert simple rough sketches into fully realized 3D assets.
- **Advanced AI Pipeline**: Leverages text-to-image and image-to-3D diffusion models for high-fidelity results.
- **Interactive 3D Viewer**: Inspect generated models with multiple visualization modes:
  - **Textured Mesh**: Full PBR rendering.
  - **Wireframe**: View the underlying geometry.
  - **Point Cloud**: Visualize data distribution.
- **Model Registry**: Access a library of pre-generated high-quality models across various categories (Furniture, Gadgets, Vehicles, etc.).
- **Responsive Design**: Seamless experience across desktop and touch devices.

## AI Generation Pipeline

This project utilizes a two-stage generative process to bridge the gap between abstract 2D sketches and concrete 3D geometry:

1.  **Sketch Analysis & Refinement (ControlNet)**
    - The user's input sketch is processed using **ControlNet** (conditioned on Stable Diffusion).
    - ControlNet preserves the structural lines of the sketch while generating a photorealistic or stylized 2D interpretation.
    - This step adds necessary texture, shading, and detail that is missing from a raw line drawing.

2.  **3D Mesh Reconstruction (TripoSR)**
    - The refined 2D image is fed into **TripoSR**, a state-of-the-art fast feed-forward 3D reconstruction model.
    - TripoSR estimates depth and geometry to generate a high-quality 3D mesh (GLB/OBJ) in seconds.
    - The result is a fully textured 3D model ready for use in games or visualization.

## Tech Stack

### Frontend
- **React.js** (Next.js App Router) for the UI framework.
- **React Three Fiber (Three.js)** for hardware-accelerated 3D rendering.
- **Tailwind CSS** for modern, responsive styling.
- **TypeScript** for robust type-safe code.

### AI & Backend Services
- **ControlNet** (for Sketch-to-Image synthesis).
- **TripoSR** (for Image-to-3D reconstruction).
- **Python / FastAPI** (for model serving and orchestration).
- **PyTorch** (Deep learning framework).

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SUBHADIPMAITI-DEV/2D-to-3D-Image-Converter.git
cd 2d-to-3d-converter
```

### 2. Frontend Setup
```bash
# Install dependencies
npm install

# Run the development server
npm run dev
```

The application will be available at `http://localhost:3000`.
