#!/usr/bin/env python3
"""
Analyze sk2d directory and generate TypeScript model registry
"""
import os
import json
from pathlib import Path

def analyze_sk2d_directory(sk2d_path):
    """Scan sk2d directory and extract all available models"""
    models = {}
    categories = {}
    
    for category_dir in sorted(os.listdir(sk2d_path)):
        category_path = os.path.join(sk2d_path, category_dir)
        
        # Skip non-directories and hidden files
        if not os.path.isdir(category_path) or category_dir.startswith('.'):
            continue
        
        category_models = []
        files_in_category = os.listdir(category_path)
        
        # Group files by base name
        model_files = {}
        for filename in files_in_category:
            if filename.startswith('.'):
                continue
                
            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            if ext in ['.glb', '.ply']:
                if name not in model_files:
                    model_files[name] = {'model': None, 'preview': None}
                model_files[name]['model'] = filename
                model_files[name]['ext'] = ext[1:]  # Remove the dot
            elif ext in ['.jpg', '.jpeg', '.png']:
                if name not in model_files:
                    model_files[name] = {'model': None, 'preview': None}
                model_files[name]['preview'] = filename
        
        # Create model entries
        for model_name, files in model_files.items():
            if files['model']:  # Only include if there's a 3D model file
                key = model_name.lower()
                
                # Find preview image (might have different name)
                preview_file = files['preview']
                if not preview_file:
                    # Look for any image with similar name
                    for f in files_in_category:
                        if f.lower().startswith(model_name.lower()) and f.lower().endswith(('.jpg', '.png', '.jpeg')):
                            preview_file = f
                            break
                
                models[key] = {
                    'name': model_name.replace('_', ' ').title(),
                    'category': category_dir,
                    'modelPath': f'/assets/.v0-internal-data/{category_dir}/{files["model"]}',
                    'previewPath': f'/assets/.v0-internal-data/{category_dir}/{preview_file}' if preview_file else f'/assets/.v0-internal-data/{category_dir}/{files["model"]}',
                    'fileType': files['ext']
                }
                category_models.append(key)
        
        if category_models:
            categories[category_dir] = category_models
    
    return models, categories

def generate_typescript_registry(models, categories):
    """Generate TypeScript code for model registry"""
    
    ts_code = """// Registry of all available 3D models from sk2d directory
// Auto-generated - do not edit manually

export interface ModelInfo {
    name: string
    category: string
    modelPath: string
    previewPath: string
    fileType: 'glb' | 'ply'
}

export const MODEL_REGISTRY: Record<string, ModelInfo> = {
"""
    
    # Group by category for better organization
    for category, model_keys in sorted(categories.items()):
        category_label = category.replace('_', ' ').title()
        ts_code += f"\n    // {category_label}\n"
        
        for key in sorted(model_keys):
            model = models[key]
            ts_code += f"    '{key}': {{\n"
            ts_code += f"        name: '{model['name']}',\n"
            ts_code += f"        category: '{model['category']}',\n"
            ts_code += f"        modelPath: '{model['modelPath']}',\n"
            ts_code += f"        previewPath: '{model['previewPath']}',\n"
            ts_code += f"        fileType: '{model['fileType']}'\n"
            ts_code += f"    }},\n"
    
    ts_code += """}\n
export const CATEGORIES = [
"""
    for category in sorted(categories.keys()):
        ts_code += f"    '{category}',\n"
    
    ts_code += """] as const

export type Category = typeof CATEGORIES[number]

export const CATEGORY_LABELS: Record<Category, string> = {
"""
    for category in sorted(categories.keys()):
        label = category.replace('_', ' ').title()
        ts_code += f"    '{category}': '{label}',\n"
    
    ts_code += """}

export function getModelsByCategory(category: Category): ModelInfo[] {
    return Object.values(MODEL_REGISTRY).filter(model => model.category === category)
}

export function getModelByName(name: string): ModelInfo | undefined {
    return MODEL_REGISTRY[name.toLowerCase()]
}

export function getAllModelNames(): string[] {
    return Object.keys(MODEL_REGISTRY)
}

export function getModelNamesForCategory(category: Category): string[] {
    return Object.entries(MODEL_REGISTRY)
        .filter(([_, model]) => model.category === category)
        .map(([key, _]) => key)
}
"""
    
    return ts_code

def main():
    # Path to source directory
    sk2d_path = '/Users/karthikm/2D-to-3D-Image-Converter/.internal-assets-cache'
    
    print("Analyzing sk2d directory...")
    models, categories = analyze_sk2d_directory(sk2d_path)
    
    print(f"\nFound {len(models)} models across {len(categories)} categories:")
    for category, model_list in sorted(categories.items()):
        print(f"  {category}: {len(model_list)} models")
        print(f"    Models: {', '.join(sorted(model_list))}")
    
    # Generate TypeScript code
    ts_code = generate_typescript_registry(models, categories)
    
    # Write to file
    output_path = '/Users/karthikm/2D-to-3D-Image-Converter/backend/lib/model-registry.ts'
    with open(output_path, 'w') as f:
        f.write(ts_code)
    
    print(f"\n✓ Generated TypeScript registry at: {output_path}")
    
    # Also save JSON for reference
    json_output = {
        'models': models,
        'categories': categories
    }
    json_path = '/Users/karthikm/2D-to-3D-Image-Converter/models_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"✓ Saved analysis JSON at: {json_path}")

if __name__ == '__main__':
    main()
