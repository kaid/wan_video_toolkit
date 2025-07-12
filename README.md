# Wan Video Toolkit

A comprehensive, type-safe toolkit for building video generation applications using Wan models.

## 🚀 Features

- **🔧 Modular Architecture**: Composition-based design with reusable core components
- **🛡️ Type Safety**: Comprehensive type annotations throughout with mypy compatibility
- **🎯 Easy to Use**: Simple, intuitive API for quick integration
- **🔌 Extensible**: Built-in support for LoRA adapters and future extensions
- **💾 Memory Efficient**: Smart pipeline management with automatic cleanup
- **📊 Production Ready**: Comprehensive error handling and validation

## 📦 Installation

### From Git Repository (Development)

```bash
pip install git+https://github.com/kaid/wan_video_toolkit.git
```

### Local Development

```bash
git clone https://github.com/kaid/wan_video_toolkit.git
cd wan_video_toolkit
pip install -e .
```

## 🎬 Quick Start

### Basic FLF2V Generation

```python
from wan_video_toolkit import WanFLF2VModel
from PIL import Image

# Initialize model
model = WanFLF2VModel()

# Load images
first_frame = Image.open("first.jpg")
last_frame = Image.open("last.jpg")

# Generate video
result = model.generate(
    first_frame=first_frame,
    last_frame=last_frame,
    prompt="smooth transition with cinematic motion",
    duration_seconds=2.0,
    steps=30
)

print(f"Video saved to: {result['video_path']}")
print(f"Generation took: {result['generation_time']:.1f}s")
```

### With LoRA Adapters

```python
from wan_video_toolkit import WanFLF2VModel, LoRAConfig

# Configure LoRA
lora_config = LoRAConfig(
    repo_id="Kijai/WanVideo_comfy",
    filename="Wan21_CausVid_14B_T2V_lora_rank32.safetensors",
    adapter_name="causvid_lora",
    weight=0.95
)

# Initialize model with LoRA
model = WanFLF2VModel()
model.add_lora(lora_config)

# Generate with LoRA applied
result = model.generate(
    first_frame=first_frame,
    last_frame=last_frame,
    prompt="fast motion with CausVid style",
    steps=4  # CausVid enables fast generation
)
```

## 🏗️ Architecture

### Core Components

- **`PipelineManager`**: Type-safe pipeline lifecycle management with LRU caching
- **`ImageProcessor`**: Comprehensive image processing with aspect ratio preservation
- **`GenerationManager`**: Parameter validation and video export functionality
- **`LoRAAdapter`**: Easy LoRA loading and management

### Model Implementations

- **`WanFLF2VModel`**: Complete FLF2V (First+Last Frame to Video) implementation
- More models coming soon (I2V, T2V)

### Type System

```python
from wan_video_toolkit.types import (
    ModelConfig,        # Model configuration
    GenerationParams,   # Generation parameters
    LoRAConfig,        # LoRA configuration
    GenerationResult   # Generation results
)
```

## 🎛️ Configuration

### Default FLF2V Configuration

```python
config = ModelConfig(
    model_id="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    model_type=ModelType.FLF2V,
    dtype="float16",
    device_strategy=DeviceStrategy.BALANCED,
    max_area=1280 * 720,
    default_height=720,
    default_width=1280,
    min_frames=8,
    max_frames=81,
    fps=24
)

model = WanFLF2VModel(config=config)
```

### Custom Configuration

```python
from wan_video_toolkit.types import ModelConfig, ModelType, DeviceStrategy

# High quality configuration
hq_config = ModelConfig(
    model_id="Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    model_type=ModelType.FLF2V,
    dtype="float16",
    device_strategy=DeviceStrategy.CUDA,
    max_area=1920 * 1080,  # Higher resolution
    default_height=1080,
    default_width=1920,
    min_frames=16,
    max_frames=120,
    fps=30
)
```

## 🔧 Advanced Usage

### Memory Management

```python
# Get memory information
memory_info = model.get_memory_info()
print(f"CUDA memory: {memory_info['pipeline_manager']['cuda_memory_allocated']} bytes")

# Manual cleanup
model.unload()
```

### Multiple LoRAs

```python
from wan_video_toolkit.adapters import LoRAManager

manager = LoRAManager()

# Add multiple LoRAs
manager.add_adapter_from_config(lora_config_1)
manager.add_adapter_from_config(lora_config_2)

# Apply with different weights
manager.apply_adapters(
    pipeline, 
    adapter_names=["lora1", "lora2"],
    weights=[0.8, 0.5]
)
```

### Batch Processing

```python
from wan_video_toolkit.core import ImageProcessor

# Process multiple image pairs
processor = ImageProcessor()
processed_pairs = []

for first, last in image_pairs:
    first_processed, last_processed, h, w = processor.prepare_image_pair(
        first, last, max_area=1280*720
    )
    processed_pairs.append((first_processed, last_processed, h, w))
```

## 🎨 Gradio Integration

```python
import gradio as gr
from wan_video_toolkit import WanFLF2VModel

model = WanFLF2VModel()

def generate_video(first_frame, last_frame, prompt, **kwargs):
    try:
        result = model.generate(
            first_frame=first_frame,
            last_frame=last_frame, 
            prompt=prompt,
            **kwargs
        )
        return result["video_path"], f"✅ Generated in {result['generation_time']:.1f}s"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Create Gradio interface
interface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Image(type="pil", label="First Frame"),
        gr.Image(type="pil", label="Last Frame"),
        gr.Textbox(label="Prompt"),
        gr.Slider(1, 50, value=30, label="Steps"),
        gr.Slider(0.5, 3.0, value=2.0, label="Duration")
    ],
    outputs=[
        gr.Video(label="Generated Video"),
        gr.Textbox(label="Status")
    ]
)

interface.launch()
```

## 🧪 Development

### Type Checking

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run type checking
mypy wan_video_toolkit
pyright wan_video_toolkit
```

### Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=wan_video_toolkit tests/
```

### Code Formatting

```bash
# Format code
black wan_video_toolkit
isort wan_video_toolkit

# Check formatting
black --check wan_video_toolkit
isort --check wan_video_toolkit
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- diffusers 0.24+
- transformers 4.30+
- CUDA-capable GPU (recommended)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper type annotations
4. Add tests for new functionality
5. Run type checking and tests
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Wan Video Models](https://huggingface.co/Wan-AI) - The foundation models
- [🤗 Diffusers](https://github.com/huggingface/diffusers) - The diffusion framework
- [CausVid](https://github.com/tianweiy/CausVid) - Fast video generation LoRA

## 📚 API Reference

### Core Classes

#### `WanFLF2VModel`

Main model class for FLF2V video generation.

**Methods:**
- `generate(first_frame, last_frame, prompt, **kwargs) -> GenerationResult`
- `add_lora(lora_config: LoRAConfig) -> LoRAAdapter`
- `load(force_reload: bool = False) -> None`
- `unload() -> None`
- `get_model_info() -> Dict[str, Any]`

#### `PipelineManager`

Type-safe pipeline lifecycle management.

**Methods:**
- `get_or_load(pipeline_id, loader_func, force_reload=False) -> Pipeline`
- `unload(pipeline_id: str) -> bool`
- `get_memory_info() -> Dict[str, int]`

#### `ImageProcessor`

Comprehensive image processing utilities.

**Methods:**
- `validate_image(image: ImageInput) -> Image.Image`
- `prepare_image_pair(first, second, max_area, **kwargs) -> Tuple[Image, Image, int, int]`
- `aspect_resize(image, max_area, **kwargs) -> Tuple[Image, int, int]`

#### `GenerationManager`

Parameter validation and generation utilities.

**Methods:**
- `validate_parameters(**kwargs) -> GenerationParams`
- `prepare_seed(seed, randomize) -> int`
- `export_video(frames, fps, output_path) -> str`

See the type definitions in `wan_video_toolkit.types` for complete parameter specifications.