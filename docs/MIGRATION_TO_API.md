# Migration Guide: Using the New GroundedDINO-VL Public API

This guide helps you migrate from the old GroundingDINO API to the new, simplified GroundedDINO-VL public API.

---

## Quick Start

### Old Way (GroundingDINO)
```python
import torch
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
import groundingdino.datasets.transforms as T
from PIL import Image

# Load config
args = SLConfig.fromfile("config.py")
args.device = "cuda"
model = build_model(args)

# Load checkpoint
checkpoint = torch.load("weights.pth", map_location="cpu")
model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
model.eval()

# Prepare image
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
image_pil = Image.open("photo.jpg").convert("RGB")
image, _ = transform(image_pil, None)

# Run inference - many manual steps...
```

### New Way (GroundedDINO-VL) ⭐
```python
from groundeddino_vl import load_model, predict

# Load model - ONE LINE!
model = load_model("config.py", "weights.pth", device="cuda")

# Run inference - ONE LINE!
result = predict(model, "photo.jpg", "car . person . dog")

# Use results - CLEAN INTERFACE!
print(f"Found {len(result)} objects:")
for label, score in zip(result.labels, result.scores):
    print(f"  {label}: {score:.2f}")
```

**Result**: 80% less code, much clearer!

---

## Migration Examples

### Example 1: Basic Inference

**Before**:
```python
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
import torch

args = SLConfig.fromfile("config.py")
args.device = "cuda"
model = build_model(args)
checkpoint = torch.load("weights.pth", map_location="cpu")
model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
model.eval()
model = model.to("cuda")

# ... many more lines for image loading and inference ...
```

**After**:
```python
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth", device="cuda")
result = predict(model, "image.jpg", "car . person")
```

### Example 2: Working with Images from OpenCV

**Before**:
```python
import cv2
from PIL import Image
import groundingdino.datasets.transforms as T

image_bgr = cv2.imread("photo.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
image_tensor, _ = transform(image_pil, None)
```

**After**:
```python
import cv2
from groundeddino_vl import preprocess_image, predict

image_bgr = cv2.imread("photo.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Just one function call!
result = predict(model, image_rgb, "car . person")
```

### Example 3: Processing Results

**Before**:
```python
# Extract and filter results manually
logits = outputs["pred_logits"].cpu().sigmoid()[0]
boxes = outputs["pred_boxes"].cpu()[0]
filt_mask = logits.max(dim=1)[0] > box_threshold
logits_filt = logits[filt_mask]
boxes_filt = boxes[filt_mask]

# Get phrases manually
tokenlizer = model.tokenizer
tokenized = tokenlizer(caption)
pred_phrases = []
for logit, box in zip(logits_filt, boxes_filt):
    pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
    pred_phrases.append(pred_phrase)

# Work with tuples
for box, label, score in zip(boxes_filt, pred_phrases, logits_filt.max(dim=1)[0]):
    print(f"{label}: {score}")
```

**After**:
```python
# Get structured results
result = predict(model, image, "car . person", box_threshold=0.35)

# Clean interface
print(f"Found {len(result)} objects")
for label, score in zip(result.labels, result.scores):
    print(f"{label}: {score:.2f}")

# Easy box format conversion
boxes_xyxy = result.to_xyxy(denormalize=True)  # Get pixel coordinates
```

### Example 4: Annotating Images

**Before**:
```python
import cv2
import numpy as np
from torchvision.ops import box_convert
import supervision as sv

# Manual annotation setup
h, w, _ = image.shape
boxes = boxes_filt * torch.Tensor([w, h, w, h])
xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
detections = sv.Detections(xyxy=xyxy)

labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(pred_phrases, logits_filt.max(dim=1)[0])]

box_annotator = sv.BoxAnnotator()
annotated_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
```

**After**:
```python
from groundeddino_vl import annotate

# One function call!
annotated = annotate(image_np, result)

# Save or display
import cv2
cv2.imwrite("output.jpg", annotated)  # Already in BGR format
```

---

## API Reference

### Core Functions

#### `load_model(config_path, checkpoint_path, device="cuda")`
Load a model from config and checkpoint files.

**Parameters**:
- `config_path` (str | Path): Path to config file (.py)
- `checkpoint_path` (str | Path): Path to checkpoint file (.pth)
- `device` (str): Device to use ("cuda" or "cpu")

**Returns**: `torch.nn.Module` - Model ready for inference

**Example**:
```python
model = load_model("config.py", "weights.pth", device="cuda")
```

---

#### `predict(model, image, text_prompt, box_threshold=0.35, text_threshold=0.25, device="cuda")`
Run object detection on an image.

**Parameters**:
- `model` (torch.nn.Module): Loaded model
- `image` (str | Path | np.ndarray | PIL.Image | torch.Tensor): Input image
- `text_prompt` (str): Detection prompt (e.g., "car . person . dog")
- `box_threshold` (float): Box confidence threshold (0.0-1.0)
- `text_threshold` (float): Text confidence threshold (0.0-1.0)
- `device` (str): Device to use

**Returns**: `DetectionResult` - Structured detection results

**Example**:
```python
# From file path
result = predict(model, "photo.jpg", "car . person")

# From numpy array
import cv2
img = cv2.imread("photo.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = predict(model, img, "dog . cat")

# From PIL Image
from PIL import Image
img = Image.open("photo.jpg")
result = predict(model, img, "car")
```

---

#### `load_image(image_path)`
Load and preprocess an image from file.

**Parameters**:
- `image_path` (str | Path): Path to image file

**Returns**: `Tuple[np.ndarray, torch.Tensor]`
- `[0]`: Original image as RGB numpy array
- `[1]`: Preprocessed tensor for model input

**Example**:
```python
image_np, image_tensor = load_image("photo.jpg")
result = predict(model, image_tensor, "car")
```

---

#### `preprocess_image(image, max_size=1333, size=800)`
Preprocess an in-memory image.

**Parameters**:
- `image` (np.ndarray | PIL.Image | torch.Tensor): Input image
- `max_size` (int): Maximum dimension
- `size` (int): Target size for shorter edge

**Returns**: `torch.Tensor` - Preprocessed tensor

**Example**:
```python
import cv2
img = cv2.imread("photo.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tensor = preprocess_image(img)
```

---

#### `annotate(image, result, show_labels=True, show_confidence=True)`
Draw detection results on an image.

**Parameters**:
- `image` (np.ndarray): RGB image array
- `result` (DetectionResult): Detection results
- `show_labels` (bool): Show labels on boxes
- `show_confidence` (bool): Show confidence scores

**Returns**: `np.ndarray` - Annotated image in BGR format

**Example**:
```python
import cv2
annotated = annotate(image_np, result)
cv2.imwrite("output.jpg", annotated)
```

---

### Data Structures

#### `DetectionResult`
Container for detection results.

**Attributes**:
- `boxes` (torch.Tensor): Bounding boxes in cxcywh format, normalized [0, 1]
- `labels` (List[str]): Detected labels
- `scores` (torch.Tensor): Confidence scores
- `image_size` (Optional[Tuple[int, int]]): Image size (height, width)

**Methods**:
- `to_xyxy(denormalize=True)`: Convert boxes to xyxy format
- `__len__()`: Get number of detections
- `__repr__()`: String representation

**Example**:
```python
result = predict(model, "photo.jpg", "car . person")

print(len(result))  # Number of detections
print(result.labels)  # ["car", "person", "car"]
print(result.scores)  # tensor([0.95, 0.87, 0.82])

# Get boxes in pixel coordinates
boxes_xyxy = result.to_xyxy(denormalize=True)
for i, (label, box) in enumerate(zip(result.labels, boxes_xyxy)):
    x1, y1, x2, y2 = box
    print(f"{label}: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
```

---

## Common Workflows

### Workflow 1: Simple Image Detection
```python
from groundeddino_vl import load_model, predict

# Load model
model = load_model("config.py", "weights.pth")

# Detect objects
result = predict(model, "photo.jpg", "car . person . dog")

# Print results
for label, score in zip(result.labels, result.scores):
    print(f"{label}: {score:.2f}")
```

### Workflow 2: Batch Processing
```python
from groundeddino_vl import load_model, predict
from pathlib import Path

model = load_model("config.py", "weights.pth")

image_dir = Path("images/")
for image_path in image_dir.glob("*.jpg"):
    result = predict(model, str(image_path), "car . person")
    print(f"{image_path.name}: {len(result)} objects detected")
```

### Workflow 3: Video Processing
```python
import cv2
from groundeddino_vl import load_model, predict, annotate

model = load_model("config.py", "weights.pth")

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect
    result = predict(model, frame_rgb, "person . car")

    # Annotate
    annotated = annotate(frame_rgb, result)

    # Display
    cv2.imshow("Detections", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Workflow 4: Export Results to JSON
```python
import json
from groundeddino_vl import load_model, predict

model = load_model("config.py", "weights.pth")
result = predict(model, "photo.jpg", "car . person")

# Convert to JSON-serializable format
detections = []
for i in range(len(result)):
    detections.append({
        "label": result.labels[i],
        "score": float(result.scores[i]),
        "box": {
            "cx": float(result.boxes[i][0]),
            "cy": float(result.boxes[i][1]),
            "w": float(result.boxes[i][2]),
            "h": float(result.boxes[i][3]),
        }
    })

with open("detections.json", "w") as f:
    json.dump(detections, f, indent=2)
```

---

## Backward Compatibility

The old `groundingdino` namespace still works for backward compatibility:

```python
# This still works (with a deprecation warning)
from groundingdino import load_model, predict

model = load_model("config.py", "weights.pth")
result = predict(model, "image.jpg", "car")
```

But we recommend migrating to the new `groundeddino_vl` namespace:

```python
# Recommended
from groundeddino_vl import load_model, predict
```

---

## Troubleshooting

### Issue: "Module 'groundeddino_vl' has no attribute 'load_model'"

**Solution**: Make sure you have the latest version installed:
```bash
pip install --upgrade groundeddino_vl
# or
pip install -e .  # if installing from source
```

### Issue: "TypeError: image must be str, Path, np.ndarray, PIL.Image, or torch.Tensor"

**Solution**: Make sure your image is in one of the supported formats:
```python
# Good
result = predict(model, "image.jpg", "car")  # str path
result = predict(model, Path("image.jpg"), "car")  # Path object
result = predict(model, np.array(image), "car")  # numpy array
result = predict(model, Image.open("image.jpg"), "car")  # PIL Image

# Bad
result = predict(model, 12345, "car")  # Invalid type
```

### Issue: Boxes are in wrong format

**Solution**: Use the `to_xyxy()` method to convert:
```python
result = predict(model, image, "car")

# Normalized xyxy [0, 1]
boxes_norm = result.to_xyxy(denormalize=False)

# Pixel coordinates xyxy
boxes_pixels = result.to_xyxy(denormalize=True)
```

---

## Getting Help

- **Documentation**: See [API_DESIGN_PHASE.md](groundeddino_vl/summaries/API_DESIGN_PHASE.md)
- **Examples**: Check [demo/simple_inference.py](demo/simple_inference.py)
- **Issues**: Report bugs at [GitHub Issues](https://github.com/ghostcipher1/GroundedDINO-VL/issues)

---

## Summary

The new GroundedDINO-VL API provides:
- ✅ **80% less code** for common tasks
- ✅ **Clean, typed interface** with full documentation
- ✅ **Multiple input formats** supported
- ✅ **Structured results** with helpful methods
- ✅ **Backward compatibility** with old code

**Happy detecting!** 🚀
