#!/usr/bin/env python3
"""Quick test to check if dependencies are available"""
import sys

print("Checking dependencies...")
missing = []

try:
    import torch
    print("✓ torch installed")
except ImportError:
    print("✗ torch NOT installed")
    missing.append("torch")

try:
    import torchvision
    print("✓ torchvision installed")
except ImportError:
    print("✗ torchvision NOT installed")
    missing.append("torchvision")

try:
    from transformers import CLIPProcessor, CLIPModel
    print("✓ transformers installed")
except ImportError:
    print("✗ transformers NOT installed")
    missing.append("transformers")

try:
    from PIL import Image
    print("✓ Pillow installed")
except ImportError:
    print("✗ Pillow NOT installed")
    missing.append("pillow")

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    print("Install with: pip3 install " + " ".join(missing))
    sys.exit(1)
else:
    print("\n✓ All dependencies available!")
    print("You can now run: python3 test_image.py --image_path ./image.0012.png")

