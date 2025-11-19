"""
PubMedCLIP Image Analysis Script
This script uses PubMedCLIP to analyze a medical image.
"""
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import argparse

# Try importing clip, but don't fail if not available
try:
    import clip
except ImportError:
    clip = None

def load_pubmedclip(model_name="RN50", device="cpu"):
    """Load PubMedCLIP model from HuggingFace or local path"""
    try:
        # Try loading from HuggingFace first
        try:
            from transformers import CLIPProcessor, CLIPModel
            print("Attempting to load PubMedCLIP from HuggingFace...")
            model = CLIPModel.from_pretrained("sarahESL/PubMedCLIP")
            processor = CLIPProcessor.from_pretrained("sarahESL/PubMedCLIP")
            return model, processor, "huggingface"
        except ImportError:
            print("transformers not installed, trying standard CLIP...")
            raise
        except Exception as e:
            print(f"Could not load from HuggingFace: {e}")
            raise
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}")
        print("Attempting to load standard CLIP via HuggingFace (openai/clip-vit-base-patch32)...")
        try:
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            return model, processor, "huggingface-openai"
        except Exception as e_hf:
            print(f"Could not load standard CLIP via HuggingFace: {e_hf}")

        print("Attempting to load standard CLIP via OpenAI's clip package...")
        try:
            # Fallback to standard CLIP
            if clip is None:
                print("clip-by-openai not installed. Please install: pip install clip-by-openai")
                return None, None, None
            model, preprocess = clip.load("ViT-B/32", device=device)
            return model, preprocess, "clip"
        except ImportError:
            print("clip-by-openai not installed. Please install: pip install clip-by-openai")
            return None, None, None
        except Exception as e2:
            print(f"Could not load CLIP: {e2}")
            return None, None, None

def analyze_image_with_text_prompts(image_path, model, processor, model_type, device="cpu"):
    """Analyze image using medical text prompts"""
    
    # Medical image analysis prompts
    medical_prompts = [
        "a normal head CT scan",
        "an abnormal head CT scan",
        "a CT scan showing brain anatomy",
        "a CT scan showing skull structures",
        "a CT scan with visible sinuses",
        "a CT scan showing temporal bones",
        "a CT scan with mastoid air cells",
        "a CT scan showing cranial base",
        "a head CT scan with no abnormalities",
        "a head CT scan with pathology"
    ]
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    
    if model_type.startswith("huggingface"):
        # Use HuggingFace processor
        inputs = processor(text=medical_prompts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
    else:
        # Use standard CLIP
        image_input = processor(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(medical_prompts).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity
            logits_per_image = (image_features @ text_features.T) * 100
            probs = logits_per_image.softmax(dim=1)
    
    # Get top matches
    top_probs, top_indices = probs[0].topk(5)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "description": medical_prompts[idx],
            "confidence": float(prob)
        })
    
    return results

def generate_description(image_path, model, processor, model_type, device="cpu"):
    """Generate a description of the image"""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Detailed medical descriptions to match against
    detailed_prompts = [
        "axial CT scan of the head showing skull bones and brain tissue",
        "head CT scan displaying cranial anatomy with visible sinuses",
        "CT scan showing temporal bones with mastoid air cells",
        "head CT scan with visible nasal cavity and paranasal sinuses",
        "CT scan showing cranial base anatomy",
        "head CT scan displaying brain parenchyma within cranial vault",
        "CT scan showing petrous temporal bones and internal auditory canals",
        "head CT scan with well-aerated mastoid air cells",
        "CT scan displaying sella turcica and sphenoid bone",
        "head CT scan showing orbits and facial structures"
    ]
    
    if model_type.startswith("huggingface"):
        inputs = processor(text=detailed_prompts, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
    else:
        image_input = processor(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(detailed_prompts).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            logits_per_image = (image_features @ text_features.T) * 100
            probs = logits_per_image.softmax(dim=1)
    
    # Get best match
    best_idx = probs[0].argmax()
    best_description = detailed_prompts[best_idx]
    confidence = float(probs[0][best_idx])
    
    return best_description, confidence

def main():
    parser = argparse.ArgumentParser(description="Analyze medical image with PubMedCLIP")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the image file")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 60)
    print("PubMedCLIP Image Analysis")
    print("=" * 60)
    print(f"Image: {args.image_path}")
    print("=" * 60)
    
    # Load model
    print("\nLoading PubMedCLIP model...")
    model, processor, model_type = load_pubmedclip(device=device)
    
    if model is None:
        print("Error: Could not load PubMedCLIP model")
        print("Please install required packages:")
        print("  pip install transformers torch torchvision pillow")
        sys.exit(1)
    
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully! (Type: {model_type})")
    
    # Analyze image
    print("\nAnalyzing image...")
    results = analyze_image_with_text_prompts(args.image_path, model, processor, model_type, device)
    
    print("\n" + "=" * 60)
    print("Image Analysis Results:")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['description']}")
        print(f"   Confidence: {result['confidence']*100:.2f}%")
    
    # Generate description
    print("\n" + "=" * 60)
    print("Best Match Description:")
    print("=" * 60)
    description, confidence = generate_description(args.image_path, model, processor, model_type, device)
    print(f"Description: {description}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("=" * 60)
    
    # Save results
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    output_file = f"pubmedclip_analysis_{image_name}.txt"
    
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("PubMedCLIP Image Analysis Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Image: {args.image_path}\n")
        f.write(f"Model Type: {model_type}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Image Analysis Results:\n")
        f.write("-" * 60 + "\n")
        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['description']}\n")
            f.write(f"   Confidence: {result['confidence']*100:.2f}%\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Best Match Description:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{description}\n")
        f.write(f"Confidence: {confidence*100:.2f}%\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()

