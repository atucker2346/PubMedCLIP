# PubMedCLIP Image Analysis Setup

## Quick Start

1. **Install Dependencies:**
   ```bash
   cd /Users/aj/Documents/GitHub/PubMedCLIP
   pip install transformers torch torchvision pillow
   ```

2. **Run the Analysis:**
   ```bash
   python test_image.py --image_path ./image.0012.png
   ```

## What the Script Does

The script uses PubMedCLIP (or standard CLIP as fallback) to:
- Analyze your medical image against medical text descriptions
- Find the best matching description
- Generate a confidence score for each match
- Save results to a text file

## Output

The script will:
1. Try to load PubMedCLIP from HuggingFace (sarahESL/PubMedCLIP)
2. If that fails, use standard CLIP as fallback
3. Compare your image against medical image descriptions
4. Show top 5 matches with confidence scores
5. Save results to `pubmedclip_analysis_image.0012.txt`

## Requirements

- Python 3.7+
- torch
- torchvision  
- transformers
- pillow
- (optional) clip-by-openai (for standard CLIP fallback)

## Troubleshooting

If you get import errors, install missing packages:
```bash
pip install transformers torch torchvision pillow
```

If HuggingFace model download fails, the script will automatically fall back to standard CLIP.

