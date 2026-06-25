# audio-portrait

Animate a portrait image to match any speech audio. Give it your face photo + a voice recording → get a talking-head video with synced lip movements and natural blinking.

## How it works

```
Audio (.wav/.mp3)
    ↓  Whisper → word timestamps → phoneme openness
Per-frame lip_ratio + blink schedule
    ↓  expression_generator.py
LivePortrait motion template (exp, lip, eye params)
    ↓  LivePortrait warp + SPADE decode
Animated frames → ffmpeg merge with original audio
    ↓
output/result.mp4
```

## Setup

### 1. Clone both repos side by side

```bash
cd ~/git
git clone https://github.com/KlingAIResearch/LivePortrait.git
# audio-portrait is already here
```

Your folder structure should be:
```
~/git/
  LivePortrait/
  audio-portrait/
```

### 2. Install dependencies

```bash
cd ~/git/audio-portrait

# Install LivePortrait deps first
pip install -r ../LivePortrait/requirements.txt

# Then audio-portrait deps
pip install -r requirements.txt
```

### 3. Download LivePortrait weights

Follow instructions in `../LivePortrait/README.md` — weights go into:
```
LivePortrait/pretrained_weights/liveportrait/
  base_models/
    appearance_feature_extractor.pth
    motion_extractor.pth
    warping_module.pth
    spade_generator.pth
  retargeting_models/
    stitching_retargeting_module.pth
```

### 4. Add your face photo

```bash
mkdir -p assets
cp /path/to/your/face.jpg assets/my_face.jpg
```

Use a clear, front-facing photo with good lighting.

## Usage

```bash
# Using your default face (assets/my_face.jpg)
python run.py --audio speech.wav

# Custom image
python run.py --image /path/to/face.jpg --audio speech.wav --output output/result.mp4

# Faster (skip Whisper, energy-only lip sync)
python run.py --audio speech.wav --no-whisper

# Better accuracy (larger Whisper model, slower)
python run.py --audio speech.wav --whisper-model small
```

## Output

`output/result.mp4` — animated video with original audio merged in.

## Notes

- Works with English and German audio (Whisper auto-detects language)
- Natural eye blinking is added automatically
- No GPU VRAM limits — M4 Max unified memory handles everything
- Processing time: ~2-4× real-time on M4 Max (25s audio → ~60-100s render)
