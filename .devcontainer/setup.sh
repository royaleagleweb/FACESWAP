#!/usr/bin/env bash
set -euo pipefail

echo ">>> Installing system dependencies"
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends ffmpeg unzip

echo ">>> Installing Python dependencies"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ">>> Pre-downloading face models (one-time, ~825 MB total)"
mkdir -p models ~/.insightface/models

if [ ! -f models/inswapper_128.onnx ]; then
  curl -fL --retry 3 \
    -o models/inswapper_128.onnx \
    https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
fi

if [ ! -f ~/.insightface/models/buffalo_l/det_10g.onnx ]; then
  curl -fL --retry 3 \
    -o ~/.insightface/models/buffalo_l.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  unzip -q -o ~/.insightface/models/buffalo_l.zip -d ~/.insightface/models/buffalo_l
  rm ~/.insightface/models/buffalo_l.zip
fi

echo ">>> Setup complete. Desktop app: python -m videoswa"
