# Videoswa

Videoswa is a Windows desktop app for swapping faces in a video. The simple
path uses one source photo for the face you select. A Multiple faces mode is
there when you want a different source for each person. The result is an MP4,
with the original audio copied back when FFmpeg can read it.

The swap engine is InsightFace: `buffalo_l` for detection and recognition,
and `inswapper_128` for the identity swap. Inference goes through ONNX Runtime.
On Windows the provider order is **TensorRT, then CUDA, then DirectML, then CPU**.

## Desktop app

```bat
python -m videoswa
```

`python run.py` opens the same window. The command-line swap tool is still
there for scripts (`python run.py swap ...`).

1. Choose a target video. Anything longer than **5 minutes** is rejected
   immediately, with an explanation, and nothing is processed.
2. Drag the sample-frame slider to a moment where the face is visible, then
   click **Detect faces**. Each thumbnail is labeled from InsightFace
   gender/age, for example **Face 1 — Male** or **Face 2 — Female**.
3. **Swap mode** opens on **Single face**. Choose one source image. The
   largest face is selected; click another thumbnail to replace that person
   instead. **Apply this source to every face** uses the same source for
   everyone. Switch to **Multiple faces** to pick a source on each thumbnail
   and leave a person empty to keep their face.
4. Click **Preview swap on this frame** to swap only the sample frame. The
   preview shows the original and the swapped frame side by side. Nothing is
   written. Click **Run swap** to export the MP4. A progress bar tracks
   frames. **Cancel** stops between frames and does not leave an output file
   behind.

The window is PySide6. There is no web UI.

## Ethics

Videoswa is for creative work, research, and accessibility where you have the
right to depict the people involved. Do not use it to harass, defame, deceive,
or create sexual content of a real person without their consent. When you
publish a result, say that it is synthetic media if the law or the platform
asks you to. You are responsible for what you do with it.

## Windows setup (RTX 4070)

These steps match the ONNX Runtime GPU builds that ship TensorRT 10.9 against
CUDA 12.x. An RTX 4070 (Ada, compute capability 8.9) is in that range. Do this
on 64-bit Windows 10 22H2 or Windows 11.

### 1. Python, FFmpeg, and the Visual C++ runtime

- Install [Python 3.11 64-bit](https://www.python.org/downloads/windows/) and
  check **Add python.exe to PATH**.
- Install the [Visual C++ 2015–2022 redistributable (x64)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
  ONNX Runtime's Windows wheels need it.
- Install FFmpeg and make sure `ffmpeg` and `ffprobe` are on PATH:

  ```bat
  winget install Gyan.FFmpeg
  ```

  Open a new terminal afterwards. `ffprobe -version` should print a version.

### 2. NVIDIA driver

Install the current Game Ready or Studio driver for the RTX 4070 from
[NVIDIA's driver download](https://www.nvidia.com/Download/index.aspx).
Then, in a terminal:

```bat
nvidia-smi
```

The driver has to be new enough for CUDA 12.8. If the CUDA installer in the
next step says the driver is too old, update the driver and reboot before
continuing.

### 3. CUDA Toolkit 12.8

Install [CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-12-8-0-download-archive)
(Windows, x86_64, exe local). The default install adds this directory to PATH:

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
```

Confirm in a **new** terminal:

```bat
nvcc --version
```

`onnxruntime-gpu` 1.21 through 1.26 is built with CUDA 12.8 and cuDNN 9.
`onnxruntime-gpu` 1.27 and newer are built with **CUDA 13** instead. This
guide pins 1.22 so the TensorRT version below matches. If you intentionally
install a CUDA 13 wheel, install CUDA 13 and a TensorRT build for that CUDA
major version, then check that `TensorrtExecutionProvider` shows up in the
provider list. The app code does not change.

### 4. cuDNN 9 for CUDA 12

Download cuDNN 9.x for CUDA 12 from the
[cuDNN archive](https://developer.nvidia.com/cudnn) (NVIDIA account required).
Use the Windows zip, not a cuDNN 8 build. Either:

- add that zip's `bin` directory to PATH, or
- copy `bin`, `lib`, and `include` into the CUDA 12.8 toolkit folders.

cuDNN 9 is not interchangeable with cuDNN 8.

You can also let pip supply the CUDA and cuDNN DLLs (this does **not**
include TensorRT):

```bat
pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
```

Videoswa calls `onnxruntime.preload_dlls()` when that function exists, before
it creates a session, so those wheel libraries are picked up.

### 5. TensorRT 10.9

ONNX Runtime 1.22's TensorRT execution provider is built for **TensorRT 10.9**.
Nearby pairs from the ONNX Runtime docs:

| onnxruntime-gpu | TensorRT | CUDA |
| --- | --- | --- |
| 1.22.x | 10.9 | 12.0–12.8 |
| 1.21.x | 10.8 | 12.0–12.8 |
| 1.20.x | 10.4 | 12.0–12.6 |

Download the TensorRT 10.9 Windows zip for CUDA 12 from
[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt/download). Extract it,
for example to `C:\TensorRT-10.9.0.34`. Add the `lib` folder — the one that
contains `nvinfer_10.dll` — to the system PATH:

```text
C:\TensorRT-10.9.0.34\lib
```

Open a new terminal so the updated PATH is visible.

### 6. Python packages

From the repository root:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip uninstall -y onnxruntime onnxruntime-directml
pip install -r requirements-windows-gpu.txt
```

`onnxruntime` (CPU) and `onnxruntime-gpu` cannot be installed together. The
second file replaces the CPU wheel with `onnxruntime-gpu==1.22.0`.

Optional face sharpening (not required to run the app):

```bat
pip install -r requirements-enhance.txt
```

### 7. Check the providers, then open the app

```bat
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
python -m videoswa
```

A working RTX 4070 install prints `TensorrtExecutionProvider` before
`CUDAExecutionProvider` and `CPUExecutionProvider`.

In the app, leave execution on **Auto**. On Windows that is TensorRT, then
CUDA, then DirectML, then CPU. The window shows the provider that is actually
running. **Running on CPU** means the export will be slow. The first
swap compiles TensorRT engines into `models\trt_cache`. That can take several
minutes and is reused on the next run. FP16 engines are the default on this
path. Set `VIDEOSWA_TRT_FP16=0` before launching if you need FP32. The workspace
limit is `VIDEOSWA_TRT_WORKSPACE_MB` (default `2048`). Delete `models\trt_cache`
after you change ONNX Runtime, TensorRT, or the FP16 switch so engines are
built again.

If `TensorrtExecutionProvider` is missing, Videoswa still runs: it retries
with CUDA, then CPU, and the log says which one came up. The usual cause is
`nvinfer_10.dll` not on PATH, or a TensorRT build that does not match the
ONNX Runtime wheel.

Auto mode still requests TensorRT, then CUDA, then CPU. A CUDA session can
start and then die inside a convolution (`CUDNN_FE` / `GRAPH_EXECUTION_FAILED`
on an RTX 4070). Videoswa treats that as a runtime failure, drops both the
detector and InSwapper sessions, and continues on CPU. The status line says
**CUDA failed; using CPU**.

ONNX Runtime does not offer a provider option that disables the cuDNN
frontend. The switch it does honor is the environment variable, and the CUDA
option it does expose is `cudnn_conv_algo_search=DEFAULT` (the fallback
heuristic). Videoswa sets `ORT_DISABLE_CUDNN_FRONTEND=1` when you have not set
it, before ONNX Runtime is imported. A desktop shortcut or `Videoswa` launcher
should do the same if it starts Python itself:

```bat
set ORT_DISABLE_CUDNN_FRONTEND=1
.venv\Scripts\python.exe -m videoswa
```

Set `ORT_DISABLE_CUDNN_FRONTEND=0` before launch to keep the faster HEURISTIC
search. `onnxruntime-gpu==1.19.2` is the last Windows build without the cuDNN
9 frontend conv path; it does not match TensorRT 10.9. `1.20.1` still uses
that frontend (TensorRT 10.4). Keep `1.22.0` when you want TensorRT 10.9.

### DirectML when CUDA is broken

DirectML is the fast Windows GPU path that does not use CUDA or the cuDNN
frontend. `onnxruntime-directml` cannot sit next to `onnxruntime-gpu`. From
the same virtualenv:

```bat
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
pip install -r requirements-windows-directml.txt
```

That file pins `onnxruntime-directml==1.22.0`. Restart Videoswa and leave
execution on Auto, or choose **DirectML**. The banner should say **Running on
DirectML**. If DirectML is not installed, a CUDA graph failure still continues
on CPU and the banner says so.

**Export speed** defaults to **Full quality**. **Half resolution (faster)**
runs the swap on a smaller frame and writes an MP4 at the original size. Use
it when the banner says CPU. **Sharpen swapped faces (GFPGAN)** stays off
until you tick it. If GFPGAN is missing, the checkbox explains
`pip install -r requirements-enhance.txt`. When the package is installed,
`GFPGANv1.4.pth` downloads into `models/` on the first enhanced swap.
Detection and swapping reuse the same InsightFace sessions; turning sharpening
on does not reload InSwapper.

### CPU-only Windows

Skip CUDA and TensorRT. `pip install -r requirements.txt` is enough, then
`python -m videoswa`, and choose **CPU only** (or leave Auto; it will land on
CPU). Expect slower frames.

## Command line

Every face in the video, one source photo:

```bat
python run.py swap -t input.mp4 -s alice.jpg -o out.mp4
```

Two people, two sources. Each reference image is a still of that person
**in the target video** (a frame export is enough):

```bat
python run.py swap -t party.mp4 --pair alice.jpg=ref_alice.jpg --pair bob.jpg=ref_bob.jpg -o party_videoswa.mp4
```

| Flag | Meaning |
| --- | --- |
| `--execution auto` | TensorRT, then CUDA, then DirectML, then CPU on Windows |
| `--execution cuda` | CUDA, then CPU. Skips TensorRT |
| `--cpu` | CPU only |
| `--coverage full` | Jaw and beard replacement (default) |
| `--coverage normal` | Tight inner-face oval |
| `--similarity 0.32` | First-lock cosine threshold. A tracked face is kept below this |
| `--enhance` | GFPGAN on each swapped face, if that extra is installed |
| `--no-audio` | Drop the original audio track |
| `--crf 18 --preset medium` | x264 quality and speed |

Videos longer than 5 minutes are rejected here too, before models load.

## Beard and jaw coverage

InSwapper's own paste is a tight oval. It erodes the 128×128 crop, and the
mouth already sits near the bottom of that crop, so the chin and beard stay
on the original person.

Videoswa's default is **full coverage**. The eyes, nose, and mouth still come
from the aligned InSwapper crop. Below that crop, the swapped chin is stretched
over a landmark footprint that runs about **two eye-to-mouth lengths below the
mouth** and about **one eye-to-eye width out to each side of the jaw**. Inside
that footprint the replacement is complete (mask value 1). The soft rim is a
thin band, and only a light color correction is applied there, so the original
face does not show back through the cheeks, jaw, or beard.

The desktop control is **Face coverage**. **Full, including beard** is
selected when the window opens. **Normal (tight face)** is the smaller oval.
The CLI flag is `--coverage full` (default) or `--coverage normal`.

## Quality tips

A usable swap shows the source identity on the target person for almost every
frame, including when the head turns a little.

- Use a sharp, frontal source photo. The face should be unobstructed.
- Sample a video frame where that person's face is clear and large.
- Leave **Face coverage** on **Full, including beard**.
- The match threshold starts at **32%**. Raise it if the wrong person is
  swapped. Lower it only if the right person is left as the original.
- Videoswa locks a face to its source while the box still overlaps and the
  embedding is still that person, so a brief pose change does not flicker
  back to the original clip. A different person in that box is not swapped.
- **Sharpen swapped faces (GFPGAN)** stays off. Turn it on when the result
  looks soft.

## Single face and multiple faces

`buffalo_l` detects faces and reads a gender/age attribute with the embedding.
InsightFace encodes gender as 0 (female) or 1 (male). The desktop window shows
that next to the thumbnail (`Face 1 — Male`). A missing or unexpected value
is labeled **Unknown**.

**Single face** is the default. One source image is mapped to the selected
face from the sample frame (the largest face until you click another
thumbnail). **Apply this source to every face** is the wildcard: that source
replaces every detection. **Multiple faces** keeps the per-person source
list. On later frames each detection is compared with those reference
embeddings and swapped only when the closest match clears the threshold,
using that person's source photo and `inswapper_128`. A person with no source
photo is left alone. The CLI `--source` flag is the same wildcard as the
checkbox.

## Models

The first run downloads:

- `models/inswapper_128.onnx` (554,253,681 bytes), SHA256
  `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af`
- InsightFace `buffalo_l` under `~/.insightface/models/`

Mirrors are tried in order, starting with `Chuchuwa2/inswap` and
`crw-dev/Deepinsightinswapper`. A download that does not match is discarded
and the next mirror is tried. An older redistributed copy,
`a290273ed497312095dac48cdef20feec9d5208298223dd01288ab202b54bea7`, is also
accepted. If every mirror fails, the error lists each URL, the hash it
produced, and the hashes Videoswa will accept.

You can place `inswapper_128.onnx` in `models/` yourself when it matches one
of those hashes and the download is skipped. Override directories with
`FACESWAP_MODELS_DIR`, `FACESWAP_TEMP_DIR`, and `FACESWAP_OUTPUTS_DIR`.

## Programmatic use

```python
import cv2
from faceswap import FaceAnalyzer, FaceSwapper, FaceSwapEngine
from faceswap.video import process_video
from pathlib import Path

analyzer = FaceAnalyzer(execution="auto")
swapper = FaceSwapper(execution="auto")
engine = FaceSwapEngine(analyzer=analyzer, swapper=swapper)
mappings = engine.build_mappings([
    (cv2.imread("alice.jpg"), cv2.imread("ref_alice.jpg")),
    (cv2.imread("bob.jpg"), cv2.imread("ref_bob.jpg")),
])
process_video(engine, mappings, Path("input.mp4"), Path("out.mp4"))
```

`execution="cpu"` forces the CPU provider. `use_gpu=False` does the same.

## Tests

```bat
pip install pytest
pytest
```

The suite covers the TensorRT → CUDA → CPU provider order, Windows DirectML
before CPU, CUDA conv failures falling back to CPU, the 5-minute
rejection (no frames swapped), cancel, gender labels, single-face versus
per-face mapping, multi-face matching, and that the desktop window opens on
Single face with full beard coverage. It does not download the swap models
and it does not require an NVIDIA GPU. TensorRT execution itself needs the
Windows stack above.

## Layout

```text
faceswap/           InsightFace engine, video I/O, CLI
videoswa/           PySide6 desktop window
requirements.txt    CPU install, including PySide6
requirements-windows-gpu.txt
                    onnxruntime-gpu pin for TensorRT / CUDA
requirements-windows-directml.txt
                    Windows GPU path when CUDA is broken
```

## Troubleshooting

- **Video is too long.** Trim it under 5 minutes. The check uses `ffprobe`
  when it is installed, otherwise frame count divided by frame rate.
- **Duration unknown.** Install FFmpeg so `ffprobe` is on PATH, or re-encode
  the file to MP4. The swap will not start without a duration.
- **No audio in the output.** Install FFmpeg. The app says so in the status
  line when `ffmpeg` is missing.
- **Provider list has no TensorRT.** Confirm `nvinfer_10.dll` is on PATH and
  that TensorRT, CUDA, and `onnxruntime-gpu` are the versions in the table.
  CUDA-only is still a supported fallback.
- **Detect faces fails with `CUDNN_FE` / `GRAPH_EXECUTION_FAILED`.** The CUDA
  provider loaded, then cuDNN rejected a convolution. Videoswa rebuilds on
  DirectML when that provider is installed, otherwise on CPU. The banner
  names the provider. Install `requirements-windows-directml.txt` for a GPU
  that does not use CUDA. Launch with the project `.venv\Scripts\python.exe`.
- **The swap is slow.** Read the provider banner. CPU is the slow path.
  Half resolution is the faster export. DirectML is the Windows GPU
  alternative when CUDA is unhealthy.
- **First GPU run is very slow.** TensorRT is building engines in
  `models/trt_cache`. The next run of the same model should start promptly.
