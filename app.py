"""Hugging Face Spaces entry point.

Spaces auto-detects an `app.py` at the repo root and runs it. We just
delegate to the Gradio UI builder in `ui/app.py`. The Space's
`requirements.txt` already covers the deps; the inswapper model and
buffalo_l detector are downloaded on first request.
"""

from ui.app import build_app

if __name__ == "__main__":
    build_app().queue().launch()
