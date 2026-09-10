"""Optional Colab setup helper.

Run this before importing MediaPipe if Colab hits MediaPipe audio import issues.
Package installation is still best done with:
    pip install -r requirements.txt
"""

import sys
import types


def patch_mediapipe_audio_imports():
    for mod_name in [
        "mediapipe.tasks.python.audio",
        "mediapipe.tasks.python.audio.audio_classifier",
        "mediapipe.tasks.python.audio.audio_embedder",
        "mediapipe.tasks.python.audio.core",
    ]:
        sys.modules[mod_name] = types.ModuleType(mod_name)


if __name__ == "__main__":
    patch_mediapipe_audio_imports()
    print("MediaPipe audio monkey-patch applied")

