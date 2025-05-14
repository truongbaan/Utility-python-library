# freeai-utils

A lightweight collection of Python utility functions that can be installed directly using `pip` — no external `.exe` downloads or complicated setups required. This library trades some accuracy for ease of implementation. It's designed for developers who prefer quick functionality over manual setup. Just call the functions — they’ll handle the rest.

## Features

- **Audio**: record WAV/MP3 (fixed, toggle, silence-triggered)  
- **Speech-to-Text**: OpenAI Whisper transcription & language detection   
- **Web Search**: scrape & summarize Google results  
- **Image**: caption generation & OCR (EasyOCR)  
- **TTS**: text-to-speech via gTTS or pyttsx3
- **PDF-DOCX-Reader**: extract text and images from pdf and docx files
* **Gemini API**: Interact with Google Cloud Gemini models via your API key

  * *Note: This works best with Google accounts that have no billing method added yet (completely free to use with limits).*

## Installation

```bash
pip install freeai-utils
```

> No need to install extra executables or clone large repositories — everything works out of the box with `pip`.

---

## 📖 Full API Reference

For a detailed list of all classes and methods, see [API.md](https://github.com/truongbaan/Utility-python-library/blob/main/API.md).

## Acknowledgements & References

See [THIRD_PARTY.md](https://github.com/truongbaan/Utility-python-library/blob/main/THIRD_PARTY.md) for a full list of third-party libraries and their licenses.

## Inspiration

This project was inspired by the GitHub repository:
[awesome-python](https://github.com/vinta/awesome-python)

## Note: 
This project is currently private; it will be made public in a future release.