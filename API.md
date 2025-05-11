# API Guide

Use the table below to navigate quickly to each feature section.

| Feature               | Description                                          | Jump To                                 |
| --------------------- | ---------------------------------------------------- | --------------------------------------- |
| Audio Recording (WAV) | Record WAV audio via fixed, toggle, or silence modes | [WAV Recorder](#wav-recorder)           |
| Audio Recording (MP3) | Record MP3 audio via fixed, toggle, or silence modes | [MP3 Recorder](#mp3-recorder)           |
| Speech-to-Text        | Transcribe audio and detect language                 | [OpenAI Whisper](#openai-whisper)       |
| Gemini Client (LLM)   | Interact with Google Gemini using memory             | [GeminiClient](#geminiclient)           |
| Web Search            | Grab information from Google                         | [Google Searcher](#google-searcher)     |
| Image Captioning      | Generate captions for images                         | [Image Captioner](#image-captioner)     |
| OCR                   | Extract text from images or screen regions           | [EasyOCR Extractor](#easyocr-extractor) |
| Text-to-Speech        | Convert text to speech via gTTS or pyttsx3           | [Text-to-Speech](#text-to-speech)       |

---

## WAV Recorder

**Record WAV audio (fixed duration, toggle, or silence-based).**

```python
import freeai_utils

rec_w = freeai_utils.WavRecorder()
rec_w.record_fixed(3, output_filename="fixed_record.wav")  # record for 3 seconds
rec_w.record_toggle(toggle_key='`', output_filename="toggle_record.wav")
rec_w.record_silence(silence_threshold=800, max_silence_seconds=3, output_filename="silence_record.wav")
```

---

## MP3 Recorder

**Record MP3 audio (same API surface as WAV).**

```python
import freeai_utils

rec_m = freeai_utils.MP3Recorder()
rec_m.record_fixed(4, output_filename="fixed_record.mp3")  # record for 4 seconds
# Other methods: record_toggle(), record_silence()
```

---

## OpenAI Whisper

**Transcribe audio files and detect language.**

```python
import freeai_utils

transcriber = freeai_utils.OpenAIWhisper()
output = transcriber.transcribe("output_pyaudio.wav")
print("Detected language:", output["language"])
print("Transcript:", output["text"])

print(output)  # full details

lang_detect = transcriber.get_lang_detect("output_pyaudio.wav")
print(lang_detect)

translation = transcriber.get_translation("output_pyaudio.wav")
print(translation)
```

---

## GeminiClient

**Connect to Google Gemini with optional memory storage.**

```python
import freeai_utils

client = freeai_utils.GeminiClient(api_key="your_api_key")
client.list_models()
answer = client.ask("What land animal do you think is the best?")
print(answer)

answer = client.ask_and_copy_to_clipboard("Could you write a hello world python script?")
answer = client.ask_with_memories("My name is An")
print(answer)
answer = client.ask_with_memories("Do you remember what we're talking about?")
print(answer)
```

---

## Google Searcher

**Search Google and retrieve snippets.**

```python
import freeai_utils

google_search = freeai_utils.GoogleSearcher(num_results=5, limit_word_per_url=500)
results = google_search.search("What is the capital of Vietnam?")
print(results)
```

---

## Image Captioner

**Generate a descriptive caption for an image.**

```python
import freeai_utils

cap = freeai_utils.ImageCaptioner()
text = cap.write_caption("your_image.png")
print("Caption:", text)
```

---

## EasyOCR Extractor

**Extract text from screen captures or image files.**

```python
import freeai_utils

extractor = freeai_utils.Text_Extractor_EasyOCR(language='en')
text = extractor.get_text_from_screen()
print(text)

extractor.set_capture_region(crop_left=20, crop_right=5, crop_up=20, crop_down=5)
text = extractor.get_text_from_screen(image_name="your_image.png")
print(text)

text = extractor.get_text_from_screen(capture_region=(0,0,1920,1080), image_name="55.png")
print(text)
```

---

## Text-to-Speech

**Convert text to speech using gTTS or pyttsx3.**

```python
from freeai_utils import gtts_speak, Text_To_Speech_Pyttsx3

# gTTS version
gtts_speak("Xin chào người đẹp, em tên là gì thế", "vi")
gtts_speak("Hello, how are you?", "en")

# pyttsx3 version
speaker = Text_To_Speech_Pyttsx3()
speaker.speak("Hello there my friend")
speaker.config_voice(rate=165, volume=0.8)
```
