# API Guide

Use the table below to navigate quickly to each feature section.

| Feature               | Description                                          | Jump To                                 |
| --------------------- | ---------------------------------------------------- | --------------------------------------- |
| Audio Recording (WAV) | Record WAV audio via fixed, toggle, or silence modes | [WAV Recorder](#wav-recorder)           |
| Audio Recording (MP3) | Record MP3 audio via fixed, toggle, or silence modes | [MP3 Recorder](#mp3-recorder)           |
| Speech-to-Text        | Transcribe audio and detect language                 | [OpenAI Whisper](#openai-whisper)       |
|                       |                                                      | [VN Whisper](#vn-whisper)               |
| Gemini Client (LLM)   | Interact with Google Gemini using memory             | [GeminiClient](#geminiclient)           |
| Web Search            | Grab information from Google                         | [Google Searcher](#google-searcher)     |
| Image Captioning      | Generate captions for images                         | [Image Captioner](#image-captioner)     |
| OCR                   | Extract text from images or screen regions           | [EasyOCR Extractor](#easyocr-extractor) |
| Text-to-Speech        | Convert text to speech via gTTS or pyttsx3           | [Text-to-Speech](#text-to-speech)       |
| PDF-DOCX-Reader       | Extract text and images from pdf/docx file           | [PDF-DOCX-Reader](#pdf-docx-reader)     |

---

## WAV Recorder

**Record WAV audio (fixed duration, toggle, or silence-based).**

```python
import freeai_utils

rec_w = freeai_utils.WavRecorder()
rec_w.record_fixed(3, output_filename="fixed_record.wav")  # record for 3 seconds
rec_w.record_toggle(toggle_key='`', output_filename="toggle_record.wav") #record until press toggle_key
rec_w.record_silence(silence_threshold=800, max_silence_seconds=3, output_filename="silence_record.wav") #record until your voice < silence+threshold more than max_silence_seconds seconds.
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

## VN Whisper

**Transcribe audio files and detect language.**

```python
from freeai_utils.audio_to_text_vn import VN_Whisper

vn_transcriber = VN_Whisper() #init the model
text = vn_transcriber.transcribe_audio("your_audio.wav") #return str
print(text)

```

---

## GeminiClient

**Connect to Google Gemini with optional memory storage.**

```python
import freeai_utils

client = freeai_utils.GeminiClient(api_key="your_api_key") # if you have .env file with a var GEMINI_API_KEY = "your_key", you could just do client = freeai_utils.GeminiClient() and it would get the key in .env
client.list_models()
answer = client.ask("What land animal do you think is the best?") #answer only (no memory add)
print(answer)

answer = client.ask_and_copy_to_clipboard("Could you write a hello world python script?") #answer with copy to clipboard (use CTRL+V to paste)
answer = client.ask_with_memories("My name is An") #answer and add in memory
print(answer)
answer = client.ask_with_memories("Do you remember what we're talking about?") #answer with knowledge about the previous conversation
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

cap = freeai_utils.ImageCaptioner(device = "cuda")
text = cap.write_caption("your_image.png")
print("Caption:", text)
```

---

## EasyOCR Extractor

**Extract text from screen captures or image files.**

```python
import freeai_utils

extractor = freeai_utils.Text_Extractor_EasyOCR(language='en')
text = extractor.get_text_from_screen() #capture screen and do ocr
print(text)

extractor.set_capture_region(crop_left=20, crop_right=5, crop_up=20, crop_down=5)#change setting for the region (screenshot)
text = extractor.get_text_from_screen(image_name="your_image.png") #use the new setting for capturing, and make specific image_name
print(text)

text = extractor.get_text_from_screen(capture_region=(0,0,1920,1080), image_name="55.png") #do a fixed capture_region size 
print(text)
```

---

## Text-to-Speech

**Convert text to speech using gTTS or pyttsx3.**

```python
from freeai_utils import gtts_speak, Text_To_Speech_Pyttsx3

# gTTS version
gtts_speak(text = "Xin chào người đẹp, em tên là gì thế", lang = "vi")
gtts_speak(text = "Hello, how are you?", lang = "en")

# pyttsx3 version
speaker = Text_To_Speech_Pyttsx3()
speaker.speak("Hello there my friend") #speak with default voice
speaker.config_voice(rate=165, volume=0.8, voice_index = 1)#modify voice setting
speaker.speak("Hello there my friend") # speak with modified voice
```

## PDF-DOCX-Reader

** Return text and images from pdf or docx files. **

```python
from freeai_utils import PDF_DOCX_Reader

reader = PDF_DOCX_Reader(start_page = 0) #init the class, support only pdf and docx
print(reader.get_text_label("example.pdf")) #return str (text) base on the label
print(reader.get_all_text("example.pdf")) #return str (text) in the file
print(reader.extract_images("example.docx")) #return the number of images found in the file
```