import os
import time
import pytest
import pyaudio
import keyboard
import gc
from freeai_utils.audio_record import WavRecorder, check_wav_length_and_size, MP3Recorder, check_mp3_length_and_size

current = os.path.dirname(os.path.abspath(__file__))

def simulate_toggle(monkeypatch, toggle_key: str = "`"):
    holder = {}
    def fake_add_hotkey(key, callback):
        assert key == toggle_key
        holder['cb'] = callback
    monkeypatch.setattr(keyboard, "add_hotkey", fake_add_hotkey)
    monkeypatch.setattr(keyboard, "remove_hotkey", lambda k: None)

    real_sleep = time.sleep
    def fake_sleep(sec):
        real_sleep(sec)
        if 'cb' in holder:
            holder['cb']()
            holder.clear()
    monkeypatch.setattr(time, "sleep", fake_sleep)

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wav_rec():
    rec = WavRecorder()
    yield rec
    del rec
    gc.collect()

def test_init_wav_defaults(wav_rec):
    assert wav_rec.channels == 1
    assert wav_rec.rate == 44100
    assert wav_rec.chunk == 1024
    assert wav_rec.format == pyaudio.paInt16


def test_record_fixed(wav_rec):
    out = os.path.join(current, "fixed_record.wav")
    wav_rec.record_fixed(duration=1, output_filename=out)
    assert os.path.exists(out)
    assert check_wav_length_and_size(out, period=1.2)


def test_record_toggle(wav_rec, monkeypatch):
    out = os.path.join(current, "toggle_record.wav")
    simulate_toggle(monkeypatch, toggle_key="`")
    wav_rec.record_toggle(toggle_key="`", output_filename=out)
    assert os.path.exists(out)
    assert check_wav_length_and_size(out, period=1.2)


def test_record_silence(wav_rec):
    out = os.path.join(current, "silence_record.wav")
    wav_rec.record_silence(output_filename=out, silence_threshold=100000, max_silence_seconds=2)
    assert os.path.exists(out)
    assert check_wav_length_and_size(out,2.2)

@pytest.fixture(scope="module")
def mp3_rec():
    rec = MP3Recorder()
    yield rec
    del rec
    gc.collect()

def test_init_mp3_default(mp3_rec):
    assert mp3_rec.channels == 1
    assert mp3_rec.rate == 44100
    assert mp3_rec.chunk == 1024
    assert mp3_rec.bitrate == 192


def test_record_fixed_mp3(mp3_rec):
    out = os.path.join(current, "fixed_record.mp3")
    mp3_rec.record_fixed(duration=1, output_filename=out)
    assert os.path.exists(out)
    assert check_mp3_length_and_size(out, period=1.2)


def test_record_toggle_mp3(mp3_rec, monkeypatch):
    out = os.path.join(current, "toggle_record.mp3")
    simulate_toggle(monkeypatch, toggle_key="`")
    mp3_rec.record_toggle(toggle_key="`", output_filename=out)
    assert os.path.exists(out)
    assert check_mp3_length_and_size(out, period=1.2)


def test_record_silence_mp3(mp3_rec):
    out = os.path.join(current, "silence_record.mp3")
    mp3_rec.record_silence(output_filename=out, silence_threshold=100000, max_silence_seconds=2)
    assert os.path.exists(out)
    assert check_mp3_length_and_size(out,2.2)

def test_clean_up(): #not really a test, just a clean up :D
    from freeai_utils.cleaner import Cleaner
    cleaner = Cleaner(current)
    cleaner.remove_all_files_end_with(".wav")
    cleaner.remove_all_files_end_with(".mp3")