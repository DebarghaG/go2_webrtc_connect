import asyncio
import logging
import os
import sys
import time
import io
import numpy as np
import pyaudio
import speech_recognition as sr
from gtts import gTTS
from openai import OpenAI
from dotenv import load_dotenv
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from aiortc.contrib.media import MediaPlayer
import tempfile

# Load environment variables (OpenAI API Key)
load_dotenv()

# --- Configuration ---
ROBOT_IP = "192.168.8.181"  # Replace with your Go2's IP address
SAMPLE_RATE = 16000 # We use 16000 for speech recognition.  48000 is unnecessarily high.
CHANNELS = 1  # Mono for microphone input
CHUNK_SIZE = 1024
VAD_THRESHOLD = 300  # Adjust this based on your environment's noise level.  Higher = less sensitive.
SILENCE_LIMIT = 1.0  # Seconds of silence to consider speech ended.

# --- Setup ---
logging.basicConfig(level=logging.INFO) # Set logging to INFO for better visibility
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
recognizer = sr.Recognizer()
audio = pyaudio.PyAudio()

# --- Voice Activity Detection (VAD) ---
def is_speaking(audio_data):
    """Simple energy-based VAD."""
    audio_as_np_int16 = np.frombuffer(audio_data, dtype=np.int16)
    energy = np.sqrt(np.mean(audio_as_np_int16**2))
    return energy > VAD_THRESHOLD

async def capture_audio(timeout=5):
    """Captures audio from the microphone until silence is detected or timeout."""
    logging.info("Listening...")
    stream = audio.open(format=pyaudio.paInt16,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

    frames = []
    silence_start = None
    try:
        while True:
            data = stream.read(CHUNK_SIZE)
            if is_speaking(data):
                frames.append(data)
                silence_start = None  # Reset silence timer
            elif frames:  # If we have recorded something and now it's silent
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_LIMIT:
                    break
            if timeout is not None and len(frames) * CHUNK_SIZE / SAMPLE_RATE >= timeout:
                  break
    except OSError as e:
      if "Stream closed" in str(e):
        logging.warning("Stream closed while capturing audio.")
        return b""
      else:
        raise

    finally:
        stream.stop_stream()
        stream.close()

    return b"".join(frames)

async def transcribe_audio(audio_data):
    """Transcribes audio data using Google Web Speech API."""
    if not audio_data:
      return ""
    logging.info("Transcribing...")
    audio_data_sr = sr.AudioData(audio_data, SAMPLE_RATE, 2)  # 2 for sample width (16-bit)
    try:
        text = recognizer.recognize_google(audio_data_sr)  # Use Google Web Speech API
        logging.info(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        logging.warning("Could not understand audio")
        return ""
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Web Speech service; {e}")
        return ""

async def generate_response(text):
    """Generates a response using OpenAI's GPT-3.5 Turbo."""
    if not text:
      return "I didn't hear anything."
    logging.info("Generating response...")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="gpt-4o-mini",
        )
        response = chat_completion.choices[0].message.content
        logging.info(f"Robot: {response}")
        return response
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I'm having trouble thinking right now."


async def text_to_speech(text):
    """Converts text to speech using gTTS and returns a file path."""
    logging.info("Converting text to speech...")
    try:
        # Use a NamedTemporaryFile to automatically handle deletion.
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            tts = gTTS(text=text, lang='en')
            tts.save(fp.name)
            return fp.name  # Return the path to the temporary file.
    except Exception as e:
        logging.error(f"Error in text-to-speech: {e}")
        return None

async def play_audio_webrtc(conn, file_path):
    """Plays audio over WebRTC using aiortc."""
    if file_path is None:
      return
    logging.info(f"Playing audio file: {file_path}")

    try:
        player = MediaPlayer(file_path)
        if player.audio:
            conn.pc.addTrack(player.audio)
            await conn.pc.setLocalDescription(await conn.pc.createOffer())
            await conn.sendOffer() #ensure the offer is sent again after adding the track
            # No need to sleep here, as the main loop will keep things running
            # await asyncio.sleep(10)  # Play for a short time
    except Exception as e:
        logging.error(f"Error playing audio over WebRTC: {e}")

async def main():
    try:
        conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
        await conn.connect()
        conn.audio.switchAudioChannel(True)

        while True:
            audio_data = await capture_audio()
            text = await transcribe_audio(audio_data)
            if text:
                response = await generate_response(text)
                audio_file_path = await text_to_speech(response)
                if audio_file_path:
                    await play_audio_webrtc(conn, audio_file_path)
            else:
                logging.info("No speech detected, listening again...")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        await conn.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)