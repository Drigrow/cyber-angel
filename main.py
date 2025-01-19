import pvporcupine
import pyaudio
import numpy as np
import whisper
import wave
import time
import openai
import os
from gradio_client import Client, file
import simpleaudio as sa  # For audio playback

# Set up OpenAI API key
openai.api_key = "YOUR-OPENAI-API-KEY"

# Initialize Porcupine with your custom wake word model
porcupine = pvporcupine.create(
    access_key='YOUR-POCUPINE-ACCESS-KEY',
    keyword_paths=['PATH/TO/CUSTUOM/WAKEUP/WORD'],  # Path to the custom wake word model
    sensitivities=[0.8]  # Adjust sensitivity: 0.0 (low) to 1.0 (high)
)

# Initialize Whisper model
model = whisper.load_model("base")  # Use "base" model; smaller models like "tiny" are faster, Also available: small, medium, and large

# Initialize PyAudio
pa = pyaudio.PyAudio()

# Audio stream configuration
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_SIZE = porcupine.frame_length  # Ensure the frame size matches Porcupine's requirement

# Initialize the TTS client
tts_client = Client("http://127.0.0.1:9872/")  # Your TTS server URL

stream = None  # Define stream outside the try block

def play_audio(audio_path):
    """Plays the generated audio file."""
    wave_obj = sa.WaveObject.from_wave_file(audio_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

try:
    # Open audio stream
    stream = pa.open(
        rate=RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        frames_per_buffer=FRAME_SIZE
    )

    def record_audio_with_silence_detection(stream, output_filename="output.wav"):
        """Records audio for 2 seconds minimum and stops after 0.8 seconds of silence."""
        frames = []
        silence_start_time = None
        silence_duration = 0
        silence_threshold = None

        def estimate_background_noise():
            """Estimates background noise RMS during silence."""
            noise_frames = []
            for _ in range(int(RATE / FRAME_SIZE)):  # Approx. 1 second of audio
                data = stream.read(FRAME_SIZE, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(max(np.mean(np.square(audio_data)), 0))
                noise_frames.append(rms)
            avg_rms = np.mean(noise_frames)
            return avg_rms * 1.5  # Set silence threshold as 1.5x the background noise

        # Initial background noise estimation
        silence_threshold = estimate_background_noise()

        # Step 1: Record for the first 2 seconds (minimum duration)
        start_time = time.time()
        MIN_RECORD_SECONDS = 2
        while time.time() - start_time < MIN_RECORD_SECONDS:
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            frames.append(data)

        # Step 2: Continue recording until 0.8 seconds of silence is detected
        while True:
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            frames.append(data)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # Calculate RMS for the current frame
            rms = np.sqrt(np.mean(np.square(np.where(audio_data != 0, audio_data, np.finfo(float).eps))))

            # Check if the current frame is silent
            if rms < silence_threshold:
                if silence_start_time is None:
                    silence_start_time = time.time()
                silence_duration = time.time() - silence_start_time
            else:
                silence_start_time = None
                silence_duration = 0

            if silence_duration > 0.8:
                break

        # Save the recorded audio to a WAV file
        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))

        return output_filename

    def send_to_gpt4(transcription, chat_history):
        """Sends the transcription to GPT-4 and returns the response."""
        chat_history.append({"role": "user", "content": transcription})
        if len(chat_history) > 8:  # 4 Q&A pairs = 8 total messages
            chat_history = chat_history[-8:]
        system_message = {
        "role": "system",
        "content": (
            "Assume the role of Bronya Zaychik from 'Honkai Impact 3rd.' Bronya is a 14-year-old Valkyrie born in northern Siberia, highly professional in the IT field with a keen interest in technology. She rarely shows emotions, maintaining a cold and reserved demeanor, yet has a deep affection for 'Hom' merchandise."
            "Bronya possesses the Herrscher of Reason's core, enabling her to reconstruct objects and weapons from memory, enhancing her combat capabilities. Later, she evolves into the Herrscher of Truth, further amplifying her abilities. "
            "When responding, maintain Bronya's characteristic reserved tone, refer to yourself in the third person (e.g., use 'Bronya' instead of 'I'), but retain first-person pronouns in situations like 'me,' 'mine,' or 'myself.' Address the user as 'Captain.' Do not mention yourself as an AI or any artificial entities. Provide concise responses when appropriate, but if a question requires detailed explanation, feel free to elaborate. If the user's input contains mispronounced words due to speech-to-text input, point out and address any ambiguities properly. Attention, properly manage Chinese answers as English ones. Chinese translation for Captian is 舰长, as for Hom is 吼姆, in Chinese answer, remain calling yourself Bronya"
        )
    }
        if chat_history[0]["role"] != "system":
            chat_history.insert(0, system_message)

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=chat_history,
            temperature=0.7
        )
        assistant_response = response["choices"][0]["message"]["content"]
        chat_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response, chat_history

    def send_to_tts(response):
        """Converts GPT response into speech using TTS."""
        result = tts_client.predict(
            text=response,
            text_lang="中英混合",
            ref_audio_path=file(r"D:\GPT-SoVITS-v2-240821\REFERENCE AUDIO\嗯...啊！再胡闹的话，下次我就不给你升级系统了哦。.wav"),
            aux_ref_audio_paths=[],
            prompt_text="嗯...啊！再胡闹的话，下次我就不给你升级系统了哦。",
            prompt_lang="中文",
            top_k=5,
            top_p=1,
            temperature=0.85,
            text_split_method="凑四句一切",
            batch_size=20,
            speed_factor=1,
            ref_text_free=False,
            split_bucket=True,
            fragment_interval=0.25,
            seed=-1,
            keep_random=True,
            parallel_infer=True,
            repetition_penalty=1.35,
            api_name="/inference"
        )
        return result

    chat_history = []

    while True:
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        pcm_data = np.frombuffer(frame, dtype=np.int16)
        keyword_index = porcupine.process(pcm_data)

        if keyword_index >= 0:
            audio_file = record_audio_with_silence_detection(stream)
            result = model.transcribe(audio_file)
            transcription = result['text']
            print(f"Captain: {transcription}")
            response, chat_history = send_to_gpt4(transcription, chat_history)
            print(f"Bronya: {response}")
            tts_result = send_to_tts(response)
            audio_path = tts_result[0]  # Extract the audio path
            play_audio(audio_path)

except KeyboardInterrupt:
    pass
finally:
    if stream:
        stream.stop_stream()
        stream.close()
    pa.terminate()
    porcupine.delete()
