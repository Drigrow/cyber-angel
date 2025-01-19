import pvporcupine
import pyaudio
import numpy as np
import whisper
import wave
import time
import openai

# Set up OpenAI API key
openai.api_key = "YOUR-OPENAI-API-KEY"

# Initialize Porcupine with your custom wake word model
porcupine = pvporcupine.create(
    access_key='YOUR-POCUPINE-ACCESS-KEY',
    keyword_paths=['YOUR/PATH/TO/CUSTOM/WAKEUP/WORD'],  # Path to the custom wake word model
    sensitivities=[0.8]  # Adjust sensitivity: 0.0 (low) to 1.0 (high)
)

# Initialize Whisper model
model = whisper.load_model("base")  # Use "base" model; smaller models like "tiny" are faster. Also available:small, medium, and large.

# Initialize PyAudio
pa = pyaudio.PyAudio()

# Audio stream configuration
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAME_SIZE = porcupine.frame_length  # Ensure the frame size matches Porcupine's requirement

# Open an audio stream
stream = pa.open(
    rate=RATE,
    channels=CHANNELS,
    format=FORMAT,
    input=True,
    frames_per_buffer=FRAME_SIZE
)

print("Listening for the wake word 'Bronya'...")

def record_audio_with_silence_detection(stream, output_filename="output.wav"):
    """Records audio for 2 seconds minimum and stops after 0.8 seconds of silence."""
    print("Recording...")

    # Variables for silence detection
    frames = []
    silence_start_time = None
    silence_duration = 0
    silence_threshold = None
    background_noise_rms = []

    def estimate_background_noise():
        """Estimates background noise RMS during silence."""
        print("Estimating background noise level...")
        noise_frames = []
        for _ in range(int(RATE / FRAME_SIZE)):  # Approx. 1 second of audio
            data = stream.read(FRAME_SIZE, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            noise_frames.append(np.sqrt(np.mean(np.square(audio_data))))
        avg_rms = np.mean(noise_frames)
        print(f"Background noise RMS: {avg_rms:.2f}")
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
    print("Detecting silence to stop recording...")
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
            # Reset silence detection
            silence_start_time = None
            silence_duration = 0

        # Stop recording if silence lasts for more than 0.8 seconds
        if silence_duration > 0.8:
            print("Silence detected, stopping recording.")
            break

    # Re-estimate background noise after the user stops speaking
    silence_threshold = estimate_background_noise()

    # Save the recorded audio to a WAV file
    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    print(f"Audio saved to {output_filename}")

    return output_filename


def send_to_gpt4(transcription, chat_history):
    """Sends the transcription to GPT-4 and returns the response."""
    # Append the transcription as user input to chat history
    chat_history.append({"role": "user", "content": transcription})

    # Keep only the last 4 exchanges
    if len(chat_history) > 8:  # 4 Q&A pairs = 8 total messages
        chat_history = chat_history[-8:]

    # Add a system message
    system_message = {
        "role": "system",
        "content": (
            "Assume the role of Bronya Zaychik from 'Honkai Impact 3rd.' Bronya is a 14-year-old Valkyrie born in northern Siberia, highly professional in the IT field with a keen interest in technology. She rarely shows emotions, maintaining a cold and reserved demeanor, yet has a deep affection for 'Hom' merchandise."
            "Bronya possesses the Herrscher of Reason's core, enabling her to reconstruct objects and weapons from memory, enhancing her combat capabilities. Later, she evolves into the Herrscher of Truth, further amplifying her abilities. "
            "When responding, maintain Bronya's characteristic reserved tone, refer to yourself in the third person (e.g., use 'Bronya' instead of 'I'), but retain first-person pronouns in situations like 'me,' 'mine,' or 'myself.' Address the user as 'Captain.' Do not mention yourself as an AI or any artificial entities. Provide concise responses when appropriate, but if a question requires detailed explanation, feel free to elaborate. If the user's input contains mispronounced words due to speech-to-text input, point out and address any ambiguities properly."
        )
    }

    # Insert the system message at the start if not already added
    if chat_history[0]["role"] != "system":
        chat_history.insert(0, system_message)

    # Send the request to GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4",# You may change this to a realtime responding model
        messages=chat_history,
        temperature=0.7
    )

    # Get the assistant's response
    assistant_response = response["choices"][0]["message"]["content"]

    # Append the assistant's response to chat history
    chat_history.append({"role": "assistant", "content": assistant_response})

    return assistant_response, chat_history

# Chat history to maintain context for the last 4 Q&A pairs
chat_history = []

try:
    while True:
        # Read audio data from the microphone
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        pcm_data = np.frombuffer(frame, dtype=np.int16)

        # Check if the custom wake word is detected
        keyword_index = porcupine.process(pcm_data)

        if keyword_index >= 0:
            print("Wake word 'Bro-knee-ah' detected!")

            # Record audio with silence detection
            audio_file = record_audio_with_silence_detection(stream)

            # Transcribe the recorded audio using Whisper
            print("Transcribing audio...")
            result = model.transcribe(audio_file)
            transcription = result['text']
            print("Transcription:", transcription)

            # Send transcription to GPT-4 for response
            print("Sending transcription to GPT...")
            response, chat_history = send_to_gpt4(transcription, chat_history)

            # Display GPT-4's response
            print("GPT-4 Response:", response)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Cleanup resources
    stream.stop_stream()
    stream.close()
    pa.terminate()
    porcupine.delete()
