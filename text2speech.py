# import os
# from deepgram import DeepgramClient, SpeakOptions
# import time

# def synthesize_text(question,audio_path):
#     try:
#         deepgram = DeepgramClient("ee7539cab75622bfe711d89af434462ba6829ce0")
#         options = SpeakOptions(
#                     model="aura-asteria-en",
#                     encoding="linear16",
#                     container="wav",
#                     sample_rate=16000
#                 )
#         SPEAK_OPTIONS = {"text": question}
#         filename = audio_path
#         deepgram.speak.v("1").save(filename, SPEAK_OPTIONS, options)
#     except Exception as e:
#         print(f"Exception: {e}")

# synthesize_text("This is a test to see how the Aphra Avatars behave when we use different photos", "./test.wav")


from elevenlabs.client import ElevenLabs
from elevenlabs import play
from elevenlabs import save

def synthesize_text(text, audio_path):
    client = ElevenLabs(
    api_key="sk_cb1f2c68cf0b51c567879902f1e245b8ba12e1168d63009c", # Defaults to ELEVEN_API_KEY
    )
    audio = client.generate(
    text=text,
    voice="Rachel",  #Rachel
    model="eleven_multilingual_v2"
    )

    save(audio, audio_path)