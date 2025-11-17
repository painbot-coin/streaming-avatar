import os
from deepgram import DeepgramClient, SpeakOptions
import time



def synthesize_text(question):

    try:
        deepgram = DeepgramClient("ee7539cab75622bfe711d89af434462ba6829ce0")
        options = SpeakOptions(
                    model="aura-asteria-en",
                    encoding="linear16",
                    container="wav"
                )

        SPEAK_OPTIONS = {"text": question}
        filename = "./audio.wav"
        deepgram.speak.v("1").save(filename, SPEAK_OPTIONS, options)
        
    except Exception as e:
        print(f"Exception: {e}")
b = time.time()
synthesize_text("Dropbox Replay allows creative teams to better create, collaborate on and deliver content. No Dropbox account is required to give feedback.")
a = time.time()
print("Latency for audio is ", a - b)