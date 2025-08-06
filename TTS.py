# tts_module.py
import pyttsx3


class TTSManager:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)

    def speak(self, text):
        """Convert text to speech"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def speak_async(self, text):
        """Convert text to speech asynchronously"""
        self.tts_engine.say(text)
        self.tts_engine.startLoop(False)
        self.tts_engine.iterate()
        self.tts_engine.endLoop()
