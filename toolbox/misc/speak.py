import pyttsx3

def say_done():
    engine = pyttsx3.init()
    engine.setProperty('volume',1.0)
    engine.say("Hey Willi, I'm done")
    engine.runAndWait()