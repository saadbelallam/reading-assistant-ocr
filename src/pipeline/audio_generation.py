################
#    Step 7    #
################

from gtts import gTTS
import os


def generate(text, output_dir):
    """
    Generate an audio file for the extracted text.
    """
    print("Running audio generation ...")
    audio_path = os.path.join(output_dir, f"7_audio.mp3")
    myobj = gTTS(text=text, lang="fr", slow=False)
    myobj.save(audio_path)