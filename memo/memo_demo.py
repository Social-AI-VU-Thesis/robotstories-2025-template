import json
import wave
from os import environ
from os.path import abspath, join
from subprocess import call

import numpy as np
from sic_framework.core.message_python2 import AudioMessage, AudioRequest
from sic_framework.devices import Nao, Pepper, device
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest

from sic_framework.services.openai_gpt.gpt import GPT, GPTConf, GPTRequest
from dotenv import load_dotenv

from sic_framework.devices.desktop import Desktop
from sic_framework.services.dialogflow.dialogflow import (
    Dialogflow,
    DialogflowConf,
    GetIntentRequest,
)

"""
This is a demo show casing a agent-driven conversation utalizating Google Dialogflow, Google TTS, and OpenAI's GTP4

IMPORTANT
First, you need to set-up dialogflow:

1. Dialogflow: https://socialrobotics.atlassian.net/wiki/spaces/CBSR/pages/2205155343/Getting+a+google+dialogflow+key 
2. Create a keyfile as instructed in (1) and save it conf/dialogflow/google_keyfile.json
2a. note: never share the keyfile online. 
3. In your empty dialogflow agent do the following things:
3a. remove all default intents
3b. go to settings -> import and export -> and import the memo/resources/dialogflow_agent.zip into your
dialogflow agent. That gives all the necessary intents and entities that are part of this example (and many more)

Secondly, you need to install espeak
Windows]
4. download and install espeak: http://espeak.sourceforge.net/
4a. add eSpeak/command-line to PATH
[Linux]
4. sudo apt-get install espeak libespeak-dev
[MacOS]
4. brew install espeak

Thirdly, you need an openAI key:
5. Generate your personal openai api key here: https://platform.openai.com/api-keys
6. Either add your openai key to your systems variables or
create a .openai_env file in the conf/openai folder and add your key there like this:
OPENAI_API_KEY="your key"

Forth, the redis server, Dialogflow, and OpenAI gpt service need to be running:

7. pip install --upgrade social-interaction-cloud[dialogflow,openai-gpt]
8. run: conf/redis/redis-server.exe conf/redis/redis.conf
9. run in new terminal: run-dialogflow 
10. run in new terminal: run-gpt
11. connect a device e.g. desktop, nao, pepper
12. Run this script
"""


class ConversationDemo:
    def __init__(self, device, google_keyfile_path, sample_rate_dialogflow_hertz=44100, lang="nl", openai_key_path=None):

        # device
        self.device = device
        self.lang = lang

        # Setup GPT client
        if openai_key_path:
            load_dotenv(openai_key_path)

        conf = GPTConf(openai_key=environ["OPENAI_API_KEY"])
        self.gpt = GPT(conf=conf)
        print("OpenAI GPT4 Ready")

        # set up the config for dialogflow
        dialogflow_conf = DialogflowConf(keyfile_json=json.load(open(google_keyfile_path)),
                                         sample_rate_hertz=sample_rate_dialogflow_hertz,
                                         language=self.lang)

        # initiate Dialogflow object
        self.dialogflow = Dialogflow(ip="localhost", conf=dialogflow_conf)
        # connect the output of DesktopMicrophone as the input of DialogflowComponent
        self.dialogflow.connect(self.device.mic)

        # register a callback function to act upon arrival of recognition_result
        self.dialogflow.register_callback(self.on_dialog)
        print("Dialogflow Ready")

        # flag to signal when the app should listen (i.e. transmit to dialogflow)
        self.request_id = np.random.randint(10000)

    @staticmethod
    def on_dialog(message):
        if message.response:
            if message.response.recognition_result.is_final:
                print("Transcript:", message.response.recognition_result.transcript)

    def say(self, text, animated=True):
        if isinstance(self.device, Desktop):
            call(["espeak", "-s140", f"-v{self.lang}", "-z", text])
        elif isinstance(self.device, Pepper) or isinstance(device, Nao):
            self.device.tts.request(
                NaoqiTextToSpeechRequest(text, animated=animated))
        else:
            print(f"Error: device {self.device} not recognized")

    def play_audio(self, audio_file):
        with wave.open(audio_file, 'rb') as wf:
            # Get parameters
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            # Ensure format is 16-bit (2 bytes per sample)
            if sample_width != 2:
                raise ValueError("WAV file is not 16-bit audio. Sample width = {} bytes.".format(sample_width))

            self.device.speaker.request(AudioRequest(wf.readframes(n_frames), framerate))

    def ask_yesno(self, question, max_attempts=2):
        attempts = 0
        while attempts < max_attempts:
            # ask question
            self.say(question)

            # listen for answer
            reply = self.dialogflow.request(GetIntentRequest(self.request_id, {'answer_yesno': 1}))

            print("The detected intent:", reply.intent)

            # return answer
            if reply.intent:
                if "yesno_yes" in reply.intent:
                    return "yes"
                elif "yesno_no" in reply.intent:
                    return "no"
                elif "yesno_dontknow" in reply.intent:
                    return "dontknow"
            attempts += 1
        return None

    def ask_entity(self, question, context, target_intent, target_entity, max_attempts=2):
        attempts = 0

        while attempts < max_attempts:
            # ask question
            self.say(question)

            # listen for answer
            reply = self.dialogflow.request(GetIntentRequest(self.request_id, {context: 1}))

            print("The detected intent:", reply.intent)

            # Return entity
            if reply.intent:
                if target_intent in reply.intent:
                    if reply.response.query_result.parameters and target_entity in reply.response.query_result.parameters:
                        return reply.response.query_result.parameters[target_entity]
            attempts += 1
        return None

    def ask_open(self, question, max_attempts=2):
        attempts = 0

        while attempts < max_attempts:
            # ask question
            self.say(question)

            # listen for answer
            reply = self.dialogflow.request(GetIntentRequest(self.request_id))

            print("The detected intent:", reply.intent)

            # Return entity
            if reply.response.query_result.query_text:
                return reply.response.query_result.query_text
            attempts += 1
        return None

    def llm_request(self, prompt):
        gpt_response = self.gpt.request(
            GPTRequest(prompt))
        return gpt_response.response

    def run(self):
        self.say("Hallo, ik ben Memo de robot.")

        wil_kletsen = self.ask_yesno("Hou je van dieren?")
        if wil_kletsen and wil_kletsen == "yes":
            lievelingsdier = self.ask_entity("Wat is jouw lievelingsdier?", 'animals', 'animals', 'animals')
            if lievelingsdier:
                self.say(f"Oh een {lievelingsdier}!")
                waarom_lievelingsdier = self.ask_open("Wat vind je daar zo leuk aan?")
                if waarom_lievelingsdier:
                    vervolg_vraag = self.llm_request(
                        f'Je bent een robot die praat met een persoon met dementie in een verzorgingstehuis.'
                        f'De robot heeft zo juist gevraagd naar de lievelingsdier van de persoon.'
                        f'Het antwoord was een {lievelingsdier}'
                        f'Vervolgens vroeg de robot "wat vind je daar zo leuk aan?"'
                        f'Daar reageerde de persoon op met: "{waarom_lievelingsdier}"'
                        f'Genereer nu een gepaste reactie afsluitend met een enkele relevante vervolgvraag'
                    )
                    if vervolg_vraag:
                        waarom_lievelingsdier_2 = self.ask_open(vervolg_vraag)
                        if waarom_lievelingsdier_2:
                            self.say(self.llm_request(
                                f'Je bent een robot die praat met een persoon met dementie in een verzorgingstehuis.'
                                f'De robot heeft zo juist gevraagd naar de lievelingsdier van de persoon.'
                                f'Het antwoord was een {lievelingsdier}'
                                f'Vervolgens vroeg de robot "wat vind je daar zo leuk aan?"'
                                f'Daar reageerde de persoon op met: "{waarom_lievelingsdier}"'
                                f'Daarna stelde je deze vervolg vraag "{vervolg_vraag}"'
                                f'En dit was de reactie daar weer op "{waarom_lievelingsdier_2}"'
                                f'Geneer nu een gepaste reactie van de robot. Deze reactie mag geen vraag bevatten.'
                            ))
                    else:
                        # In case nothing was generated
                        self.say("Wat interessant zeg.")
                else:
                    # Om case we are not sure if anything was said or not.
                    self.say("Oke.")
            else:
                # in case nothing was recognized
                self.say("Oh dat begreep ik even niet. laten we verder gaan.")
        else:
            # In case they did not like animals.
            self.say("Helemaal prima, dan kletsen we over wat anders")

        self.say("Wat fijn dat we even konden praten. Ik ga nu weer even rusten.")


if __name__ == '__main__':
    # Select your device
    desktop = Desktop()
    # nao = Nao(ip="10.0.0.xxx")
    # pepper = Pepper(ip="10.0.0.xxx")

    demo = ConversationDemo(
        device=desktop,
        google_keyfile_path=abspath(join("..", "conf", "dialogflow", "google_keyfile.json")),
        openai_key_path=abspath(join("..", "conf", "openai", ".openai_env")))

    demo.run()
