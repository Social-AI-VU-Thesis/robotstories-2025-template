import queue
from json import load
from os import environ
from os.path import abspath, join
from pathlib import Path
from threading import Thread
from time import strftime

import numpy as np
from dotenv import load_dotenv
from sic_framework.devices import Pepper
from sic_framework.devices.common_naoqi.naoqi_text_to_speech import NaoqiTextToSpeechRequest
from sic_framework.services.dialogflow import DialogflowConf, Dialogflow, GetIntentRequest
from sic_framework.services.openai_gpt import GPTRequest, GPTConf, GPT


class MemoLLM:
    def __init__(self, pepper_ip, google_keyfile, openai_key_path):
        self.pepper = Pepper(pepper_ip)

        # set up the config
        conf = DialogflowConf(keyfile_json=google_keyfile, sample_rate_hertz=16000, language='nl-NL')

        # initiate Dialogflow object
        self.dialogflow = Dialogflow(ip="localhost", conf=conf)

        # connect the output of NaoqiMicrophone as the input of DialogflowComponent
        self.dialogflow.connect(self.pepper.mic)

        # register a callback function to act upon arrival of recognition_result
        self.dialogflow.register_callback(self.on_dialog)

        self.request_id = np.random.randint(10000)

        if openai_key_path:
            load_dotenv(openai_key_path)

        # Setup GPT client
        conf = GPTConf(openai_key=environ["OPENAI_API_KEY"])
        self.gpt = GPT(conf=conf)

        self.log_queue = None
        self.log_thread = None

    def start_logging(self, log_id):
        folder = Path("logs")
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / f"{log_id}.log"
        self.log_queue = queue.Queue()
        self.log_thread = Thread(target=self.log_writer, args=(log_path,), daemon=True)
        self.log_thread.start()

        timestamp = strftime("%Y-%m-%d %H:%M:%S")
        self.log_queue.put(f'[{timestamp}] ### START NEW LOG ###')

    def stop_logging(self):
        if self.log_queue:
            self.log_queue.put(None)
        if self.log_thread:
            self.log_thread.join()

    def log_writer(self, log_path):
        with open(log_path, 'a', encoding='utf-8') as f:
            while True:
                item = self.log_queue.get()
                if item is None:
                    break  # Exit signal
                f.write(item + '\n')
                f.flush()

    def log_utterance(self, speaker, text):
        if self.log_queue:
            timestamp = strftime("%Y-%m-%d %H:%M:%S")
            self.log_queue.put(f"[{timestamp}] {speaker}: {text}")

    def on_dialog(self, message):
        if message.response:
            print("Transcript:", message.response.recognition_result.transcript)

    def say(self, text, speed=100):
        self.pepper.tts.request(NaoqiTextToSpeechRequest(fr' \rspd={str(speed)}\ ' + text,
                                                         language='Dutch',
                                                         animated=True))

    def listen(self, question, speed=100, max_attempts=2):
        attempts = 0

        while attempts < max_attempts:
            # ask question
            self.say(question, speed)

            # listen for answer
            reply = self.dialogflow.request(GetIntentRequest(self.request_id))

            # Return text
            if reply.response.query_result.query_text:
                return reply.response.query_result.query_text
            attempts += 1
        return None

    def llm_action(self, dialog_history=''):

        base_prompt = (f"We willen graag dat je een rol speelt. Je bent Memo. Memo is een zorgrobot die graag praatjes maakt met mensen."
                       f"Je bent heel geïnteresseerd in je gesprekspartner. Je hebt al 5 jaar gewerkt in huizen waar ouderen wonen."
                       f"Je mag niet naar het bos en ook niet naar het strand, want dan komt er zand in je wieltjes."
                       f"Je vertelt graag gedichten."
                       f""
                       f"Het gesprek vindt plaats in de woonkamer van een huis waar oudere mensen wonen."
                       f"Er zijn ongeveer een stuk of 10 mensen. Zij drinken koffie."
                       f"Jij drinkt geen koffie, want jij haalt jouw energie uit het stopcontact."
                       f""
                       f"Je bent nieuwsgierig naar de verhalen die jouw gesprekspartners aan jou willen vertellen."
                       f"Dat is het doel van het gesprek."
                       f""
                       f"Je hebt zelf ook een doel. Af en toe vraag je of jij een gedichtje mag voordragen."
                       f"Als het antwoord ‘ja’ is, dan maak je een gedichtje van 2 strofen,"
                       f"in een AABB CCBB rijmschema, op basis van de input die je tot dan toe hebt ontvangen."
                       f""
                       f"Als je antwoord geeft, dan doe je dat in korte zinnen. Maximaal 2 zinnen per keer."
                       f"Je gebruikt simpele taal. Niet moeilijker dan A1 niveau."
                       f"Je mag herhalen wat je hebt gehoord."
                       f"Na een vraag kan je een vervolgvraag stellen, of over iets soortgelijks een vraag stellen."
                       f"Alles wat in de conversatie gezegd is, onthoud je voor een latere vraag of een gedichtje."
                       f"Stuur alleen jouw tekst terug zonder de 'Memo: prefix'"
                       f""
                       f"Als je denkt dat de persoon wil stoppen met het gesprek kun je dit vragen."
                       f"Is dat het geval of er wordt duidelijk proactief al gecommuniceerd dat de persoon wil of "
                       f"moet stoppen met het gesprek, dan stuur je enkel het woord 'stop' terug."
                       f"Vergeet ook je gedichtjes af en toe niet.")

        if dialog_history:
            complete_prompt = base_prompt + f"Dit is het gesprek tot nu toe: {dialog_history}"
        else:
            complete_prompt = base_prompt

        gpt_response = self.gpt.request(GPTRequest(complete_prompt))
        return gpt_response.response

    def run(self, speed=100):
        try:
            self.start_logging()
            dialog_history = ''
            llm_response = ''
            while llm_response.lower() != 'stop':
                llm_response = self.llm_action(dialog_history)
                print(f'Memo: {llm_response}')
                if llm_response.lower() != 'stop':
                    dialog_history += f'Memo: {llm_response}\n'
                    self.log_utterance('Memo', llm_response)
                    user_response = self.listen(llm_response, speed=speed)
                    dialog_history += f'Persoon: {user_response}'
                    self.log_utterance('Persoon', user_response)
            self.say('Oke dat was het weer, bedankt voor dit fijne gesprek. Tot ziens!', speed=speed)
            self.stop_logging()
        except KeyboardInterrupt:
            self.say('Oke dat was het weer, bedankt voor dit fijne gesprek. Tot ziens!', speed=speed)
            self.stop_logging()


if __name__ == '__main__':
    memo = MemoLLM(pepper_ip='192.168.1.110',
                   google_keyfile=load(open(abspath(join("..", "conf", "dialogflow", "google_keyfile.json")))),
                   openai_key_path=abspath(join("../conf", "openai", ".openai_env")))
    memo.run(speed=75)

