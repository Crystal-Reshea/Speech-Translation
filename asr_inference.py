import torch
import soundfile as sf
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from nltk.tokenize import sent_tokenize


class ASRInference:
    def __init__(self, model_name='patrickvonplaten/wav2vec2-base-100h-with-lm'):
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
    
    def inference(self, audio):
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        self.text = self.processor.decode(predicted_ids[0]).lower()
        return self.text
        
    def get_languages(self, source, target):
        source = source.strip().lower()
        target = target.strip().lower()
        if source == 'english' and target == 'spanish':
            return self.translate('en-es')
        else: 
            return "One or both of the selected languages are not available. Please choose different language(s)"
        
    def translate(self,mod): 
        if mod == 'en-es':
            translator = pipeline('translation', model='Helsinki-NLP/opus-mt-en-es')
        translated_text = ''
        sentences = sent_tokenize(self.text)
        for sent in sentences:
            tkey = translator(sent)
            translated_text += " " + (tkey[0])['translation_text']
        return translated_text

    
    
# if __name__=='__main__':
#     asr = ASRInference()
# #     speech, rate = librosa.load("sample1.flac",sr=16000)
#     speech, rate = sf.read("sample1.flac")
# #     audio = np.random.rand(16000)
#     text = asr.inference(speech)
#     translations = asr.get_languages('english', 'spanish')
#     print("Translation: " + translations)
                