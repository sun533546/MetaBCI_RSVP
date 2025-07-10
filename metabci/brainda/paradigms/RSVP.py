from .base import BaseTimeEncodingParadigm

class RSVP(BaseTimeEncodingParadigm):
    def is_valid(self, dataset):
        
        ret = True
        if dataset.paradigm != "RSVP":
            ret = False
        return ret