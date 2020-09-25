import kfserving
from typing import List, Dict
import logging
import io
import argparse
import sys,json
import requests
import os
import logging

# User can add other libaries as per the requirement, the package must be present in the serving image. 

DEFAULT_MODEL_NAME = "model"

parser = argparse.ArgumentParser(parents=[kfserving.kfserver.parser])
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')
parser.add_argument('--predictor_host', help='The URL for the model predict function', required=True)

args, _ = parser.parse_known_args()

def sample_print():
    print("Using additional function")

class ImageTransformer(kfserving.KFModel):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host

    def preprocess(self, inputs: Dict) -> Dict:
        try:
            json_data = inputs
        except ValueError:
            return json.dumps({ "error": "Recieved invalid json" })
        inp_data = json_data["signatures"]["inputs"] # the input key value depends on how user has formatted the input.
        '''
        Do the required preprocessing on the inp_data
        '''
        sample_print()  # to show that user can have additional functions. 
        res = {"instances":inp_data,"token":inputs['token']} # the key values depends on the model input.
        return res

    def postprocess(self, inputs: List) -> List:
        out_data = inputs['predictions'] # the keyvalue can vary according to the model output.
        '''
        Add your postprocessing code here
        '''
        sample_print()  # to show that user can have additional functions. 
        return out_data

if __name__ == "__main__":
    transformer = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    kfserver = kfserving.KFServer()
    kfserver.start(models=[transformer])