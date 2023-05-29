import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
import transformers


from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    
    # set verbosity to suppress warning messages
    transformers.logging.set_verbosity_error()

    # load LLM fine-tuned on ELI5 dataset using huggingface pipeline
    generator = transformers.pipeline("conversational", model = "fine_tuned_science_llm/fine_tuned_lm", tokenizer="fine_tuned_science_llm/tokenizer")
    
    # parallel tokenization for faster output
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    for text in request.text:
        chat = generator(transformers.Conversation(text), pad_token_id = 50256)
        result = str(chat)
        # get only model output part of the response
        response = result[result.find("bot >> ")+6:].strip()
        output.append(response)

    return SimpleText(dict(text=output))