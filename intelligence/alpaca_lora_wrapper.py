import os
import logging

from intelligence.llm_wrapper import LLMWrapper
from dotenv import load_dotenv

from gradio_client import Client

load_dotenv()
logger = logging.getLogger('alpaca_lora_wrapper')


class AlpacaLoraWrapper(LLMWrapper):
    """
    Class for wrapping around the Alpaca Lora LLM.
    """

    def __init__(self):
        self._client = Client(os.environ["GRADIO_URL"])
        self._client.view_api()

    def make_request(self, message: str) -> str:
        """
        Makes a request to the alpaca lora service and return the response.

        :param message: an input to the GPT.
        :return: response from the GPT
        """
        logger.debug(f"alpaca_lora_input=\"{message}\"")

        try:
            # The following parameter order reflects the API parameter sequence and type from the view_api() call above.
            response = self._client.predict(message, "", 0.1, 0.75, 40, 4, 256)
        except Exception as e:
            return e

        logger.debug(f"alpaca_lora_output=\"{response}\"")

        return response
