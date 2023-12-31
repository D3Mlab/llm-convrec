import logging

import requests

from intelligence.llm_wrapper import LLMWrapper

from gradio_client import Client

logger = logging.getLogger('alpaca_lora_wrapper')


class AlpacaLoraWrapper(LLMWrapper):
    """
    Class for wrapping around the Alpaca Lora LLM.

    :param gradio_url: url generated from gradio that hosts alpaca lora
    :param temperature: temperature used for the model
    """

    def __init__(self, gradio_url: str, temperature: float=0.1):
        super().__init__()
        try:
            self._client = Client(gradio_url)
        except requests.exceptions.HTTPError as e:
            raise Exception("The provided Gradio URL is invalid. Please input a correct url and retry.")
        self._client.view_api()
        self._temperature = temperature

    def make_request(self, message: str) -> str:
        """
        Makes a request to the alpaca lora service and return the response.

        :param message: an input to the GPT.
        :return: response from the GPT
        """
        logger.debug(f"alpaca_lora_input=\"{message}\"")

        # The following parameter order reflects the API parameter sequence and type from the view_api() call above.
        response = self._client.predict(message, "", self._temperature, 0.75, 40, 4, 1000)

        logger.debug(f"alpaca_lora_output=\"{response}\"")
        return response

