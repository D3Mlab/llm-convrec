import logging

from intelligence.llm_wrapper import LLMWrapper

from gradio_client import Client

logger = logging.getLogger('alpaca_lora_wrapper')


class AlpacaLoraWrapper(LLMWrapper):
    """
    Class for wrapping around the Alpaca Lora LLM.
    """

    def __init__(self, gradio_url: str, temperature: float=0.1):
        try:
            self._client = Client(gradio_url)
        except:
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

        try:
            # The following parameter order reflects the API parameter sequence and type from the view_api() call above.
            response = self._client.predict(message, "", self._temperature, 0.75, 40, 4, 1000)
        except Exception as e:
            return e

        logger.debug(f"alpaca_lora_output=\"{response}\"")

        return response
