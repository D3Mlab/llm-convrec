from typing import Optional, Callable

from warning_observer import WarningObserver
from intelligence.llm_wrapper import LLMWrapper
import logging
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    Future,
    RetryCallState,
)

logger = logging.getLogger('gpt_wrapper')


class GPTWrapper(LLMWrapper):
    """
    Class for wrapping around the GPT LLM.

    :param openai_api_key: api key for open ai used to access GPT
    :param model_name: name of the llm model
    :param temperature: temperature used for the model
    :param observers: list of observers that will get notified when gpt retry have occurred
    :param max_attempt: maximum number of attempts that wrapper can make.
                        If max_attempt is None, there is no maximum number of attempts
    :param min_sleep: minimum number of seconds to sleep when retrying
    :param max_sleep: maximum number of seconds to sleep when retrying
    :param timeout: number of seconds for each retry until it raises error
    """

    _model_name: str
    _temperature: Optional[float]
    _observers: list[WarningObserver]
    _max_attempt: int
    _min_sleep: int
    _max_sleep: int
    _timeout: float | None

    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo",
                 temperature: Optional[float] = None,
                 observers=None, max_attempt=5, min_sleep=3, max_sleep=60, timeout=15):
        super().__init__()
        if observers is None:
            observers = []
        self._model_name = model_name
        self._temperature = temperature
        self._observers = observers
        self._max_attempt = max_attempt
        self._min_sleep = min_sleep
        self._max_sleep = max_sleep
        self._timeout = timeout
        openai.api_key = openai_api_key

    def make_request(self, message: str) -> str:
        """
        Makes a request to the GPT and return the response.

        :param message: an input to the GPT.
        :return: response from the GPT
        """
        logger.debug(f"gpt_input=\"{message}\"")
        response = self.completion_with_backoff(
                model=self._model_name,
                temperature=self._temperature,
                messages=[{"role": "user", "content": message}],
                max_tokens=1000
            )

        if response is None:
            return ""

        # prompt_tokens = response["usage"]["prompt_tokens"]
        # completion_tokens = response["usage"]["completion_tokens"]
        tokens_used = response["usage"]["total_tokens"]
        cost_of_response = tokens_used * 0.000002

        self.total_tokens_used += tokens_used
        self.total_cost += cost_of_response
        logger.debug(f"gpt_output=\"{response['choices'][0]['message']['content']}\"")
        return response['choices'][0]['message']['content']

    def _notify_observers(self, attempt_number: int, outcome: Future | None) -> None:
        """
        Notify the observer that re-request have occurred.

        :param attempt_number: number of attempts so far
        :param outcome: result of the request
        """
        for observer in self._observers:
            observer.notify_gpt_retry({
                'attempt number': attempt_number,
                'outcome': outcome
            })

    def _before_completion_sleep(self, retry_state: RetryCallState) -> None:
        """
        Perform steps that should be done before re-requesting GPT.
        Log the retry details and notify the observers.

        :param retry_state: state of the retry
        """
        logger.warning(f'Retrying {retry_state.fn}: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}')
        self._notify_observers(retry_state.attempt_number, retry_state.outcome)

    @staticmethod
    def _custom_retry(func: Callable) -> Callable:
        """
        custom decorator for retry

        :param func: function called
        :return: wrapper for the custom decorator
        """
        def wrapped(self, *args, **kwargs):
            if self._max_attempt is None:
                t_decorator = retry(
                    wait=wait_random_exponential(min=self._min_sleep, max=self._max_sleep),
                    retry=retry_if_exception_type((openai.error.RateLimitError, openai.error.Timeout, openai.APIError,
                        openai.error.APIConnectionError, openai.error.ServiceUnavailableError)),
                    before_sleep=self._before_completion_sleep,
                    retry_error_callback=lambda retry_state: None
                )
            else:
                t_decorator = retry(
                    wait=wait_random_exponential(min=self._min_sleep, max=self._max_sleep),
                    retry=retry_if_exception_type((
                        openai.error.RateLimitError, openai.error.Timeout, openai.APIError,
                        openai.error.APIConnectionError, openai.error.ServiceUnavailableError
                    )),
                    stop=stop_after_attempt(self._max_attempt),
                    before_sleep=self._before_completion_sleep,
                    retry_error_callback=lambda retry_state: None
                )
            decorated = t_decorator(func)
            return decorated(self, *args, **kwargs)
        return wrapped

    @_custom_retry
    def completion_with_backoff(self, *args, **kwargs) -> dict:
        """
        Wrapper for openai.ChatCompletion.create that retries when RateLimitError have occurred or if it takes
        too long to get the response.
        """
        return openai.ChatCompletion.create(*args, **{**kwargs, **{'request_timeout': self._timeout}})






