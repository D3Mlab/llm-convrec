
class GPTWrapperObserver:
    """
    Observer for GPTWRapper that gets notified when GPTWrapper re-requests to GPT due to Rate limit error.
    """
    def notify_gpt_retry(self, retry_info: dict):
        """
        Notify this object that gpt re-requested due to Rate Limit Error.

        :param retry_info: dictionary that contains information about retry
        """
        raise NotImplementedError()
