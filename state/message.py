class Message:
    """
    Class representing one message from the user or recommender.

    Preconditions:
        - role in {"recommender", "user"}

    :param role: role of the messenger
    :param content: text content of this message
    """

    _role: str
    _content: str

    def __init__(self, role: str, content: str):
        self._role = role
        self._content = content

    def get_role(self):
        """
        Returns the role ("recommender" or "user") of the messanger for this message.

        :return: role of the messanger for this message
        """
        return self._role

    def get_content(self):
        """
        Returns the text content of this message.

        :return: text content of this message.
        """
        return self._content

    def __str__(self):
        """
        Returns the string representation of this message in the following format:
            "{role: <role>, content: <content>}"

        :return: string representation of this message
        """
        # Implementation Note: couldn't use fstring because of "{"
        return "{role: " + self._role + ", " + "content: " + self._content + "}"
