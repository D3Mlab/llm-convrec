class UserInterface:
    """Abstract class representing the basic text-based user interface."""

    def get_user_input(self, message: str) -> str:
        """
        Get text input from the user.

        :param message: prompt message displayed to the user when getting input
        :return: given user's input
        """
        raise NotImplementedError()

    def display_to_user(self, response: str) -> None:
        """
        Display the given text to the user

        :param response: text displayed to the user
        """
        raise NotImplementedError()

    def display_warning(self, warning_message: str) -> None:
        """
        Display the given warning text to the user

        :param warning_message: warning text displayed to the user
        """
        raise NotImplementedError()

