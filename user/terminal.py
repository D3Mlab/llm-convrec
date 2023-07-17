from user.user_interface import UserInterface


class Terminal(UserInterface):
    """
    Class that implements UserInterface with terminal.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_user_input(self, message: str) -> str:
        """
        Get text input from the user in the terminal.

        :param message: prompt message displayed to the terminal when getting input
        :return: given user's input
        """
        return input(message)

    def display_to_user(self, message: str = f'Recommender: Hello there! I am a recommender. Please provide me with some preferences for what you are looking for. Thanks!'
):
        """
        Display the given text to the user in the terminal.

        :param message: text displayed in the terminal to the user
        """
        print(message)
