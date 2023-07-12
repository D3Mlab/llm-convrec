from user.user_interface import UserInterface
import gradio as gr

class Gradio(UserInterface):
    """
    Class that implements UserInterface with terminal.
    """

    def __init__(self) -> None:
        super().__init__()

    def display_warning(self, warning_message: str) -> None:
        """
        Display the given warning text to the user

        :param warning_message: warning text displayed to the user
        """
        gr.Warning(warning_message)