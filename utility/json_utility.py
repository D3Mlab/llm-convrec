import json
from typing import Union
import regex
import logging
logger = logging.getLogger('convrec')

def extract_json(text: str) -> Union[dict, None]:
    """
    Find and extract the json from the given text and return it as a dict.
    Return None if json cannot be extracted from the text.

    :param text: text to extract the json
    :return: resulting dictionary corresponding to the json or None if json cannot be extract from the text
    """
    try:
        # convert from single quotes to double quotes
        text = regex.sub(r"(?<!\w)'(?![s:])|'(?!\w)", '"', text)
        return json.loads(regex.search(r'\{(?:[^{}]|(?R))*\}', text).group(0))
    except json.JSONDecodeError | IndexError:
        logger.warning(f"The following text couldn't converted to JSON format:\n{text}")
        return None
