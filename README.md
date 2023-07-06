# llm-convrec
LLM-based Conversational Recommendation Architecture

pip install -r requirements.txt

Install streamlit chat gpt

1. git clone https://github.com/joeychrys/streamlit-chatGPT.git
2. pip install -r streamlit-chatGPT/requirements.txt

# To run Colab service (note public URL changes each time you restart the server)

1. Open Colab (https://colab.research.google.com/drive/1FfKTLmVV0rQSQWkvoGpiyb1RuK7E1l6k?usp=sharing#scrollTo=9_rc9X75fFT5) and run it

2. Copy the public URL (should be something like this https://8b4a0f826a0deb0ec1.gradio.live)

3. Change GRADIO_URL under the Gradio.live API call cell to the public URL you copied.

You must update this based on the URL listed in the output cell above
GRADIO_URL = "https://8b4a0f826a0deb0ec1.gradio.live" <- change this URL

3. Add public URL to .env where the key is GRADIO_URL
   example: GRADIO_URL = "https://8b4a0f826a0deb0ec1.gradio.live"

# How to run unit tests

inside terminal:
pytest .\test\filename.py

# Test Format

- **constraints_extractor_test_v2.csv**: test file used to test constraints extractor

  - 1-3rd columns (utterance 1-3): past 3 utterances in the conversation history starting from left to right
  - 4-11th columns: expected constraints extracted where value in the header is the corresponding key (e.g. location). Each value is a list separated by commas (e.g. italian, indian).

- **constraints_classifier_test.csv**: test file used to test constraints classifier
  - 1-3rd columns (utterance 1-3): past 3 utterances in the conversation history starting from left to right
  - 4-11th columns (ones that starts with "input:"): inputted constraints extracted where value (with "input:" removed) in the header is the corresponding key (e.g. location). Each value is a list separated by commas (e.g. italian, indian).
  - 12-19th columns (ones that starts with "hard_constraints:") expected hard constraints extracted where value (with "hard_constraints:" removed) in the header is the corresponding key (e.g. location). Each value is a list separated by commas (e.g. italian, indian).
  - 20-28th columns (ones that starts with "soft_constraints:") expected soft constraints extracted where value (with "soft_constraints:" removed) in the header is the corresponding key (e.g. location). Each value is a list separated by commas (e.g. italian, indian).
- **accepted_restaurants_extractor_test.csv / rejected_restaurants_extractor_test.csv**: test file used to test accepted / rejected restaurants extractor

  - 1st column (utterance): utterance from the user provided as an input
  - 2nd column (all mentioned restaurants): list of all restaurant names mentioned in the conversation provided as an input. Value is separated by commas (e.g. Restaurant A, Restaurant B)
  - 3rd column (recently mentioned restaurants): list of restaurant names mentioned most recently in the conversation provided as an input. Value is separated by commas (e.g. Restaurant A, Restaurant B)
  - 4th column (accepted / rejected restaurants): list of expected accepted / rejected restaurant names extracted. Value is separated by commas (e.g. Restaurant A, Restaurant B)

- **dialogue_manager_state_management_test.csv**: test file used to test whole state management in the dialogue manager
  - 1st column (test number): number corresponding to the single test. If test number are same, that row represents data in the same conversation.
  - 2nd column (user utterance): utterance from the user provided as an input
  - 3rd column (recommender utterance): utterance from the recommender provided as an input
  - 4th column (user utterance): utterance from the user provided as an input
  - 5th column (user intents): user intents corresponding to the user utterance, provided as an input. Value is separated by commas.
  - 6th column (recommender actions): recommender actions corresponding to the recommender utterance, provided as an input. Value is separated by commas.
  - 7-28th columns: expected value for the fields in the state_manager. Value in the header represents the key in the state. For nested dictionary, keys are separated with ":" in the header (e.g. hard_constraints:wait times represents data in state_manager.get("hard_constraints").get("soft_constraints")).
- **Thresholduserintenttest.csv**: test files used to test multilabel user intent classifier

  - 1st column (Input): utterance from the user provided as an input
  - 2nd column (Output1): 1st expected classification of user intent
  - 3rd column (Output2): 2nd expected classification of user intent

- **Singleuserintenttest.csv**: test files used to test multilabel user intent classifier
  - 1st column (Input): utterance from the user provided as an input
  - 2nd column (Output1): 1st expected classification of user intent
