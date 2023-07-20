# llm-convrec
## Introduction: A Semi-Structured Conversational Recommendation System
LLM-ConvRec is a prompting-based, semi-structured conversational system that leverages the generative power of GPT to provide flexible and natural interaction. Unlike fully-structured conversational systems such as Siri, where utterances are often predefined and inflexible, LLM-ConvRec is designed for versatility and the production of more natural responses. Moreover, it incorporates past memory into the conversation, a feature often lacking in fully-structured systems.

While unstructured conversational systems like ChatGPT can produce fluid, engaging responses, their approach to utterance handling is often a "black box", which can lead to the generation of inappropriate or incorrect responses, or cause the conversation to go off the rails. This is where LLM-ConvRec distinguishes itself: although it provides the flexibility and naturalness of an unstructured system, its semi-structured nature ensures that utterance handling is not opaque, and that inappropriate responses can be avoided through structural constraints.

The system retains important information about the conversation, ensuring that context and past interactions are reflected in the responses. This makes LLM-ConvRec not just a conversational system, but a conversational partner capable of delivering precise, personalized recommendations across diverse domains.

## Table of Content


## Installation and Running the System

pip install -r requirements.txt



## To run Colab service (note public URL changes each time you restart the server)

1. Open Colab (https://colab.research.google.com/drive/1FfKTLmVV0rQSQWkvoGpiyb1RuK7E1l6k?usp=sharing#scrollTo=9_rc9X75fFT5) and run it

2. Copy the public URL (should be something like this https://8b4a0f826a0deb0ec1.gradio.live)

3. Change GRADIO_URL under the Gradio.live API call cell to the public URL you copied.

You must update this based on the URL listed in the output cell above
GRADIO_URL = "https://8b4a0f826a0deb0ec1.gradio.live" <- change this URL

3. Add public URL to .env where the key is GRADIO_URL
   example: GRADIO_URL = "https://8b4a0f826a0deb0ec1.gradio.live"


## Overall Conversation Flow

The LLM-ConvRec system follows a precise process during each conversation turn to ensure the highest quality of responses. Let's break down this process, from the initial user input to the final recommender response:

Examples to demonstrate these categories are from the restaurant recommendation domain.

### 1.Intent Classification
The conversation begins with user input. This input is passed to an Intent Classifier, which determines the user's intent. The intent could be any or multiple of the following:

- **Provide Preference:** The user expresses a preference or interest. (Example: "I'd like to eat some sushi.")
- **Inquire:** The user asks a question about a specific item or detail. (Example: "What's on their menu? Does the restaurant have a patio?")
- **Accept/Reject Recommendation:** The user responds to a previous recommendation made by the system.(Example: "Sure, that first one sounds good!")

### 2.State Update
After identifying the user's intent, the system updates its internal state. This state stores critical information gathered during the conversation, including:

- **User's Preferences and Constraints:** These include their location, budget, dietary restrictions, etc.
- **Current Item of Interest:** The specific item (e.g., restaurant) the user is currently referring to or interested in.
- **Accepted and Rejected Items:** The system keeps track of the items that the user has accepted or rejected.

### 3.Action Classification
With the updated state, the system then decides the action to take next. This could be any of the following:

- **Request More Information:** The system may ask the user for additional details to refine its understanding or recommendations. (Example: "Can you provide your location?")
- **Make a Recommendation:** The system might suggest an item that matches the user's stated preferences.
- **Answer a Question:** If the user asked a question in their last utterance, the system would provide an appropriate answer.

### Action Generation
Once the action is chosen, the system generates a structured response that aligns with the decided action. The system ensures this response is in line with the ongoing conversation context and adheres to the system's semi-structured conversational style.

This process is repeated at each turn of the conversation, enabling LLM-ConvRec to provide a dynamic, interactive, and engaging conversational recommendation experience.


## Domain Initialization and Customization

1.Few shots

2.Constraints

3.Hard coded responses

4.filter configs

5.domain specific config
- domain name
- file path to files, shouldnt change normally

6.data

7.User defined classes
  

## Testing






## How to run unit tests

inside terminal:
pytest .\test\filename.py

## Test Format

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

# How to implement another domain
Make a folder in domain_configs directory.In that folder, you need following files.
- **domain_specific_config.yaml**
  - DOMAIN: domain name
  - followings should remain the same if you name files as in this documentation
  - CONSTRAINTS_CATEGORIES: file name that has constraints categories
  - CONSTRAINTS_UPDATER_FEWSHOTS: file name that has fewshots for constraints updater prompt
  - ACCEPTED_ITEMS_EXTRACTOR_FEWSHOTS_FILE: file name that has fewshots for accepted items extractor prompt
  - REJECTED_ITEMS_EXTRACTOR_FEWSHOTS_FILE: file name that has fewshots for rejected items extractor prompt
  - CURRENT_ITEMS_EXTRACTOR_FEWSHOTS_FILE: file name that has fewshots for current items etractor prompt
  - ANSWER_EXTRACT_CATEGORY_FEWSHOTS_FILE: file name that has fewshots for answer recaction's extract category prompt
  - ANSWER_IR_FEWSHOTS_FILE: file name that has fewshots for answer recaction's ir prompt
  - ANSWER_SEPARATE_QUESTIONS_FEWSHOTS_FILE: file name that has fewshots for answer recaction's separate questions prompt
  - ANSWER_VERIFY_METADATA_RESP_FEWSHOTS_FILE: file name that has fewshots for answer recaction's verify metadata response prompt
- **constraints_config.csv**
- **constraints_updater_fewshots.csv**
- **accepted_items_extractor_fewshots.csv**
- **rejected_items_extractor_fewshots.csv**
- **current_items_extractor_fewshots.csv**
  - requires columns named user_input and response
  - user_input: user input
  - response: items extracted from the user input
- **answer_extract_category_fewshots.csv**
  - requires columns named input and output
  - input: question from user
  - output: metadata category that has the answer for the user input
- **answer_ir_fewshots.csv**
  - requires columns named question, information, and answer
  - question: question from user
  - information: reviews retrieved by information retrieval
  - answer: answer to the question based on the information given
- **answer_separate_questions_fewshots.csv**
  - requires columns named question and individual_questions
  - question: question from user
  - individual_questions: individual questions in the user's question separated by a new line character (i.e. "\n")
- **answer_verify_metadata_resp_fewshots.csv**
  - requires columns named question, answer, and response
  - question: question from user
  - answer: answer created by the system
  - response: "Yes." if the answer actually answer the question. "No." if the answer doesn't answer the question
