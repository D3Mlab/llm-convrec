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