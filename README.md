# llm-convrec
## Introduction: A Semi-Structured Conversational Recommendation System
LLM-ConvRec is a prompting-based, semi-structured conversational system that leverages the generative power of GPT to provide flexible and natural interaction. Unlike fully-structured conversational systems such as Siri, where utterances are often predefined and inflexible, LLM-ConvRec is designed for versatility and the production of more natural responses. Moreover, it incorporates past memory into the conversation, a feature often lacking in fully-structured systems.

While unstructured conversational systems like ChatGPT can produce fluid, engaging responses, their approach to utterance handling is often a "black box", which can lead to the generation of inappropriate or incorrect responses, or cause the conversation to go off the rails. This is where LLM-ConvRec distinguishes itself: although it provides the flexibility and naturalness of an unstructured system, its semi-structured nature ensures that utterance handling is not opaque, and that inappropriate responses can be avoided through structural constraints.

The system retains important information about the conversation, ensuring that context and past interactions are reflected in the responses. This makes LLM-ConvRec not just a conversational system, but a conversational partner capable of delivering precise, personalized recommendations across diverse domains.

## Table of Content


## Installation and Running the System

Before you can use the system, you must first ensure that you have Python (version 3.7 or higher) installed on your machine. You will also need pip for installing the necessary Python packages.

## 1. Clone the GitHub Repository

Clone the repository from GitHub to your local machine by running the following command in your terminal:

**git clone https://github.com/<your_username>/LLM-ConvRec.git**

Please replace `<your_username>` with your actual GitHub username.

## 2. Navigate to the Project Directory

Once you've cloned the repository, use the command line to navigate into the project's directory:

**cd LLM-ConvRec**

## 3. Install the Required Packages

The project has a number of dependencies that need to be installed. These are listed in the `requirements.txt` file. To install these dependencies, run the following command in your terminal:

**pip install -r requirements.txt**

## 4. Run the System





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

- **Request Information:** The system may ask the user for additional details to refine its understanding or recommendations. (Example: "Can you provide your location?")
- **Give Recommendation:** The system might suggest an item that matches the user's stated preferences.
- **Answer a Question:** If the user asked a question in their last utterance, the system would provide an appropriate answer.

### 4.Action Generation
Once the action is chosen, the system generates a structured response that aligns with the decided action. The system ensures this response is in line with the ongoing conversation context and adheres to the system's semi-structured conversational style.

This process is repeated at each turn of the conversation, enabling LLM-ConvRec to provide a dynamic, interactive, and engaging conversational recommendation experience.


## Domain Initialization and Customization

## 1. Constraints Configuration

To provide personalized recommendations, the LLM-ConvRec system takes into account user constraints that can be both explicit (provided directly by the user) or implicit (derived from the user's input). For efficient constraint management, it is crucial to set up a `constraints_config.csv` that defines the various constraints and their properties.

The `constraints_config.csv` file should include the following columns:

- **key**: The constraint's key name.
- **description**: Description of the constraint.
- **is_cumulative**: A Boolean value (TRUE or FALSE) indicating if the constraint's value is cumulative. If `is_cumulative` is TRUE, the system appends the newly extracted values to the existing values of the constraint rather than overwriting them. If FALSE, any newly identified value replaces the previous value.
- **default_value**: The default value of the constraint when it is not specified by the user.

Below is an example of how the `constraints_config.csv` file should look:

| key | description | is_cumulative | default_value |
|-----|-------------|---------------|---------------|
| location | The desired location of the restaurants. | FALSE | None |
| cuisine type | The desired specific style of cooking or cuisine offered by the restaurants (e.g., "Italian", "Mexican", "Chinese"). This can be implicitly provided through dish type (e.g "italian" if dish type is "pizza"). | FALSE | None |
| dish type | The desired menu item or dish in the restaurant that user shows interests. | TRUE | None |
| type of meal | The desired category of food consumption associated with specific times of day (e.g., "breakfast", "lunch", "dinner"). | TRUE | None |

The configuration of these constraints will allow the system to capture user preferences more accurately, leading to more personalized and relevant recommendations.

## 2.Few shots for prompts
Few-shot examples are crucial in training the LLM-ConvRec system. They are a set of input-output pairs that demonstrate the type of behavior we want the system to exhibit. In the context of our system, few-shot examples help train the classifiers and provide the necessary prompts for information extraction.

Few-shot learning is a powerful tool in AI because it enables models to understand and perform tasks after seeing just a few examples. This is crucial for conversational systems where a diverse array of utterances are possible. By providing the LLM-ConvRec system with few-shot examples, the model can learn to generate appropriate responses to a wide range of user inputs.

When providing few-shot examples, make sure that they are representative of the tasks you want the model to perform. For instance, if you want the system to recognize when a user is expressing a preference, include examples where users express preferences in different ways.

Few-shot examples should be provided in CSV format. Each row in the file should correspond to a unique example, with separate columns for the input and the desired output.

Remember, the quality of the few-shot examples can significantly impact the performance of the system. Carefully curating these examples will lead to a more responsive and accurate conversational system.

### 2.1 Few-shots for Intent Classification Prompts

For effective intent classification, few-shot examples must be provided for each intent. This should be done in the form of CSV files with two columns: 'User Input' and 'Response'. 'User Input' should contain examples of user utterances, while 'Response' indicates whether the input corresponds to the respective intent (True) or not (False).

#### 2.1.1 `accept_classification_fewshots.csv`: 
This file should contain examples of user utterances that express acceptance of a recommendation.

| User Input | Response |
|------------|----------|
| That sounds good, let's go there | True |
| I don't like that type of food | False |

#### 2.1.2 `reject_classification_fewshots.csv`: 
This file should contain examples where the user rejects a recommendation.

| User Input | Response |
|------------|----------|
| No, I don't want to go to that place | True |
| Sure, that sounds nice | False |

#### 2.1.3 `inquire_classification_fewshots.csv`: 
This file should contain examples where the user is inquiring or asking a question.

| User Input | Response |
|------------|----------|
| What's on their menu? | True |
| I think we should try something else | False |

### 2.2 Few-shots for Constraint Updater prompt

The 'constraints_updater_fewshots.csv' file plays an instrumental role in enhancing the personalization of recommendations by keeping track of explicit and implicit user preferences. It's fundamental in identifying and storing the constraints from the user's conversation, categorizing them as hard or soft constraints. Hard constraints are strict requirements, such as a specific location or cuisine type, which must be adhered to. On the other hand, soft constraints denote user preferences that are desirable but not necessarily mandatory, like a preference for a place offering free parking or a patio.

#### 2.2.1 `constraints_updater_fewshots.csv`:
This file should be structured with five columns: 'user_input', 'old_hard_constraints', 'old_soft_constraints', 'new_hard_constraints', and 'new_soft_constraints'. 

- 'user_input' represents a user's utterance.
- 'old_hard_constraints' and 'old_soft_constraints' denote the previously identified hard and soft constraints, respectively.
- 'new_hard_constraints' and 'new_soft_constraints' represent the updated set of hard and soft constraints after processing the user's input.

Here is an example of the structure of this CSV file:

| user_input | old_hard_constraints | old_soft_constraints | new_hard_constraints | new_soft_constraints |
|------------|----------------------|----------------------|----------------------|----------------------|
| pizza and pasta | "location=[""Toronto""]" |  | "location=[""toronto""], cuisine type=[""italian""], dish type=[""pizza"", ""pasta""]" |  |
| does it have a patio? | "location=[""jasper avenue, edmonton""], cuisine type=[""japanese""]" | "price range=[""moderate""], others=[""free parking""]" | "location=[""jasper avenue, edmonton""], cuisine type=[""japanese""]" | "price range=[""moderate""], others=[""free parking"", ""patio""]" |
| What kind of menu does I Love Sushi offer? | "location=[""jasper avenue""], cuisine type=[""japanese""]" |  | "location=[""jasper avenue""], cuisine type=[""japanese""]" 

The ability to track and update these evolving constraints allows the system to fine-tune its recommendations, significantly enhancing the overall conversation experience.



### 2.4 Few-shots for 'Answer' Recommender Action Prompts

This section provides details about the few-shot prompt CSV files required for the 'Answer' recommender action.

#### 2.4.1 `answer_extract_category_fewshots.csv`
This file helps in mapping user queries to metadata categories. It needs two columns: 'input' (user's question) and 'output' (metadata category that corresponds to the user's question).

| input | output |
|-------|--------|
| What's their addresses? | address |
| Can you recommend any dishes or specialties? | none |
| Can I make a reservation? | HasReservations |
| What are the meals it's known for? | PopularMeals |

#### 2.4.2 `answer_ir_fewshots.csv`
This file trains the model to extract answers from reviews based on the user's question. It requires 'question' (user's question), 'information' (reviews retrieved by the system), and 'answer' (the answer to the question derived from the provided information).

| question | information | answer |
|----------|-------------|--------|
| Do they have a slide in the restaurant? | I really like this place. They have great food. | I do not know. |

#### 2.4.3 `answer_separate_questions_fewshots.csv`
This file aids the system in breaking down complex user queries into simpler, individual questions. It requires 'question' (user's question) and 'individual_questions' (decomposed questions).

| question | individual_questions |
|----------|----------------------|
| Do they have wine? | Do they have wine? |
| What are dishes, cocktails and types of wine do you recommend? | What dishes do you recommend?\nWhat cocktails do you recommend?\nWhat types of wine do you recommend? |

#### 2.4.4 `answer_verify_metadata_resp_fewshots.csv`
This file trains the model to verify if a system-generated response accurately answers a user's query. It needs 'question' (user's question), 'answer' (the system's generated answer), and 'response' (indicates whether the generated answer meets the user's query).

| question | answer | response |
|----------|--------|----------|
| Do they have a high chair? | Subway is kid friendly. | No. |
| Do they serve vodka? | They have a full bar. | No. |
| Are there gluten free options? | Yes, there are gluten free options. | Yes. |

Please note that these examples are illustrative. The content of your files will be determined by the specific nature of your domain and the complexity of the user's queries. 

## 3.Hard-Coded Responses

The `hard_coded_responses.csv` file includes specific, structured responses that the system should use under certain conditions. This file has three columns: 'Action', 'Response', and 'Constraints'.

- **Action**: This column refers to the action that the system should take.
- **Response**: This column provides the exact response that should be given when the action is chosen.
- **Constraints**: This column specifies one or more constraints. If any of these constraints are missing in the system's state, then the corresponding response will be prioritized. If no constraints are applicable, this field should be left empty.

Descriptions of each action:

- **PostAcceptanceAction**: The response given after the user accepts a recommendation.
- **PostRejectionAction**: The response given after the user rejects a recommendation.
- **RequestInformation**: The system is requesting additional information from the user. The required information type depends on the 'Constraints' column.
- **DefaultResponse**: Used when the system failed to classify an intent and cannot decide on any specific action until the user provides more information.
- **NoRecommendation**: This response is given when the system cannot find a restaurant that meets the user's constraints.
- **NoAnswer**: Used when the system cannot answer the user's question with the information it has retrieved.

Here are some examples:

| Action | Response | Constraints |
| --- | --- | --- |
| PostAcceptanceAction | "Great! If you need any more assistance, feel free to ask." |  |
| PostRejectionAction | I'm sorry that you did not like the recommendation. Is there anything else I can assist you with? |  |
| RequestInformation | Could you provide the location? | location |
| RequestInformation | Could you provide the cuisine type or dish type? | "cuisine type, dish type" |
| RequestInformation | Do you have any other preferences? |  |
| DefaultResponse | Could you provide more information? |  |
| NoRecommendation | "Sorry, there is no restaurant that matches your constraints." |  |
| NoAnswer | Please only ask questions about previously recommended restaurant. |  |

By providing these hard-coded responses, you can control the behavior of the system and ensure that the conversation flow remains on track.



4.filter configs

| type_of_filter | key_in_state | metadata_field | default_max_distance_in_km | distance_type |
| -------------- | ------------ | -------------- | -------------------------- | ------------- |
| word in | "cuisine type, dish type" | categories | | |
| item | recommended_items | name | | |


5.domain specific config
- domain name
- file path to files, shouldnt change normally

## 6. Data

The LLM-ConvRec system requires two main types of data: metadata and reviews.

### 6.1 Metadata

The metadata must include unique item identifiers (item ID) as a key. Each item can have various other keys representing different types of metadata, such as location, type of cuisine, cost, etc. It is not necessary for all items to have a value for every metadata field. The metadata fields could be populated based on the information available for each item.

An example of a metadata structure is as follows:


{
    "item_id": "-3GD07waps96fB_okEwFqw",
    "name": "Brits Fish & Chips",
    "address": "6940 77 Street NW",
    "city": "Edmonton",
    "categories": ["Fish & Chips", "Restaurants"]
}
review has to be csv, has itemid and reviews

ordershould match



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
