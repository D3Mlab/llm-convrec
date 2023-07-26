# llm-convrec

## Table of Content

- [1. Introduction](#Introduction:)
- [2. Installation and Running the System](#2-installation-and-running-the-system)
- [3. Overall Conversation Flow](#3-overall-conversation-flow)
- [4. Domain Initialization and Customization](#4-domain-initialization-and-customization)


## Introduction: A Semi-Structured Conversational Recommendation System
LLM-ConvRec is a prompting-based, semi-structured conversational system that leverages the generative power of GPT to provide flexible and natural interaction. Unlike fully-structured conversational systems such as Siri, where utterances are often predefined and inflexible, LLM-ConvRec is designed for versatility and the production of more natural responses. Moreover, it incorporates past memory into the conversation, a feature often lacking in fully-structured systems.

While unstructured conversational systems like ChatGPT can produce fluid, engaging responses, their approach to utterance handling is often a "black box", which can lead to the generation of inappropriate or incorrect responses, or cause the conversation to go off the rails. This is where LLM-ConvRec distinguishes itself: although it provides the flexibility and naturalness of an unstructured system, its semi-structured nature ensures that utterance handling is not opaque, and that inappropriate responses can be avoided through structural constraints.

The system retains important information about the conversation, ensuring that context and past interactions are reflected in the responses. This makes LLM-ConvRec not just a conversational system, but a conversational partner capable of delivering precise, personalized recommendations across diverse domains.

## Installation and Running the System

Before you can use the system, you must first ensure that you have Python (version 3.7 or higher) installed on your machine. You will also need pip for installing the necessary Python packages.

### 1. Clone the GitHub Repository

Clone the repository from GitHub to your local machine by running the following command in your terminal:

**git clone https://github.com/<your_username>/LLM-ConvRec.git**

Please replace `<your_username>` with your actual GitHub username.

### 2. Navigate to the Project Directory

Once you've cloned the repository, use the command line to navigate into the project's directory:

**cd LLM-ConvRec**

### 3. Install the Required Packages

The project has a number of dependencies that need to be installed. These are listed in the `requirements.txt` file. To install these dependencies, run the following command in your terminal:

**pip install -r requirements.txt**

### 4 Obtaining an OpenAI API Key and Configuring the .env file

Before you can run the system, you need an API key from OpenAI. This key enables the model to interact with the OpenAI's servers to process and generate conversational responses.

Here are the steps to obtain an API key:

1. Visit OpenAI's website and create an account or log in if you already have an account.

2. Navigate to the 'API Keys' section in your account settings.

3. Click on the 'Create a new API key' button.

4. Name your key and click on 'Create secret key' to generate your new API key.

After you have the API key, you need to configure your `.env` file:

1. Create a new file in your project root directory and name it `.env`.

2. Inside the `.env` file, create a new line and write `OPENAI_API_KEY=`, and then paste your API key after the equals sign. For example:

OPENAI_API_KEY='sk1234567890abcdef`.

4. Save the `.env` file.

Please ensure you do not upload your `.env` file to public repositories to keep your OpenAI API key secure.

After this step, you are ready to install the dependencies and run the system.

### 5. Configuring gradio URL

1. Open Colab (https://colab.research.google.com/drive/1FfKTLmVV0rQSQWkvoGpiyb1RuK7E1l6k?usp=sharing#scrollTo=9_rc9X75fFT5) and run first 4 and 11th cells

2. Copy the public URL (should be something like this https://8b4a0f826a0deb0ec1.gradio.live)

3. Change GRADIO_URL under the Gradio.live API call cell to the public URL you copied.

You must update this based on the URL listed in the output cell above
GRADIO_URL = "https://8b4a0f826a0deb0ec1.gradio.live" <- change this URL

3. Add public URL to .env where the key is GRADIO_URL
   example: GRADIO_URL = "https://8b4a0f826a0deb0ec1.gradio.live"


### 6. Run the System
If you want to run the restaurant demo, execute following command in the terminal: 

```
python restaurant_main.py
```

If you want to run the clothing demo, execute following command in the terminal: 

```
python clothing_main.py
```


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

### 4.Response Generation
Once the action is chosen, the system generates a structured response that aligns with the decided action. The system ensures this response is in line with the ongoing conversation context and adheres to the system's semi-structured conversational style.

This process is repeated at each turn of the conversation, enabling LLM-ConvRec to provide a dynamic, interactive, and engaging conversational recommendation experience.


## Domain Initialization and Customization
This system is designed to be flexible and adaptable, allowing you to initialize and customize your own domain. With a configuration process involving providing some key files, you can utilize our robust system architecture tailored to your specific needs. 

### Quick Start

If you're looking to get started quickly, we've already set up two pre-configured domains: **Restaurant** and **Clothing**.

- **Restaurant Domain:** This domain utilizes a dataset containing all Edmonton restaurants. The domain is already initialized and ready to use, providing a wide range of restaurant data.

- **Clothing Domain:** For the Clothing domain, we've integrated Amazon data related to clothing items. This domain is fully initialized and can provide insights into a broad spectrum of clothing items.


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

When providing few-shot examples, make sure that they are representative of the tasks you want the model to perform. For instance, if you want the system to recognize when a user is expressing a preference, include examples where users express preferences in different ways.

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
This file instructs the model to extract answers from reviews based on the user's question. It requires 'question' (user's question), 'information' (reviews retrieved by the system), and 'answer' (the answer to the question derived from the provided information).

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
This file instructs the model to verify if a system-generated response accurately answers a user's query. It needs 'question' (user's question), 'answer' (the system's generated answer), and 'response' (indicates whether the generated answer meets the user's query).

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



## 4.filter configs

### 4.1 `filter_config.csv`

The `filter_config.csv` is a configuration file that allows you to specify the filters you want to apply in the system. This CSV file consists of three columns: `type_of_filter`, and `item`.

- `type_of_filter`: This column specifies the type of filter you want to use. Valid types include `"exact word matching"`, `"word in"`, `"value range"`, `"item"`, and `"location"`.

- `key_in_state`: This column lists the keys in the hard constraint that you want to check against `metadata_field`. Depending on the filter type, this could be a list of keys or a single key.

- `metadata_field`: This column contains the field you want to check against `key_in_state`.

Here is an example of what this file might look like:

| type_of_filter | key_in_state | metadata_field |
| -------------- | ------------ | -------------- |
| word in | "cuisine type, dish type" | categories |
| item | recommended_items | name |




Below, we describe several types of filters that can be used in the system. They allow you to filter out items based on various criteria.

### 1.Exact Word Matching Filter

This filter retains an item if any value in `key_in_state` exactly matches a value in `metadata_field`. This filter is case insensitive.

**Type of filter:** `"exact word matching"`

**key_in_state:** List of keys in the hard constraint you want to check against `metadata_field`.

**metadata_field:** The field you want to check against `key_in_state`. The value in this field should be a string, a list-like string (e.g. “A, B, C”), or a list. If it is neither a string nor a list, the item will be retained.

### 2.Word In Filter

This filter retains an item if any value in `key_in_state` is present in `metadata_field`, or vice versa. It also considers plural forms. This filter is case insensitive.

**Type of filter:** `"word in"`

**key_in_state:** List of keys in the hard constraint you want to check against `metadata_field`.

**metadata_field:** The field you want to check against `key_in_state`. The value in this field should be a string, a list-like string (e.g. “A, B, C”), or a list. If it is neither a string nor a list, the item will be retained.

### 3.Value Range Filter

This filter retains an item if any value in `metadata_field` falls within a value range in `key_in_state`, or if any value range in `metadata_field` overlaps with a value range in `key_in_state`.

**Type of filter:** `"value range"`

**key_in_state:** A key in the hard constraint containing the value ranges to check against `metadata_field`. Value ranges should be in the format of "(lower bound value (with units))-(upper bound value (with units))".

**metadata_field:** The field you want to check against `key_in_state`. The value in this field should be a string (either a value or a value range), a list-like string of values (e.g. “A, B, C”), or a list of values. If it is neither a string nor a list, the item will be retained.

### 4.Item Filter

This filter retains an item if it is not in the list of recommended items specified by `key_in_state`.

**Type of filter:** `"item"`

**key_in_state:** A key in the state manager that contains a list of recommended items to check against `metadata_field`.

**metadata_field:** The field to use for checking whether an item is not in the value of `key_in_state`. Must be either `item_id` or `name`.

### Location Filter

This filter retains an item if it is close enough to one of the locations in `constraint_key`. "Close enough" means that the item falls within a circle whose radius is half the diagonal length of the location's boundary (or the `default_max_distance_in_km` if it is larger), with the location at the center.

**constraint_key:** A key in the hard constraint that contains location information.

**metadata_field:** A list with two elements, where the first element is the metadata field containing item latitude, and the second element contains item longitude.

**default_max_distance_in_km:** Default maximum distance in km. If half the diagonal length of the location boundary is smaller than this value, the filter will use the default distance as the circle radius.

**geocoder_wrapper:** The geocoder wrapper to use.



## 5.domain specific config
- domain name
- file path to files, shouldnt change normally

## 6. Data

The LLM-ConvRec system requires two main types of data: metadata and reviews.

### 6.1 Metadata

The metadata must include name and unique item identifiers (item_id) as keys. Each item can have various other keys representing different types of metadata, such as location, type of cuisine, cost, etc. It is not necessary for all items to have a value for every metadata field. The metadata fields could be populated based on the information available for each item.

An example of a metadata structure is as follows:


{
    "item_id": "-3GD07waps96fB_okEwFqw",
    "name": "Brits Fish & Chips",
    "address": "6940 77 Street NW",
    "city": "Edmonton",
    "categories": ["Fish & Chips", "Restaurants"]
}

### 6.2 Review data


## 7.User defined classes
  

## Example Conversations:

Demonstration using the restaurant domain:

User: I am looking for Japanese restaurants that serve excellent sushi, preferably in a casual setting.

Recommender: Can you provide the location?

User: Tower Road NW & Kingsway NW

Recommender: How about trying Washoku Bistro for a comfortable and laid-back vibe while enjoying some delicious Japanese sushi? But if you're looking for something quick and easy without sacrificing taste, Tokyo Express is another great option near the same location.

User: What kind of menu do they offer?

Recommender: For Washoku Bistro, they offer a lunch menu with bento boxes and entrées, with limited options. They also have sushi rolls, such as spicy salmon rolls and California rolls. As for Tokyo Express, they have a wide variety of options including bento boxes, sushi combos, feature rolls, tempura, and noodles.

User: Ok, then the first one doesn't seem to match my preference.

Recommender: I'm sorry that you did not like the recommendation. Is there anything else I can assist you with?

User: Does Tokyo Express have a parking lot?

Recommender: Tokyo Express has a parking lot.

User: Great. I will go there then. Thanks!

Recommender: Great! Enjoy your meal! If you need any more assistance, feel free to ask.


