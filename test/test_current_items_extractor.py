import pandas as pd
import pytest
import os
import dotenv
import yaml

from domain_specific_config_loader import DomainSpecificConfigLoader
from information_retriever.item.item_loader import ItemLoader
from intelligence.gpt_wrapper import GPTWrapper
from user_intent.extractors.current_items_extractor import CurrentItemsExtractor
from state.common_state_manager import CommonStateManager
from state.message import Message
from intelligence.alpaca_lora_wrapper import AlpacaLoraWrapper

dotenv.load_dotenv()

test_file_path = 'test/current_restaurants_extractor_test.csv'
path_to_domain_configs = "domain_specific/configs/restaurant_configs"
test_df = pd.read_csv(test_file_path)

recommended_items = []
test_data = []

item_loader = ItemLoader()
for col in test_df.to_dict("records"):
    row = [col["user_input"]]
    if col["current_item_names"] != "None.":

        list_curr_item_names = col["current_item_names"][:-1].split(
            ',')
        one_turn_reccommended_items = []

        for curr_item_name in list_curr_item_names:
            dictionary_info = {
                "item_id": "",
                "name": curr_item_name,
                "optional": {}
            }
            recommended_item = item_loader.create_recommended_item("", dictionary_info, [])
            one_turn_reccommended_items.append(recommended_item)

        row.append(one_turn_reccommended_items)
        recommended_items.append(one_turn_reccommended_items)
    else:
        row.append([])

    test_data.append(row)

for row in test_data:
    row.append(recommended_items)


class TestCurrItemsExtractor:

    @pytest.mark.parametrize('user_input,list_curr_item_objs,recommended_items', tuple(test_data))
    @pytest.mark.parametrize('llm_wrapper', [GPTWrapper(os.environ['OPENAI_API_KEY']), AlpacaLoraWrapper(os.environ['GRADIO_URL'])])
    def test_extract_category_from_input(self, llm_wrapper, user_input, list_curr_item_objs, recommended_items) -> None:
        with open('system_config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['PATH_TO_DOMAIN_CONFIGS'] = path_to_domain_configs
        domain_specific_config_loader = DomainSpecificConfigLoader(config)

        state_manager = CommonStateManager(set())
        state_manager.update_conv_history(Message('user', user_input))

        conv_history = state_manager.get("conv_history")

        extractor = CurrentItemsExtractor(
            llm_wrapper,
            domain_specific_config_loader.load_domain(),
            domain_specific_config_loader.load_current_items_fewshots(),
            domain_specific_config_loader.system_config
        )

        answer = extractor.extract(recommended_items, conv_history)
        assert list_curr_item_objs == answer
