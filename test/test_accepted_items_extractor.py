import os

import pytest
import pandas as pd
import yaml

from domain_specific_config_loader import DomainSpecificConfigLoader
from information_retriever.item.item_loader import ItemLoader
from intelligence.gpt_wrapper import GPTWrapper
from state.message import Message
from user_intent.extractors.accepted_items_extractor import AcceptedItemsExtractor
from intelligence.alpaca_lora_wrapper import AlpacaLoraWrapper

import dotenv
dotenv.load_dotenv()

test_file_path = 'test/accepted_clothing_extractor_test.csv'
path_to_domain_configs = "domain_specific/configs/clothing_configs"
test_df = pd.read_csv(test_file_path, encoding='latin1')
test_data = [
    (
        row["utterance"],
        list(map(lambda x: x.strip(), row['all mentioned items'].split(',')))
        if isinstance(row['all mentioned items'], str) else [],
        list(map(lambda x: x.strip(), row['recently mentioned items'].split(',')))
        if isinstance(row['recently mentioned items'], str) else [],
        list(map(lambda x: x.strip(), row['accepted items'].split(',')))
        if isinstance(row['accepted items'], str) else [],
    )
    for row in test_df.to_dict("records")
]


class TestAcceptedItemsExtractor:

    @pytest.fixture(params=[GPTWrapper(os.environ['OPENAI_API_KEY']), AlpacaLoraWrapper(os.environ['GRADIO_URL'])])
    def accepted_items_extractor(self, request):
        with open('system_config.yaml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['PATH_TO_DOMAIN_CONFIGS'] = path_to_domain_configs
        domain_specific_config_loader = DomainSpecificConfigLoader(config)
        yield AcceptedItemsExtractor(request.param, domain_specific_config_loader.load_domain(),
                                     domain_specific_config_loader.load_accepted_items_fewshots(),
                                     domain_specific_config_loader.system_config)

    @pytest.mark.parametrize("utterance,all_mentioned_item_names,recently_mentioned_item_names,accepted_item_names", test_data)
    def test_extract(self, accepted_items_extractor, utterance, all_mentioned_item_names, recently_mentioned_item_names, accepted_item_names):
        item_loader = ItemLoader()
        _recently_mentioned_item_names = set(recently_mentioned_item_names)
        _accepted_item_names = set(accepted_item_names)
        all_mentioned_items = [item_loader.create_recommended_item("", {'item_id': '', 'name': name, 'optional': {}}, []) for name in all_mentioned_item_names]
        recently_mentioned_items = [item for item in all_mentioned_items if item.get_name() in _recently_mentioned_item_names]
        accepted_items = [item.get_name() for item in all_mentioned_items if item.get_name() in _accepted_item_names]

        conv_history = [Message("user", utterance)]

        actual = accepted_items_extractor.extract(conv_history, all_mentioned_items, recently_mentioned_items)

        assert [item.get_name() for item in actual] == accepted_items
