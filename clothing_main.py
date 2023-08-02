from conv_rec_system import ConvRecSystem
from dotenv import load_dotenv
import logging.config
import warnings
import yaml
import os

"""
Runs clothing conversational recommendation system in terminal 
"""

warnings.simplefilter("default")
logging.config.fileConfig('logging.conf')
with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config['PATH_TO_DOMAIN_CONFIGS'] = "domain_specific/configs/clothing_configs"
load_dotenv()

openai_api_key_or_gradio_url = os.environ['OPENAI_API_KEY']

conv_rec_system = ConvRecSystem(
    config, openai_api_key_or_gradio_url)

conv_rec_system.run()
