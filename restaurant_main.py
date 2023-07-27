from domain_specific.classes.restaurants.location_constraint_merger import LocationConstraintMerger
from domain_specific.classes.restaurants.location_status import LocationStatus
from domain_specific.classes.restaurants.location_filter import LocationFilter

from conv_rec_system import ConvRecSystem
from dotenv import load_dotenv
import logging.config
import warnings
import yaml
import os


"""
Runs restaurant conversational recommendation system in terminal 
"""


warnings.simplefilter("default")
logging.config.fileConfig('logging.conf')
with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

user_constraint_merger_objects = [LocationConstraintMerger()]
user_constraint_status_objects = [LocationStatus()]
user_filter_objects = [LocationFilter("location", ["latitude", "longitude"], 2)]

load_dotenv()

if config['LLM'] == "Alpaca Lora":
    openai_api_key_or_gradio_url = os.environ['GRADIO_URL']
else:
    openai_api_key_or_gradio_url = os.environ['OPENAI_API_KEY']

conv_rec_system = ConvRecSystem(
    config, openai_api_key_or_gradio_url,
    user_defined_constraint_mergers=user_constraint_merger_objects,
    user_constraint_status_objects=user_constraint_status_objects,
    user_defined_filter=user_filter_objects)

conv_rec_system.run()
