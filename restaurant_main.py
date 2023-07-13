from domain_specific.classes.restaurants.location_constraint_merger import LocationConstraintMerger
from domain_specific.classes.restaurants.location_status import LocationStatus


from conv_rec_system import ConvRecSystem
import logging.config
import warnings
import yaml

warnings.simplefilter("default")
logging.config.fileConfig('logging.conf')
with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

user_constraint_merger_objects = [LocationConstraintMerger()]
user_constraint_status_objects = [LocationStatus()]

conv_rec_system = ConvRecSystem(config, user_constraint_merger_objects, user_constraint_status_objects)
conv_rec_system.run()