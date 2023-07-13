from domain_specific.classes.restaurants.location_constraint_merger import LocationConstraintMerger


from conv_rec_system import ConvRecSystem
import logging.config
import warnings
import yaml

warnings.simplefilter("default")
logging.config.fileConfig('logging.conf')
with open('system_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

user_merge_constraint_objects = [LocationConstraintMerger()]

conv_rec_system = ConvRecSystem(config, user_merge_constraint_objects)
conv_rec_system.run()