from information_retrievers.item import Item
from information_retrievers.item_loader import ItemLoader
import ast


class RestaurantItemLoader(ItemLoader):

    def create_item(self, item_dict: dict) -> Item:
        item_info = {
            'category': ast.literal_eval(item_dict['category']),
            'price': item_dict['price'],
            'brand': item_dict['brand'],
            'feature': ast.literal_eval(item_dict['feature']),
            'rating': float(item_dict['price']),
            'num_reviews': float(item_dict['num_reviews']),
            'rank': item_dict['rank'],
        }

        return Item(item_dict['id'], item_dict['name'], item_info, ast.literal_eval(item_dict['optional']))