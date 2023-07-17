from information_retrievers.item.item import Item
from information_retrievers.item.item_loader import ItemLoader
import ast


class RestaurantItemLoader(ItemLoader):

    def create_item(self, item_dict: dict) -> Item:
        item_info = {
            "address": item_dict['address'],
            "city": item_dict['city'],
            "state": item_dict['state'],
            "postal_code": item_dict['postal_code'],
            "latitude": float(item_dict['latitude']),
            "longitude": float(item_dict['longitude']),
            "stars": float(item_dict['stars']),
            "review_count": int(item_dict['review_count']),
            "is_open": bool(item_dict['is_open']),
            "categories": list(item_dict['categories'].split(",")),
            "hours": ast.literal_eval(item_dict['hours']),
        }
        return Item(item_dict['item_id'], item_dict['name'],
                    item_info, ast.literal_eval(item_dict['optional']))
