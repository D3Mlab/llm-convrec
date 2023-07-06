from information_retrievers.filter.check_already_recommended_restaurant import CheckAlreadyRecommendedRestaurant
from information_retrievers.recommended_item import RecommendedItem
from information_retrievers.item import Item
import pytest

check_already_recommended_restaurant = CheckAlreadyRecommendedRestaurant()

dictionary_info1 = {"name": "Famoso Neapolitan Pizzeria",
                    "address": "15745 97th Street NW",
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "latitude": 0,
                    "longitude": 0,
                    "stars": 0,
                    "review_count": 0,
                    "is_open": True,
                    "attributes": {},
                    "categories": [],
                    "hours": {}}
restaurant1 = Item("0CLCzdedGT2DPjkYM52Tqg", dictionary_info1)
rec_restaurant1 = RecommendedItem(restaurant1, "", [])

dictionary_info2 = {"name": "Famoso Neapolitan Pizzeria",
                    "address": "1437 99 Street",
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "latitude": 0,
                    "longitude": 0,
                    "stars": 0,
                    "review_count": 0,
                    "is_open": True,
                    "attributes": {},
                    "categories": [],
                    "hours": {}}
restaurant2 = Item("0IG9w6YCkh-z5pnfSip6Nw", dictionary_info2)
rec_restaurant2 = RecommendedItem(restaurant2, "", [])

dictionary_info3 = {"name": "Uccellino",
                    "address": "",
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "latitude": 0,
                    "longitude": 0,
                    "stars": 0,
                    "review_count": 0,
                    "is_open": True,
                    "attributes": {},
                    "categories": [],
                    "hours": {}}
restaurant3 = Item("2Z8LWDUbrUbe-o3KZE2r7Q", dictionary_info3)
rec_restaurant3 = RecommendedItem(restaurant3, "", [])

dictionary_info4 = {"name": "Pazzo Pazzo Italian Cuisine",
                    "address": "",
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "latitude": 0,
                    "longitude": 0,
                    "stars": 0,
                    "review_count": 0,
                    "is_open": True,
                    "attributes": {},
                    "categories": [],
                    "hours": {}}
restaurant4 = Item("2fTfpN5SggLgW4LlzptMPg", dictionary_info4)
rec_restaurant4 = RecommendedItem(restaurant4, "", [])

dictionary_info5 = {"name": "Cafe Amore Bistro",
                    "address": "",
                    "city": "",
                    "state": "",
                    "postal_code": "",
                    "latitude": 0,
                    "longitude": 0,
                    "stars": 0,
                    "review_count": 0,
                    "is_open": True,
                    "attributes": {},
                    "categories": [],
                    "hours": {}}
restaurant5 = Item("3gTZAOmML02T1L5gYIHYqQ", dictionary_info5)
rec_restaurant5 = RecommendedItem(restaurant5, "", [])


class TestCheckAlreadyRecommendedRestaurant:
    @pytest.mark.parametrize("restaurant_name, already_recommended_restaurant, expected_tf",
                             [("Famoso Neapolitan Pizzeria", [], True),
                              ("Famoso Neapolitan Pizzeria", [rec_restaurant1], False),
                              ("Famoso Neapolitan Pizzeria", [rec_restaurant1, rec_restaurant2], False),
                              ("Famoso Neapolitan Pizzeria", [rec_restaurant2, rec_restaurant1], False),
                              ("Famoso Neapolitan Pizzeria", [rec_restaurant3, rec_restaurant4, rec_restaurant5], True),
                              ("Cafe Amore Bistro", [rec_restaurant3, rec_restaurant4, rec_restaurant5], False)])
    def test_is_not_recommended(self, restaurant_name: str,
                                already_recommended_restaurant: list[RecommendedItem],
                                expected_tf: bool):
        """
        checks whether the restaurant is already rejected by the user or not.

        :param restaurant_name: restaurant name
        :param already_recommended_restaurant: restaurants that are already recommended
        :param expected_tf: expected true or false
        """
        restaurant_name = restaurant_name.lower().strip().replace(" ", "")
        assert check_already_recommended_restaurant.is_not_recommended(
            restaurant_name, already_recommended_restaurant) == expected_tf
