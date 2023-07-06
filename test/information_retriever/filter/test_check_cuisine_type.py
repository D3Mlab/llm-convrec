from information_retrievers.filter.check_cuisine_dish_type import CheckCuisineDishType
import pytest

check_cuisine_type = CheckCuisineDishType()
class TestCheckCuisineType:
    @pytest.mark.parametrize("cuisine_type, categories_restaurant, expected_tf",
                             [(["japanese"], ["american", "french"], False),
                              (["japanese"], ["american", "japanese"], True),
                              (["american", "italian"], ["american", "french"], True),
                              (["italian", "korean"], ["american", "french"], False),
                              (["american", "french"], ["american", "french"], True),
                              (["sushi"], ["sushipizza", "japanese"], True),
                              (["steak"], ["steakhouses", "american"], True),
                              (["hamburger"], ["burgers", "american"], True),
                              (["bakery"], ["bakeries", "italian", "french"], True)])
    def test_does_cuisine_type_match(self, cuisine_type: list[str], categories_restaurant: list[str],
                                     expected_tf: bool):
        """
        Test the function that checks whether the cuisine type specified by the user match a restaurant using its categories from metadata.

        :param cuisine_type: cuisine type specified by the user (all lowercase, no spaces)
        :param categories_restaurant: categories of restaurant in metadata (all lowercase, no spaces)
        :param expected_tf: expected true or false
        """
        assert check_cuisine_type.does_cuisine_dish_type_match(
            cuisine_type, categories_restaurant) == expected_tf

