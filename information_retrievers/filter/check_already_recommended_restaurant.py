from information_retrievers.recommended_item import RecommendedItem


class CheckAlreadyRecommendedRestaurant:
    """
        Class that checks whether the restaurant is already recommended or not
    """

    def is_not_recommended(self, restaurant_name: str,
                           already_recommended_restaurant: list[RecommendedItem]) -> bool:
        """
        checks whether the restaurant is already rejected by the user or not.

        :param restaurant_name: restaurant name (lower case, stripped, no spaces)
        :param already_recommended_restaurant: restaurants that are already recommended
        :return: true if the restaurant is not already recommended otherwise, false
        """
        if not already_recommended_restaurant:
            return True

        for rec_restaurant in already_recommended_restaurant:
            if restaurant_name == rec_restaurant.get("name").lower().strip().replace(" ", ""):
                return False

        return True
