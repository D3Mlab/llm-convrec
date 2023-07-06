class CheckCuisineDishType:
    """
        Class that checks whether the cuisine type specified by the user match a restaurant using its categories from metadata.
    """

    def does_cuisine_dish_type_match(self, cuisine_dish_type: list[str],
                                     categories_restaurant: list[str]) -> bool:
        """
        Check whether the cuisine type specified by the user match a restaurant using its categories from metadata.

        :param cuisine_dish_type: cuisine type and dish type specified by the user
        :param categories_restaurant: categories of restaurant in metadata
        :return: true or false on whether the cuisine type matches or not
        """
        if cuisine_dish_type is None:
            return True

        for category in categories_restaurant:
            for cuisine_dish in cuisine_dish_type:
                if cuisine_dish in category or category in cuisine_dish \
                        or self._convert_to_plural(cuisine_dish) in category \
                        or category in self._convert_to_plural(cuisine_dish):
                    return True

        return False

    def _convert_to_plural(self, word):
        """
        Try to convert the word to plural.

        :param word: word to be converted to plural
        :return: word in plural form
        """
        plural_rules = [
            (["s", "sh", "ch", "x", "z"], "es"),  # Words ending in these letters take "es"
            (["ay", "ey", "iy", "oy", "uy"], "s"),  # Words ending in these letter combinations take "s"
            (["y"], "ies")  # Words ending in "y" with a consonant before it, replace "y" with "ies"
        ]

        for rule in plural_rules:
            endings, plural_ending = rule
            for end in endings:
                if word.endswith(end):
                    return word.rstrip(end) + plural_ending

        return word + "s"  # Default rule, simply add "s" at the end
