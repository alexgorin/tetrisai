from typing import Iterable, List, Tuple

from features import Feature


class Utility:
    def __init__(self, features: Iterable[Feature], coefficients: Iterable[float]):
        self.features = features
        self.coefficients = coefficients

    def value(self, world) -> Tuple[float, List[float], List[float]]:
        feature_values = [feature.value(world) for feature in self.features]
        weighted_feature_values = [
            coefficient * feature_value
            for feature_value, coefficient in zip(feature_values, self.coefficients)
        ]
        return sum(weighted_feature_values), feature_values, weighted_feature_values
