from enum import Enum
from Models import UCBNormalModel
from functools import partial


class ModelType(Enum):
    class ConstructorWrapper:
        def __init__(self, f):
            self.f = f

        def __call__(self, *args, **kwargs):
            return self.f(*args, **kwargs)

    IN_ALGORITHMS_LAMBDA = ConstructorWrapper(UCBNormalModel)
    IN_ALGORITHMS_UBC_BASED_THOMPSON = ConstructorWrapper(UCBNormalModel)
    IN_ALGORITHMS_STOCHASTIC = ConstructorWrapper(UCBNormalModel)
    BETA_LAMBDA = ConstructorWrapper(UCBNormalModel)
    BETA_UBC_BASED_THOMPSON = ConstructorWrapper(UCBNormalModel)
    BETA_STOCHASTIC = ConstructorWrapper(UCBNormalModel)
    UCB_NORMAL = ConstructorWrapper(UCBNormalModel)
    UCB_ENTROPY = ConstructorWrapper(UCBNormalModel)
    THOMPSON_NORMAL = ConstructorWrapper(UCBNormalModel)
    THOMPSON_ENTROPY = ConstructorWrapper(UCBNormalModel)


def getModel(model_type, machines_list, K, T, rewards):
    for model in ModelType:
        if model == model_type:
            return model_type.value(machines_list, K, T, rewards)
    return None
