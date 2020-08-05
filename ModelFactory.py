from enum import Enum

from Models import *


class ModelType(Enum):
    class ConstructorWrapper:
        """
        Wrapper class for constructor method - required for use of class name as constructor in Enum value
        """

        def __init__(self, f):
            self.f = f

        def __call__(self, *args, **kwargs):
            return self.f(*args, **kwargs)

    IN_ALGORITHMS_LAMBDA = None
    IN_ALGORITHMS_UBC_BASED_THOMPSON = None
    IN_ALGORITHMS_STOCHASTIC = None

    LAMBDA = ConstructorWrapper(LambdaModel)
    LAMBDA_BETA = ConstructorWrapper(LambdaBetaModel)

    UCB_BASED_THOMPSON = ConstructorWrapper(UCBBasedThompsonModel)
    BETA_UBC_BASED_THOMPSON = ConstructorWrapper(UCBBasedThompsonBetaModel)

    STOCHASTIC = ConstructorWrapper(StochasticThompsonUCBModel)
    BETA_STOCHASTIC = ConstructorWrapper(StochasticThompsonUCBBetaModel)

    UCB_NORMAL = ConstructorWrapper(UCBNormalModel)
    UCB_ENTROPY = ConstructorWrapper(UCBEntropyModel)
    UCB_ENTROPY_GAIN = ConstructorWrapper(UCBEntropyGainModel)
    UCB_ENTROPY_NORMALIZED = ConstructorWrapper(UCBEntropyNormalizedModel)

    THOMPSON_NORMAL = ConstructorWrapper(ThompsonNormalModel)
    THOMPSON_ENTROPY = ConstructorWrapper(ThompsonEntropyModel)

    BASELINE_MODEL = ConstructorWrapper(RandomModel)
    ENTROPY_GAIN_MODEL = ConstructorWrapper(EntropyGainModel)


def getModel(model_type, machines_list, K, T, rewards):
    return model_type.value(machines_list, K, T, rewards)
