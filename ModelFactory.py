from enum import Enum
import UCBModels
import ThompsonModels
import HybridModels
import BaselineModels


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

    LAMBDA = ConstructorWrapper(HybridModels.LambdaModel)
    LAMBDA_BETA = ConstructorWrapper(HybridModels.LambdaBetaModel)
    LAMBDA_BETA_PLUS = ConstructorWrapper(HybridModels.LambdaBetaModelPlus)

    UCB_BASED_THOMPSON = ConstructorWrapper(HybridModels.UCBBasedThompsonModel)
    BETA_UBC_BASED_THOMPSON = ConstructorWrapper(HybridModels.UCBBasedThompsonBetaModel)
    BETA_UBC_BASED_THOMPSON_PLUS = ConstructorWrapper(HybridModels.UCBBasedThompsonBetaPlusModel)

    STOCHASTIC = ConstructorWrapper(HybridModels.StochasticThompsonUCBModel)
    BETA_STOCHASTIC = ConstructorWrapper(HybridModels.StochasticThompsonUCBBetaModel)

    UCB_NORMAL = ConstructorWrapper(UCBModels.UCBNormalModel)
    UCB_ENTROPY_GAIN = ConstructorWrapper(UCBModels.UCBEntropyGainModel)
    UCB_ENTROPY_NORMALIZED = ConstructorWrapper(UCBModels.UCBEntropyNormalizedModel)

    THOMPSON_NORMAL = ConstructorWrapper(ThompsonModels.ThompsonNormalModel)
    THOMPSON_ENTROPY = ConstructorWrapper(ThompsonModels.ThompsonEntropyGainModel)

    BASELINE_MODEL = ConstructorWrapper(BaselineModels.RandomModel)
    ENTROPY_GAIN_MODEL = ConstructorWrapper(BaselineModels.EntropyGainModel)
