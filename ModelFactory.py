from enum import Enum
import DirectedExplorationModels
import RandomExplorationModels
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
    LAMBDA_BETA_PLUS_NORMALIZED = ConstructorWrapper(HybridModels.LambdaBetaModelPlusNormalized)

    UCB_BASED_THOMPSON = ConstructorWrapper(HybridModels.UCBBasedThompsonModel)
    BETA_UBC_BASED_THOMPSON = ConstructorWrapper(HybridModels.UCBBasedThompsonBetaModel)
    BETA_UBC_BASED_THOMPSON_PLUS = ConstructorWrapper(HybridModels.UCBBasedThompsonBetaPlusModel)

    STOCHASTIC = ConstructorWrapper(HybridModels.StochasticThompsonUCBModel)
    BETA_STOCHASTIC = ConstructorWrapper(HybridModels.StochasticThompsonUCBBetaModel)

    UCB_NORMAL = ConstructorWrapper(DirectedExplorationModels.UCBNormalModel)
    UCB_ENTROPY_GAIN = ConstructorWrapper(DirectedExplorationModels.UCBEntropyGainModel)
    UCB_ENTROPY_NORMALIZED = ConstructorWrapper(DirectedExplorationModels.UCBEntropyNormalizedModel)

    THOMPSON_NORMAL = ConstructorWrapper(RandomExplorationModels.ThompsonNormalModel)
    THOMPSON_ENTROPY = ConstructorWrapper(RandomExplorationModels.ThompsonEntropyGainModel)

    BASELINE_MODEL = ConstructorWrapper(BaselineModels.RandomModel)
    ENTROPY_GAIN_MODEL = ConstructorWrapper(BaselineModels.EntropyGainModel)
    OPTIMAL_MODEL = ConstructorWrapper(BaselineModels.OptimalModel)
