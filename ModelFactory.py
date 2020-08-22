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

    LH = ConstructorWrapper(HybridModels.LambdaModel)
    LEG_LH = ConstructorWrapper(HybridModels.LambdaBetaModel)
    AEG_LH = ConstructorWrapper(HybridModels.LambdaBetaPlusModel)
    NOAEG_LH = ConstructorWrapper(HybridModels.LambdaBetaModelPlusNormalized)

    # UCB_BASED_THOMPSON = ConstructorWrapper(HybridModels.UCBBasedThompsonModel)
    # BETA_UBC_BASED_THOMPSON = ConstructorWrapper(HybridModels.UCBBasedThompsonBetaModel)
    # BETA_UBC_BASED_THOMPSON_PLUS = ConstructorWrapper(HybridModels.UCBBasedThompsonBetaPlusModel)

    BH = ConstructorWrapper(HybridModels.StochasticThompsonUCBModel)
    UBH = ConstructorWrapper(HybridModels.StochasticThompsonUCBUpdateModel)
    LEG_BH = ConstructorWrapper(HybridModels.StochasticThompsonUCBBetaModel)
    LEG_UBH = ConstructorWrapper(HybridModels.StochasticThompsonUCBUpdateBetaModel)
    AEG_BH = ConstructorWrapper(HybridModels.StochasticThompsonUCBBetaPlusModel)
    AEG_UBH = ConstructorWrapper(HybridModels.StochasticThompsonUCBUpdateBetaPlusModel)

    UCB = ConstructorWrapper(DirectedExplorationModels.UCBNormalModel)
    LEG_UCB = ConstructorWrapper(DirectedExplorationModels.UCBEntropyGainModel)
    AEG_UCB = ConstructorWrapper(DirectedExplorationModels.UCBEntropyGainPlusModel)
    # UCB_ENTROPY_NORMALIZED = ConstructorWrapper(DirectedExplorationModels.UCBEntropyNormalizedModel)

    TS = ConstructorWrapper(RandomExplorationModels.ThompsonNormalModel)
    LEG_TS = ConstructorWrapper(RandomExplorationModels.ThompsonEntropyGainModel)
    AEG_TS = ConstructorWrapper(RandomExplorationModels.ThompsonEntropyGainPlusModel)

    RANDOM_BASELINE = ConstructorWrapper(BaselineModels.RandomModel)
    EG_BASELINE = ConstructorWrapper(BaselineModels.EntropyGainModel)
    OPTIMAL_BASELINE = ConstructorWrapper(BaselineModels.OptimalModel)

    NLEG_LH = ConstructorWrapper(HybridModels.LambdaBetaNoiseModel)
    NAEG_LH = ConstructorWrapper(HybridModels.LambdaBetaPlusNoiseModel)
    NLEG_BH = ConstructorWrapper(HybridModels.StochasticThompsonUCBBetaNoiseModel)
    NLEG_UBH = ConstructorWrapper(HybridModels.StochasticThompsonUCBUpdateBetaNoiseModel)
    NAEG_BH = ConstructorWrapper(HybridModels.StochasticThompsonUCBBetaPlusNoiseModel)
    NAEG_UBH = ConstructorWrapper(HybridModels.StochasticThompsonUCBUpdateBetaPlusNoiseModel)
