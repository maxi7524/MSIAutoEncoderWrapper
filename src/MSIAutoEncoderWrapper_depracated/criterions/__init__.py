from .base_criterion import MSIABaseAutoEncoderCriterion

from .ContrastiveCriterion import ContrastiveCriterion

CRITERIONS_REGISTRY = {
    "ContrastiveLoss": ContrastiveCriterion
    }