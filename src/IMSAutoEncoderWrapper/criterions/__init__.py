from .base import IMSABaseAutoEncoderCriterion

# TODO list of import (for implemented methods ??) 
from .ContrastiveCriterion import ContrastiveCriterion

CRITERIONS_REGISTRY = {
    "ContrastiveLoss": ContrastiveCriterion
    }