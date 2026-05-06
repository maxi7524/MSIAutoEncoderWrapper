from .base import MSIABaseAutoEncoderCriterion

# TODO list of import (for implemented methods ??) 
from .ContrastiveCriterion import ContrastiveCriterion

CRITERIONS_REGISTRY = {
    "ContrastiveLoss": ContrastiveCriterion
    }