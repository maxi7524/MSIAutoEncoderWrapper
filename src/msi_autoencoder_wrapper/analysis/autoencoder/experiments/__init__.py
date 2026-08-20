"""Generic, domain-agnostic reading of on-disk grid-experiment campaigns."""

from .campaign_reader import CampaignTask, read_campaign

__all__ = ["CampaignTask", "read_campaign"]
