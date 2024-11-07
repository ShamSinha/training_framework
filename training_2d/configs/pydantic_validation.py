from pydantic import BaseModel, Field, root_validator
from typing import List, Dict


class ClassificationConfig(BaseModel):
    sampling_tags: List[str] = Field(..., min_items=1)
    heads: List[str] = Field(..., min_items=1)
    user_class_wts: Dict[str, float]
    loss_wts: Dict[str, float]
    alpha: int

    @root_validator(pre=True)
    def validate_matching_elements(cls, values):
        sampling_tags = values.get("sampling_tags", [])
        heads = values.get("heads", [])
        user_class_wts = values.get("user_class_wts", {})
        loss_wts = values.get("loss_wts", {})

        """
        # Ensure all heads are in sampling tags
        if not all(item in sampling_tags for item in heads):
            raise ValueError("All heads must be included in sampling_tags")
        """

        # Ensure keys in user_class_wts match sampling tags
        if sorted(user_class_wts.keys()) != sorted(sampling_tags):
            raise ValueError(
                "Keys in user_class_wts must match the elements in sampling_tags"
            )

        # Ensure keys in loss_wts match heads
        if sorted(loss_wts.keys()) != sorted(heads):
            raise ValueError("Keys in loss_wts must match the elements in heads")

        print("Overall cls Configuration Validation Success.")

        return values


class FullConfig(BaseModel):
    cls: ClassificationConfig
