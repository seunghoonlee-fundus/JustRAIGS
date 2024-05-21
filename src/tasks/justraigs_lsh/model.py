from typing import Optional

import timm
from torch import nn

import torch
from timm.models.layers import trunc_normal_
from flair import FLAIRModel


class TIMMBase(nn.Module):
    CLASSIFIER_NAMES = iter(["classifier", "fc", "head"])

    def __init__(
        self,
        model_name: str,
        num_classes: Optional[int] = None,
        pretrained: Optional[bool] = None,
        classifier_name: Optional[str] = None,
        **model_params,
    ):
        super(TIMMBase, self).__init__()
        self.model = timm.create_model(
            model_name, num_classes=num_classes, pretrained=pretrained, **model_params
        )
        self.classifier_name = classifier_name
        self._setup()

    def forward(self, x):
        return self.model(x)

    def _setup(self):
        if self.classifier_name is None:
            classifier = None
            while True:
                try:
                    classifier_name = next(self.CLASSIFIER_NAMES)
                    classifier = getattr(self.model, classifier_name)
                    break

                except AttributeError:
                    continue

                except StopIteration:
                    break

            if classifier is None:
                raise ValueError(
                    f"Could not find any of the classifier names {self.CLASSIFIER_NAMES}"
                )

            self.classifier_name = classifier_name


class TIMMCLSModel(TIMMBase):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = False,
        linear_probing: bool = False,
        classifier: Optional[nn.Module] = None,
        **model_params,
    ):
        super(TIMMCLSModel, self).__init__(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            **model_params,
        )
        self._num_classes = num_classes
        self.model_ready(linear_probing, classifier)

    def model_ready(
        self, linear_probing: bool = False, classifier: Optional[nn.Module] = None
    ):
        if classifier is not None:
            setattr(self.model, self.classifier_name, classifier)

        if linear_probing:
            for param in self.model.parameters():
                param.requires_grad = False

            for name, param in self.model.named_parameters():
                if name.startswith(f"{self.classifier_name}."):
                    param.requires_grad = True

            classifier = getattr(self.model, self.classifier_name)
            for name, param in classifier.named_parameters():
                if "weight" in name:
                    trunc_normal_(param, std=0.01)

            setattr(self.model, self.classifier_name, classifier)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def classifier(self):
        return getattr(self.model, self.classifier_name)
    