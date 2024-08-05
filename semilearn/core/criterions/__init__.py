
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .cross_entropy import ce_loss, CELoss, entropyLoss
from .consistency import consistency_loss, ConsistencyLoss
from .information_bottleneck import IBLoss, vaeLoss
from .contrastive import SupConLoss