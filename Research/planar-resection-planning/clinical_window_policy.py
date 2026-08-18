"""Compact convolutional feature extractor for five-action clinical PPO."""

from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ClinicalGridExtractor(BaseFeaturesExtractor):
    """Encode the padded 30x40 state into a compact global representation."""

    def __init__(self, observation_space, features_dim: int = 256) -> None:
        channels = int(observation_space.shape[0])
        super().__init__(observation_space, features_dim=features_dim)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 5)),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 5, features_dim),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


class LocalGlobalClinicalExtractor(BaseFeaturesExtractor):
    """当前位置四邻域局部特征 + 全局压缩状态的策略特征提取器。

    纯全局压缩 CNN（``ClinicalGridExtractor``）把 30x40 网格压到 4x5 后丢失了
    “当前在哪、四邻域是什么状态”的精确关系，导致导航可学习性不足（v5 Pilot
    Validation 前 16 场景完成率 0%）。本提取器把决策信息拆成两路：

    - **局部分支**：从 ``current_position`` 通道定位当前格，显式读取当前格及其
      上、下、左、右四邻域（顺序与动作 0-3 一致）的全部 25 通道值，拼接后过
      两层 MLP。策略可以直接看到“往每个方向走目标格是已切/前沿/血管/出血多少”。
    - **全局分支**：卷积压缩整个 25 通道网格，并显式拼接 7 个全局标量
      （clamped_phase、clamp_elapsed_fraction、unclamp_remaining_fraction、
      elapsed_time_fraction、no_progress_streak_fraction、same_edge_streak_fraction、
      clinical_cost_fraction），提供夹闭阶段、
      时间进度与停滞上下文。

    两路融合后输出 ``features_dim`` 维向量，供 PPO 的 pi/vf 共享 MLP 使用。
    观测为 ``(25, 30, 40)`` 的 padded 网格，其中 ``current_position`` 通道为
    one-hot，因此用该通道的 argmax 定位当前格是确定性的。
    """

    #: CLINICAL_OBSERVATION_CHANNELS 中 current_position 的下标（channel 7）。
    CURRENT_POSITION_CHANNEL = 7
    #: 夹闭、时间、停滞与临床成本的七个整层常量通道。
    GLOBAL_SCALAR_CHANNELS = (17, 18, 19, 20, 21, 23, 24)
    #: 当前格 + 四邻域，顺序与 ACTION 上/下/左/右 (0-3) 一致。
    NEIGHBORHOOD = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        local_hidden: int = 128,
        global_channels: int = 32,
    ) -> None:
        channels = int(observation_space.shape[0])
        super().__init__(observation_space, features_dim=features_dim)
        needed = max(self.GLOBAL_SCALAR_CHANNELS)
        if channels <= needed:
            raise ValueError(
                f"observation has {channels} channels; need at least {needed + 1}"
            )
        self._global_scalar_indices = torch.tensor(
            self.GLOBAL_SCALAR_CHANNELS, dtype=torch.long
        )
        local_in = len(self.NEIGHBORHOOD) * channels
        self.local_net = torch.nn.Sequential(
            torch.nn.Linear(local_in, local_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(local_hidden, local_hidden),
            torch.nn.ReLU(),
        )
        self.global_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, global_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(global_channels, global_channels * 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(global_channels * 2, global_channels * 2, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 5)),
            torch.nn.Flatten(),
        )
        global_dim = global_channels * 2 * 4 * 5
        fusion_in = local_hidden + global_dim + len(self.GLOBAL_SCALAR_CHANNELS)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(fusion_in, features_dim),
            torch.nn.ReLU(),
        )

    def _local_neighborhood_features(self, observations: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = observations.shape
        position_channel = observations[:, self.CURRENT_POSITION_CHANNEL]
        flat = position_channel.reshape(batch, -1)
        index = flat.argmax(dim=1)
        rows = index // width
        cols = index % width
        batch_index = torch.arange(batch, device=observations.device)
        parts: list[torch.Tensor] = []
        for delta_row, delta_col in self.NEIGHBORHOOD:
            neighbor_row = rows + delta_row
            neighbor_col = cols + delta_col
            in_bounds = (
                (neighbor_row >= 0)
                & (neighbor_row < height)
                & (neighbor_col >= 0)
                & (neighbor_col < width)
            )
            safe_row = neighbor_row.clamp(0, height - 1)
            safe_col = neighbor_col.clamp(0, width - 1)
            features = observations[batch_index, :, safe_row, safe_col]
            if (delta_row, delta_col) != (0, 0):
                features = features * in_bounds.to(features.dtype).unsqueeze(1)
            parts.append(features)
        return torch.cat(parts, dim=1)

    def _global_scalars(self, observations: torch.Tensor) -> torch.Tensor:
        # 这七个通道在 reset 后用 fill() 填满整层常量，任意格子取值一致。
        return observations[:, self._global_scalar_indices, 0, 0]

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        local = self._local_neighborhood_features(observations)
        local_features = self.local_net(local)
        global_features = self.global_encoder(observations)
        scalars = self._global_scalars(observations)
        return self.fusion(torch.cat([local_features, global_features, scalars], dim=1))
