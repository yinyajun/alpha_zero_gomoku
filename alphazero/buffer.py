import random
from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class Sample:
    state: torch.Tensor  # [2, H, W], float
    search_rate: torch.Tensor  # [H, W], float (动作访问分布)
    win_rate: torch.Tensor  # [], float （这里装的是 z ∈ {-1,0,1}）


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.storage: List[Sample] = []
        self.ptr = 0

    def __len__(self):
        return len(self.storage)

    def add(self, s: Sample) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(s)
        else:
            self.storage[self.ptr] = s
            self.ptr = (self.ptr + 1) % self.capacity

    def extend(self, samples: List):
        for x in samples:
            self.add(x)

    def save(self, path: str):
        torch.save(self.storage, path)

    def load(self, path: str) -> "ReplayBuffer":
        try:
            items = torch.load(path, map_location="cpu")
            self.extend(items)
        except Exception as e:
            print(f"未加载数据，原因：{e}")
        return self

    @torch.no_grad()
    def sample_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
          states: [B, 2, H, W]       (s)
          target_policies: [B, H*W]  (π)
          target_values: [B, 1]      (z)
        """
        assert len(self.storage) > 0
        batch = random.sample(self.storage, min(batch_size, len(self.storage)))

        # 数据增强：随机旋转 k∈{0,1,2,3}，50% 左右翻转
        k = random.randint(0, 3)
        do_flip = random.random() < 0.5

        states, pis, zs = [], [], []
        for s in batch:
            state = s.state
            pi = s.search_rate
            z = s.win_rate

            state = torch.rot90(state, k=k, dims=(1, 2))  # [2,size,size]
            pi = torch.rot90(pi, k=k, dims=(0, 1))  # [size,size]
            if do_flip:
                state = torch.flip(state, dims=[2])
                pi = torch.flip(pi, dims=[1])

            states.append(state)
            pis.append(pi.reshape(-1))  # [H*W]
            zs.append(z.view(1))  # [1]

        states = torch.stack(states, dim=0).to(device)  # [B, 2, size, size]
        target_policies = torch.stack(pis, dim=0).to(device)  # [B, size*size]
        target_values = torch.cat(zs, dim=0).unsqueeze(-1).to(device)  # [B, 1]

        return states, target_policies, target_values
