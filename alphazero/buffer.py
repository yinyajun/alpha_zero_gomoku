import torch
import random
from typing import List, Tuple


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, resume_path: str = None):
        self.capacity = capacity
        self.buffer: List = []
        self.idx = 0

        if resume_path:
            self.load(resume_path)

    def __len__(self):
        return len(self.buffer)

    def add(self, s, pi, z) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append([s, pi, z])
        else:
            self.buffer[self.idx] = [s, pi, z]
            self.idx = (self.idx + 1) % self.capacity

    def add_trajectory(self, trajectory: dict):
        s, pi, z = trajectory["s"], trajectory["pi"], trajectory["z"]
        for s, pi, z in zip(s, pi, z):
            self.add(s, pi, z)

    def save(self, path: str):
        torch.save(self.buffer, path)

    def load(self, path: str) -> "ReplayBuffer":
        try:
            items = torch.load(path, map_location="cpu")
            for i in items:
                self.add(*i)
        except Exception as e:
            print(f"未加载数据，原因：{e}")
        return self

    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
          states: [B, 2, H, W]       (s)
          target_policies: [B, H*W]  (π)
          target_values: [B, 1]      (z)
        """
        assert len(self.buffer) > 0
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states, pis, zs = [], [], []
        # 数据增强：随机旋转 k∈{0,1,2,3}，50% 左右翻转
        k = random.randint(0, 3)
        do_flip = random.random() < 0.5

        for [s, pi, z] in batch:
            state = torch.rot90(s, k=k, dims=(1, 2))  # [2,h,w]
            pi = torch.rot90(pi, k=k, dims=(0, 1))  # [h,w]
            if do_flip:
                state = torch.flip(state, dims=[2])
                pi = torch.flip(pi, dims=[1])

            states.append(state)
            pis.append(pi.reshape(-1))  # [h*w]
            zs.append(z.view(1))  # [1]

        states = torch.stack(states, dim=0).to(device)  # [B, 2, h, w]
        target_policies = torch.stack(pis, dim=0).to(device)  # [B, h*w]
        target_values = torch.cat(zs, dim=0).unsqueeze(-1).to(device)  # [B, 1]

        return states, target_policies, target_values
