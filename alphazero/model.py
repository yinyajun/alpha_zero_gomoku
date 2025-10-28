import time

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from buffer import ReplayBuffer


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x, inplace=True)
        return out


class Alpha0Module(torch.nn.Module):
    def __init__(
            self,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            resume_path: str = None,
    ):
        super(Alpha0Module, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_step = 0

        # backbone: 输入 [B, 2, H, W] 输出 [B, 128, H, W]
        self.conv1 = nn.Conv2d(2, 64, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = ResidualBlock(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.res2 = ResidualBlock(128)
        # self.res3 = ResidualBlock(128)
        # self.res4 = ResidualBlock(128)

        # policy head: 输出 [B, H, W]
        self.policy_conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1))
        self.policy_bn1 = nn.BatchNorm2d(64)
        self.policy_conv2 = nn.Conv2d(64, 1, kernel_size=(1, 1))

        # value head (1x1 降维 + BN + ReLU -> GAP -> MLP -> tanh): 输出 [B, 1]
        self.value_conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1))  # 先降维保持空间信息
        self.value_bn1 = nn.BatchNorm2d(64)
        self.value_conv2 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.value_bn2 = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32, 128)
        self.value_fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU(inplace=True)

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        if resume_path is not None:
            try:
                self.load_checkpoint(resume_path)
                print(f"成功加载模型: {resume_path}")
            except Exception as e:
                print(f"未加载模型，原因: {e}")

        self.to(self.device)
        self.eval()

    def forward(self, x):
        # backbone
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, H, W]
        x = self.res1(x)
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 128, H, W]
        x = self.res2(x)
        # x = self.res3(x)
        # x = self.res4(x)

        # policy head
        p = self.relu(self.policy_bn1(self.policy_conv1(x)))  # [B, 64, H, W]
        p = self.policy_conv2(p)  # [B, 1, H, W]
        policy_logits = p.flatten(1)  # [B, H*W]

        # value head
        v = self.relu(self.value_bn1(self.value_conv1(x)))  # [B, 64, H, W]
        v = self.relu(self.value_bn2(self.value_conv2(v)))  # [B, 32, H, W]
        v = F.adaptive_avg_pool2d(v, 1)  # [B, 32, 1, 1]
        v = v.flatten(1)  # [B, 32]
        v = self.relu(self.value_fc1(v))  # [B, 128]
        value = torch.tanh(self.value_fc2(v))  # [B, 1]

        return policy_logits, value

    def train_model(
            self,
            buffer: ReplayBuffer,
            epochs: int,
            batch_size: int = 128,
            log_interval: int = 100,
            value_coef: float = 0.4,
    ):

        if len(buffer) == 0:
            print("[train] 回放缓冲为空，跳过训练。")
            return

        self.train()

        start = time.time()
        for epoch in range(1, epochs + 1):
            states, target_policies, target_values = buffer.sample(batch_size, self.device)

            # forward
            policy_logits, value_logit = self(states)

            # loss
            log_policy_prob = F.log_softmax(policy_logits, dim=-1)  # todo: need mask?
            policy_loss = -(target_policies * log_policy_prob).sum(dim=-1).mean()  # H(pi, p)
            value_loss = F.mse_loss(value_logit, target_values)
            entropy = - (log_policy_prob * log_policy_prob.exp()).sum(dim=-1).mean()
            loss = policy_loss + value_coef * value_loss

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            # grad norm for metric only
            grad_norm = clip_grad_norm_(self.parameters(), max_norm=float("inf"))
            self.optimizer.step()
            self.global_step += 1

            with torch.no_grad():
                value_mae = (value_logit - target_values).abs().mean()
                # target_policy entropy
                safe_log = torch.where(target_policies > 0, target_policies.log(), torch.zeros_like(target_policies))
                target_entropy = - (target_policies * safe_log).sum(dim=-1).mean()

            if epoch % log_interval == 0:
                log_dict = {
                    "train/loss_total": round(loss.item(), 4),
                    "train/loss_policy": round(policy_loss.item(), 4),
                    "train/loss_value": round(value_loss.item(), 4),
                    "train/policy_entropy": round(entropy.item(), 4),
                    "train/target_entropy": round(target_entropy.item(), 4),
                    "train/value_mae": round(value_mae.item(), 4),
                    "train/grad_norm": round(float(grad_norm.item()), 4),
                    "train/step": self.global_step,
                }
                wandb.log(log_dict)
                print(f"[Train-{self.global_step}] {log_dict}")

        print(f"[Train] consume: {time.time() - start: .2f} sec, epochs: {epochs}")
        self.eval()

    def save_checkpoint(self, path: str):
        ckpt = {
            "model": self.state_dict(),
            "opt": self.optimizer.state_dict(),
            "global_step": int(self.global_step),
        }
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.load_state_dict(ckpt["model"], strict=False)
        if "opt" in ckpt:
            self.optimizer.load_state_dict(ckpt["opt"])
        self.global_step = int(ckpt.get("global_step", 0))

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
