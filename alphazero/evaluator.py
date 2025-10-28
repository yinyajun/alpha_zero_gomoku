import time
import torch
import queue
import threading
import numpy as np
import torch.nn.functional as F

from model import Alpha0Module


class SingleEvaluator:
    def __init__(self, model):
        self.model = model

    def predict(self, state, mask) -> tuple[np.ndarray, float]:
        state = state.unsqueeze(0)  # [1,2,h,w]

        with torch.no_grad():
            state = state.to(self.model.device)
            policy_out, value_out = self.model(state)  # logits: [1,A], values: [1,1]

        policy_out = policy_out[0].detach().cpu()
        value_out = value_out.detach().cpu()
        policy_out = policy_out.masked_fill(mask, -1e9)
        policy_out = F.softmax(policy_out, dim=-1)
        # logits: [1,A], values: [1,1]
        return policy_out.numpy().astype(np.float32), float(value_out.item())


class BatchEvaluator:
    """
    达到 batch_size 或超时 timeout_ms，才真正计算
    """

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 max_batch_size: int = 1,
                 max_timeout_ms: float = 0.1):
        self.model = model  # .to(device).eval()
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_timeout_ms = max_timeout_ms

        self.q = queue.Queue()

        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, state: torch.Tensor, mask: torch.Tensor):
        """
        异步计算；返回 future（queue.Queue(maxsize=1)）
        """
        if state.device.type == "cpu" and not state.is_pinned() and self.device.type != "cpu":
            state = state.pin_memory()

        fut = queue.Queue(maxsize=1)
        self.q.put((state, mask, fut))
        return fut  # fut.get() -> (policy_prob [A], value [])

    def close(self):
        self._stop.set()
        self._worker.join(timeout=1.0)

    @torch.inference_mode()
    def _forward(self, state: torch.Tensor, mask: torch.Tensor):  # [B, 2, h, w] / [B, hw] or [hw]
        state = state.to(self.model.device, non_blocking=self.device.type != "cpu")

        policy_out, value_out = self.model(state)  # logits: [B,A], values: [B,1]
        policy_out = policy_out.detach().cpu()
        value_out = value_out.detach().cpu()
        policy_out = policy_out.masked_fill(mask, -1e9)  # 屏蔽非法动作
        policy_out = F.softmax(policy_out, dim=-1)
        return policy_out, value_out.squeeze(-1)  # logits: [B,A], values: [B]

    def _loop(self):
        while not self._stop.is_set():
            try:
                first = self.q.get(timeout=0.1)  # 等到首个请求
            except queue.Empty:
                continue

            batch = [first]
            deadline = time.perf_counter() + self.max_timeout_ms / 1000.0

            # 阻塞式按剩余时间取样本，零自旋
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self.q.get(timeout=remaining)
                    batch.append(item)
                except queue.Empty:
                    break

            xs = [x for (x, _, _) in batch]
            ms = [m for (_, m, _) in batch]
            states = torch.stack(xs, 0).contiguous()  # [B,2,h,w]
            masks = torch.stack(ms, 0).contiguous()  # [B, hw]

            policies, values = self._forward(states, masks)  # logits: [B,A], values: [B]

            # 逐个回填
            for i, (_, __, fut) in enumerate(batch):
                policy_out = policies[i].numpy().astype(np.float32)  # [A]
                value_out = float(values[i].item())  # []
                fut.put((policy_out, value_out))


def build_pv_fn(model: Alpha0Module):
    return SingleEvaluator(model).predict


def build_pv_fn_batch(model: Alpha0Module, max_batch_size: int = 1, max_timeout_ms: float = 0.1):
    evaluator = BatchEvaluator(model, model.device, max_batch_size=max_batch_size, max_timeout_ms=max_timeout_ms)

    def predict(state: torch.Tensor, mask: torch.Tensor):
        return evaluator.submit(state, mask).get()

    return predict


