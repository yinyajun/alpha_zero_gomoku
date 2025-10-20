import time
import torch
import queue
import threading
import torch.nn.functional as F


class DefaultPolicyValueFn:
    def __init__(self, model):
        self.model = model

    def run(self, state, mask):
        state = state.unsqueeze(0)  # [1,2,h,w]
        k = time.time()

        with torch.no_grad():
            policy_out, value_out = self.model(state.to(self.model.device))

        print(44444444444, time.time() - k)

        policy_out = policy_out[0].detach().cpu()
        value_out = value_out.detach().cpu()
        policy_out = policy_out.masked_fill(mask, -1e9)
        policy_out = F.softmax(policy_out, dim=-1)
        return policy_out, value_out


class Evaluator:
    """
    达到 batch_size 或超时 timeout_ms，才真正计算
    """

    def __init__(self,
                 model: torch.nn.Module,
                 # device: str = "cuda",
                 batch_size: int = 1,
                 timeout_ms: float = 0.1):
        self.model = model  # .to(device).eval()
        # self.dev = device
        self.batch_size = batch_size
        self.t_ms = timeout_ms

        self.q = queue.Queue()

        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, state: torch.Tensor, mask: torch.Tensor):
        """
        异步计算；返回 future（queue.Queue(maxsize=1)）
        """
        if state.device.type == "cpu" and not state.is_pinned() and self.model.device.type != "cpu":
            state = state.pin_memory()
        if mask.device.type == "cpu" and not mask.is_pinned() and self.model.device.type != "cpu":
            mask = mask.pin_memory()

        fut = queue.Queue(maxsize=1)
        self.q.put((state, mask, fut))
        return fut  # fut.get() -> (policy_prob [A], value [])

    def close(self):
        self._stop.set()
        self._worker.join(timeout=1.0)

    @torch.inference_mode()
    def _forward(self, state: torch.Tensor, mask: torch.Tensor):  # [B, 2, h, w] / [B, hw] or [hw]
        non_blocking = self.model.device.type != "cpu"
        state = state.to(self.model.device, non_blocking=non_blocking)
        mask = mask.to(self.model.device, non_blocking=non_blocking)

        logits, values = self.model(state)  # logits: [B,A], values: [B,1]
        logits = logits.masked_fill(mask, -1e9)  # 屏蔽非法动作
        policy = torch.softmax(logits, dim=-1)
        return policy.detach().cpu(), values.squeeze(-1).detach().cpu()  # [B, A] / [B]

    def inference_one(self, state: torch.Tensor, mask: torch.Tensor):
        state = state.unsqueeze(0)  # [1, 2, h, w]

        # k = time.time()
        with torch.no_grad():
            policy_out, value_out = self.model(state.to(self.model.device))
        # print(2222222, time.time() - k)

        policy_out = policy_out[0].detach().to("cpu")
        value_out = value_out
        policy_out = policy_out.masked_fill(mask, -1e9)
        policy_out = F.softmax(policy_out, dim=-1)
        return policy_out, value_out
        # policies, values =  self._forward(state, mask) # [1, A] / [1]
        # policy= policies[0]
        # value = values[0]
        # return policy,value # [A] / []

    def _loop(self):
        while not self._stop.is_set():
            try:
                first = self.q.get(timeout=0.1)  # 等到首个请求
            except queue.Empty:
                continue

            batch = [first]
            deadline = time.perf_counter() + self.t_ms / 1000.0

            # 阻塞式按剩余时间取样本，零自旋
            while len(batch) < self.batch_size:
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

            policies, values = self._forward(states, masks)
            # 逐个回填
            for i, (_, __, fut) in enumerate(batch):
                fut.put((policies[i], values[i]))  # [A] / []


def build_pv_fn(evaluator: Evaluator):
    def predict(state: torch.Tensor, mask: torch.Tensor):
        assert state.device.type == "cpu", f"state on {state.device}"
        assert mask.device.type == "cpu", f"mask on {mask.device}"
        fut = evaluator.submit(state, mask)
        return fut.get()

    return predict


def build_pv_fn2(evaluator: Evaluator):
    def predict(state: torch.Tensor, mask: torch.Tensor):
        return evaluator.inference_one(state, mask)

    return predict
