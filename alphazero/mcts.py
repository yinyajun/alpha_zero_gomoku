import math
import time

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from game import Game, Move
from buffer import Sample
from net import Alpha0Module


class MCTSNode:
    """
    显式的四阶段：
      - Selection: 在“完全展开”的节点间，用 PUCT 选择 child
      - Expansion: 在“未完全展开”的节点上，从动作池中取 1 个动作扩展为新子
      - Rollout:   从该子节点开始随机模拟到终局（这里替换为用价值估计）
      - Backprop:  将结果回传
    """

    def __init__(
            self,
            state: Game,
            player: int,
            model: Alpha0Module,
            last_move: Optional[Move] = None,
            parent: Optional["MCTSNode"] = None,
    ):
        self.state = state  # 确保上一手last_move已经落子
        self.player = player  # 当前轮次
        self.last_move: Move = last_move  # 上一手，
        # 因为MCTS是统计边，而实现是将统计放到子节点中，所以要记一下通过上一手来到该子节点
        self.parent = parent
        self.children: list[MCTSNode] = []

        # lazy calculate
        self.model = model
        self.priors: Optional[np.ndarray] = None
        self.value: Optional[float] = None  # 当前棋面的价值, node.player 视角

        self.visits = 0
        self.wins = 0.0  # 累计价值, node.player 视角
        # player - 1/2; last_player: 2/1
        self.end: int = self.state.check_win_from(last_move, 3 - player)  # 上一手的棋局结果；
        self.untried_moves: list[Move] = []
        if not self.is_terminal():
            self.untried_moves = self.state.feasible_moves()

    def is_terminal(self):
        return self.end != 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def ensure_priors_and_value(self):
        # 终局直接返回，非终局用模型的价值预估
        if self.priors is not None:  # 已经计算过
            return

        if self.is_terminal():
            self.priors = np.zeros(Game.size * Game.size, dtype=np.float32)
            self.value = -1.0 if self.end == 1 else 0.0  # 上一手胜 => 我方输 => -1；平 => 0
            return

        # 非终局，跑模型计算，value的值域[-1, 1]
        s = self.state.get_state(self.player).unsqueeze(0)
        k = time.time_ns()

        with torch.no_grad():
            policy_out, value_out = self.model(s.to(self.model.device))
        # print(66666666, time.time_ns() - k)

        policy_out = policy_out[0].detach().to("cpu")
        value_out = value_out.item()
        mask = torch.from_numpy(self.state.board.flatten() != 0).bool()
        policy_out = policy_out.masked_fill(mask, -1e9)
        policy_out = F.softmax(policy_out, dim=-1).numpy().astype(np.float32)

        self.priors = policy_out
        self.value = float(value_out)

    def best_child(self, c: float) -> "MCTSNode":
        # PUCT: Q + c * prior * sqrt(N) / (1+n)
        N = max(1, self.visits)
        self.ensure_priors_and_value()

        def puct(ch: "MCTSNode") -> float:
            p = float(self.priors[ch.last_move[0] * Game.size + ch.last_move[1]])
            exploit = - (ch.wins / ch.visits) if ch.visits > 0 else 0.0  # Q_child(parent视角) = - Q_child(child视角)
            explore = p * math.sqrt(N) / (1.0 + ch.visits)
            return exploit + c * explore

        return max(self.children, key=puct)

    def expand(self, use_prior: bool = False) -> "MCTSNode":
        """
        只在expand的过程中落子
        """
        idx = np.random.randint(len(self.untried_moves))

        if use_prior:
            self.ensure_priors_and_value()
            weights = np.array([self.priors[move[0] * Game.size + move[1]] for move in self.untried_moves])
            weights_sum = weights.sum()
            if weights_sum > 0:
                weights /= weights_sum
                idx = np.random.choice(len(self.untried_moves), p=weights)

        move = self.untried_moves.pop(idx)
        game = self.state.clone()
        game.apply(move, self.player)  # 当前player执行的move得到子节点

        child = MCTSNode(game, 3 - self.player, self.model, move, self)
        self.children.append(child)
        return child

    def rollout(self) -> float:
        """
        在alpha0中，不进行随机模拟，用价值估计代替
        """
        self.ensure_priors_and_value()

        return float(self.value)

    def backprop(self, result: float):
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            result *= -1  # 父子换手，翻转视角
            node = node.parent

    def add_noise(self,
                  noise_eps: float = 0.25,
                  dirichlet_alpha: float = 0.1):
        """
        只有根节点才要加噪声
        根噪声：P' = (1-ε)P + ε Dir(α)
        仅对合法位注入，再散射回全局
        """
        if self.is_terminal():
            return

        legal_mask = self.state.board.flatten() == 0
        legal_idx = np.flatnonzero(legal_mask)
        if legal_idx.size == 0:  # 无合法位
            return

        self.ensure_priors_and_value()
        priors = self.priors

        noise_legal = np.random.dirichlet([dirichlet_alpha] * legal_idx.size).astype(np.float32)  # sum=1 on legal
        noise_full = np.zeros_like(priors, dtype=np.float32)
        noise_full[legal_idx] = noise_legal

        mixed = (1.0 - noise_eps) * priors + noise_eps * noise_full
        mixed[~legal_mask] = 0.0  # 额外保险：非法位归零 + 归一化
        s = mixed.sum()
        self.priors = mixed / s if s > 0 else priors

    def to_sample(self) -> Sample:
        search_rate = torch.zeros((Game.size, Game.size), dtype=torch.float)
        assert self.children
        for ch in self.children:
            search_rate[ch.last_move[0], ch.last_move[1]] = ch.visits / self.visits if self.visits > 0 else 0.0

        win_rate = torch.tensor(0.0, dtype=torch.float)  # 等待self_play的时候回填真实结果

        return Sample(
            state=self.state.get_state(self.player),  # [2, size, size] float tensor
            search_rate=search_rate,  # [size, size] float tensor
            win_rate=win_rate,  # [] float tensor
        )

    def select_move(self, tau: float = 0.0):
        # 利用探索结果，来获取最佳动作
        if not self.children:
            return None, []

        visits = np.array([ch.visits for ch in self.children], dtype=np.float32)  # 利用visits而不是winrate
        if tau <= 0:
            idx = int(np.argmax(visits))
        else:
            probs = visits ** (1 / tau)
            probs_sum = np.sum(probs)
            if probs_sum <= 0:
                idx = np.random.randint(len(self.children))
            else:
                probs /= np.sum(probs)
                idx = np.random.choice(len(self.children), p=probs)

        chosen = self.children[idx].last_move
        stats = [(ch.last_move, ch.visits, ch.wins / ch.visits if ch.visits > 0 else 0.0) for ch in self.children]
        return chosen, stats

    def search(self, iterations: int = 500, c: float = 0.3):

        t1, t2, t3, t4 = 0, 0, 0, 0
        for _ in range(iterations):
            start = time.time()

            # 1) Selection
            node = self
            while not node.is_terminal() and node.is_fully_expanded():  # 已经完全展开，并且没有终局，select最佳child
                node = node.best_child(c)

            t1 += (time.time() - start)

            start = time.time()
            # 2) Expansion
            if not node.is_terminal():  # is_terminal() or not is_fully_expanded()
                node = node.expand()
            t2 += (time.time() - start)

            start = time.time()
            # 3) Rollout
            result = node.rollout()
            t3 += (time.time() - start)

            start = time.time()
            # 4) Backprop
            node.backprop(result)
            t4 += (time.time() - start)


def mcts_build(
        game: Game,
        model: Alpha0Module,
        player: int,
        iterations: int = 300,
        c: float = 0.8,
        noise_eps: float = 0.25,
        dirichlet_alpha: float = 0.3
):
    root = MCTSNode(game.clone(), player, model)
    if noise_eps > 0:
        root.add_noise(noise_eps, dirichlet_alpha)

    root.search(iterations, c)

    return root
