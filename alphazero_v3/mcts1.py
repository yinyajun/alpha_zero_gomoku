import math
import torch
import numpy as np
from typing import Optional

from game import Game, Move, Nobody


class TreeNode:
    """
    显式的四阶段：
      - Selection: 在“完全展开”的节点间，用 PUCT 选择 child
      - Expansion: 在“未完全展开”的节点上，从动作池中取 1 个动作扩展为新子
      - Rollout:   从该子节点开始随机模拟到终局（这里替换为用价值估计）
      - Backprop:  将结果回传
    """

    def __init__(self, game: Game, parent: Optional["TreeNode"] = None):
        # 状态
        self.game = game

        # TreeNode成员
        self.parent = parent
        self.children = {}  # move -> TreeNode
        self.untried_moves = game.available_moves()

        # 统计量
        self.N = 0  # N(s,a) 动作访问次数
        self.W = 0.0  # W(s,a) 累计价值总和
        self.Q = 0.0  # Q(s,a) 平均价值

        # 估计量
        self.priors: Optional[np.ndarray] = None  # P(s,a) 下一手动作先验概率（策略网络)
        self.value: Optional[float] = None  # V(s) 当前棋面的价值(来自价值网络的输出)

    def is_terminal(self):
        return self.game.is_end is True

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0  # 每次展开一个子节点，直到没有可扩展的子节点

    def ensure_priors_and_value(self, pv_fn):
        # 终局直接返回，非终局用模型的价值预估
        if self.priors is not None:  # 已经计算过
            return

        if self.is_terminal():
            self.priors = np.zeros(Game.size * Game.size, dtype=np.float32)
            self.value = 0.0 if self.game.winner == Nobody else -1.0  # 当前行动方在终局节点必然是输家（因为对手刚刚赢）或者平手
            return

        # 非终局，跑模型计算，value的值域[-1, 1]
        state = self.game.get_state()  # [2, h, w]  cpu
        mask = torch.from_numpy(self.game.board.flatten() != 0).bool()  # 非空位的mask
        self.priors, self.value = pv_fn(state, mask)

    def select(self, c_puct: float) -> "TreeNode":
        """
        PUCT: Q + c * P * sqrt(N) / (1+n)
        """
        assert self.priors is not None
        parent_sqrt = math.sqrt(self.N + 1e-8)

        def puct(item) -> float:
            move, ch = item
            P = float(self.priors[move[0] * Game.size + move[1]])  # P(s,a)
            exploit = - ch.Q  # Q_child(parent视角) = - Q_child(child视角)
            explore = P * parent_sqrt / (1.0 + ch.N)
            return exploit + c_puct * explore

        return max(self.children.items(), key=puct)

    def expand(self) -> "TreeNode":
        """
        只在expand的过程中落子
        """
        move_idx = np.random.randint(len(self.untried_moves))  # 随机扩展
        if self.priors is not None:  # 根据prior采样扩展
            # self.ensure_priors_and_value()
            weights = np.array([self.priors[move[0] * Game.size + move[1]] for move in self.untried_moves])
            weights_sum = weights.sum()
            if weights_sum > 0:
                weights /= weights_sum
                move_idx = np.random.choice(len(self.untried_moves), p=weights)

        move = self.untried_moves.pop(move_idx)
        game = self.game.clone()
        game.step(move)

        child = TreeNode(game=game, parent=self)
        self.children[move] = child
        return child

    def rollout(self) -> float:
        """
        在alpha0中，不进行随机模拟，用价值估计代替
        """
        assert self.value is not None
        return self.value

    def backprop(self, v: float):
        node = self
        while node is not None:
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v *= -1  # 父子换手，翻转视角
            node = node.parent

    def play_out(self, pv_fn, c_puct: float, expand_with_prior: bool = True):
        node = self

        # 1) Selection
        while not node.is_terminal() and node.is_fully_expanded():  # 已经完全展开，并且没有终局，select最佳child
            node.ensure_priors_and_value(pv_fn)
            _, node = node.select(c_puct)

        # 2) Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            if expand_with_prior:
                node.ensure_priors_and_value(pv_fn)
            node = node.expand()

        # 3) Rollout
        node.ensure_priors_and_value(pv_fn)
        v = node.rollout()

        # 4) Backprop
        node.backprop(v)


class MCTSTree:
    def __init__(self, game: Game, pv_fn):
        self.root = TreeNode(game=game)
        self.pv_fn = pv_fn

    def add_noise(self, noise_eps: float, dirichlet_alpha: float):
        """
        只有根节点才要加噪声: 根噪声：P' = (1-ε)P + ε Dir(α)
        仅对合法位注入，再散射回全局
        """
        node = self.root
        pv_fn = self.pv_fn
        if node.is_terminal():
            return

        legal_mask = node.game.board.flatten() == 0
        legal_idx = np.flatnonzero(legal_mask)
        if legal_idx.size == 0:  # 无合法位
            return

        node.ensure_priors_and_value(pv_fn)
        priors = node.priors  # 除非终局，不会全0

        noise_legal = np.random.dirichlet([dirichlet_alpha] * legal_idx.size).astype(np.float32)  # sum=1 on legal
        noise_full = np.zeros_like(priors, dtype=np.float32)
        noise_full[legal_idx] = noise_legal

        mixed = (1.0 - noise_eps) * priors + noise_eps * noise_full
        mixed[~legal_mask] = 0.0  # 额外保险：非法位归零 + 归一化
        s = mixed.sum()
        node.priors = mixed / s if s > 0 else priors

    def search_move(
            self,
            iterations: int,
            c_puct: float,
            warm_moves: int,
            tau: float,
            noise_moves: int,
            noise_eps: float,
            dirichlet_alpha: float,
            expand_with_prior: bool = True,
    ) -> Move:
        assert iterations > 0

        if self.root.game.move_count < noise_moves:
            self.add_noise(noise_eps, dirichlet_alpha)

        for _ in range(iterations):
            self.root.play_out(pv_fn=self.pv_fn, c_puct=c_puct, expand_with_prior=expand_with_prior)

        # get move
        assert self.root.children
        child_list = list(self.root.children.values())
        visits = np.array([ch.N for ch in child_list], dtype=np.float32)

        if self.root.game.move_count >= warm_moves:
            tau = 0.0

        if tau <= 0:
            idx = int(np.argmax(visits))
        else:
            probs = visits ** (1 / tau)
            probs_sum = np.sum(probs)
            assert probs_sum > 0
            probs /= probs_sum
            idx = np.random.choice(len(child_list), p=probs)

        chosen = child_list[idx].game.last_move
        return chosen

    def reuse(self, new_game: Game):
        """
        在对局进行中复用搜索树
        """
        assert new_game.move_count - self.root.game.move_count == 1

        new_game = new_game.clone()
        last_move = new_game.last_move

        # 尝试在现有孩子里找到这步棋
        for ch in self.root.children.values():
            if ch.game.last_move == last_move:
                ch.parent = None
                self.root.children = {}
                self.root = ch
                return

        # 没找到: 新建根
        self.root = TreeNode(new_game)

    @property
    def search_prob(self):
        assert self.root.children
        total = sum(ch.N for ch in self.root.children.values())
        assert total > 0
        pi = torch.zeros((Game.size, Game.size), dtype=torch.float)
        for ch in self.root.children.values():
            last_move = ch.game.last_move
            pi[last_move[0], last_move[1]] = ch.N / total
        return pi
