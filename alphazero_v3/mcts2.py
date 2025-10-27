import math
import numpy as np
import torch
from typing import Optional

from game import Game, Nobody, Move


class TreeNode:

    def __init__(self, game: Game, prior: float, parent: Optional["TreeNode"] = None):
        # 状态
        self.game = game

        # TreeNode成员
        self.parent = parent
        self.children = {}  # move -> TreeNode

        # 统计量
        self.N = 0  # N(s,a) 动作访问次数
        self.W = 0.0  # W(s,a) 累计价值总和
        self.Q = 0.0  # Q(s,a) 平均价值

        # 估计量
        self.P: Optional[float] = prior  # P(s,a) 下一手动作先验概率（策略网络)
        self.value: Optional[float] = None  # V(s) 当前棋面的价值(来自价值网络的输出)

    def is_terminal(self):
        return self.game.is_end is True

    def is_expanded(self):
        return len(self.children) > 0  # 采用的是“全展开”，有孩子就表示已展开

    def select(self, c_puct: float) -> tuple[Move, "TreeNode"]:
        """
        PUCT: Q + c * P * sqrt(N) / (1+n)
        """
        assert self.children
        parent_sqrt = math.sqrt(self.N + 1e-8)

        def puct(item) -> float:
            move, ch = item
            exploit = - ch.Q  # Q_child(parent视角) = - Q_child(child视角)
            explore = ch.P * parent_sqrt / (1.0 + ch.N)
            return exploit + c_puct * explore

        return max(self.children.items(), key=puct)

    def evaluate(self, pv_fn):
        if self.is_terminal():
            priors = np.zeros(Game.size * Game.size, dtype=np.float32)
            value = 0.0 if self.game.winner == Nobody else -1.0  # 当前行动方在终局节点必然是输家（因为对手刚刚赢）或者平手
            return priors, value

        # 非终局，跑模型计算，value的值域[-1, 1]
        state = self.game.get_state()  # [2, h, w]  cpu
        mask = torch.from_numpy(self.game.board.flatten() != 0).bool()  # 非空位的mask
        policy_out, value_out = pv_fn(state, mask)
        return policy_out, value_out

    def expand_all(self, priors):
        if len(self.children) > 0:
            return

        for move in self.game.available_moves():
            p = priors[move[0] * Game.size + move[1]]
            game = self.game.clone()
            game.step(move)
            self.children[move] = TreeNode(game=game, prior=p, parent=self)

    def backprop(self, v: float):
        node = self
        while node is not None:
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v *= -1  # 父子换手，翻转视角
            node = node.parent

    def play_out(self, pv_fn, c_puct: float):
        node = self

        # 1) select
        while not node.is_terminal() and node.is_expanded():  # 已经完全展开，并且没有终局，select最佳child
            move, node = node.select(c_puct)
        # is_terminal or not expanded

        # 2) rollout
        priors, value = node.evaluate(pv_fn)
        node.value = value

        # 3) expand all
        if not node.is_terminal():
            node.expand_all(priors)

        # 4) backprop
        node.backprop(node.value)


class MCTSTree:
    def __init__(self, game: Game, pv_fn):
        self.root = TreeNode(game=game.clone(), prior=0.0)
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

        priors, value = node.evaluate(pv_fn)  # 除非终局，不会全0
        # 由于prior存储在子节点中，根节点需要先展开才能添加噪声
        node.value = value
        if not node.is_expanded():
            node.expand_all(priors)

        noise_legal = np.random.dirichlet([dirichlet_alpha] * legal_idx.size).astype(np.float32)  # sum=1 on legal
        noise_full = np.zeros_like(priors, dtype=np.float32)
        noise_full[legal_idx] = noise_legal

        mixed = (1.0 - noise_eps) * priors + noise_eps * noise_full
        mixed[~legal_mask] = 0.0  # 额外保险：非法位归零 + 归一化
        s = mixed.sum()
        priors = mixed / s if s > 0 else priors

        assert len(node.children) == legal_idx.size
        for move, ch in node.children.items():
            ch.P = priors[move[0] * Game.size + move[1]]

    def search_move(
            self,
            iterations: int,
            c_puct: float,
            warm_moves: int,
            tau: float,
            noise_moves: int,
            noise_eps: float,
            dirichlet_alpha: float,
    ) -> Move:
        assert iterations > 0

        if self.root.game.move_count < noise_moves:
            self.add_noise(noise_eps, dirichlet_alpha)

        for _ in range(iterations):
            self.root.play_out(pv_fn=self.pv_fn, c_puct=c_puct)

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
        self.root = TreeNode(new_game, 0.0)

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
