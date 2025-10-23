import math
import numpy as np
import torch
from typing import Optional

from game import Game, Nobody, Move


class TreeNode:

    def __init__(self, prior: float, parent: Optional["TreeNode"] = None):
        self.parent = parent
        self.children = {}  # move -> TreeNode

        # 统计量
        self.P: Optional[float] = prior
        self.N = 0  # N(s,a) 动作访问次数
        self.W = 0.0  # W(s,a) 累计价值总和
        self.Q = 0.0  # Q(s,a) 平均价值

    def is_terminal(self, game: Game):
        return game.is_end is True

    def is_fully_expanded(self):
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

    def evaluate(self, game: Game, pv_fn):
        if self.is_terminal(game):
            priors = np.zeros(Game.size * Game.size, dtype=np.float32)
            value = 0.0 if game.winner == Nobody else -1.0  # 上一手胜 => 我方输 => -1；平 => 0
            return priors, value

        # 非终局，跑模型计算，value的值域[-1, 1]
        state = game.get_state()  # [2, h, w]  cpu
        mask = torch.from_numpy(game.board.flatten() != 0).bool()  # 非空位的mask
        policy_out, value_out = pv_fn(state, mask)
        return policy_out, value_out

    def expand_all(self, priors, game: Game):
        if len(self.children) > 0:
            return

        for move in game.available_moves():
            p = priors[move[0] * Game.size + move[1]]
            self.children[move] = TreeNode(prior=p, parent=self)

    def backprop(self, v: float):
        node = self
        while node is not None:
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v *= -1  # 父子换手，翻转视角
            node = node.parent

    def play_out(self, game: Game, pv_fn, c_puct: float):
        node = self
        game = game.clone()

        # 1) select
        while not node.is_terminal(game) and node.is_fully_expanded():  # 已经完全展开，并且没有终局，select最佳child
            move, node = node.select(c_puct)
            game.step(move)
        # is_terminal or not expanded

        # 2) roll
        priors, value = node.evaluate(game, pv_fn)

        # 3) expand all
        if not node.is_terminal(game):
            node.expand_all(priors, game)

        # 4) backprop
        node.backprop(value)
