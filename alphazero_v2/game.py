import torch
import numpy as np

Move = tuple[int, int]
# 四个方向向量：水平(1,0)、竖直(0,1)、主对角(1,1)、副对角(1,-1)
directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

# 棋盘落子状态
UnPlayed = 0  # 未落子
Player1 = 1  # 玩家1落子
Player2 = 2  # 玩家2落子

# 棋盘终结状态
Unfinished = 0  # 未终局
Win = 1  # 有人获胜
Draw = 2  # 平局


class Game:
    size = 7  # 棋盘边长
    win_num = 4  # 连子数

    def __init__(self, first_player: int = Player1):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)  # 0=空, 1/2=玩家
        self.move_count = 0  # 落子数
        self.player = first_player  # 当前玩家
        self.last_move = None  # 上一手
        self.result = Unfinished  # 当前棋盘终结状态

    @staticmethod
    def is_legal_move(i: int, j: int) -> bool:
        return 0 <= i < Game.size and 0 <= j < Game.size

    def feasible_moves(self) -> list[Move]:  # 可落子位置
        ii, jj = np.where(self.board == 0)
        return list(zip(ii.tolist(), jj.tolist()))

    def do_move(self, move: Move):
        i, j = move
        assert Game.is_legal_move(i, j)
        assert self.board[i, j] == UnPlayed
        self.board[i, j] = self.player
        self.move_count += 1
        self.last_move = move
        self.result = self.check_win_from(move, self.player)
        self.player = 3 - self.player

    def check_win_from(self, move: Move, player: int) -> int:
        """
        上一手 move 已经由 player 落下，仅围绕该点检查胜负。

        返回 end:
            0: 未结束
            1: 刚落子者获胜
            2: 平局（满盘）
        """
        if move is not None:
            i, j = move
            for di, dj in directions:
                cnt = 1
                # 正向
                ii, jj = i + di, j + dj
                while Game.is_legal_move(ii, jj) and self.board[ii, jj] == player:
                    cnt += 1
                    ii += di
                    jj += dj
                # 反向
                ii, jj = i - di, j - dj
                while Game.is_legal_move(ii, jj) and self.board[ii, jj] == player:
                    cnt += 1
                    ii -= di
                    jj -= dj

                if cnt >= self.win_num:
                    return Win

        return Draw if self.move_count >= self.board.size else Unfinished

    def get_state(self) -> torch.Tensor:
        """
        [2,H,W]：通道0=player的子，通道1=对手的子。
        """
        out = torch.zeros(2, Game.size, Game.size, dtype=torch.float)
        board = torch.from_numpy(self.board)
        out[0] = (board == self.player).float()
        out[1] = (board == (3 - self.player)).float()
        return out

    def clone(self) -> "Game":
        g = Game()
        g.move_count = self.move_count
        g.player = self.player
        g.last_move = self.last_move
        g.result = self.result
        g.board = np.copy(self.board)
        return g

    def __repr__(self):
        out = "\n"
        for row in self.board:
            for col in row:
                out += f" {int(col)} "
            out += "\n"
        return out + "\n"


if __name__ == '__main__':
    g = Game()
    print(g)
