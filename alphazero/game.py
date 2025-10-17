import torch
import numpy as np

Move = tuple[int, int]

# 四个方向向量：水平(1,0)、竖直(0,1)、主对角(1,1)、副对角(1,-1)
directions = [(1, 0), (0, 1), (1, 1), (1, -1)]


class Game:
    size = 9  # 棋盘边长
    win_num = 5  # 连子数

    def __init__(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)  # 0=空, 1/2=玩家
        self.empty_space = self.board.size

    @staticmethod
    def is_legal_move(i: int, j: int) -> bool:
        return 0 <= i < Game.size and 0 <= j < Game.size

    def feasible_moves(self) -> list[Move]:
        ii, jj = np.where(self.board == 0)
        return list(zip(ii.tolist(), jj.tolist()))

    def apply(self, move: Move, player: int) -> bool:
        i, j = move
        if not Game.is_legal_move(i, j) or self.board[i, j] != 0:
            return False

        self.board[i, j] = player
        self.empty_space -= 1
        return True

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
                    return 1

        return 2 if self.empty_space <= 0 else 0

    def get_state(self, player: int) -> torch.Tensor:
        """
        [2,H,W]：通道0=player的子，通道1=对手的子。
        """
        out = torch.zeros(2, Game.size, Game.size, dtype=torch.float)
        board = torch.from_numpy(self.board)
        out[0] = (board == player).float()
        out[1] = (board == 3 - player).float()
        return out

    def clone(self) -> "Game":
        g = Game()
        g.empty_space = int(self.empty_space)
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
