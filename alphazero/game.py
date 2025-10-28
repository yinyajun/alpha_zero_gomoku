import torch
import numpy as np

Move = tuple[int, int]
directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 四个方向向量：水平(1,0)、竖直(0,1)、主对角(1,1)、副对角(1,-1)

# 棋盘落子状态
Nobody = 0  # 无人
Player1 = 1  # 玩家1落子
Player2 = 2  # 玩家2落子


class Game:
    size = 9  # 棋盘边长
    win_num = 5  # 连子数

    def __init__(self, first_player: int = Player1):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)  # 0=空, 1/2=玩家
        self.move_count = 0  # 落子数
        self.player = first_player  # 当前玩家
        self.last_move = None  # 上一手
        self.is_end = False  # 是否结束
        self.winner = Nobody  # 胜者

    @staticmethod
    def is_legal_move(i: int, j: int) -> bool:
        return 0 <= i < Game.size and 0 <= j < Game.size

    def available_moves(self) -> list[Move]:  # 可落子位置
        if self.is_end:
            return []
        ii, jj = np.where(self.board == Nobody)
        return list(zip(ii.tolist(), jj.tolist()))

    def step(self, move: Move):
        i, j = move
        assert Game.is_legal_move(i, j)
        assert self.board[i, j] == Nobody
        self.board[i, j] = self.player
        self.move_count += 1
        self.last_move = move
        self.is_end, self.winner = self.is_game_over(move, self.player)
        self.player = 3 - self.player

    def is_game_over(self, move: Move, player: int) -> (bool, int):
        """
        player落下move后的棋面，围绕该手来判断棋面是否结束
        上一手 move 已经由 player 落下，仅围绕该点检查胜负。

        返回:
            是否结束
            winner是谁
        """
        if self.is_end:
            return self.is_end, self.winner

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
                    return True, player

        if self.move_count >= self.board.size:  # 平局
            return True, Nobody

        return False, Nobody

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
        g.is_end = self.is_end
        g.winner = self.winner
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
