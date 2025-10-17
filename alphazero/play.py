import time
import pygame
import random
from abc import abstractmethod

from game import Game, Move
from mcts import MCTSNode, Alpha0Module, mcts_build
from render import Renderer, edge, cell_size


class PlayPolicy:
    @staticmethod
    def wait_click():
        while True:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if e.type == pygame.MOUSEBUTTONDOWN:
                return e.pos
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:  # 按下 R 键
                    return "R"  # 或者返回一个特殊标志表示“重开”

    @abstractmethod
    def select_move(self, game: Game, player: int, root: MCTSNode = None) -> tuple[Move, MCTSNode]:
        pass


class HumanClickPolicy(PlayPolicy):
    """鼠标点哪下哪：只负责把屏幕点击坐标转换为棋盘坐标并返回。"""

    def select_move(self, game: Game, player: int, root: MCTSNode = None):
        while True:
            pos = HumanClickPolicy.wait_click()
            if not isinstance(pos, tuple):
                continue
            pos = (pos[0] - edge, pos[1] - edge)
            move = (int(pos[0] / cell_size), int(pos[1] / cell_size))
            # 不直接 apply，这里只返回 move；合法性交给外层处理（或这里先做校验）
            if game.is_legal_move(move[0], move[1]) and game.board[move[0], move[1]] == 0:
                return move, root


class RandomPolicy(PlayPolicy):
    """随机合法落子（用于 Render 自检）。"""

    def __init__(self, pause_ms: int = 800):
        self.pause_ms = pause_ms

    def select_move(self, game: Game, player: int, root: MCTSNode = None):
        pygame.time.wait(self.pause_ms)
        moves = game.feasible_moves()
        return random.choice(moves), root


class MCTSPolicy(PlayPolicy):
    """MCTS 策略：负责调用 MCTS 搜索，并返回结果。"""

    def __init__(
            self,
            model: Alpha0Module,
            iterations: int = 300,
            c: float = 0.3,
            noise_eps: float = 0.0,
            dirichlet_alpha: float = 0.3,
            pause_ms: int = 500
    ):
        self.model = model
        self.iterations = iterations
        self.c = c
        self.noise_eps = noise_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.pause_ms = pause_ms

    def select_move(self, game: Game, player: int, root: MCTSNode = None) -> tuple[Move, MCTSNode]:
        start = time.time()
        root = mcts_build(
            game, self.model, player,
            iterations=self.iterations, c=self.c,
            noise_eps=self.noise_eps, dirichlet_alpha=self.dirichlet_alpha
        )
        move, _ = root.select_move()
        consume_ms = int(1000 * (time.time() - start))
        pause_ms = self.pause_ms - consume_ms
        if pause_ms > 0:
            pygame.time.wait(pause_ms)

        return move, root


def play(g: Game, player1: PlayPolicy, player2: PlayPolicy):
    end = 0
    step = 0
    player = 1
    game = g.clone()

    render = Renderer()
    policies = {
        1: player1,
        2: player2
    }
    root = None
    render.draw(game, root, player, end)

    while not end:
        step += 1
        policy = policies[player]
        move, root = policy.select_move(game, player, root)

        ok = game.apply(move, player)
        if not ok:
            continue

        end = game.check_win_from(move, player)  # （0 未终, 1 胜, 2 平）

        if end > 0:
            render.draw(game, root, player, end)
            policy.wait_click()
            play(g, player1, player2)
            break

        player = 3 - player
        render.draw(game, root, player, end)




if __name__ == '__main__':
    # # Human vs Random
    # play(Game(), HumanClickPolicy(), RandomPolicy())

    # # Human vs MCTS
    # play(Game(), HumanClickPolicy(), MCTSPolicy(Alpha0Module()))

    # MCTS vs Human
    play(Game(), MCTSPolicy(Alpha0Module()), HumanClickPolicy())
