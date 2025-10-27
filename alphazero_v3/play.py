import time
import torch
import random
from tqdm import tqdm
from typing import Optional

from game import Game, Player1, Player2, Move
from mcts1 import MCTSTree


class BasePlayer:
    def __init__(self):
        self.game: Game = Game()

    def load(self, game: Game) -> "BasePlayer":
        self.game = game
        return self

    def search(self) -> Move:  # myself
        pass

    def observe(self, game: Game):  # opponent_move
        pass


class MCTSPlayer(BasePlayer):
    """
     MCTS 玩家
     """

    def __init__(
            self,
            pv_fn,
            iterations: int,
            c_puct: float,
            warm_moves: int,
            tau: float,
            noise_moves: int,
            noise_eps: float,
            dirichlet_alpha: float,
            tree_cls=MCTSTree
    ):
        super(MCTSPlayer, self).__init__()
        self.tree: Optional["MCTSTree"] = None
        self.pv_fn = pv_fn
        self.iterations = iterations
        self.c_puct = c_puct
        self.warm_moves = warm_moves
        self.tau = tau
        self.noise_moves = noise_moves
        self.noise_eps = noise_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.tree_cls = tree_cls

    def load(self, game: Game) -> "MCTSPlayer":
        super(MCTSPlayer, self).load(game)
        self.tree: "MCTSTree" = self.tree_cls(game, self.pv_fn)
        return self

    def search_move(self):
        move = self.tree.search_move(
            iterations=self.iterations, c_puct=self.c_puct,
            warm_moves=self.warm_moves, tau=self.tau,
            noise_moves=self.noise_moves, noise_eps=self.noise_eps, dirichlet_alpha=self.dirichlet_alpha,
        )
        assert move is not None
        return move

    def search(self) -> Move:
        # print("ssss1", self.tree.root.N, self.tree.root.game.last_move)
        move = self.search_move()
        # print("ssss2", self.tree.root.N, move)
        # for ch in self.tree.root.children:
        #     print("\tssss2-ch", ch.game.last_move, ch.N)
        return move

    def observe(self, game: Game):
        # print("oooo1", self.tree.root.N, self.tree.root.game.last_move)
        self.tree.reuse(game)
        # print("oooo2", move, self.tree.root.N, self.tree.root.game.last_move)


class RandomPlayer(BasePlayer):
    """随机合法落子，测试用。"""

    def search(self) -> Move:
        time.sleep(0.5)
        moves = self.game.available_moves()
        move = random.choice(moves)
        return move


class HumanPlayer(BasePlayer):
    """鼠标点哪下哪：只负责把屏幕点击坐标转换为棋盘坐标并返回。"""

    def __init__(self, render: "Renderer"):
        from render import cell_size, edge
        super(HumanPlayer, self).__init__()
        self.render = render
        self.cell_size = cell_size
        self.edge = edge

    def search(self) -> Move:
        while True:
            pos = self.render.wait_click()
            pos = (pos[0] - self.edge, pos[1] - self.edge)
            move = (int(pos[0] / self.cell_size), int(pos[1] / self.cell_size))
            if Game.is_legal_move(move[0], move[1]) and self.game.board[move[0], move[1]] == 0:
                return move


def self_play(game: Game, player: MCTSPlayer, force_center: bool = False, enable_tqdm: bool = True):
    """
    自我对弈（收集轨迹）
    """
    game = game.clone()
    step = 0
    player = player.load(game)
    trajectory = dict(s=[], pi=[], z=[], turns=[])  # z: [], float （这里装的是 z ∈ {-1,0,1}）
    pbar = tqdm(desc="self-play", unit="step", disable=not enable_tqdm,
                bar_format="{desc}: [{elapsed_s:.2f}s,{rate_fmt}{postfix}]")

    while not game.is_end:
        step += 1

        if step == 1 and force_center:  # empty board
            center = game.size // 2
            move = (center, center)
        else:
            move = player.search()

            trajectory["s"].append(game.get_state())  # [2, H, W], float
            trajectory["pi"].append(player.tree.search_prob)  # [H, W], float (动作访问分布)
            trajectory["turns"].append(game.player)
        game.step(move)

        player.observe(game)

        pbar.update(1)
        pbar.set_postfix({"step": step, "player": 3 - game.player})

    pbar.close()
    winner = game.winner

    if winner > 0:
        z = [torch.tensor(1.0 if p == winner else -1.0, dtype=torch.float) for p in trajectory["turns"]]
    else:  # draw
        z = [torch.tensor(0.0) for _ in trajectory["turns"]]
    trajectory["z"] = z
    # print(game)

    return step, winner, trajectory


def eval_play(game: Game, player1: MCTSPlayer, player2: MCTSPlayer):
    """
    新旧对弈
    """
    game = game.clone()
    step = 0
    players = {
        Player1: player1.load(game),
        Player2: player2.load(game)
    }

    while not game.is_end:
        step += 1
        player, opponent = players[game.player], players[3 - game.player]

        move = player.search()
        game.step(move)
        player.observe(game)
        opponent.observe(game)

    return step, game.winner


def play_and_render(game: Game, player1: BasePlayer, player2: BasePlayer, render: "Renderer"):
    """
    可视化对弈
    """
    game = game.clone()
    step = 0
    players = {
        Player1: player1.load(game),
        Player2: player2.load(game)
    }
    render.draw_game(game)

    while not game.is_end:
        step += 1

        player, opponent = players[game.player], players[3 - game.player]
        render.draw_mcts_numbers(player.tree if isinstance(player, MCTSPlayer) else None)

        move = player.search()
        game.step(move)

        player.observe(game)
        opponent.observe(game)
        render.draw_game(game)

    # end
    render.wait_click()


def example_human_vs_human():
    from render import Renderer
    g = Game()
    r = Renderer()
    p1 = HumanPlayer(r)
    p2 = HumanPlayer(r)
    while True:
        play_and_render(g, p1, p2, r)


def example_human_vs_random():
    from render import Renderer
    g = Game()
    r = Renderer()
    p1 = HumanPlayer(r)
    p2 = RandomPlayer()
    while True:
        play_and_render(g, p1, p2, r)


def example_human_vs_ai():
    from model import Alpha0Module
    from render import Renderer
    from evaluator import build_pv_fn, build_pv_fn_batch

    g = Game()
    r = Renderer()
    model = Alpha0Module(resume_path="model.pt")
    pv_fn = build_pv_fn(model)

    p1 = HumanPlayer(r)
    p2 = MCTSPlayer(
        pv_fn=pv_fn,
        iterations=600,
        c_puct=0.5,
        warm_moves=6,
        tau=1.0,
        noise_moves=10,
        noise_eps=0.25,
        dirichlet_alpha=0.2,
    )

    while True:
        play_and_render(g, p1, p2, r)


if __name__ == '__main__':
    example_human_vs_human()
    # example_human_vs_ai()
