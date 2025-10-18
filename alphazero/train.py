import time

from tqdm import tqdm
from itertools import chain, zip_longest
from dataclasses import dataclass

from game import Game
from net import Alpha0Module
from mcts import mcts_build
from buffer import ReplayBuffer


def self_play(
        model: Alpha0Module,
        round_num: int,
        iterations: int = 300,
        c: float = 0.3,
        noise_eps: float = 0.25,
        dirichlet_alpha: float = 0.1,
        verbose: bool = False
):
    game = Game()
    player = 1
    end = 0
    step = 0
    buffer = {1: [], 2: []}

    if verbose:
        print(f"\n======= Game {round_num} =======")
    pbar = tqdm(desc="self-play", unit="step", bar_format="{desc}: [{elapsed_s:.2f}s,{rate_fmt}{postfix}]")

    while not end:
        step += 1
        root = mcts_build(game, model, player,
                          iterations=iterations, c=c,
                          noise_eps=noise_eps, dirichlet_alpha=dirichlet_alpha)
        buffer[player].append(root.to_sample())

        # schedule
        tau = 0.0 if step > 8 else 1.0
        force_center = round_num < 200
        if step == 1 and force_center:  # empty board
            center = game.size // 2
            move = (center, center)
        else:
            move, _ = root.select_move(tau)

        if move is None:
            print("[Error] self_play failed: invalid move")
            break

        game.apply(move, player)
        end = game.check_win_from(move, player)  # 0未终、1胜、2平

        player = 3 - player  # 轮次切换

        pbar.update(1)
        pbar.set_postfix({"round": round_num, "step": step, "player": player})

    pbar.close()

    won = -1  # 0 平局，1/2 胜方
    if end == 1:
        won = 3 - player  # 落子的player是胜方
        if verbose:
            print(f"[Result] Player {won} Win")
        for s in buffer[won]:  # 胜方：回填奖励
            s.win_rate.fill_(1.0)
        for s in buffer[3 - won]:  # 败方：回填奖励
            s.win_rate.fill_(-1.0)
    elif end == 2:
        # 平局，s.win_rate默认为0，什么不用做
        won = 0
        if verbose:
            print(f"[Result] Draw")

    if verbose:
        print(game)

    samples = chain.from_iterable(zip_longest(buffer[1], buffer[2]))
    samples = [s for s in samples if s is not None]
    return step, won, samples


@dataclass
class PlayStat:
    cnt_p1: int = 0
    cnt_p2: int = 0
    cnt_draw: int = 0
    total_steps: int = 0
    play_num: int = 0

    def update(self, won: int, step: int):
        assert won in [1, 2, 0]
        self.play_num += 1
        if won == 1:
            self.cnt_p1 += 1
            self.total_steps += step
        elif won == 2:
            self.cnt_p2 += 1
            self.total_steps += step
        elif won == 0:
            self.cnt_draw += 1
            self.total_steps += step

    def log(self):
        if self.play_num == 0:
            return
        print(f"[Stat @ {self.play_num}] "
              f"WinRateP1: {self.cnt_p1 / self.play_num: .2%}, "
              f"WinRateP2: {self.cnt_p2 / self.play_num: .2%}, "
              f"DrawRate: {self.cnt_draw / self.play_num: .2%}, "
              f"AvgSteps: {self.total_steps / self.play_num: .1f}")


def train():
    stat = PlayStat()

    model_path = "model.pt"
    buffer_path = "buffer.pt"

    model = Alpha0Module(lr=1e-3, weight_decay=1e-4, path=model_path)
    buffer = ReplayBuffer(capacity=50_000).load(buffer_path)
    mcts_config = dict(iterations=600, c=0.3, noise_eps=0.25, dirichlet_alpha=0.12)  # 10/board_size

    total_num = 100000
    train_interval = 10
    start = time.time()

    for i in range(1, 1 + total_num):
        step, won, samples = self_play(model, i, **mcts_config)
        buffer.extend(samples)
        stat.update(won, step)

        if i % train_interval == 0:
            print(8888888, time.time() - start)
            # stat.log()
            model.train_model(
                buffer, epochs=2000, batch_size=128, log_interval=100,
                value_coef=0.4, entropy_coef=0.0, save_path=model_path)

        if i % 300 == 0:
            buffer.save(buffer_path)


if __name__ == '__main__':
    train()
