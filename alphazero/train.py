import os
import time
from datetime import datetime

import wandb
import torch
from tqdm import tqdm
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

from game import Game, Player1, Player2
from model import Alpha0Module
from evaluator import BatchEvaluator, build_pv_fn, build_pv_fn_batch
from buffer import ReplayBuffer
from play import self_play, MCTSPlayer, eval_play


@dataclass
class PlayMetric:
    cnt_p1: int = 0
    cnt_p2: int = 0
    cnt_draw: int = 0
    total_steps: int = 0
    play_num: int = 0

    def update(self, winner: int, step: int):
        self.play_num += 1
        if winner == Player1:
            self.cnt_p1 += 1
            self.total_steps += step
        elif winner == Player2:
            self.cnt_p2 += 1
            self.total_steps += step
        elif winner == 0:
            self.cnt_draw += 1
            self.total_steps += step

        record = {
            "selfplay/episode": self.play_num,
            "selfplay/episode_length": step,
            "selfplay/agg/p1_win_rate": round(self.cnt_p1 / self.play_num, 2),
            "selfplay/agg/p2_win_rate": round(self.cnt_p2 / self.play_num, 2),
            "selfplay/agg/draw_rate": self.cnt_draw / self.play_num,
            "selfplay/agg/avg_steps": round(self.total_steps / self.play_num, 2),
        }
        return record


@dataclass
class TrainConfig:
    # game
    board_size: int = Game.size
    win_num: int = Game.win_num
    # mcts
    iterations: int = 300
    c_puct: float = 0.5
    noise_moves: int = 7
    noise_eps: float = 0.25
    dirichlet_alpha: float = 0.2  # # 10/board_size
    warm_moves: int = 11
    tau: float = 1.0
    # model_train
    save_dir: str = "output"
    resume_model_path: str = None
    resume_buffer_path: str = None
    train_epochs: int = 1000
    log_interval: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    value_coef: float = 1.0
    # train
    center_round: int = 400  # 首子居中局数
    total_round: int = 100000  # 总训练局数
    collect_round: int = 10  # 收集局数
    collect_actors: int = 10  # 收集并发
    # eval
    eval_round: int = 20  # eval局数
    eval_interval: int = 20  # eval间隔局数 = collect_round * eval_interval


def eval(round: int, conf: TrainConfig):
    if round % (conf.eval_interval * conf.collect_round) != 0:
        return

    new_model_path = os.path.join(conf.save_dir, f"model_{round}.pt")
    ref_model_path = os.path.join(conf.save_dir, f"model_{round - conf.collect_round}.pt")

    new_model = MCTSPlayer(
        pv_fn=build_pv_fn(Alpha0Module(resume_path=new_model_path)),
        iterations=conf.iterations,
        c_puct=conf.c_puct,
        warm_moves=0,
        tau=conf.tau,
        noise_moves=0,
        noise_eps=conf.noise_eps,
        dirichlet_alpha=conf.dirichlet_alpha,
        tree_cls=MCTSTree,
    )

    ref_model = MCTSPlayer(
        pv_fn=build_pv_fn(Alpha0Module(resume_path=ref_model_path)),
        iterations=conf.iterations,
        c_puct=conf.c_puct,
        warm_moves=0,
        tau=conf.tau,
        noise_moves=0,
        noise_eps=conf.noise_eps,
        dirichlet_alpha=conf.dirichlet_alpha,
        tree_cls=MCTSTree,
    )

    new_win = 0
    ref_win = 0

    for i in tqdm(range(conf.eval_round), desc="Eval", unit="round"):
        if i % 2 == 0:
            _, winner = eval_play(Game(), player1=new_model, player2=ref_model)
            new_win += int(winner == Player1)
            ref_win += int(winner == Player2)
        else:
            _, winner = eval_play(Game(), player1=ref_model, player2=new_model)
            new_win += int(winner == Player2)
            ref_win += int(winner == Player1)

    print(f"[Eval] new_win: {new_win}, ref_win: {ref_win}")

    # === Remove old files ===
    try:
        # 保留最近的版本，删除更早的
        keep_n = 10
        model_files = sorted(
            [f for f in os.listdir(conf.save_dir) if f.startswith("model_") and f.endswith(".pt")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        buffer_files = sorted(
            [f for f in os.listdir(conf.save_dir) if f.startswith("buffer_") and f.endswith(".pt")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )

        # 删除旧的模型和 buffer 文件
        for old_file in model_files[:-keep_n]:
            os.remove(os.path.join(conf.save_dir, old_file))
            print(f"[cleanup] removed old model file: {old_file}")
        for old_file in buffer_files[:-keep_n]:
            os.remove(os.path.join(conf.save_dir, old_file))
            print(f"[cleanup] removed old buffer file: {old_file}")

    except Exception as e:
        print(f"[cleanup] failed to remove old files: {e}")


def parallel_train(conf: TrainConfig):
    """两阶段同步训练"""
    print("同步并行训练")
    os.makedirs(conf.save_dir, exist_ok=True)
    model = Alpha0Module(lr=conf.lr, weight_decay=conf.weight_decay, resume_path=conf.resume_model_path)
    buffer = ReplayBuffer(capacity=50_000, resume_path=conf.resume_buffer_path)
    pv_fn = build_pv_fn_batch(model, max_batch_size=conf.collect_actors, max_timeout_ms=0.04)
    metric = PlayMetric()

    wandb.init(
        project=f"alpha_zero_gomoku",
        name=f"run-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config=asdict(conf),
        mode="disabled"
    )
    wandb.define_metric("selfplay/episode")
    wandb.define_metric("selfplay/*", step_metric="selfplay/episode")
    wandb.define_metric("selfplay/agg/*", step_metric="selfplay/episode")
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")

    for i in range(0, conf.total_round, conf.collect_round):
        start = time.time()

        # === A) 收集阶段：并发跑自我对弈 ===
        def collect_one(round: int):
            player = MCTSPlayer(
                pv_fn=pv_fn,
                iterations=conf.iterations,
                c_puct=conf.c_puct,
                warm_moves=conf.warm_moves,
                tau=conf.tau,
                noise_moves=conf.noise_moves,
                noise_eps=conf.noise_eps,
                dirichlet_alpha=conf.dirichlet_alpha,
                tree_cls=MCTSTree,
            )
            step, winner, trajectory = self_play(
                Game(), player, force_center=round <= conf.center_round, enable_tqdm=False)
            return step, winner, trajectory

        max_round = i + conf.collect_round
        rounds = range(i, max_round)

        metrics = []
        with ThreadPoolExecutor(max_workers=conf.collect_actors) as executor:
            futures = [executor.submit(collect_one, r) for r in rounds]
            for fut in tqdm(futures, desc="Collecting", unit="round"):
                step, winner, trajectory = fut.result()
                buffer.add_trajectory(trajectory)
                metrics.append(metric.update(winner, step))

        for r in metrics:
            # print(r)
            wandb.log(r)

        print(f"[Collect] consume: {time.time() - start: .2f}sec, rounds: {len(rounds)}")

        # === B) 训练阶段 ===
        model_path = os.path.join(conf.save_dir, f"model_{max_round}.pt")
        buffer_path = os.path.join(conf.save_dir, f"buffer_{max_round}.pt")
        model.train_model(
            buffer,
            epochs=conf.train_epochs,
            batch_size=conf.batch_size,
            log_interval=conf.log_interval,
            value_coef=conf.value_coef,
        )
        model.save_checkpoint(model_path)
        buffer.save(buffer_path)
        eval(max_round, conf)
        torch.cuda.empty_cache()

    wandb.finish()


def train(conf: TrainConfig):
    """两阶段同步训练"""
    print("同步串行训练")
    os.makedirs(conf.save_dir, exist_ok=True)
    model = Alpha0Module(lr=conf.lr, weight_decay=conf.weight_decay, resume_path=conf.resume_model_path)
    buffer = ReplayBuffer(capacity=50_000, resume_path=conf.resume_buffer_path)
    pv_fn = build_pv_fn(model)
    metric = PlayMetric()

    wandb.init(
        project=f"alpha_zero_gomoku",
        name=f"run-{datetime.now().strftime('%Y%m%d-%H%M')}",
        config=asdict(conf),
        # mode="disabled"
    )
    wandb.define_metric("selfplay/episode")
    wandb.define_metric("selfplay/*", step_metric="selfplay/episode")
    wandb.define_metric("selfplay/agg/*", step_metric="selfplay/episode")
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")

    player = MCTSPlayer(
        pv_fn=pv_fn,
        iterations=conf.iterations,
        c_puct=conf.c_puct,
        warm_moves=conf.warm_moves,
        tau=conf.tau,
        noise_moves=conf.noise_moves,
        noise_eps=conf.noise_eps,
        dirichlet_alpha=conf.dirichlet_alpha,
        tree_cls=MCTSTree,
    )

    for i in range(0, conf.total_round, conf.collect_round):

        start = time.time()

        # === A) 收集阶段：自我对弈 ===
        max_round = i + conf.collect_round
        rounds = range(i, max_round)
        for r in rounds:
            step, winner, trajectory = self_play(Game(), player, force_center=r < conf.center_round)
            buffer.add_trajectory(trajectory)
            record = metric.update(winner, step)
            # print(record)
            wandb.log(record)

        print(f"[Collect] consume: {time.time() - start: .2f}sec, rounds: {len(rounds)}")

        # === B) 训练阶段 ===
        model_path = os.path.join(conf.save_dir, f"model_{max_round}.pt")
        buffer_path = os.path.join(conf.save_dir, f"buffer_{max_round}.pt")
        model.train_model(
            buffer,
            epochs=conf.train_epochs,
            batch_size=conf.batch_size,
            log_interval=conf.log_interval,
            value_coef=conf.value_coef,
        )
        model.save_checkpoint(model_path)
        buffer.save(buffer_path)
        eval(max_round, conf)
        torch.cuda.empty_cache()

    wandb.finish()


if __name__ == '__main__':
    # mcts tree两种实现方式，通过import来切换
    # from mcts2 import MCTSTree
    from mcts1 import MCTSTree

    conf = TrainConfig()
    print(conf)

    # train(conf=conf)
    parallel_train(conf=TrainConfig())
