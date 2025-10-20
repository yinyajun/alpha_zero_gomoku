import wandb
import torch
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

from game import Game
from model import Alpha0Module
from evaluator import Evaluator, build_pv_fn, build_pv_fn2, DefaultPolicyValueFn
from buffer import ReplayBuffer
from play import self_play, MCTSPlayer


@dataclass
class TrainConfig:
    # game
    board_size: int = Game.size
    win_num: int = Game.win_num
    # mcts
    iterations: int = 600
    c_puct: float = 0.5
    noise_moves: int = 1
    noise_eps: float = 0.25
    dirichlet_alpha: float = 0.20  # # 10/board_size
    warm_moves: int = 6
    tau: float = 1.0
    # model_train
    resume_model_path: str = None
    resume_buffer_path: str = None
    train_epochs: int = 2000
    log_interval: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    # train
    center_round: int = 200
    total_round: int = 50000
    collect_round: int = 5
    collect_actors: int = 5
    eval_batch: int = 10
    eval_timeout_ms: float = 0.1


def train3(conf: TrainConfig):
    """两阶段同步训练"""
    model = Alpha0Module(lr=conf.lr, weight_decay=conf.weight_decay, resume_path=conf.resume_model_path)
    buffer = ReplayBuffer(capacity=50_000, resume_path=conf.resume_buffer_path)
    fn = DefaultPolicyValueFn(model=model).run
    # wandb.init(project=f"alpha_zero_gomoku",
    #            name=f"run-{datetime.now().strftime('%Y%m%d-%H%M')}",
    #            config=asdict(conf))
    # wandb.define_metric("selfplay/episode")
    # wandb.define_metric("selfplay/*", step_metric="selfplay/episode")
    # wandb.define_metric("selfplay/agg/*", step_metric="selfplay/episode")
    # wandb.define_metric("train/step")
    # wandb.define_metric("train/*", step_metric="train/step")

    player = MCTSPlayer(
        policy_value_fn=fn,
        iterations=conf.iterations,
        c_puct=conf.c_puct,
        warm_moves=conf.warm_moves,
        tau=conf.tau,
        noise_moves=conf.noise_moves,
        noise_eps=conf.noise_eps,
        dirichlet_alpha=conf.dirichlet_alpha,
    )

    for i in range(1, 1 + conf.total_round, conf.collect_round):

        # === A) 收集阶段：自我对弈 ===
        rounds = range(i, i + conf.collect_round)
        for r in rounds:
            step, winner, trajectory = self_play(Game(), player, force_center=r < conf.center_round)
            buffer.add_trajectory(trajectory)

        # === B) 训练阶段 ===
        model_path = "model.pt"
        buffer_path = "buffer.pt"
        model.train_model(
            buffer,
            epochs=conf.train_epochs,
            batch_size=conf.batch_size,
            log_interval=conf.log_interval,
            value_coef=conf.value_coef,
            entropy_coef=conf.entropy_coef,
        )
        model.save_checkpoint(model_path)
        buffer.save(buffer_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    train3(conf=TrainConfig())
