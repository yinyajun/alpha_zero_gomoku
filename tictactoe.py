#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCTS Tic-Tac-Toe (single-file, runnable)
- Player 1 (X = 1): MCTS
- Player 2 (O = 2): configurable (random or MCTS)
- Returns moves and prints board per turn.
- Pure Python, no dependencies.

Usage examples:
  python mcts_tictactoe.py                           # MCTS vs Random (default)
  python mcts_tictactoe.py --iters 1000              # increase Player1 search iters
  python mcts_tictactoe.py --p2 mcts --p2-iters 300  # MCTS vs MCTS
  python mcts_tictactoe.py --seed 0                  # deterministic demo
"""
import math
import random
import argparse
from typing import List, Tuple, Optional

Board = List[List[int]]  # 0 empty, 1 X, 2 O
Move = Tuple[int, int]  # (row, col)


def copy_board(state: Board) -> Board:
    return [row[:] for row in state]


def count_player(state: Board, p: int) -> int:
    return sum(row.count(p) for row in state)


def current_player_from_state(state: Board) -> int:
    x_count = count_player(state, 1)
    o_count = count_player(state, 2)
    return 1 if x_count == o_count else 2


def empty_actions(state: Board) -> List[Move]:
    return [(i, j) for i in range(3) for j in range(3) if state[i][j] == 0]


def apply_move(state: Board, move: Move, player: int) -> Board:
    ns = copy_board(state)
    ns[move[0]][move[1]] = player
    return ns


def check_winner_on(state: Board) -> Optional[int]:
    # rows and cols
    for i in range(3):
        if state[i][0] == state[i][1] == state[i][2] != 0:
            return state[i][0]
        if state[0][i] == state[1][i] == state[2][i] != 0:
            return state[0][i]
    # diagonals
    if state[0][0] == state[1][1] == state[2][2] != 0:
        return state[0][0]
    if state[0][2] == state[1][1] == state[2][0] != 0:
        return state[0][2]
    return None


class MCTSNode:
    def __init__(self, state: Board, parent: Optional["MCTSNode"] = None, action: Optional[Move] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.wins = 0.0  # IMPORTANT: from Player-1 (X) perspective
        self.untried_actions = empty_actions(state)

    # ---- Game helpers ----
    def get_current_player(self) -> int:
        return current_player_from_state(self.state)

    def is_terminal(self) -> bool:
        return (check_winner_on(self.state) is not None) or (len(empty_actions(self.state)) == 0)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    # ---- MCTS steps ----
    def expand(self) -> "MCTSNode":
        action = self.untried_actions.pop()  # choose one untried action
        player = self.get_current_player()
        new_state = apply_move(self.state, action, player)
        child = MCTSNode(new_state, parent=self, action=action)
        self.children.append(child)
        return child

    def best_child(self, c: float = 1.4) -> "MCTSNode":
        # UCB1: Q + c * sqrt( ln(N) / n )
        assert self.children, "best_child called on node with no children"
        lnN = math.log(self.visits + 1e-12)

        def ucb(node: "MCTSNode") -> float:
            if node.visits == 0:
                return float(
                    "inf")  # ensure unvisited child is picked if any (should rarely happen because of expansion rule)
            exploit = node.wins / node.visits
            explore = math.sqrt(lnN / node.visits)
            return exploit + c * explore

        return max(self.children, key=ucb)

    def rollout(self) -> float:
        # Random playout from current state until terminal.
        # Return value from Player-1 (X) perspective: 1 win, 0 loss, 0.5 draw.
        state = copy_board(self.state)
        player = current_player_from_state(state)
        while True:
            winner = check_winner_on(state)
            if winner is not None:
                return 1.0 if winner == 1 else 0.0
            actions = empty_actions(state)
            if not actions:
                return 0.5  # draw
            move = random.choice(actions)
            state[move[0]][move[1]] = player
            player = 1 if player == 2 else 2

    def backpropagate(self, result: float) -> None:
        # result is from Player-1 perspective
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)


def tree_policy_select(node: MCTSNode) -> MCTSNode:
    # Selection: go down while node is terminal==False and fully expanded==True
    while True:
        if node.is_terminal():
            return node  # terminal
        if not node.is_fully_expanded():
            return node  # stop at first not-fully-expanded
        node = node.best_child()
    # unreachable


def mcts_search(root_state: Board, iterations: int = 500, c: float = 1.4):
    root = MCTSNode(root_state)

    for _ in range(iterations):
        # 1) Selection
        node = tree_policy_select(root)
        # 2) Expansion
        if not node.is_terminal():
            node = node.expand()
        # 3) Simulation
        result = node.rollout()
        # 4) Backpropagation
        node.backpropagate(result)

    # choose best move at root without exploration
    if not root.children:
        return None, []

    def stat(n: MCTSNode):
        wr = n.wins / n.visits if n.visits > 0 else 0.0
        return n.action, n.visits, wr

    best = max(root.children, key=lambda ch: ch.wins / ch.visits if ch.visits > 0 else -1)
    stats = [stat(ch) for ch in root.children]
    return best.action, stats


def print_board(state: Board) -> None:
    symbols = {0: ".", 1: "X", 2: "O"}
    for r in state:
        print(" ".join(symbols[v] for v in r))


def play_game(iters_p1: int = 500, p2: str = "random", iters_p2: int = 300, c: float = 1.4, seed: Optional[int] = None,
              verbose: bool = True):
    if seed is not None:
        random.seed(seed)

    board: Board = [[0] * 3 for _ in range(3)]
    current_player = 1

    if verbose:
        print("MCTS Tic-Tac-Toe Demo")
        print("Legend: . empty, X=1 (MCTS), O=2 ({})\n".format("MCTS" if p2 == "mcts" else "Random"))

    for turn in range(9):
        if verbose:
            print_board(board);
            print()

        if current_player == 1:
            move, stats = mcts_search(board, iterations=iters_p1, c=c)
            if verbose:
                print(f"P1 (MCTS) chooses: {move}")
                if stats:
                    print("Root children stats (move, visits, winrate_for_X):")
                    for a, v, wr in sorted(stats):
                        print(f"  {a}: visits={v}, wr={wr:.3f}")
        else:
            if p2 == "mcts":
                move, _ = mcts_search(board, iterations=iters_p2, c=c)
                if verbose: print(f"P2 (MCTS) chooses: {move}")
            else:
                empties = empty_actions(board)
                move = random.choice(empties) if empties else None
                if verbose: print(f"P2 (Random) chooses: {move}")

        if move is None:
            if verbose: print("No moves left.")
            break

        board = apply_move(board, move, current_player)

        winner = check_winner_on(board)
        if winner is not None:
            if verbose:
                print_board(board)
                print(f"\nPlayer {winner} wins!")
            return winner

        current_player = 1 if current_player == 2 else 2

    if verbose:
        print_board(board)
        print("\nDraw!")
    return 0  # draw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=500, help="Iterations for Player 1 (MCTS).")
    parser.add_argument("--p2", type=str, choices=["random", "mcts"], default="random",
                        help="Opponent type for Player 2.")
    parser.add_argument("--p2-iters", type=int, default=300, help="Iterations for Player 2 if p2==mcts.")
    parser.add_argument("--c", type=float, default=1.4, help="Exploration constant for UCB1.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = parser.parse_args()

    play_game(iters_p1=args.iters, p2=args.p2, iters_p2=args.p2_iters, c=args.c, seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()
