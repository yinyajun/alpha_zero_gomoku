import pygame
from typing import Optional

from game import Game
from mcts import MCTSNode

cell_size = 40
edge = int(1.5 * cell_size)
black = (0, 0, 0)
white = (255, 255, 255)
brown = (222, 184, 135)
bg_color = (253, 246, 227)


class Panel:
    """简单的棋盘面板：负责静态底图(背景+网格)和标题的贴图。"""

    def __init__(self, pos, size, title, font, grid_color=black, bg=brown):
        self.pos = pos  # (x, y)
        self.size = size  # (w, h)
        self.title = title
        self.font = font
        self.grid_color = grid_color
        self.bg = bg

        # surface 的 convert() 需要在 set_mode 之后调用；本类实例应在 set_mode 之后创建
        self.surface = pygame.Surface(size).convert()
        self.surface.fill(self.bg)
        self.title_surface = self.font.render(self.title, True, black)

    def draw_grid(self, game_size: int, cell: int, line_w: int = 3):
        """把网格画到自身 surface（仅一次）"""
        w, h = self.size
        for i in range(game_size + 1):
            y = i * cell
            pygame.draw.line(self.surface, self.grid_color, (0, y), (w, y), line_w)
        for i in range(game_size + 1):
            x = i * cell
            pygame.draw.line(self.surface, self.grid_color, (x, 0), (x, h), line_w)

    def blit_to(self, screen: pygame.Surface):
        """把自身 surface 和标题贴到主屏"""
        bx, by = self.pos
        screen.blit(self.surface, (bx, by))
        tx = bx + (self.size[0] * 4) // 10
        ty = by - int(edge * 0.4)
        screen.blit(self.title_surface, (tx, ty))


class Renderer:
    """渲染器：管理四个面板、缓存坐标、画棋子与 MCTS 数字。"""

    def __init__(self, size_n=Game.size, cell=cell_size, edge=edge):
        # 确保 pygame 初始化
        pygame.init()
        self.size_n = size_n
        self.cell = cell
        self.font = pygame.font.Font(None, 24)

        # 画布大小
        self.BOARD_W = self.cell * self.size_n
        self.BOARD_H = self.BOARD_W
        self.screen = pygame.display.set_mode((self.BOARD_W * 2 + 3 * edge, self.BOARD_H * 2 + 4 * edge))

        pygame.display.set_caption("AlphaZero Demo")

        # 四个面板左上角位置
        self.bias = [
            (edge, edge),
            (edge * 2 + self.BOARD_W, edge),
            (edge, edge * 2 + self.BOARD_H),
            (edge * 2 + self.BOARD_W, edge * 2 + self.BOARD_H),
        ]
        titles = ["board", "value_out", "search_num", "policy_out"]

        # 四个面板（结构一样，标题不同），需在 set_mode 之后创建
        panel_size = (self.BOARD_W, self.BOARD_H)
        self.panels = [
            Panel(self.bias[0], panel_size, titles[0], self.font),
            Panel(self.bias[1], panel_size, titles[1], self.font),
            Panel(self.bias[2], panel_size, titles[2], self.font),
            Panel(self.bias[3], panel_size, titles[3], self.font),
        ]

        # 静态网格只画一次
        for p in self.panels:
            p.draw_grid(self.size_n, self.cell)

        # 预计算各格落子中心点（局部坐标）
        self.centers = [
            [(self.cell // 2 + i * self.cell, self.cell // 2 + j * self.cell)
             for j in range(self.size_n)] for i in range(self.size_n)
        ]

        # 预计算各格数字中心点（局部坐标）
        self.num_anchor = [
            [(i * self.cell + self.cell // 10, j * self.cell + self.cell // 5)
             for j in range(self.size_n)] for i in range(self.size_n)
        ]

    def draw_stones(self, game: Game):
        """棋子画在左上面板"""
        bx, by = self.bias[0]
        for i in range(self.size_n):
            for j in range(self.size_n):
                v = game.board[i][j]
                if v == 0:
                    continue
                color = black if v == 1 else white
                cx, cy = self.centers[i][j]
                pygame.draw.circle(self.screen, color, (bx + cx, by + cy), 14)

    def draw_mcts_numbers(self, node: MCTSNode):
        """把 MCTS 的数值画到三个面板：value/visits/prior；node 为 None 则跳过。"""
        if node is None:
            return
        size = self.size_n
        priors = node.priors

        bx1, by1 = self.panels[1].pos
        bx2, by2 = self.panels[2].pos
        bx3, by3 = self.panels[3].pos

        for ch in node.children:
            i, j = ch.last_move
            ax, ay = self.num_anchor[i][j]

            # value_out 面板=1
            val_str = f"{-ch.value: .2f}"
            self.screen.blit(self.font.render(val_str, True, black), (bx1 + ax, by1 + ay))

            # search_num 面板=2
            vis_str = str(ch.visits)
            self.screen.blit(self.font.render(vis_str, True, black), (bx2 + ax, by2 + ay))

            # policy_out 面板=3
            p_str = f"{priors[i * size + j]: .2f}"
            self.screen.blit(self.font.render(p_str, True, black), (bx3 + ax, by3 + ay))

    def draw_banner(self, player: int, end: int):
        color = "Black" if player == 1 else "White"
        bias = (edge, self.BOARD_H * 2 + 3 * edge)

        if end == 1:
            text = f"[Info] {color} win!"
        elif end == 2:
            text = "[Info] The game is draw"
        else:
            text = f"[Info] Player {player}'s turn"

        surf = self.font.render(text, True, 1)
        self.screen.blit(surf, bias)

    def draw(self, game: Game, node: Optional[MCTSNode], player: int, end: int):
        """一次完整渲染：清屏 -> 四个面板底图 -> 棋子 -> 数值 -> flip。"""
        self.screen.fill(bg_color)
        for p in self.panels:
            p.blit_to(self.screen)
        self.draw_stones(game)
        self.draw_mcts_numbers(node)
        self.draw_banner(player, end)
        pygame.display.flip()
