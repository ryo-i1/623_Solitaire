import os
import numpy as np
import pandas as pd
from enum import IntEnum

from Card import Card, Suit
from SolitaireSolver import Action, SolitaireSolver


class Solitaire:
    """
    SlitaireSolver テスト用

    自動的に盤面情報を渡す
    """

    def __init__(self):
        self.n_columns = 7
        self.opened_cards = [[] for _ in range(self.n_columns)]
        self.closed_cards = [[] for _ in range(self.n_columns)]
        self.stock_cards = [None]
        self.stock_top_idx = 0
        self.foundation_cards = [0] * 4

        self.game_init()

    def game_init(self):
        """
        盤面の初期化
        """

        # 割り振るカード
        cards = Card.get_all_cards()
        np.random.shuffle(cards)

        # 場札の表のカード
        # 枚数 : [1, 1, 1, 1, 1, 1, 1]
        for i in range(self.n_columns):
            self.opened_cards[i].append(cards.pop())

        # 場札の裏のカード
        # 枚数 : [0, 1, 2, 3, 4, 5, 6]
        for i in range(self.n_columns):
            self.closed_cards[i] = cards[:i]
            cards = cards[i:]

        # 山札
        self.stock_cards = [None] + cards

    def send_top_cards(self):
        """
        山札の一番上のカード，場札の一番上のカードを渡す
        """

        # 山札の一番上のカード
        stock_top_card = self.stock_cards[self.stock_top_idx]
        # 場札の一番上のカード
        tableau_top_cards = [
            cards[-1] if len(cards) > 0 else None for cards in self.opened_cards
        ]

        return stock_top_card, tableau_top_cards

    def update(self, action: Action, args: tuple[int]):
        """
        行動を受け取り，場の情報を更新する
        """

        # 場札から組札に移動
        if action == Action.MOVE_TABLEAU_TO_FOUNDATION:
            # 移動元 列
            i_col = args[0]

            # 盤面を更新
            card = self.opened_cards[i_col].pop()
            self.foundation_cards[card.suit] = card.value

        # 山札から組札に移動
        elif action == Action.MOVE_STOCK_TO_FOUNDATION:
            # 移動元 山札
            card = self.stock_cards.pop(self.stock_top_idx)
            self.stock_top_idx -= 1

            # 盤面を更新
            self.foundation_cards[card.suit] = card.value

        # 場札から場札に移動
        elif action == Action.MOVE_TABLEAU_TO_TABLEAU:
            # 移動元, 移動先
            i_col = args[0]
            j_col = args[1]

            # 盤面を更新
            cards = self.opened_cards[i_col]
            self.opened_cards[i_col] = []
            # 移動するカード
            card = cards[0]

            self.opened_cards[j_col].extend(cards)

        # 山札から場札に移動
        elif action == Action.MOVE_STOCK_TO_TABLEAU:
            # 移動元 山札
            card = self.stock_cards.pop(self.stock_top_idx)
            self.stock_top_idx -= 1
            # 移動先 列
            j_col = args[0]

            # 盤面を更新
            self.opened_cards[j_col].append(card)

        # 山札をめくる
        elif action == Action.DRAW_STOCK:
            # 何枚めくるか
            # n_draw = args[0]

            # カードを1枚めくる
            self.stock_top_idx = (self.stock_top_idx + 1) % len(self.stock_cards)

        # 何もしない
        elif action == Action.DO_NOTHING:
            # ゲーム終了らしい
            return False

        # 移動後
        for i_col in range(self.n_columns):
            # openedがなく，closedがある場合，裏カードをめくる
            if len(self.opened_cards[i_col]) == 0 and len(self.closed_cards[i_col]) > 0:
                card = self.closed_cards[i_col].pop()
                self.opened_cards[i_col].append(card)

        return True

    def draw(self):

        print()
        print("GAME", "=" * 20)
        print(
            f"[山札]\t{self.stock_cards[self.stock_top_idx]} ({self.stock_top_idx + 1}/{len(self.stock_cards)})"
        )
        print("[組札]", end='')
        for i in range(4):
            print(f"\t{Suit(i)}\t: {self.foundation_cards[i]}")
        print()

        for i_col, cards in enumerate(self.opened_cards):
            print(f"[場札 {i_col + 1}]", end='')
            print(f"({len(self.closed_cards[i_col])})\t", end='')
            for card in cards:
                print(card, end=', ')
            print()
        print("=" * 20)
        print()


def main():
    game = Solitaire()
    solver = SolitaireSolver()

    is_end = True
    while is_end:
        # 盤面情報を取得
        stock_top_card, opened_top_cards = game.send_top_cards()

        # solver にわたす
        action, args = solver.update(stock_top_card, opened_top_cards)

        # game にactionを渡す
        is_end = game.update(action, args)

        # 描画
        # game.draw()
        solver.draw()

    # 終了イベント
    game.game_end()


if __name__ == "__main__":
    main()
