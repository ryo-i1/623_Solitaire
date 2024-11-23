import os
import numpy as np
import pandas as pd
from enum import IntEnum


class Suit(IntEnum):

    HEART = 0
    DIAMOND = 1
    CLUB = 2
    SPADE = 3

    def is_red(self):
        return self in [Suit.HEART, Suit.DIAMOND]

    def is_black(self):
        return self in [Suit.CLUB, Suit.SPADE]

    def __str__(self):
        return self.name.capitalize()


class Card:

    def __init__(self, suit: Suit, value: int):
        if not isinstance(suit, Suit):
            raise TypeError("Invalid suit provided.")
        if not 1 <= value <= 13:
            raise ValueError("The value of the card is out of range.")

        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.value} of {self.suit}"

    def __add__(self, value: int):
        new_value = self.value + value
        if not 1 <= new_value <= 13:
            raise ValueError("The value of the card is out of range.")

        return Card(self.suit, new_value)

    def __sub__(self, value: int):
        return self.__add__(-value)

    def __eq__(self, card: "Card"):
        return self.suit == card.suit and self.value == card.value

    def __ne__(self, card: "Card"):
        return not self.__eq__(card)

    def is_red(self):
        return self.suit.is_red()

    def is_black(self):
        return self.suit.is_black()

    @staticmethod
    def get_contrast_cards(card: "Card") -> list["Card"]:
        """
        cardが赤なら黒のカード，黒なら赤のカードを返す
        """

        contrast_suits = (
            [Suit.CLUB, Suit.SPADE] if card.is_red() else [Suit.HEART, Suit.DIAMOND]
        )
        return [Card(suit, card.value) for suit in contrast_suits]

    @staticmethod
    def get_all_cards(
        suits: list[Suit] = None, values: list[int] = None
    ) -> list["Card"]:
        """
        条件に合うカードを返す
        """

        if suits is None:
            suits = list(Suit)
        if values is None:
            values = list(range(1, 14))

        return [Card(suit, val) for suit in suits for val in values]


class Action(IntEnum):
    """
    - 場札のカードを組札に移動する (i列 -> 組札)
    - 山札のカードを組札に移動する (山札 -> 組札)
    - 場札のカードを場札に移動する (i列 -> j列)
    - 山札のカードを場札に移動する (山札 -> j列)
    - 山札をめくる
    """

    MOVE_TABLEAU_TO_FOUNDATION = 0
    MOVE_STOCK_TO_FOUNDATION = 1
    MOVE_TABLEAU_TO_TABLEAU = 2
    MOVE_STOCK_TO_TABLEAU = 3
    DRAW_STOCK = 4


class SolitaireSolver:
    """
    SolitaireSolver

    Attributes
    ----------
    opened_cards : 2d list of Card
        場札の表カード
        -1番目が一番上
    n_closed : 1d list of int
        場札の裏カードの枚数
    stock_top_card : Card
        山札の一番上のカード
    foundation_cards : 1d list of Card
        組札の枚数
    n_stock : int
        山札の残り枚数
    """

    # 場におけるカードの列数
    N_COLUMNS = 8

    def __init__(self):
        # 場札
        self.opened_cards = [[] for _ in range(self.N_COLUMNS)]
        self.n_closed = list(range(self.N_COLUMNS))

        # 山札の一番上のカード
        self.stock_top_card = None
        # 山札の残り枚数
        self.n_stock = 24

        # 組札
        self.foundation_cards = [0] * 4

    def update(self):
        """
        場の状態を更新する
        """

        # 場札と山札の一番上のカード情報を受け取る
        opened_top_cards, stock_top_card = self.get_top_cards()

        # 受け取ったカード情報をもとに盤面を更新
        self.set_top_cards(opened_top_cards, stock_top_card)

        # 行動を決定
        action, args = self.solve()

        # 行動を実行
        self.execute_action(action, args)

    def get_top_cards(self):
        """場札と山札の一番上のカード情報を取得する"""

        pass

    def set_top_cards(self, opened_top_cards, stock_top_card):
        """場札と山札の一番上のカードを更新する"""

        # 場札の表カード
        for i_col, card in enumerate(opened_top_cards):
            # 新しいカードがめくられた場合
            if self.opened_cards[i_col] == [] and card is not None:
                self.opened_cards[i_col].append(card)
                self.n_closed[i_col] -= 1

        # 山札の一番上のカード
        self.stock_top_card = stock_top_card

    def solve(self) -> tuple[Action, tuple[int]]:
        """
        現在の状態から次の手を決定する
        """

        # 1. 組札に移動できるカードが場札か山札にあるか

        # 組札に移動できるカード
        desire_cards = []
        for i in range(4):
            if self.foundation_cards[i] is None:
                # 組札がない場合，Aを移動できる
                desire_cards.append(Card(i, 1))
            elif self.foundation_cards[i].value == 13:
                # 組札がKの場合，移動できるカードなし
                continue
            else:
                # 組札がA~Qの場合，次のカードを移動できる
                desire_cards.append(self.foundation_cards[i] + 1)

        # 場札の表カードを順に見ていく
        for i_col, cards in enumerate(self.opened_cards):
            # 列が空の場合skip
            if len(cards) == 0:
                continue

            # その列の一番上のカード
            top_card = cards[-1]

            if top_card in desire_cards:
                # i列のカードを組札に移動する
                return Action.MOVE_TABLEAU_TO_FOUNDATION, (i_col,)

        if self.stock_top_card in desire_cards:
            # 山札の一番上のカードを組札に移動する
            return Action.MOVE_STOCK_TO_FOUNDATION, ()

        # 2. 場札内で移動できるカードがあるか

        # 移動先カードリスト
        desire_cards = []
        for cards in self.opened_cards:
            # 列が空の場合
            if len(cards) == 0:
                # Kを移動できる
                desire_cards.append(Card.get_all_cards(value=[13]))
            else:
                # その列の一番上の表カード
                top_card = cards[-1]
                # その列に移動できるカード
                desire_cards.append(Card.get_contrast_cards(top_card - 1))

        # 移動元カード
        for i_col, i_cards in enumerate(self.opened_cards):
            # 列が空の場合skip
            if len(i_cards) == 0:
                continue

            # その列の一番下の表カード
            i_bottom_card = i_cards[0]

            # 移動先カード
            for j_col, j_desire_cards in enumerate(desire_cards):
                # 移動元と移動先が同じ列の場合skip
                if i_col == j_col:
                    continue

                if i_bottom_card in j_desire_cards:
                    # i_bottom_cardをj列に移動する
                    return Action.MOVE_TABLEAU_TO_TABLEAU, (i_col, j_col)

        # 3. 山札の一番上のカードを場札に移動できるか

        # 山札の一番上にカードがある場合
        if self.stock_top_card is None:
            # 移動先カード
            for j_col, j_desire_cards in enumerate(desire_cards):
                if self.stock_top_card in j_desire_cards:
                    # 山札の一番上のカードをj列に移動する
                    return Action.MOVE_STOCK_TO_TABLEAU, (j_col,)

        # 4. 山札をめくる
        return Action.DRAW_STOCK, ()

    def execute_action(self, action: Action, args: tuple[int]):
        """
        行動を実行する
        """

        # 場札から組札に移動
        if action == Action.MOVE_TABLEAU_TO_FOUNDATION:
            # 移動元 列
            i_col = args[0]

            # 盤面を更新
            # 移動元 カードをpop
            card = self.opened_cards[i_col].pop()
            # 移動先組札を更新
            self.foundation_cards[card.suit] = card

            # プレイヤーに通知
            send_msg(f"場札の {i_col + 1} 列の {card} を組札に移動する")

        # 山札から組札に移動
        elif action == Action.MOVE_STOCK_TO_FOUNDATION:
            # 移動元 山札のカード
            card = self.stock_top_card

            # 盤面を更新
            # 移動先 組札を更新
            self.foundation_cards[card.suit] = card

            # プレイヤーに通知
            send_msg(f"山札の {card} を組札に移動する")

        # 場札から場札に移動
        elif action == Action.MOVE_TABLEAU_TO_TABLEAU:
            # 移動元 列
            i_col = args[0]
            # 移動先 列
            j_col = args[1]

            # 盤面を更新
            # 移動元
            cards = self.opened_cards[i_col]
            self.opened_cards[i_col] = []
            # 移動するカード
            card = cards[0]

            # 移動先
            self.opened_cards[j_col].extend(cards)

            # プレイヤーに通知
            send_msg(f"場札の {i_col + 1} 列の {card} を {j_col + 1} 列に移動する")

        # 山札から場札に移動
        elif action == Action.MOVE_STOCK_TO_TABLEAU:
            # 移動元 山札のカード
            card = self.stock_top_card
            # 移動先 列
            j_col = args[0]

            # 盤面を更新
            # 移動先 カードを追加
            self.opened_cards[j_col].append(card)
            # 移動元 山札のカードをpop
            self.stock_top_card = None

            # プレイヤーに通知
            send_msg(f"山札の {card} を {j_col + 1} 列に移動する")

        # 山札をめくる
        elif action == Action.DRAW_STOCK:
            # 山札のカード
            card = self.stock_top_card

            # 盤面を更新
            # 山札のカードをめくる
            self.stock_top_card = None

            # プレイヤーに通知
            send_msg("山札をめくる")


def send_msg(msg: str):
    print(msg)


def main():
    pass


if __name__ == "__main__":
    main()