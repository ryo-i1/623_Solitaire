import numpy as np
from enum import IntEnum

from Card import Suit, Card


class Action(IntEnum):
    """
    - 場札のカードを組札に移動する (i列 -> 組札)
    - 山札のカードを組札に移動する (山札 -> 組札)
    - 場札のカードを場札に移動する (i列 -> j列)
    - 山札のカードを場札に移動する (山札 -> j列)
    - 山札をめくる
    - 何もできない
    """

    MOVE_TABLEAU_TO_FOUNDATION = 0
    MOVE_STOCK_TO_FOUNDATION = 1
    MOVE_TABLEAU_TO_TABLEAU = 2
    MOVE_STOCK_TO_TABLEAU = 3
    DRAW_STOCK = 4
    DO_NOTHING = 5


class SolitaireSolver:
    """
    SolitaireSolver

    Attributes
    ----------
    n_columns : int
        場札の列数
    opened_cards : 2d list of Card
        場札の表カード
        -1番目が一番上
    n_closed : 1d list of int
        場札の裏カードの枚数
    foundation_cards : 1d list of Card
        組札の枚数
    n_stock : int
        山札の残り枚数
    stock_cards : list of Card
        山札のカード
    stock_top_idx : int
        山札の一番上のカードのインデックス
    """

    def __init__(self):
        # 場におけるカードの列数
        self.n_columns = 7

        # 場札
        self.opened_cards = [[] for _ in range(self.n_columns)]
        self.n_closed = [1, 2, 3, 4, 5, 6, 7]

        # 山札
        self.stock_cards = [None]
        self.stock_top_idx = 0
        self.n_stock = 52 - np.sum(self.n_closed)

        # 組札
        self.foundation_cards = [0] * 4

    def update(
        self, stock_top_card: Card, opened_top_cards: list[Card]
    ) -> tuple[Action, tuple[int]]:
        """
        場の状態を更新する
        """

        # 受け取ったカード情報をもとに盤面を更新
        self.set_top_cards(stock_top_card, opened_top_cards)

        # 行動を決定
        action, args = self.solve()

        # 行動を実行
        self.execute_action(action, args)

        # テスト用に行動を返す
        return action, args

    def set_top_cards(self, stock_top_card, opened_top_cards):
        """場札と山札のカードを更新する"""

        # 場札の表カード
        for i_col, card in enumerate(opened_top_cards):
            # 新しいカードがめくられた場合
            if self.opened_cards[i_col] == [] and card is not None:
                self.opened_cards[i_col].append(card)

                # 裏カードの枚数を減らす
                if self.n_closed[i_col] > 0:
                    self.n_closed[i_col] -= 1

        # 山札に未知のカードがある場合
        if len(self.stock_cards) - 1 < self.n_stock:
            # 山札に表示されているカードが変わった場合
            if self.stock_cards[self.stock_top_idx] != stock_top_card:
                self.stock_cards.append(stock_top_card)
                self.stock_top_idx += 1

        # 山札が全て既知の場合
        else:
            # 山札に表示されているカードが変わった場合
            if self.stock_cards[self.stock_top_idx] != stock_top_card:
                self.stock_top_idx = (self.stock_top_idx + 1) % len(self.stock_cards)

    def solve(self) -> tuple[Action, tuple[int]]:
        """
        現在の状態から次の手を決定する
        """

        # 1. 組札に移動できるカードが場札か山札にあるか

        # 場札の表カードを順に見ていく
        for i_col, cards in enumerate(self.opened_cards):
            # 列が空の場合skip
            if len(cards) == 0:
                continue

            # その列の一番上のカード
            top_card = cards[-1]

            # top_cardを組札に移動できるか
            if self.can_move_to_foundation(top_card):
                # i列のカードを組札に移動する
                return Action.MOVE_TABLEAU_TO_FOUNDATION, (i_col,)

        # 山札にカードが表示されている場合
        if self.stock_top_idx != 0:
            # 山札の一番上のカード
            stock_top_card = self.stock_cards[self.stock_top_idx]

            # 組札に移動できるか
            if self.can_move_to_foundation(stock_top_card):
                # 山札のカードを組札に移動する
                return Action.MOVE_STOCK_TO_FOUNDATION, ()

        # 2. 場札内で移動できるカードがあるか

        # 移動元カード
        for i_col, i_cards in enumerate(self.opened_cards):
            # 列が空の場合skip
            if len(i_cards) == 0:
                continue
            # 列の裏カードがなく，Kが一番下の場合skip
            if self.n_closed[i_col] == 0 and i_cards[0].value == 13:
                continue

            # その列の一番下の表カード
            i_bottom_card = i_cards[0]

            # 場札に移動できるか
            j_col = self.can_move_to_tableau(i_bottom_card, i_col)
            # i列目のカードをj列目に移動する
            if j_col != -1:
                return Action.MOVE_TABLEAU_TO_TABLEAU, (i_col, j_col)

        # 3. 山札の一番上のカードを場札に移動できるか

        # 山札にカードが表示されている場合
        if self.stock_top_idx != 0:
            # 山札の一番上のカード
            stock_top_card = self.stock_cards[self.stock_top_idx]

            # 場札に移動できるか
            j_col = self.can_move_to_tableau(stock_top_card, -1)
            # 山札のカードをj列に移動する
            if j_col != -1:
                return Action.MOVE_STOCK_TO_TABLEAU, (j_col,)

        # 4. 山札をめくる

        # 山札に未知のカードがある場合
        if len(self.stock_cards) - 1 < self.n_stock:
            # 山札を1枚めくる
            return Action.DRAW_STOCK, (1,)

        # 山札が全て既知の場合
        else:
            # 山札を何枚めくるか
            for i in range(1, len(self.stock_cards)):
                # 探索対象
                i_stock_idx = (self.stock_top_idx + i) % len(self.stock_cards)
                if i_stock_idx == 0:
                    continue
                # 探索対象のカード
                card = self.stock_cards[i_stock_idx]

                # 組札に移動できるか
                if self.can_move_to_foundation(card):
                    return Action.DRAW_STOCK, (i,)

                # 場札に移動できるか
                if self.can_move_to_tableau(card, -1) != -1:
                    return Action.DRAW_STOCK, (i,)

        # 5. これ以上アクションできない

        return Action.DO_NOTHING, ()

    def can_move_to_foundation(self, card: Card) -> bool:
        """
        cardを組札に移動できるか
        """

        # 組札に移動できるカード
        desire_cards = []
        for i in range(4):
            # 組札がない場合，Aを移動できる
            if self.foundation_cards[i] == 0:
                desire_cards.append(Card(i, 1))

            # 組札がKの場合，移動できるカードなし
            elif self.foundation_cards[i] == 13:
                continue

            # 組札がA~Qの場合，次のカードを移動できる
            else:
                desire_cards.append(Card(i, self.foundation_cards[i] + 1))

        return card in desire_cards

    def can_move_to_tableau(self, card: Card, ignore_col: int = -1) -> int:
        """
        cardを場札に移動できるか

        Parameters
        ----------
        ignore_col : int
            無視する移動先の列インデックス

        Returns
        -------
        moveable index : int
            移動先の列インデックス
        """

        # 場札に移動できるカード
        desire_cards = []
        for cards in self.opened_cards:
            # 列が空の場合
            if len(cards) == 0:
                # Kを移動できる
                desire_cards.append(Card.get_all_cards(values=[13]))
            else:
                # その列の一番上の表カード
                top_card = cards[-1]
                # その列に移動できるカード
                desire_cards.append(Card.get_contrast_cards(top_card - 1))

        for i, cards in enumerate(desire_cards):
            # ignore_col列目は無視
            if i == ignore_col:
                continue

            # i列目に移動できる
            if card in cards:
                return i

        # 移動できない
        return -1

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
            self.foundation_cards[card.suit] = card.value

            # プレイヤーに通知
            send_msg(f"場札の {i_col + 1} 列目の {card} を組札に移動する")

        # 山札から組札に移動
        elif action == Action.MOVE_STOCK_TO_FOUNDATION:
            # 移動元 山札のカード
            card = self.stock_cards.pop(self.stock_top_idx)
            self.stock_top_idx -= 1
            self.n_stock -= 1

            # 盤面を更新
            # 移動先 組札を更新
            self.foundation_cards[card.suit] = card.value

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
            send_msg(f"場札の {i_col + 1} 列目の {card} を {j_col + 1} 列目に移動する")

        # 山札から場札に移動
        elif action == Action.MOVE_STOCK_TO_TABLEAU:
            # 移動元 山札のカード
            card = self.stock_cards.pop(self.stock_top_idx)
            self.stock_top_idx -= 1
            self.n_stock -= 1
            # 移動先 列
            j_col = args[0]

            # 盤面を更新
            # 移動先 カードを追加
            self.opened_cards[j_col].append(card)

            # プレイヤーに通知
            send_msg(f"山札の {card} を場札の {j_col + 1} 列目に移動する")

        # 山札をめくる
        elif action == Action.DRAW_STOCK:
            # 何枚めくるか
            n_draw = args[0]

            # カードを1枚めくる
            # self.stock_top_idx = (self.stock_top_idx + 1) % len(self.stock_cards)

            # プレイヤーに通知
            send_msg(f"山札をあと {n_draw} 枚めくる")

        # 何もできない
        elif action == Action.DO_NOTHING:
            # ゲームクリア判定
            if self.foundation_cards == [13, 13, 13, 13]:
                self.gameclear()
            else:
                self.gameover()

    def draw(self):

        print()
        print("SOLVER", "=" * 20)
        print(
            f"[山札]\t{self.stock_cards[self.stock_top_idx]} ({self.stock_top_idx + 1}/{len(self.stock_cards)})"
        )

        print("[組札]", end='')
        for i in range(4):
            print(f"\t{Suit(i)}\t: {self.foundation_cards[i]}")
        print()

        for i_col, cards in enumerate(self.opened_cards):
            print(f"[場札 {i_col + 1}]", end='')
            print(f"({self.n_closed[i_col]})\t", end='')
            for card in cards:
                print(card, end=', ')
            print()
        print("=" * 20)
        print()

    def gameover(self):
        """
        ゲームオーバー
        """

        send_msg("カードをこれ以上移動できません")
        exit()

    def gameclear(self):
        """
        ゲームクリア
        """

        send_msg("ゲームクリア")
        exit()


def send_msg(msg: str):
    print(msg)


def main():
    pass


if __name__ == "__main__":
    main()
