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
            if isinstance(suit, int):
                suit = Suit(suit)
            else:
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
        if card is None:
            return False
        if not isinstance(card, Card):
            raise TypeError("Invalid card provided.")

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

        contrast_suits = [suit for suit in Suit if suit.is_red() != card.is_red()]
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
