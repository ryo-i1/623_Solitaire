import numpy as np
import cv2
import pyautogui as pag
import time

from pathlib import Path

from Card import Card, Suit
from SolitaireSolver import SolitaireSolver, Action
from Identifier import Identifier

work_dir = Path.cwd().resolve().parent
log_dir = work_dir.joinpath("log")

# 画面をキャプチャする間隔と上限 (sec)
CAPTURE_INTERVAL = 0.3
CAPTURE_LIMIT = 10


def main():
    solver = SolitaireSolver()
    identifier = Identifier()

    # waiting 3 seconds
    send_log("Waiting 3 seconds before starting...")
    time.sleep(1)
    send_log("Waiting 2 seconds before starting...")
    time.sleep(1)
    send_log("Waiting 1 seconds before starting...")
    time.sleep(1)

    # capture the window
    img_capture = identifier.capture_solitaire_window()
    # init gameboard area
    identifier.init_area_position(img_capture)

    i_game = 0
    while True:
        i_game += 1
        send_log(f"Step {i_game}:")

        # capture the window
        img_capture = identifier.capture_solitaire_window()
        for i_cap in range(np.ceil(CAPTURE_LIMIT / CAPTURE_INTERVAL).astype(int)):
            time.sleep(CAPTURE_INTERVAL)
            img_capture_prev = img_capture.copy()
            img_capture = identifier.capture_solitaire_window()
            diff = cv2.absdiff(img_capture, img_capture_prev)
            if np.mean(diff) < 0.01:
                break
            elif i_cap == CAPTURE_LIMIT // CAPTURE_INTERVAL - 1:
                send_log("The game is stuck. Continue with unstable image.")

        # identify the top cards
        stock_top_card, tableau_top_cards = identifier.get_top_cards(img_capture)

        send_log(f"\tStock: {stock_top_card}", debug=True)
        send_log("\tTableau: [", end="", debug=True)
        for card in tableau_top_cards:
            send_log(f"{card}, ", end="", debug=True)
        send_log("]", debug=True)

        # update solver
        action, args, is_end = solver.update(stock_top_card, tableau_top_cards)

        # run the action
        if action == Action.MOVE_TABLEAU_TO_FOUNDATION:
            # click the tableau card
            pos = identifier.get_tableau_top_pos(args[0])

        elif action == Action.MOVE_STOCK_TO_FOUNDATION:
            # click the stock card
            pos = identifier.get_stock_open_pos()

        elif action == Action.MOVE_TABLEAU_TO_TABLEAU:
            # click the tableau card
            n_closed_card = solver.get_n_closed_card(args[0])
            pos = identifier.get_tableau_bottom_pos(args[0], n_closed_card)

        elif action == Action.MOVE_STOCK_TO_TABLEAU:
            # click the stock card
            pos = identifier.get_stock_open_pos()

        elif action == Action.DRAW_STOCK:
            # click the stock
            pos = identifier.get_stock_closed_pos()

        elif action == Action.DO_NOTHING:
            break

        # click the position
        pag.click(x=identifier.window.left + pos[0], y=identifier.window.top + pos[1])

        # blank line
        send_log("")

        # save the captured image
        savepath = log_dir.joinpath(f"step_{i_game}.png")
        cv2.imwrite(str(savepath), cv2.cvtColor(img_capture, cv2.COLOR_RGB2BGR))

        # game end
        if is_end:
            break

        # ESC key to exit
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

    if solver.is_clear():
        send_log("Game clear!")
    else:
        send_log("Game over...")


def send_log(msg, end="\n", debug=False):
    if not debug:
        print(msg, end=end)


if __name__ == "__main__":
    main()
