import numpy as np
import cv2
import pygetwindow as gw
import mss
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from Card import Card, Suit


SOLITAIRE_WINDOW_TITLE = "Solitaire & Casual Games"
TABLEAU_COLS = 7

WORK_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = WORK_DIR.joinpath("resources", "template_images")


class Identifier:
    """
    画面上のカードを識別する
    """

    def __init__(self, template_dir=TEMPLATE_DIR):
        self.load_template_images(template_dir)
        self.find_solitaire_window()

    def load_template_images(self, template_dir: Path):
        """テンプレート画像を読み込む"""

        if not template_dir.exists():
            raise FileNotFoundError(f"Directory not found: {template_dir}")

        templates = {}
        for filepath in template_dir.glob("*.png"):
            # load as RGB
            img = cv2.imread(str(filepath))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            templates[filepath.stem] = img
        self.templates = templates

        # SIFT
        sift = cv2.SIFT_create()
        temp_descriptors = {}
        for temp_name, temp_img in self.templates.items():
            temp_img_gray = cv2.cvtColor(temp_img, cv2.COLOR_RGB2GRAY)
            kp, des = sift.detectAndCompute(temp_img_gray, None)
            temp_descriptors[temp_name] = des
        self.temp_descriptors = temp_descriptors

    def find_solitaire_window(self):
        """ソリティアのウィンドウを取得する"""

        windows = gw.getWindowsWithTitle(SOLITAIRE_WINDOW_TITLE)
        if not windows:
            print("Solitaire window not found.")
            self.window = None
            return False
        else:
            self.window = windows[0]
            return True

    def capture_solitaire_window(self) -> np.ndarray:
        """ウィンドウをキャプチャする"""

        # 指定領域をキャプチャ
        with mss.mss() as sct:
            monitor = {
                "left": self.window.left,
                "top": self.window.top,
                "width": self.window.width,
                "height": self.window.height,
            }
            screenshot = sct.grab(monitor)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return img

    def init_area_position(self, img: np.ndarray) -> None:
        """画面上の各領域位置を取得する"""
        # img は初期画面

        # ------------------------------
        # 上下の UI を除いた領域を取得
        # ------------------------------
        # 背景色（薄緑）
        bg_mask = mask_by_bg(img)

        bg_mask_y = np.mean(bg_mask, axis=1) > 0.9
        bg_mask_y_diff = np.diff(bg_mask_y.astype(int))

        # 上下の UI を除いたエリア
        self.game_area_top = np.where(bg_mask_y_diff == 1)[0][0] + 1
        self.game_area_bottom = np.where(bg_mask_y_diff == -1)[0][-1] + 1

        # ------------------------------
        # 場札の各列の位置を取得
        # ------------------------------
        # 白色
        white_mask = mask_by_white(img)
        white_mask = self.mask_game_area(white_mask)

        white_mask_x = np.mean(white_mask, axis=0) > 0.01
        white_mask_x_diff = np.diff(white_mask_x.astype(int))

        # 場札の開始位置
        self.tableau_top = np.where(bg_mask_y_diff == -1)[0][1] + 1
        tableau_cols_left = np.where(white_mask_x_diff == 1)[0] + 1
        tableau_cols_right = np.where(white_mask_x_diff == -1)[0] + 1

        # card width
        self.card_width = np.max(tableau_cols_right - tableau_cols_left).astype(int)

        # tableau_cols_left[i] - tableau_cols_right[i-1] が小さいとき、その組を削除
        col_space = np.floor(
            np.median(tableau_cols_left[1:] - tableau_cols_right[:-1])
        ).astype(int)
        available_cols = [0]
        for i in range(1, len(tableau_cols_left)):
            if tableau_cols_left[i] - tableau_cols_right[i - 1] > col_space * 0.6:
                available_cols.append(i)
        self.tableau_cols_left = tableau_cols_left[available_cols]

        # TEST: 場札の列数が TABLEAU_COLS と一致するか
        assert (
            len(self.tableau_cols_left) == TABLEAU_COLS
        ), f"Failed to detect {TABLEAU_COLS} tableau columns. Detected {len(self.tableau_cols_left)} columns."

        # ------------------------------
        # 山札の位置を取得
        # ------------------------------
        # 山札の開始位置 (y座標)
        self.stock_top = np.where(bg_mask_y_diff == -1)[0][0] + 1

        # card height
        white_mask[: self.tableau_top, :] = False

        card_heights = []
        card_bottoms = []
        for left in self.tableau_cols_left:
            mask_white_col_y = (
                np.mean(white_mask[:, left : left + self.card_width], axis=1) > 0.9
            )
            mask_white_col_y_diff = np.diff(mask_white_col_y.astype(int))

            card_heights.append(
                np.where(mask_white_col_y_diff == -1)[0][-1]
                - np.where(mask_white_col_y_diff == 1)[0][0]
            )
            card_bottoms.append(np.where(mask_white_col_y_diff == -1)[0][-1])
        self.card_height = np.median(card_heights).astype(int)

        # 場札の裏札の高さ
        card_closed_heights = []
        for i in range(1, TABLEAU_COLS):
            card_closed_heights.append(card_bottoms[i] - card_bottoms[i - 1])
        self.card_closed_height = np.median(card_closed_heights).astype(int)

    def mask_game_area(self, img: np.ndarray) -> np.ndarray:
        """マスクに対して上下の UI 部分を除外する"""

        img = img.copy()
        img[: self.game_area_top, :] = False
        img[self.game_area_bottom :, :] = False
        return img

    def get_top_cards(self, img: np.ndarray) -> tuple[Card, list[Card]]:
        """山札と場札の一番上のカードを取得する"""

        # 山札
        stock_img = self.get_stock_top_card_area(img)
        stock_card = self.identify_card(stock_img)

        # 場札
        tableau_imgs = self.get_tableau_top_cards_area(img)
        tableau_cards = [self.identify_card(img) for img in tableau_imgs]

        # save images
        for i, (card, img) in enumerate(
            zip([stock_card] + tableau_cards, [stock_img] + tableau_imgs)
        ):
            if card is not None:
                save_path = WORK_DIR.joinpath("tmp", f"{card}.png")
                res = cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if not res:
                    print(f"Failed to save {save_path}. img shape: {img.shape}")

        return stock_card, tableau_cards

    def get_stock_top_card_area(self, img: np.ndarray) -> np.ndarray:
        """山札の一番上のカードの領域を取得する"""

        mask_white = mask_by_white(img)
        # 山札の領域
        mask_area = np.full_like(mask_white, False)
        mask_area[
            self.stock_top : self.stock_top + self.card_height,
            self.tableau_cols_left[1] : self.tableau_cols_left[2]
            + self.card_width // 2,
        ] = True

        # 右から探索
        mask = mask_white & mask_area
        mask_x = np.mean(mask, axis=0) > 0.01
        mask_x_diff = np.diff(mask_x.astype(int))
        if len(np.where(mask_x_diff == -1)[0]) == 0:
            return img[
                self.stock_top : self.stock_top + self.card_height,
                self.tableau_cols_left[1] : self.tableau_cols_left[1] + self.card_width,
                :,
            ]

        right = np.where(mask_x_diff == -1)[0][-1] + 2
        if right < self.tableau_cols_left[1] + self.card_width:
            right = self.tableau_cols_left[1] + self.card_width

        return img[
            self.stock_top : self.stock_top + self.card_height,
            right - self.card_width : right,
            :,
        ]

    def get_tableau_top_cards_area(self, img: np.ndarray) -> list[np.ndarray]:
        """場札の一番上のカードの領域を取得する"""

        self.tableau_cols_bottom = self.get_tableau_cols_bottom(img)

        areas = []
        for i_col in range(TABLEAU_COLS):
            left = self.tableau_cols_left[i_col]
            bottom = self.tableau_cols_bottom[i_col]

            areas.append(
                img[
                    bottom - self.card_height : bottom, left : left + self.card_width, :
                ]
            )
        return areas

    def get_tableau_cols_bottom(self, img: np.ndarray) -> None:
        """場札の一番下のカードの位置を取得する"""

        mask_white = mask_by_white(img)
        mask_white = self.mask_game_area(mask_white)

        bottom_list = []
        for left in self.tableau_cols_left:
            # 場札の領域
            mask_area = np.full_like(mask_white, False)
            mask_area[
                self.tableau_top :,
                left : left + self.card_width,
            ] = True

            # 下から探索
            mask = mask_white & mask_area
            mask_y = np.mean(mask, axis=1) > 0.01
            mask_y_diff = np.diff(mask_y.astype(int))
            if len(np.where(mask_y_diff == -1)[0]) == 0:
                bottom = self.tableau_top + self.card_height
            else:
                bottom = np.where(mask_y_diff == -1)[0][-1] + 2

            if bottom < self.tableau_top + self.card_height:
                bottom = self.tableau_top + self.card_height
            bottom_list.append(bottom)

        return bottom_list

    def compute_similarity(self, temp_name, des1):
        temp_des = self.temp_descriptors[temp_name]
        matches = cv2.BFMatcher().knnMatch(des1, temp_des, k=2)
        dists = [m.distance for m, _ in matches]
        similarity = np.mean(dists) if len(dists) > 0 else float('inf')
        return temp_name, similarity

    def identify_card(self, img: np.ndarray) -> Card:
        """カードを識別する"""

        # gray
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 画像が緑の場合
        if np.mean((img > 50) & (img < 200)) > 0.8:
            return None

        # resize
        temp_img = list(self.templates.values())[0]
        img = cv2.resize(img, (temp_img.shape[1], temp_img.shape[0]))
        # 2値化
        # img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

        # SIFT
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img, None)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(
                self.compute_similarity,
                self.temp_descriptors.keys(),
                [des1] * len(self.temp_descriptors),
            )

        scores = {}
        for temp_name, similarity in results:
            scores[temp_name] = similarity
        best_name, best_similarity = min(scores.items(), key=lambda x: x[1])

        # カードが置かれていない場合
        if str.startswith(best_name, "nac"):
            return None
        elif str.startswith(best_name, "back"):
            return None

        return Card.from_template_name(best_name)

    def get_stock_closed_pos(self) -> tuple[int, int]:
        """山札の閉じたカードの位置を取得する"""

        x = self.tableau_cols_left[0] + self.card_width // 2
        y = self.stock_top + self.card_height // 2
        return x, y

    def get_stock_open_pos(self) -> tuple[int, int]:
        """山札の開いたカードの位置を取得する"""

        x = self.tableau_cols_left[1] + (self.card_width // 4) * 3
        y = self.stock_top + self.card_height // 2
        return x, y

    def get_tableau_top_pos(self, col: int) -> tuple[int, int]:
        """場札の指定列の表のカードのうち、一番表のカードの位置を取得する"""

        x = self.tableau_cols_left[col] + self.card_width // 2
        y = self.tableau_cols_bottom[col] - self.card_height // 2
        return x, y

    def get_tableau_bottom_pos(self, col: int, n_closed_card: int) -> tuple[int, int]:
        """場札の指定列の表のカードのうち、根本のカードの位置を取得する"""

        x = self.tableau_cols_left[col] + self.card_width // 2
        y = (
            self.tableau_top
            + self.card_closed_height * n_closed_card
            + self.card_closed_height // 2
        )
        return x, y


def mask_by_color(img, color_range):
    r_mask = (img[:, :, 0] > color_range['R'][0]) & (img[:, :, 0] < color_range['R'][1])
    g_mask = (img[:, :, 1] > color_range['G'][0]) & (img[:, :, 1] < color_range['G'][1])
    b_mask = (img[:, :, 2] > color_range['B'][0]) & (img[:, :, 2] < color_range['B'][1])
    return r_mask & g_mask & b_mask


def mask_by_white(img):
    white_range = {'R': (200, 255), 'G': (200, 255), 'B': (200, 255)}
    return mask_by_color(img, white_range)


def mask_by_bg(img):
    bg_range = {'R': (5, 50), 'G': (100, 150), 'B': (55, 100)}
    return mask_by_color(img, bg_range)
