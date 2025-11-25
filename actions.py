# actions.py
# This class handles all system actions triggered by hand gestures

import pyautogui
import time

class GestureActions:
    def __init__(self, cooldown=0.5):
        """
        cooldown: Minimum time (in seconds) between repeated actions
        """
        self.cooldown = cooldown
        self.last_action_time = {}  # Track last action time per gesture
        self.dragging = False       # Track if C gesture is active
        pyautogui.FAILSAFE = False  # Disable mouse moving to corner exit

    def _can_execute(self, gesture):
        """
        Check if enough time has passed since last execution
        """
        current_time = time.time()
        last_time = self.last_action_time.get(gesture, 0)
        if current_time - last_time >= self.cooldown:
            self.last_action_time[gesture] = current_time
            return True
        return False

    # ---------------- ONE-TIME ACTIONS ----------------
    def next_slide(self):
        if self._can_execute("01_palm"):
            pyautogui.press("right")

    def previous_slide(self):
        if self._can_execute("08_palm_moved"):
            pyautogui.press("left")

    def left_click(self):
        if self._can_execute("03_fist"):
            pyautogui.click()

    def double_click(self):
        if self._can_execute("04_fist_moved"):
            pyautogui.doubleClick()

    def right_click(self):
        if self._can_execute("05_thumb"):
            pyautogui.click(button="right")

    def close_window(self):
        if self._can_execute("10_down"):
            pyautogui.hotkey("alt", "f4")

    # ---------------- CONTINUOUS ACTIONS ----------------
    # def move_cursor(self, x, y, gesture=None, smooth=True):
    #     """
    #     Move the mouse pointer.
    #     x, y: normalized coordinates (0 to 1)
    #     gesture: optional, if C gesture, hold mouse down
    #     smooth: whether to move smoothly to avoid jitter
    #     """
    #     screen_width, screen_height = pyautogui.size()
    #     target_x = int(x * screen_width)
    #     target_y = int(y * screen_height)

    #     # If C gesture, start dragging
    #     if gesture == "09_c":
    #         if not self.dragging:
    #             pyautogui.mouseDown()  # Start holding
    #             self.dragging = True
    #     else:
    #         # If previously dragging, release
    #         if self.dragging:
    #             pyautogui.mouseUp()
    #             self.dragging = False

    #     if smooth:
    #         pyautogui.moveTo(target_x, target_y, duration=0.1)
    #     else:
    #         pyautogui.moveTo(target_x, target_y, duration=0)


    def move_cursor(self, x, y, smooth=True, drag=False):
        """
        Move the mouse pointer.
        x, y: normalized coordinates (0 to 1)
        smooth: whether to move smoothly to avoid jitter
        drag: whether to hold the mouse down while moving
        """
        screen_width, screen_height = pyautogui.size()
        target_x = int(x * screen_width)
        target_y = int(y * screen_height)

        if drag:
            # Hold mouse down while moving
            pyautogui.mouseDown()
            pyautogui.moveTo(target_x, target_y, duration=0.1 if smooth else 0)
            # Do NOT release here, keep pressed
        else:
            pyautogui.moveTo(target_x, target_y, duration=0.1 if smooth else 0)


    def scroll_up(self):
        pyautogui.scroll(300)  # Scroll up
     
    def scroll_down(self):
        pyautogui.scroll(-300)  # Scroll down

    # ---------------- MAPPING HELPER ----------------
    def perform_action(self, gesture, **kwargs):
        """
        Main interface: call this with detected gesture
        kwargs: extra info for actions like mouse movement
        """
        if gesture == "01_palm":
            self.next_slide()
        elif gesture == "08_palm_moved":
            self.previous_slide()
        elif gesture == "03_fist":
            self.left_click()
        elif gesture == "04_fist_moved":
            self.double_click()
        elif gesture == "05_thumb":
            self.right_click()
        elif gesture == "10_down":
            self.close_window()
        elif gesture in ["06_index", "09_c"]:
            x, y = kwargs.get("x", 0.5), kwargs.get("y", 0.5)
            self.move_cursor(x, y, gesture=gesture)
        elif gesture == "02_l":
            self.scroll_up()
        elif gesture == "07_ok":
            self.scroll_down()
