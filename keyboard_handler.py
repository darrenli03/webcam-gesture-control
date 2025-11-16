import pyautogui


class KeyboardHandler:
    """Handles mapping action ints to keyboard presses.

    Mapping: -1 -> 'right', 0 -> no key, 1 -> 'left'
    If `pyautogui` is not available, actions are printed instead.
    """

    def __init__(self):
        self.enabled = True
        self.last_action = None

    def perform_action(self, action):
        """Perform keyboard action for given action int.

        Only performs a key press when the action changes from the previous.
        """
        if action is None:
            return
        if action == self.last_action:
            return

        key = None
        if action == "right":
            # user tilted head to right
            key = 'down'
        elif action == "left":
            # user tilted head to left
            key = 'up'

        if key:
            if self.enabled:
                try:
                    pyautogui.press(key)
                    print("keypress:", key)
                except Exception as e:
                    print(f"pyautogui error pressing {key}: {e}")
            else:
                print(f"KeyboardHandler (dry-run): would press '{key}'")
        else:
            # explicit no-op
            print("KeyboardHandler: no action (straight/uncertain)")

        self.last_action = action

        return key
