try:
    import pyautogui
except Exception:
    pyautogui = None


class KeyboardHandler:
    """Handles mapping action ints to keyboard presses.

    Mapping: -1 -> 'left', 0 -> no key, 1 -> 'right'
    If `pyautogui` is not available, actions are printed instead.
    """

    def __init__(self):
        self.enabled = pyautogui is not None
        self._pg = pyautogui
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
        if action == -1:
            #what the camera sees as right is the user's left
            key = 'right'
        elif action == 1:
            key = 'left'

        if key:
            if self.enabled:
                try:
                    self._pg.press(key)
                except Exception as e:
                    print(f"pyautogui error pressing {key}: {e}")
            else:
                print(f"KeyboardHandler (dry-run): would press '{key}'")
        else:
            # explicit no-op
            print("KeyboardHandler: no action (straight/uncertain)")

        self.last_action = action

        return key
