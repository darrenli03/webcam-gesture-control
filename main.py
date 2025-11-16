"""Entrypoint for the gesture-based keyboard demo.

Creates a `VideoProcessor` to read frames and a `KeyboardHandler` to perform
keyboard actions derived from the processor output.
"""

from video_processor import VideoProcessor
from keyboard_handler import KeyboardHandler


def main():
	vp = VideoProcessor(source=0, window_size=10, show=True)
	kh = KeyboardHandler()

	try:
		while True:
			action, quit_flag = vp.read_action()
			action_taken = kh.perform_action(action)
			
			if quit_flag:
				break
			
			if action_taken == 'left' or action_taken == 'right':
				print(f"Key pressed: {action_taken}")
	
	finally:
		vp.release()


if __name__ == "__main__":
	main()

