#!/usr/bin/env python3
"""
Production-grade event-driven input logger with screen recording for gameplay data collection.

This script logs all keyboard and mouse events with precise timestamps, and captures
screen frames at 5 FPS for training Vision-Language Models on gameplay data.

ETHICAL USE ONLY: This tool should only be used to collect your own input data
for legitimate purposes such as ML research, UX studies, or personal analytics.
"""

import csv
import time
import threading
import queue
from pathlib import Path
from typing import Set
import pyautogui
from pynput import keyboard, mouse
import mss
from PIL import Image

class InputLogger:
    """Thread-safe event-driven input logger with screen recording."""

    # Screen recording rate
    SCREEN_RECORD_FPS = 5
    SCREEN_RECORD_INTERVAL = 1.0 / SCREEN_RECORD_FPS  # 0.2 seconds

    def __init__(self):
        """
        Initialize the input logger.

        Args:
            output_file: Path to the output CSV file
            frames_dir: Directory to save screen recording frames
        """
        i = 0
        while Path(f"episode{i}").exists():
            i += 1
        root = Path(f"episode{i}.")
        root.mkdir()
        self.events_file_path = root / "events.csv"
        self.frames_dir = root / "frames"
        self.meta_file = root / "meta.csv"
        self.start_time = None
        self.running = False

        # Thread-safe shared state
        self.state_lock = threading.Lock()
        self.pressed_keys: Set[str] = set()
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_wheel = 0

        # Event queue for lock-light event logging
        self.event_queue = queue.Queue()

        # Screen resolution (captured at startup)
        self.screen_width = 0
        self.screen_height = 0

        # Event listeners
        self.keyboard_listener = None
        self.mouse_listener = None

        # Screen recorder thread
        self.screen_recorder_thread = None

        # Event writer thread
        self.event_writer_thread = None
        self.events_file = None
        self.events_writer = None

    def _get_screen_resolution(self) -> tuple[int, int]:
        """
        Get the current screen resolution.

        Returns:
            Tuple of (width, height) in pixels
        """
        try:
            size = pyautogui.size()
            return size.width, size.height
        except Exception as e:
            print(f"Warning: Could not get screen size: {e}")
            return 1920, 1080  # Fallback to common resolution

    def _on_mouse_move(self, x: int, y: int):
        """
        Callback for mouse move events.

        Args:
            x: Mouse X coordinate
            y: Mouse Y coordinate
        """
        with self.state_lock:
            self.mouse_x = x
            self.mouse_y = y

    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int):
        """
        Callback for mouse scroll events.

        Args:
            x: Mouse X coordinate
            y: Mouse Y coordinate
            dx: Horizontal scroll amount
            dy: Vertical scroll amount
        """
        with self.state_lock:
            self.mouse_wheel += dy

    def _on_mouse_click(self, x: int, y: int, button, pressed: bool):
        """
        Callback for mouse click events.

        Args:
            x: Mouse X coordinate
            y: Mouse Y coordinate
            button: The mouse button (mouse.Button.left, mouse.Button.right, etc.)
            pressed: True if button was pressed, False if released
        """
        # Convert button to string
        button_name = f"mouse_{button.name}"

        # Get precise timestamp immediately
        t_sec = time.perf_counter() - self.start_time if self.start_time else 0

        with self.state_lock:
            if pressed:
                # Ignore OS autorepeat
                if button_name in self.pressed_keys:
                    return
                self.pressed_keys.add(button_name)
                event_type = "mouse_down"
            else:
                self.pressed_keys.discard(button_name)
                event_type = "mouse_up"

        # Queue event for writing (lock-free)
        self.event_queue.put((t_sec, event_type, button_name))

    def _on_key_press(self, key):
        """
        Callback for key press events.

        Args:
            key: The pressed key (pynput Key or KeyCode object)
        """
        key_str = self._key_to_string(key)

        # Get precise timestamp immediately
        t_sec = time.perf_counter() - self.start_time if self.start_time else 0

        with self.state_lock:
            # Ignore OS autorepeat
            if key_str in self.pressed_keys:
                return
            self.pressed_keys.add(key_str)

        # Queue event for writing (lock-free)
        self.event_queue.put((t_sec, "key_down", key_str))

    def _on_key_release(self, key):
        """
        Callback for key release events.

        Args:
            key: The released key (pynput Key or KeyCode object)
        """
        key_str = self._key_to_string(key)

        # Get precise timestamp immediately
        t_sec = time.perf_counter() - self.start_time if self.start_time else 0

        with self.state_lock:
            self.pressed_keys.discard(key_str)

        # Queue event for writing (lock-free)
        self.event_queue.put((t_sec, "key_up", key_str))

    @staticmethod
    def _key_to_string(key) -> str:
        """
        Convert a pynput key to a string representation.

        Args:
            key: pynput Key or KeyCode object

        Returns:
            String representation of the key (unmodified by shift/ctrl)
        """
        try:
            # Handle special keys (Key.ctrl, Key.shift, etc.)
            if hasattr(key, 'name'):
                return key.name
            # Handle alphanumeric keys - use vk to get unshifted key
            elif hasattr(key, 'vk'):
                # For letter keys (A-Z), convert to lowercase
                if hasattr(key, 'char') and key.char is not None:
                    # Return lowercase version to get base key
                    return key.char.lower()
                return str(key)
            else:
                return str(key)
        except AttributeError:
            return str(key)

    def _event_writer_loop(self):
        """
        Event writer loop that writes events from the queue to events.csv.

        This runs in a separate thread to minimize lock contention in callbacks.
        """
        event_count = 0

        while self.running:
            try:
                # Get event from queue with timeout
                t_sec, event_type, name = self.event_queue.get(timeout=0.1)

                # Write to events CSV with full precision
                self.events_writer.writerow([
                    f"{t_sec:.6f}",
                    event_type,
                    name
                ])

                event_count += 1

                # Flush every 30 events
                if event_count % 30 == 0:
                    self.events_file.flush()

            except queue.Empty:
                continue

        # Drain remaining events on shutdown
        while not self.event_queue.empty():
            try:
                t_sec, event_type, name = self.event_queue.get_nowait()
                self.events_writer.writerow([
                    f"{t_sec:.6f}",
                    event_type,
                    name
                ])
            except queue.Empty:
                break

        self.events_file.flush()
        print(f"Event logging stopped. Logged {event_count} events.")

    def _screen_recorder_loop(self):
        """
        Screen recording loop that captures at 5 FPS.

        Saves frames with timestamps synchronized to the input log start time.
        """
        next_capture_time = time.perf_counter()
        frame_count = 0

        print(f"Screen recording started at {self.SCREEN_RECORD_FPS} FPS...")

        with mss.mss() as sct:
            # Get the primary monitor
            monitor = sct.monitors[1]

            while self.running:
                current_time = time.perf_counter()

                # If we're at or past the next capture time
                if current_time >= next_capture_time:
                    # Calculate timestamp relative to start time
                    t_sec = current_time - self.start_time

                    # Capture screenshot
                    screenshot = sct.grab(monitor)

                    # Convert to PIL Image
                    img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)

                    # Save frame with timestamp in filename
                    frame_filename = self.frames_dir / f"frame_{t_sec:.6f}.png"
                    img.save(frame_filename, compress_level=1)

                    frame_count += 1

                    # Calculate next capture time with drift correction
                    next_capture_time += self.SCREEN_RECORD_INTERVAL
                    if next_capture_time < current_time:
                        # We've fallen behind; reset to current time
                        next_capture_time = current_time + self.SCREEN_RECORD_INTERVAL

                # Sleep for a short time to avoid busy-waiting
                sleep_time = max(0, next_capture_time - time.perf_counter() - 0.001)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        print(f"Screen recording stopped. Captured {frame_count} frames.")

    def start(self):
        """Start logging input data and screen recording."""
        if self.running:
            print("Logger is already running!")
            return

        # Record screen resolution
        self.x_resolution, self.y_resolution = self._get_screen_resolution()
        self.screen_width, self.screen_height = self.x_resolution, self.y_resolution

        with open(self.meta_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x_resolution', 'y_resolution'])
            writer.writerow([self.x_resolution, self.y_resolution])

        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")

        # Create frames directory
        self.frames_dir.mkdir(exist_ok=True)
        print(f"Frames directory: {self.frames_dir.absolute()}")

        # Initialize start time (shared between event logging and screen recording)
        self.start_time = time.perf_counter()
        self.running = True

        # Set up events CSV file (event-driven)
        self.events_file = open(self.events_file_path, 'w', newline='', encoding='utf-8')
        self.events_writer = csv.writer(self.events_file)
        self.events_writer.writerow(['t_sec', 'type', 'name'])

        print(f"Events logging to: {self.events_file_path.absolute()}")

        # Start event writer thread
        self.event_writer_thread = threading.Thread(
            target=self._event_writer_loop,
            daemon=True
        )
        self.event_writer_thread.start()

        # Start keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.keyboard_listener.start()

        # Start mouse listener
        self.mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll
        )
        self.mouse_listener.start()

        # Start screen recorder thread (synchronized start time)
        self.screen_recorder_thread = threading.Thread(
            target=self._screen_recorder_loop,
            daemon=True
        )
        self.screen_recorder_thread.start()

        print("Press Ctrl+C to stop logging.")

        try:
            # Keep main thread alive
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping event logging and screen recording...")
        finally:
            self.stop()
            self.events_file.close()
            print(f"Events saved to: {self.events_file_path.absolute()}")

    def stop(self):
        """Stop event logging and screen recording."""
        self.running = False

        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.screen_recorder_thread:
            self.screen_recorder_thread.join(timeout=1.0)
        if self.event_writer_thread:
            self.event_writer_thread.join(timeout=1.0)

def main():
    """Main entry point for the input logger with screen recording."""
    print("=" * 60)
    print("Input Event Logger + Screen Recorder")
    print("Gameplay Data Collection for ML Training")
    print("=" * 60)
    print()
    print("IMPORTANT: This tool is intended for personal use only.")
    print("Only log your own input for legitimate purposes such as")
    print("ML research, game AI development, or personal analytics.")
    print()
    print("Recording: Events + 5 FPS screen capture")
    print("=" * 60)
    print()
    # Create and start logger
    logger = InputLogger()
    logger.start()


if __name__ == "__main__":
    main()
