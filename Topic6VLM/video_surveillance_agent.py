"""
Video Surveillance Agent

This program analyzes a video file to detect when a person enters and exits the scene:
- Extracts one frame every 2 seconds using OpenCV
- Queries LLaVA per frame: "Is there a person visible in this image?"
- Uses a state machine to detect No→Yes (entered) and Yes→No (exited) transitions
- Reports timestamps in MM:SS format
"""

import sys
import base64

import cv2
import ollama


PROMPT = "Is there a person visible in this image? Reply with only Yes or No."
MAX_SIZE = 512 



def extract_frames(video_path: str) -> list:
    """
    Extract one frame every 2 seconds from the given video file.

    Args:
        video_path: Path to the video file

    Returns:
        List of frames as numpy arrays (BGR), where index i = timestamp i*2 seconds
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * 2)   # ~2-second frame interval

    frames = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval == 0:
            frames.append(frame)
        frame_num += 1
    cap.release()

    return frames


def frame_to_b64(frame, max_size: int = MAX_SIZE) -> str:
    """
    Resize and base64-encode a video frame as a JPEG string.

    Args:
        frame: numpy array (BGR) from OpenCV
        max_size: Maximum size for the longest dimension

    Returns:
        Base64-encoded JPEG string
    """
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def ask_llava(image_b64: str) -> bool:
    """
    Ask LLaVA whether a person is visible in the given image.

    Args:
        image_b64: Base64-encoded JPEG string

    Returns:
        True if LLaVA detects a person, False otherwise
    """
    response = ollama.chat(
        model="llava",
        messages=[{"role": "user", "content": PROMPT, "images": [image_b64]}],
    )
    answer = response["message"]["content"].strip().lower()
    return answer.startswith("yes")


def fmt_time(seconds: int) -> str:
    """Format seconds as MM:SS."""
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def main(video_path: str) -> None:
    print(f"Loading video: {video_path}")
    frames = extract_frames(video_path)

    if not frames:
        print("No frames extracted. Check that the video path is valid.")
        sys.exit(1)

    print(f"Extracted {len(frames)} frames (one every ~2 seconds)\n")

    person_present = False
    no_streak = 0
    DEBOUNCE = 2  # consecutive "No" frames required to confirm exit
    events = []   # list of (timestamp_str, event_label)

    for i, frame in enumerate(frames):
        timestamp = i * 2   # seconds
        image_b64 = frame_to_b64(frame)
        person_in_frame = ask_llava(image_b64)

        label = "Person detected" if person_in_frame else "No person"
        print(f"[{fmt_time(timestamp)}] Frame {i + 1}/{len(frames)} — {label}")

        if person_in_frame and not person_present:
            no_streak = 0
            events.append((fmt_time(timestamp), "Person ENTERED"))
            person_present = True
        elif not person_in_frame and person_present:
            no_streak += 1
            if no_streak >= DEBOUNCE:
                events.append((fmt_time(timestamp), "Person EXITED"))
                person_present = False
                no_streak = 0
        else:
            no_streak = 0

    print()
    if events:
        for timestamp_str, event_label in events:
            print(f"{event_label} at {timestamp_str}")
    else:
        print("No person detected in any frame.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_surveillance_agent.py <video_file>")
        sys.exit(1)
    main(sys.argv[1])
