# Markham Lee (C) 2026
# https://github.com/MarkhamLee/vision-pipeline-benchmarks
# Abstracts frame delivery from either a folder of video files
# or an RTSP stream. For folder mode, emits video lifecycle events
# with metadata so orchestrators can compute per-video summaries.
import cv2
import sys
import time
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.logging_utils import console_logging  # noqa: E402

logger = console_logging('video-source')


class VideoSource:
    def __init__(self, source_config: dict) -> None:
        self.source_type = source_config['type']
        self.path = source_config.get('path', '')
        self.rtsp_url = source_config.get('rtsp_url', '')
        self.rtsp_limit_s = source_config.get('rtsp_limit_seconds', 300)

    def frames(self):
        if self.source_type == 'folder':
            yield from self._folder_frames()
        elif self.source_type == 'rtsp':
            yield from self._rtsp_frames()
        else:
            raise ValueError(f'Unknown source type: {self.source_type}')

    def _folder_frames(self):
        extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        video_dir = Path(self.path)
        video_files = sorted([
            p for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ])

        if not video_files:
            raise FileNotFoundError(f'No video files found in: {self.path}')

        logger.info('Found %d video file(s) in %s',
                    len(video_files),
                    self.path)

        for video_path in video_files:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FileNotFoundError(f'Could not open video file: {video_path}')  # noqa: E501

            native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_s = (frame_total / native_fps) if native_fps > 0 else 0.0
            video_name = video_path.name

            logger.info('Processing: %s | %sx%s | native_fps=%.3f | frames=%d',
                        video_path, width, height, native_fps, frame_total)

            yield {
                'event': 'video_start',
                'video_path': str(video_path),
                'video_name': video_name,
                'native_fps': native_fps,
                'frame_total': frame_total,
                'width': width,
                'height': height,
                'duration_s': duration_s,
            }

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield {
                        'event': 'frame',
                        'video_path': str(video_path),
                        'video_name': video_name,
                        'frame': frame,
                    }
            finally:
                cap.release()

            yield {
                'event': 'video_end',
                'video_path': str(video_path),
                'video_name': video_name,
            }

    def _rtsp_frames(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            raise ConnectionError(f'Could not open RTSP stream {self.rtsp_url}')  # noqa: E501

        logger.info('RTSP stream opened | limit=%ds', self.rtsp_limit_s)
        start = time.monotonic()

        try:
            while True:
                if time.monotonic() - start >= self.rtsp_limit_s:
                    logger.info('RTSP time limit reached')
                    break
                ret, frame = cap.read()
                if not ret:
                    logger.warning('RTSP stream dropped, stopping')
                    break
                yield frame
        finally:
            cap.release()
