# Markham Lee (C) 2023 - 2026
# https://github.com/MarkhamLee/vision-pipeline-benchmarks
# Abstracts frame delivery from either a folder of video files
# or an RTSP stream. Yields frames as numpy arrays so orchestrators
# don't need to know the source type.
import cv2
import os
import sys
import time

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils.logging_utils import console_logging  # noqa: E402

logger = console_logging('video-source')


class VideoSource:

    def __init__(self, source_config: dict) -> None:
        self.source_type = source_config['type']          # 'folder' | 'rtsp'
        self.path = source_config.get('path', '')
        self.rtsp_url = source_config.get('rtsp_url', '')
        self.rtsp_limit_s = source_config.get('rtsp_limit_seconds', 300)

    def frames(self):
        """Yield frames from the configured source."""
        if self.source_type == 'folder':
            yield from self._folder_frames()
        elif self.source_type == 'rtsp':
            yield from self._rtsp_frames()
        else:
            raise ValueError(f'Unknown source type: {self.source_type}')

    def _folder_frames(self):
        """Iterate over all video files in a folder, yield frames."""
        extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        video_files = sorted([
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if os.path.splitext(f)[1].lower() in extensions
        ])

        if not video_files:
            raise FileNotFoundError(f'No video files found in: {self.path}')

        logger.info('Found %d video file(s) in %s',
                    len(video_files),
                    self.path)

        for video_path in video_files:
            cap = cv2.VideoCapture(video_path)
            logger.info('Processing: %s', video_path)
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    yield frame
            finally:
                cap.release()

    def _rtsp_frames(self):
        """Read from an RTSP stream up to rtsp_limit_s seconds."""
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            raise ConnectionError('Could not open RTSP stream %s:',
                                  {self.rtsp_url})

        logger.info('RTSP stream opened | limit=%ds',
                    self.rtsp_limit_s)
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
