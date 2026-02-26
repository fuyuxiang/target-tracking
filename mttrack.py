#!/usr/bin/env python3
"""
Multi-Target Tracking with VL Classification.

Usage:
    python mttrack.py --input ./data/test_multi_target_tracker_video.mp4 --output ./out/result.mp4

Environment variables:
    VLLM_BASE_URL: VLLM API base URL
    VLLM_API_KEY: VLLM API key
    VLLM_MODEL: VLLM model name
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mttrack.domain import ByteTrackTracker, SORTTracker
from mttrack.infrastructure import (
    YoloDetector,
    VllmClient,
    VideoReader,
    VideoWriter,
)
from mttrack.service import TrackerService, LabelService
from mttrack.annotators import TrackingAnnotator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Target Tracking with VL Classification"
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video path"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output video path"
    )

    parser.add_argument(
        "--tracker",
        choices=["bytetrack", "sort"],
        default="bytetrack",
        help="Tracker type (default: bytetrack)"
    )

    parser.add_argument(
        "--yolo-model",
        default="./yolo/yolo26x.pt",
        help="YOLO model path (default: ./yolo/yolo26x.pt)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for YOLO (default: cuda)"
    )

    parser.add_argument(
        "--enable-vl",
        action="store_true",
        default=True,
        help="Enable VL classification"
    )

    parser.add_argument(
        "--vl-interval",
        type=int,
        default=30,
        help="VL classification interval in frames (default: 30)"
    )

    parser.add_argument(
        "--vl-timeout",
        type=int,
        default=30,
        help="VL API timeout in seconds (default: 30)"
    )

    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Show FPS on output video"
    )

    return parser.parse_args()


def create_vllm_client(args) -> VllmClient:
    """Create VLLM client from environment or args."""
    base_url = os.getenv("VLLM_BASE_URL")
    api_key = os.getenv("VLLM_API_KEY")
    model = os.getenv("VLLM_MODEL")

    if not base_url:
        print("[Warning] VLLM_BASE_URL not set, VL classification disabled")
        return None
    if not api_key:
        print("[Warning] VLLM_API_KEY not set, VL classification disabled")
        return None
    if not model:
        print("[Warning] VLLM_MODEL not set, using default")
        model = "/models/Qwen/Qwen3-VL-8B-Instruct"

    return VllmClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=args.vl_timeout,
    )


def crop_track(frame, bbox, margin=10):
    """Crop track region from frame.

    Args:
        frame: Input frame
        bbox: Bounding box (x1, y1, x2, y2)
        margin: Margin to add around bbox

    Returns:
        Cropped image or None
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    # Add margin
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def main():
    """Main entry point."""
    args = parse_args()

    # Resolve paths
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        print(f"[Error] Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Input: {input_path}")
    print(f"[Info] Output: {output_path}")
    print(f"[Info] Tracker: {args.tracker}")
    print(f"[Info] YOLO model: {args.yolo_model}")

    # Initialize detector
    print("[Info] Loading YOLO model...")
    detector = YoloDetector(
        model_path=args.yolo_model,
        confidence_threshold=args.confidence,
        device=args.device,
    )
    detector.warmup()

    # Initialize VL client
    vllm_client = None
    if args.enable_vl:
        print("[Info] Initializing VLLM client...")
        vllm_client = create_vllm_client(args)
        if vllm_client:
            print(f"[Info] VLLM client initialized (base_url: {vllm_client.base_url})")
        else:
            print("[Warning] VL classification disabled due to missing config")
    else:
        print("[Info] VL classification disabled by user")

    # Initialize tracker service
    tracker_service = TrackerService(
        detector=detector,
        tracker_type=args.tracker,
    )

    # Initialize label service
    label_service = LabelService(
        vllm_client=vllm_client,
        enabled=args.enable_vl and vllm_client is not None,
        label_interval=args.vl_interval,
        cache_ttl=60.0,
    )

    # Initialize annotator
    annotator = TrackingAnnotator()

    # Process video
    print("[Info] Processing video...")

    frame_count = 0
    fps = 30.0  # Default, will be updated

    with VideoReader(input_path) as reader:
        fps = reader.fps
        print(f"[Info] Video: {reader.width}x{reader.height} @ {fps:.2f} fps")

        # Initialize writer
        writer = VideoWriter(
            output_path,
            fps=fps,
            frame_size=(reader.width, reader.height),
        )

        try:
            for frame_id, frame in reader:
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"[Info] Processing frame {frame_count}...")

                # Detect and track
                frame_tracks = tracker_service.process_frame(frame)

                # Get active track IDs
                active_track_ids = {t.track_id for t in frame_tracks.tracks}

                # Clean up old cache entries
                label_service.cleanup_old_tracks(active_track_ids)

                # Label tracks with VL model
                if label_service.enabled and vllm_client:
                    for track in frame_tracks.tracks:
                        # Check if should label
                        if label_service.should_label(track.track_id, frame_id):
                            # Crop image
                            crop = crop_track(frame, track.bbox)
                            if crop is not None and crop.size > 0:
                                # Get VL classification
                                result = label_service.label_track(
                                    track.track_id,
                                    crop,
                                    frame_id
                                )
                                if result and result.class_name != "unknown":
                                    # Update track with VL label
                                    track.label = result.class_name
                                    track.label_confidence = result.confidence
                                    tracker_service.update_track_label(
                                        track.track_id,
                                        result.class_name,
                                        result.confidence
                                    )

                # Get labels for tracks
                for track in frame_tracks.tracks:
                    if not track.label:
                        cached = label_service.get_cached_label(track.track_id)
                        if cached:
                            track.label = cached.class_name
                            track.label_confidence = cached.confidence

                # Annotate frame
                annotated = annotator.annotate(frame, frame_tracks.tracks)

                # Write frame
                writer.write(annotated)

        finally:
            writer.close()

    print(f"[Info] Done! Processed {frame_count} frames")
    print(f"[Info] Output saved to: {output_path}")


if __name__ == "__main__":
    main()
