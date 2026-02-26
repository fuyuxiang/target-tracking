"""
VLLM client for VL model inference.
"""

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Optional

from PIL import Image

import numpy as np


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@dataclass
class VLClassificationResult:
    """VL classification result."""

    class_name: str
    confidence: float
    raw_response: str


class VllmClient:
    """VLLM client for vision-language model inference."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize VLLM client.

        Args:
            base_url: VLLM API base URL
            api_key: API key
            model: Model name
            timeout: Request timeout in seconds
            max_retries: Max number of retries
        """
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://10.132.19.82:50100")
        self.api_key = api_key or os.getenv("VLLM_API_KEY", "sk-8fA3kP2QxR7mJ9WZC6dE0T1B4yH5VnL")
        self.model = model or os.getenv("VLLM_MODEL", "/models/Qwen/Qwen3-VL-8B-Instruct")
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = None

    def _init_client(self) -> None:
        """Initialize the OpenAI client."""
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=self.base_url + "/v1",
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
            except Exception as e:
                print(f"[VllmClient] Failed to initialize client: {e}")
                self.client = None

    def is_available(self) -> bool:
        """Check if VLLM service is available."""
        self._init_client()
        if self.client is None:
            return False
        try:
            # Simple health check
            self.client.models.list()
            return True
        except Exception:
            return False

    def classify_crop(
        self,
        image: np.ndarray,
        track_id: int,
    ) -> VLClassificationResult:
        """Classify a cropped image using VL model.

        Args:
            image: Cropped image (BGR)
            track_id: Track ID for logging

        Returns:
            VLClassificationResult
        """
        if self.client is None:
            self._init_client()

        if self.client is None:
            return VLClassificationResult(
                class_name="unknown",
                confidence=0.0,
                raw_response="Client not initialized"
            )

        # Convert BGR to RGB PIL Image
        image_rgb = image[..., ::-1]  # BGR to RGB
        pil_image = Image.fromarray(image_rgb)

        # Convert to base64
        b64_image = image_to_base64(pil_image)
        image_url = f"data:image/jpeg;base64,{b64_image}"

        prompt = """You are an object classification system.

Look at the object in the image and classify it into the most appropriate category.

Output requirements:
- Only output a JSON object, no other text
- Format: {"class": "category_name", "confidence": 0.85}
- Choose the most appropriate category based on visual appearance
- confidence should be 0.0-1.0
- You can use any reasonable category name (e.g., person, car, truck, dog, cat, etc.)
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                max_tokens=256,
                temperature=0.1,
            )

            raw = response.choices[0].message.content
            result = self._parse_response(raw)

            return VLClassificationResult(
                class_name=result.get("class", "unknown"),
                confidence=result.get("confidence", 0.0),
                raw_response=raw
            )

        except Exception as e:
            print(f"[VllmClient] Classification failed for track {track_id}: {e}")
            return VLClassificationResult(
                class_name="unknown",
                confidence=0.0,
                raw_response=str(e)
            )

    def _parse_response(self, response: str) -> dict:
        """Parse VL model JSON response."""
        # Try to extract JSON
        json_match = re.search(r'```json\s*\n(.+?)\n```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
            else:
                return {}

        # Clean JSON
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        json_str = json_str.replace("'", '"')

        try:
            import json
            return json.loads(json_str)
        except Exception:
            return {}
