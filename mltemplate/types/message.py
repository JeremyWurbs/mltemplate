"""Message type definition."""
from dataclasses import dataclass, field
from typing import List, Optional

from PIL.Image import Image


@dataclass
class Message:
    """Message class definition using the expected GPT message format."""

    sender: Optional[str] = None
    text: Optional[str] = None
    images: List[Image] = field(default_factory=list)
