"""Message type definition."""
from dataclasses import dataclass, field
from PIL.Image import Image
from typing import List, Optional


@dataclass
class Message:
    """Message class definition using the expected GPT message format."""

    sender: Optional[str] = None
    text: Optional[str] = None
    images: List[Image] = field(default_factory=list)
