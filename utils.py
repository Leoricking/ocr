import math


def format_duration(seconds) -> str:
    """Convert seconds to HH:MM:SS. Returns '--:--:--' for invalid input."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "--:--:--"
    if not math.isfinite(value) or value < 0:
        return "--:--:--"
    total = max(0, int(round(value)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
