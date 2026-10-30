"""Build stateless SimpleVLN prompts from an instruction and sampled image history.

Standalone inference uses the environment's JPEG buffers in data URLs. The
training interface uses decoded PIL images and the model's multimodal processor.
"""
SYSTEM_PROMPT = "You are a helpful assistant."

PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the current observation. "
    "Your assigned task is: '{}'. Analyze this series of images to decide your next move, "
    "which could involve turning left or right by a specific degree or moving forward a certain distance."
)
# eval hardcodes these two text pieces around the images:
SIMPLEVLN_INTRO = ("Imagine you are a robot programmed for navigation tasks. "
                "You have been given a video of historical observations")
SIMPLEVLN_BRIDGE = "and an image of the current observation"
K_HISTORY = 8


def simplevln_tail(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction).split("current observation", 1)[1]


def uniform_sample_with_ends(data: list, n: int) -> list:
    if n <= 1:
        return data[-1:] if data else []
    if len(data) <= n:
        return data
    idx = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in idx]


def build_simplevln_messages(instruction: str, pil_images: list, k_history: int = K_HISTORY):
    """verl path: returns (messages, ordered_pil_images, image_indices).

    pil_images[-1] = current frame (PIL), pil_images[:-1] = history.
    Messages use {"type": "image"} markers; caller passes images to apply_chat_template.
    image_indices: positions in the original pil_images list (for buffer-based replay).
    """
    current = pil_images[-1]
    num_frames = len(pil_images)
    current_index = num_frames - 1
    if num_frames > 1:
        historic_indices = uniform_sample_with_ends(list(range(num_frames - 1)), k_history)
        historic = [pil_images[i] for i in historic_indices]
    else:
        historic_indices = [current_index]
        historic = [current]
    images = list(historic) + [current]
    image_indices = historic_indices + [current_index]
    content = [{"type": "text", "text": SIMPLEVLN_INTRO}]
    content += [{"type": "image"} for _ in historic]
    content.append({"type": "text", "text": SIMPLEVLN_BRIDGE})
    content.append({"type": "image"})
    content.append({"type": "text", "text": simplevln_tail(instruction)})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    return messages, images, image_indices


def build_simplevln_messages_b64(instruction: str, b64_buffer: list, k_history: int = K_HISTORY):
    """b64_buffer = list of env_server JPEG base64 strings, [-1] = current frame.

    Returns OpenAI-format messages (data-URL images). Matches eval: if only the
    first frame exists, the single history slot is the current frame.
    """
    current = b64_buffer[-1]
    historic = uniform_sample_with_ends(b64_buffer[:-1], k_history) if len(b64_buffer) > 1 else [current]

    def img(b):
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}

    content = [{"type": "text", "text": SIMPLEVLN_INTRO}]
    content += [img(b) for b in historic]
    content.append({"type": "text", "text": SIMPLEVLN_BRIDGE})
    content.append(img(current))
    content.append({"type": "text", "text": simplevln_tail(instruction)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
