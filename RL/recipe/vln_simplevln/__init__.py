"""VLN Full-Episode Online DAPO recipe for verl.

Full-episode Habitat rollout -> flatten every SimpleVLN decision into an independent
training sample -> trajectory-level advantage with DAPO dynamic sampling. Prompt
and action parser are aligned with the SimpleVLN vLLM inference pipeline.
"""
