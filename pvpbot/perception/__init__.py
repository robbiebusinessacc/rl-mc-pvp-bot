"""Perception module: pixels -> PERCEPTION_DIM state vector -> OBS_DIM obs.

Submodules (import them directly; this package init stays torch-free):
    synth    -- procedural synthetic (frame, label) generator, pure NumPy
    train    -- PerceptionCNN training CLI on streamed synthetic batches
    infer    -- FrameEncoder checkpoint wrapper + latency benchmark
    adapter  -- ObsAssembler: perception stream + own actions -> 48-float obs
    distill  -- DAgger-style (frame, obs, action) collector over the stub env
"""
