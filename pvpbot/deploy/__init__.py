"""Live macOS deployment: screen capture -> perception -> policy -> input.

Submodules:
    capture       FrameSource interface, MockFrameSource, QuartzFrameSource
    input_inject  InputSink interface, MockInputSink, QuartzInputSink, Calibration
    loop          Real-time tick loop with flight recorder + kill-switch
    run           CLI entry point (``python3 -m pvpbot.deploy.run``)

Everything Quartz-specific is optional and guarded; tests run fully offline
with the Mock* implementations.
"""
