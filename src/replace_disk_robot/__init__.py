"""Backend-neutral package root for the hard-disk replacement project.

Applications select their runtime adapter explicitly; importing this module
therefore never loads a simulator, middleware, or hardware SDK.
"""

__version__ = "0.1.0"
