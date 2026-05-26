"""quantbots — reusable framework for building bots on the private Manifold clone.

Keep this module import-light: importing `quantbots` must not pull in numpy,
scipy, openai or the network. Those live behind the relevant submodules so a bot
author can use the core (client, store, sizing, runner) without optional extras.
"""

__version__ = "0.1.0"
