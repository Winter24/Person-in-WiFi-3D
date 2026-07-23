"""Compatibility entry point for regenerating the system overview.

The original bitmap patcher has been replaced by the reproducible vector
renderer, which obtains its public selected-model label from the renderer.
"""

from render_system_overview import main


if __name__ == '__main__':
    main()
