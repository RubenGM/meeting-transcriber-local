from pathlib import Path

from meeting_transcriber.gui import main
from meeting_transcriber.runtime import configure_runtime_environment


if __name__ == "__main__":
    configure_runtime_environment(Path(__file__).resolve().parents[2])
    main()
