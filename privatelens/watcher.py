"""Debounced filesystem watching for incremental indexing."""

from pathlib import Path
from threading import Event
from collections.abc import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from privatelens.utils.image_formats import SUPPORTED_IMAGE_EXTENSIONS


CHANGE_EVENT_TYPES = frozenset({"created", "modified", "moved", "deleted"})
STRUCTURAL_DIRECTORY_EVENTS = frozenset({"created", "moved", "deleted"})


class ImageChangeHandler(FileSystemEventHandler):
    """Set a shared event when a supported image path changes."""

    def __init__(self, pending: Event):
        super().__init__()
        self.pending = pending

    def on_any_event(self, event) -> None:
        if event.event_type not in CHANGE_EVENT_TYPES:
            return
        if event.is_directory:
            if event.event_type in STRUCTURAL_DIRECTORY_EVENTS:
                self.pending.set()
            return

        paths = [getattr(event, "src_path", ""), getattr(event, "dest_path", "")]
        if any(Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS for path in paths if path):
            self.pending.set()


def watch_for_changes(
    folder: Path,
    callback: Callable[[str], None],
    *,
    recursive: bool,
    debounce: float,
    initial_scan: bool,
) -> None:
    """Run callback after initial setup and debounced image changes."""
    pending = Event()
    observer = Observer()
    observer.schedule(ImageChangeHandler(pending), str(folder), recursive=recursive)
    started = False

    try:
        observer.start()
        started = True
        if initial_scan:
            callback("initial")

        while True:
            pending.wait()
            pending.clear()
            while pending.wait(debounce):
                pending.clear()
            callback("changed")
    except KeyboardInterrupt:
        return
    finally:
        if started:
            observer.stop()
            observer.join()
