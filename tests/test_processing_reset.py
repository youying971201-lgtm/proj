import sys
import unittest
from pathlib import Path
from threading import Event, Thread


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import main


class ProcessingResetTests(unittest.TestCase):
    def test_reset_waits_until_monitor_worker_has_finished(self):
        saved_state = dict(main.app.state._state)
        saved_status = dict(main.road_marking_status)
        with main.frame_broadcaster._condition:
            saved_broadcast = (
                main.frame_broadcaster._frame,
                main.frame_broadcaster._sequence,
                main.frame_broadcaster._finished,
            )
        worker_started = Event()
        release_worker = Event()
        reset_done = Event()

        def worker():
            worker_started.set()
            release_worker.wait(2)
            main.app.state.processing_started = False
            main.frame_broadcaster.finish()

        main.app.state.gates = [{}, {}, {}]
        main.app.state.monitor_worker = worker
        main.app.state.processing_started = False
        main.app.state.road_marking_reload_requested = False
        try:
            self.assertEqual(main.start_processing()["status"], "started")
            self.assertTrue(worker_started.wait(1))
            reset_thread = Thread(target=lambda: (main.reset_processing(), reset_done.set()))
            reset_thread.start()
            self.assertFalse(reset_done.wait(0.05))
            release_worker.set()
            self.assertTrue(reset_done.wait(1))
            reset_thread.join(1)
            self.assertFalse(main.app.state.processing_started)
            self.assertFalse(main.app.state.cancel_requested)
            self.assertIsNone(main.app.state.monitor_thread)
            self.assertFalse(main.frame_broadcaster._finished)
        finally:
            release_worker.set()
            thread = getattr(main.app.state, "monitor_thread", None)
            if thread is not None:
                thread.join(1)
            main.app.state._state.clear()
            main.app.state._state.update(saved_state)
            main.road_marking_status.clear()
            main.road_marking_status.update(saved_status)
            with main.frame_broadcaster._condition:
                (
                    main.frame_broadcaster._frame,
                    main.frame_broadcaster._sequence,
                    main.frame_broadcaster._finished,
                ) = saved_broadcast


if __name__ == "__main__":
    unittest.main()
