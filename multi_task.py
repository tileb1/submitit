from http.server import HTTPServer, SimpleHTTPRequestHandler
import submitit
import socketserver
import time
import json
import functools
import requests
import logging
from typing import List

PORT = 8000

logging.basicConfig()
logger = logging.getLogger("multi_task")
logger.setLevel(logging.INFO)



class Worker(submitit.helpers.Checkpointable):
    def __init__(self, n: int = 10):
        self.n = n
        self.counter = 0
        self.w_counter = 1
        self.results: List[int] = []

        # master_pause is a hack to force the job to be preempted once.
        # Only useful for debugging
        self.master_pause = True

        # We can't do the setup here because this is called before submitting the job
        self.task_id = 0
        self.workers = 0

    def __call__(self):
        # This is setup code for each task.
        # You have to be careful here because this code will be called every time
        # the job is restarted and can override value sets in `checkpoint`
        env = submitit.JobEnvironment()
        self.task_id = env.global_rank
        self.workers = env.num_tasks

        if self.task_id == 0:
            self.master_main()
        else:
            self.slave_main()
        return self.results

    def should_checkpoint(self, env: submitit.JobEnvironment) -> bool:
        """
        This is new api I'm introducing, it allows the user to chose if this task should checkpoint.
        When restarting, if no checkpoint is found for this task, we use the task "0" checkpoint.
        """
        logger.warning(f"Returning 'True' from should_checkpoint in task {env.global_rank}")
        return True

    def checkpoint(self, *args, **kwargs) -> submitit.core.utils.DelayedSubmission:
        # This is a debugging implementation,
        # users won't need to override this most of the time.
        if self.task_id == 0:
            logger.warning(f"Checkpointing master. State: {self.counter}/{self.n}, {self.w_counter}/{self.workers} (pause={self.master_pause})")
        else:
            logger.warning(f"Checkpointing worker {self.task_id}. State: {self.counter}")
        return super().checkpoint(*args, **kwargs)

    ############################################################################
    # The rest of the code implement a not so simple example:
    #   * the master task send work to slave tasks
    #   * the master collect the results and check they match what expected
    #   * the master blocks at 20% of the job to be requeued
    #   * the master properly finishes and check that the other tasks haven't desync
    # I haven't figured a proper way of killing the workers, so the job will time out
    # It's an issue because, master will exit without being checkpointed,
    # and it will restart from previous state.
    ############################################################################
    def slave_main(self) -> None:
        port = PORT + self.task_id
        httpd = WorkerHTTPServer(self, port)
        logger.warning(f"Started worker {self.task_id} at port {port}. State: {self.counter}")
        httpd.serve_forever()

    def master_main(self) -> None:
        time.sleep(2)
        logger.warning(f"Starting master. State: {self.counter}/{self.n}, {self.w_counter}/{self.workers} (pause={self.master_pause})")

        for i in range(self.counter, self.n):
            self.counter = i
            if i == 2 and self.master_pause:
                # Timeout on purpose
                self.master_pause = False
                time.sleep(200)
                raise Exception("We should have been canceled before calling this.")

            for w in range(self.w_counter, self.workers):
                self.w_counter = w
                j = self.remote_step(w)
                if j != i:
                    logger.warning(f"!!! Worker {w} is at step {j} while master is at {i}")
                    assert j == i
                self.results.append(j)

            logger.warning(f"Master has done {self.counter}/{self.n}")
            self.w_counter = 1

        logger.warning(f"Master has finished !")
        self.checkpoint()

    def step(self) -> int:
        i = self.counter
        self.counter += 1
        time.sleep(1)
        return i

    def remote_step(self, w: int) -> int:
        r = requests.get(f"http://localhost:{PORT + w}")
        r.raise_for_status()
        res = r.json()
        assert res["task_id"] == w
        return res["res"]



class WorkerHTTPServer(HTTPServer):
    def __init__(self, worker: Worker, port: int):
        self.worker = worker
        super().__init__(("", port), WorkerHttpHandler)


class WorkerHttpHandler(SimpleHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        worker: Worker = self.server.worker  # type: ignore
        response = {"res": worker.step(), "task_id": worker.task_id}
        self.wfile.write(bytes(json.dumps(response), encoding="ascii"))


def main(n: int = 10, workers: int = 3, cluster: str = "local"):
    ex = submitit.AutoExecutor("logs", cluster=cluster)
    ex.update_parameters(timeout_min=2, tasks_per_node=workers + 1, slurm_partition="dev")
    j = ex.submit(Worker())
    print(j, j.paths.stdout)
    j.wait()
    for t in j._sub_jobs:
        print(f"*** output of {t} ***")
        print(t.stdout())
        print(t.stderr())
    assert j._sub_jobs[0].result()


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(main)
