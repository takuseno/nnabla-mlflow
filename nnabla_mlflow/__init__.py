import mlflow
import gorilla
import time


from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from mlflow.utils.autologging_utils import try_mlflow_log


def _check_interval(index, flush_at, interval):
    return (index - flush_at) >= interval


def autolog():
    @gorilla.patch(MonitorSeries)
    def add_series(self, index, value):
        if _check_interval(index, self.flush_at, self.interval):
            value = sum(self.buf + [value]) / (len(self.buf) + 1)
            try_mlflow_log(mlflow.log_metric, self.name, value, step=index)
        original = gorilla.get_original_attribute(MonitorSeries, 'add')
        original(self, index, value)

    @gorilla.patch(MonitorTimeElapsed)
    def add_time_elapsed(self, index):
        if _check_interval(index, self.flush_at, self.interval):
            now = time.time()
            elapsed = now - self.lap
            try_mlflow_log(mlflow.log_metric, self.name, elapsed, step=index)
        original = gorilla.get_original_attribute(MonitorTimeElapsed, 'add')
        original(self, index)

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    patches = [
        gorilla.Patch(MonitorSeries, 'add', add_series, settings=settings),
        gorilla.Patch(MonitorTimeElapsed, 'add', add_time_elapsed,
                      settings=settings),
    ]

    for x in patches:
        gorilla.apply(x)
