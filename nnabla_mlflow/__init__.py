import nnabla as nn
import mlflow
import gorilla
import time
import os


from nnabla.monitor import Monitor, MonitorSeries, MonitorTimeElapsed
from nnabla.monitor import MonitorImage, MonitorImageTile
from mlflow.utils.autologging_utils import try_mlflow_log


def _check_interval(index, flush_at, interval):
    return (index - flush_at) >= interval

def _check_interval_image(index, interval):
    return (index + 1) % interval == 0

def autolog(with_save_parameters=False):
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

    @gorilla.patch(MonitorImage)
    def add_image(self, index, value):
        original = gorilla.get_original_attribute(MonitorImage, 'add')
        original(self, index, value)
        if _check_interval_image(index, self.interval):
            image_name_tmpl = '{:06d}-{:03d}.png'
            run_id = mlflow.active_run().info.run_id
            for j in range(min(self.num_images, value.shape[0])):
                image_name = image_name_tmpl.format(index, j)
                local_path = os.path.join(self.save_dir, image_name)
                uri = 'runs:{}/{}'.format(run_id, self.name)
                try_mlflow_log(mlflow.log_artifact, local_path, uri)

    @gorilla.patch(MonitorImageTile)
    def add_image_tile(self, index, value):
        original = gorilla.get_original_attribute(MonitorImageTile, 'add')
        original(self, index, value)
        if _check_interval_image(index, self.interval):
            image_name = '{:06d}.png'.format(index)
            local_path = os.path.join(self.save_dir, image_name)
            run_id = mlflow.active_run().info.run_id
            uri = 'runs:{}/{}'.format(run_id, self.name)
            try_mlflow_log(mlflow.log_artifact, local_path, uri)

    @gorilla.patch(nn)
    def save_parameters(path, params=None):
        original = gorilla.get_original_attribute(nn, 'save_parameters')
        original(path, params)
        run_id = mlflow.active_run().info.run_id
        uri = 'runs:{}/{}'.format(run_id, 'parameters')
        try_mlflow_log(mlflow.log_artifact, path, uri)

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    patches = [
        gorilla.Patch(MonitorSeries, 'add', add_series, settings=settings),
        gorilla.Patch(MonitorTimeElapsed, 'add', add_time_elapsed,
                      settings=settings),
        # this must come earlier than MonitorImage's patch
        gorilla.Patch(MonitorImageTile, 'add', add_image_tile,
                      settings=settings),
        gorilla.Patch(MonitorImage, 'add', add_image, settings=settings),
    ]

    if with_save_parameters:
        patch = gorilla.Patch(nn, 'save_parameters', save_parameters,
                              settings=settings)
        patches.append(patch)

    for x in patches:
        gorilla.apply(x)
