# nnabla-mlflow

MLflow utilities for NNabla.

## dependencies
- nnabla
- mlflow

## install
```
$ pip install git+https://github.com/takuseno/nnabla-mlflow
```

## usage
Calling `nnabla_mlflow.autolog` monkey-pathches NNabla's monitors to automatically save data as MLflow metrics.

```py
import mlflow
import nnabla_mlflow

from nnabla.monitor import MonitorSeries

nnabla_mlflow.autolog() # monkey-patch Monitor classes

metric_monitor = MonitorSeries('metric', interval=1)

with mlflow.start_run():
    metric_monitor.add(1, 2) # mlflow.log_metric('metric', 1, step=2) is internally called
```

Currently, the following monitors are supported.

- MonitorSeries
- MonitorTimeElapsed

## TODO
- [x] support MonitorImage and MonitorImageTile
- [ ] support `log_model` and `save_model` just like `mlflow.tensorflow` and `mlflow.pytorch`
- [ ] support NNabla's Trainer
