<!-- Dev guide -->

## Logging

We use `logr.Logger` interface for logging everywhere.
The logger instance is loaded from `context.Context` or passed around as an argument directly.
This is aligned with contextual logging as explained in [k8s instrumentation logging guidelines](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-instrumentation/logging.md).

In other words, we explicitly don't use `klog` global logging calls.
Using `klog` log value helpers like `klog.KObj` is just fine.

### Change log verbosity

We generally follow the [k8s instrumentation logging guidelines](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-instrumentation/logging.md), which states "the practical default level is V(2). Developers and QE environments may wish to run at V(3) or V(4)".

To configure logging verbosity, specify the `v` flag such as `--v=2`.

### Add logs

The [k8s instrumentation logging guidelines](https://github.com/kubernetes/community/blob/master/contributors/devel/sig-instrumentation/logging.md) has the following definitions:

- `logger.V(0).Info` = `logger.Info` - Generally useful for this to **always** be visible to a cluster operator
- `logger.V(1).Info` - A reasonable default log level if you don't want verbosity.
- `logger.V(2).Info` - Useful steady state information about the service and important log messages that may correlate to significant changes in the system. This is the recommended default log level for most systems.
- `logger.V(3).Info` - Extended information about changes
- `logger.V(4).Info` - Debug level verbosity
- `logger.V(5).Info` - Trace level verbosity

We choose to simplify to the following 3 common levels.

```
const(
    DEFAULT=2
    VERBOSE=3
    DEBUG=4
    TRACE=5
)
```

The guidelines are written in the context of a k8s controller. Our [epp](../pkg/epp/) does more things such as handling requests and scraping metrics, therefore we adapt the guidelines as follows:

1. The server startup process and configuration.

   - `logger.Info` Logging at the `V(0)` verbosity level is generally welcome here as this is only logged once at startup, and provides useful info for debugging.

2. Reconciler loops. The reconciler loops watch for CR changes such as the `InferenceObjective` CR. And given changes in these CRs significantly affect the behavior of the extension, we recommend using v=1 verbosity level as default, and sparsely use higher verbosity levels.

   - `logger.V(DEFAULT)`
     - Default log level in the reconcilers.
     - Information about config (listening on X, watching Y)
     - Errors that repeat frequently that relate to conditions that can be corrected (e.g., inference model not initialized yet)
     - System state changing (adding/removing objects in the data store)
   - `logger.V(VERBOSE)` and above: Use your best judgement.

3. Inference request handling. These requests are expected to be much higher volume than the control flow in the reconcilers and therefore we should be mindful of log spamming. We recommend using v=2 to log important info about a request, such as the HTTP response code, and higher verbosity levels for less important info.

   - `logger.V(DEFAULT)`
     - Logging the status code of an HTTP request
     - Important decision making such as picking the target model, target pod
   - `logger.V(VERBOSE)`
     - Detailed request scheduling algorithm operations, such as running the filtering logic
   - `logger.V(DEBUG)` and above: Use your best judgement.

4. Metric scraping loops. These loops run at a very high frequency, and logs can be very spammy if not handled properly.

   - `logger.V(TRACE)`
     - Transient errors/warnings, such as failure to get response from a pod.
     - Important state changes, such as updating a metric.

5. Misc
   1. Periodic (every 5s) debug loop which prints the current pods and metrics.
      - `logger.V(DEFAULT).Error` If the metrics are not fresh enough, which indicates an error occurred during the metric scraping loop.
      - `logger.V(DEBUG)`
        - This is very important to debug the request scheduling algorithm, and yet not spammy compared to the metric scraping loop logs.

### Passing Logger Around

You can pass around a `context.Context` that contains a logger or a `logr.Logger` instance directly.
You need to make the call which one to use. Passing a `context.Context` is more standard,
on the other hand you then need to call `log.FromContext` everywhere.

As `logger.V` calls are cummulative, i.e. `logger.V(2).V(3)` results in `logger.V(5)`,
a logger should be passed around with no verbosity level set so that `logger.V(DEFAULT)`
actually uses `DEFAULT` verbosity level.
