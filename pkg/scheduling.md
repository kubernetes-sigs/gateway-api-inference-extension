## Scheduling Package in Ext Proc
The scheduling package implements request scheduling algorithms for load balancing requests across backend pods in an inference gateway. The scheduler ensures efficient resource utilization while maintaining low latency and prioritizing critical requests. It applies a series of filters based on metrics and heuristics to select the best pod for a given request.

# Flowchart
<img src="../docs/schedular-flowchart.png" alt="Scheduling Algorithm" width="400" />