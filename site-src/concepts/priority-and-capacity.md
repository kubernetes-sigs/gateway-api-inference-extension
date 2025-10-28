# Priority and Capacity

The InferenceObjective creates the definition of `Priority` which describes how requests interact with each other, this naturally interacts with total pool capacity, and properly understanding and configuring these behaviors is important in allowing a pool to handle requests of different priority.

## Priority (in flow control)

It should be noted that priority is currently only used in [Capacity](#capacity), and that the description below is how Priority will be consumed in the `Flow Control` model.

Priority is a simple stack rank; the higher the number, the higher the priority. Should no priority for a request be specified, the default value is zero. Requests of higher priority are _always_ selected first when requests are queued. Requests of equal priority currently operate on a FCFS basis. 

## Capacity

The current capacity model uses configurable [thresholds](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/35b14a10a9830d1a9e3850913539066ebc8fb317/pkg/epp/saturationdetector/saturationdetector.go#L49) to determine if the entire pool is saturated. The calculation is to simply iterate through each endpoint in the pool, and if all are above all thresholds, the pool is considered `saturated`. In the event of saturation, all requests with a negative priority will be rejected, and other requests will be scheduled and queued on the model servers. 

## Future work

The Flow Control system is nearing completion and will add more nuance to the Priority and Capacity model: proper priority enforcement, more articulate capacity tracking, queuing at the Inference Gateway level, etc. This documentation will be updated when the Flow Control has finished implementation.