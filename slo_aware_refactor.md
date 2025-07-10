
The goal of the SLO aware routing refactor is to isolate the code and logic for SLO aware routing into an independent scheduing profile with plugins that perfom the same functionality that is currently hardcoded.

Current functionality:

1. Request is recieved 
2. Normal scheduling profile runs with filters, scorers, and pickers
3. if the EPP runtime flag "enable-latency-predictor" is present, we then make a call to the latency predictor sidecar, using the prefix cache score calculated in the previous step along with various pod metrics
4. We then overwrite the existing types.SchedulingResult with a new one based on the latency predictions, using the following logic:
    - if the prediction is less than the SLO, we consider the pod "valid" and score it based on it's headroom (LEAST headroom while still under = highest score, so as to pack most efficiently)
    - if the prediction is more than the SLO, we check the criticality: if it's critical we use the pod with the least negative headroom (closest to being able to serve the request), else if non-critical we shed the request
5. We then do a weighted random draw over all pods to pick the target pods, including the invalid pods, but at a very low weight (about 1% of the weight of a valid pod)
6. In director.go, in prepareRequest() we call datastore.PodAddRequest() to add the request to the pod's running request queue
7. In reponse.go, in HandleResponseBodyModelStreaming() (only streaming since we only support SLO aware routing for streamed requests), we call datastore.PodRemoveRequest() to remove the request from the pod's running request queue
8. We track and send the latency data to the training sidecar in HandleResponseBodyChunk() in director.go, which continuously trains the predictor sidecar


The refactor will make the flow look like this:

1. A new scheduling profile must be made specifically for SLO based routing
2. if the "enable-latency-predictor" is present we use this new profile which will:
3. if using this profile, skip the normal saturation detection logic
4. the profile will:
    4.1 first it will run the prefix cache scorer to get the prefix cache scores which are required inputs for the latency predictor
    4.2 second, it will run the SLO scorer, which runs has the same logical flow as the current functionality:
        - if the prediction is less than the SLO, we consider the pod "valid" and score it based on it's headroom (LEAST headroom while still under = highest score, so as to pack most efficiently)
        - if the prediction is more than the SLO, we check the criticality: if it's critical we use the pod with the least negative headroom (closest to being able to serve the request), else if non-critical we shed the request
    4.3 do a weighted random draw over all pods to pick the target pods, including the invalid pods, but at a very low weight (about 1% of the weight of a valid pod)
    4.4 once we have a choosen pod from the scheduling layer, the PreRequest plugin with add the request to the list of running requests for that pod with datastore.PodAddRequest()
    4.5 in the PostResponse() we will remove the request from the running requests with datastore.PodRemoveRequest()
5. We track and send the latency data to the training sidecar in HandleResponseBodyChunk() in director.go, which continuously trains the predictor sidecar

For step 5, we can keep the current implementation as it is impractical to move that into a profile, and it's already gated behind the "enable-latency-predictor" flag.

We are performing an refactor of the code here, the goal is to utilize plugins to perform the SLO aware routing logic currently hardcoded into pkg/epp/requestcontrol/director.go pkg/epp/handlers/response.go and several other files. It's important that we keep the changes as isolated as possible, so as to not disrupt other functionality. you can find the scoring logic in pkg/epp/requestcontrol/prediction_based_scorer.go