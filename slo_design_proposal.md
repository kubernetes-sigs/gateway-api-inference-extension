# **SLO Aware Routing IG EPP Proposal**

[Benjamin Braun](mailto:benjaminbraun@google.com) / Last updated: Jul 31, 2025

## **Context**

[\[PUBLIC\] Latency Predictor + SLO Aware routing Feature Documentation](https://docs.google.com/document/d/1q56wr3N5XGx0B21MzHu5oBsCiGi9VrbZAvyhP2VFG_c/edit?usp=sharing)  
[\[Public\] WVA Design Proposal](https://docs.google.com/document/d/1XfLkoGBwpZX2M1GzUdCG44ar3SAoI-ZodrVpYUF8cLA/edit?usp=sharing)

## **Proposal**

This proposal outlines a strategy for integrating SLO-aware routing into the existing request handling flow, leveraging latency prediction to optimize pod selection and improve service level objective (SLO) adherence.

**Current Flow** (Simplified)

* Request received by gateway.  
* Pod saturations checked (KV, queue metrics, etc.)  
* (Shed if necessary/sheddable).  
* Scorers run to determine the best pod.  
* Request forwarded to the selected pod endpoint.

**Proposed Flow with Latency Prediction**

The proposed flow aims to utilize latency prediction at an earlier stage and implement a dedicated SLO-aware routing profile as an alternative scheduling profile.

1. Request received by gateway.  
2. Check latency prediction flag: if enabled, use “slo-routing profile” instead of default  
   1. For each potential pod, run latency prediction and store in memory along the request path.  
   2. \[Saturation Detector\] Evaluate pod saturations as a function of the request's SLO and latency predictions.  
   3. (if sheddable, shed if sheddable/no valid pods capable of meeting SLO).  
   4. Proceed to use SLO-aware scheduling profile (see "SLO-Aware Scheduling Profile" below).  
   5. Once a pod is decided, store the request with predicted ttft/tpot in datastore under that pods running requests  
3. Forward request to the selected pod endpoint.  
4. Continuously add the history of actual latencies and predicted latencies to the running requests on the pod in the datastore

**SLO-Aware Scheduling Profile:**

This will be a separate scheduling profile, used when the latency prediction flag is enabled for EPP. It will prioritize pods that can meet the request's SLO with the lowest positive headroom (i.e. compact bin packing). In cases where no pods can meet the SLO, it will select from available pods based on the highest negative headroom (i.e. closest to meeting SLO) for critical requests, shedding non-critical requests.

* **Inputs:** Prediction inputs from existing scorer prefix scorer, and pod metrics like KV, queue, request length, etc. will be used for latency prediction.  
  * This **REQUIRES** the prefix caching scorer to run before the SLO based picker (scores each pod and weighted draw to pick)  
* **Output:** specific pod  
* **Prediction:** Obtain latency predictions for the given request for each potential pod.  
* **Valid Pods:** Identify "valid" pods (those predicted to serve the request within its SLO, or have no running requests).  
* **Selection Logic:**  
  * If `len(valid_pods) > 0`: Return a weighted random draw favoring pods with the lowest **OR** highest positive headroom based on EPP runtime flag:  
    * Lowest: Assign to pods that have just enough resources to meet SLO, maintaining pods with high headroom for large critical requests  
    * Highest: Assign to pods that have substantial resources to meet SLO, so as to evenly distribute load.  
      (Both options, perhaps a very small chance of choosing an invalid pod, for exploration for training purposes)  
  * If `len(valid_pods) == 0`:  
    * If request is **not critical**: Shed the request.  
    * If request is **critical**: Return a weighted random draw favoring pods with the lowest negative headroom (least “overwhelmed” pods among those not meeting SLO).

**Datastore Changes**

- Add predictions to the running requests on pods:  
  - Request id  
  - Slo  
  - Predicted ttft  
  - Predicted tpot

**Post Request**

- Add a “PostReponseBody” plugin that sends off the training request to the async latency prediction client, sending the predicted and actual request latencies  
- Have this PostReponseBody run per-chunk

**Inference Scheduling Objective**

- Integrate logic with new InferenceObjectives

4\. Key Considerations

* **Only supported with 100% streamed requests:** in order to train we need streamed request data, we are not currently supporting non-streamed requests for SLO based routing  
* **Criticality:** Criticality will be handled by the layer above scheduling, allowing the scheduler to focus on efficient bin-packing. The saturation detector will be responsible for shedding non-critical requests if SLOs cannot be met.  
* **Prefix Score Reuse:** The new SLO-aware profile can reuse the existing prefix score logic.  
* **No SLO Provided:** If the latency prediction flag is enabled in EPP, we require all requests to provide an SLO, error if otherwise.  
* **Benchmarking:** Further benchmarking scenarios, especially with critical requests, should be considered.

5\. Communication / Next Steps

* Share proposal with WVA group chat, input from key stakeholders  
* Github issue in EPP  
* Begin implementation of the proposed flow and SLO-aware scheduling profile.  
* PR in EPP  
    
* (llm-d) Share SLO-aware routing benchmarking results in the llm-d weekly meetings and slack channel and get feedback to guide a more concrete design proposal.


