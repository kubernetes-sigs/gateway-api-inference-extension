changes:
    - Split stress and functional tests
        - add setup_training_data.py for stress tests to train first, now that they no longer have the training in functional tests proceeding them
    - Model distribution through a model bundle registry
        - treelite mode requires both conformal weights and additional compiled shared libraries for each version of the model which results in more files. Also, conformal weights are paired to that version of the model, so moving everythign 
        - model caching logic, check if bundle hash gets updated before deciding to download

Future work:
    - During tests, track kubectl top pods resources so we can cross compare QPS to pod resource usage
    - Unify configation files into a shared configmap across the stack
    

list of future enhancements
    - Add rbac for the test pod to capture pod resource, so we can capture a `kubectl top pods` during tests.
    - Change tests to allow for easier aggregation of replicas of the test
    - Theres lots of conditional logic checking the mode, if we like this we should just re-write to use treelite only
    - Currently were distributing model weights from a model bundle registry with smarter caching. Its possible that with RWX PVCs we could have faster model weight transfer from training to prediction server, I just dont have it in my cluster
    - Use Kustomize configMapGenerator with hash suffixes for the shared configuration to automatically trigger rolling pod restarts when ConfigMap values change, preventing configuration drift between training and prediction servers.
    - change model downloads form polling on the prediction servers and checking if the current bundle is different to event driven
