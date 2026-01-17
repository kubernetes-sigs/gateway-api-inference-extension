changes:
    - split stress and functional tests
        - add setup_training_data.py for stress tests to train first, now that they no longer have the training in functional tests proceeding them
    - 

Future work:
    - During tests, track kubectl top pods resources so we can cross compare QPS to pod resource usage
    - Unify configation files into a shared configmap across the stack
    

list of future enhancements
    - Add rbac for the test pod to capture pod resource, so we can capture a `kubectl top pods` during tests.
    - Change tests to allow for easier aggregation of replicas of the test
    - Theres lots of conditional logic checking the mode, if we like this we should just re-write to use treelite only
