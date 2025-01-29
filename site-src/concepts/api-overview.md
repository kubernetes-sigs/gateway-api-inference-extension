# API Overview

## Background
The API design is based on these axioms:

- Pools of shared compute should be *discrete* for scheduling to properly work
- Pod-level scheduling should not be handled by a high-level gateway 
- Simple services should be simple to define (or are implicitly defined via reasonable defaults)
- This solution should be composable with other Gateway solutions and flexible to fit customer needs
- The MVP will heavily assume requests are done using the OpenAI spec, but open to extension in the future
- The Gateway should route in a way that does not generate a queue of requests at the model server level
- Model serving differs from web-serving in critical ways. One of these is the existence of multiple models for the same service, which can materially impact behavior, depending on the model served. As opposed to a web-service that has mechanisms to render implementation changes invisible to an end user 