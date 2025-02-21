# Body-Based Routing
This package provides an extension that can be deployed to write the `model`
HTTP body parameter as a header so as to enable routing capabilities on the
model name.

As per OpenAI spec, it is standard for the model name to be included in the
body of the HTTP request. However, most implementations do not support routing
based on the request body. This extension helps bridge the gap for clients that
are unable to write the model name into the headers themselves.
