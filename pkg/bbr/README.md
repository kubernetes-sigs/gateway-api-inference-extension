# Body-Based Routing
By deafult this package provides a plugable extension that can be to set the `model`
HTTP body parameter as a header (`X-Gateway-Model-Name`) so as to enable routing capabilities on the
model name.

As per OpenAI spec, it is standard for the model name to be included in the
body of the HTTP request. However, most implementations do not support routing
based on the request body. This extension helps bridge that gap for clients.
This extension works by parsing the request body. If it finds a `model` parameter in the
request body, it will copy the value of that parameter into a request header.
