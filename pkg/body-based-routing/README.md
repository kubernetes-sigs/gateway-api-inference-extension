# Body-Based Routing
This package provides an extension that can be deployed to write the `model`
HTTP body parameter as a header (X-Gateway-Model-Name) so as to enable routing capabilities on the
model name.

As per OpenAI spec, it is standard for the model name to be included in the
body of the HTTP request. However, most implementations do not support routing
based on the request body. This extension helps bridge that gap for clients.
This extension works by parsing the request body. If it finds a `model` parameter in the
request body, it will copy the value of that parameter into a request header.

This extension is intended to be paired with an `ext_proc` capable Gateway. There is not
a standard way to represent this kind of extension in Gateway API yet, so we recommend
referring to implementation-specific documentation for how to deploy this extension.
