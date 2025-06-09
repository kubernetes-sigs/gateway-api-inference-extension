# Pluggability

Author(s): @nirrozenbaum, @ahg-g, @elevran
## Proposal Status
 ***Draft***

## Summary
EPP contains several layes, each may have multiple extension points implemented by in-tree or out-of-tree plugins.  
High level EPP architecture can be found [here](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/0683-epp-architecture-proposal).  
This proposal aims to discuss the pluggability and the way that plugins are maintained and structured in the repo.

## Design Principles
- Plugins may implement more than one extension point.   
More specifically, plugins may implement multiple extension points that are defined in different layers, e.g., a plugin may implement extension point from scheduling layer and another one from requestcontrol layer.
- It should be possible to define plugins by code in an easy manner, with no code duplication.  
- It should be possible to configure the plugins via configuration file that loads plugins upon startup.  
At this point, dynamic configuration changes are not required. If one would like to change plugins it is needed to update the
configuration file and restart the EPP pod to initiate a configuration reload.
- Based on requirements, we might need to allow dynamic configuration changes in the future.  

## Definitions
- **Inference Extension Framework** - The system created to allow for pluggable extension points in the EPP.
- **Plugin** - Implementation of framework-defined interface(s) to add or extend logic across the framework.

## Proposal

This proposal aims to define how to structure and setup plugins for different layers of the system. For the sake of the discussion, the proposal focuses on extension points from the scheduling layer, which is the most stable one in terms of pluggability, and from the RequestControl layer.  
RequestControl might end up having multiple extension points, two of those are already agreed on all parties and will be used as a basis for the discussion:

 - `PreRequest` a plugin that takes the scheduling result and use it to execute logic before the request is sent to the selected endpoint.
- `PostResponse` - a plugin that is executed after the request was served successfully. The plugin receives the response in order to allow to operate on it. 

Please note that there is no intention to cover all extension points of the system, but only to specify extension points from at least two different layers in order to discuss how to maintain and structure plugins in the repo.

### Plugins maintainace
Since plugins may implement multiple extension points from different layers of EPP, it makes sense to group plugin implementations in under one directory. 
For example, instead of keeping plugins under `/pkg/epp/scheduling/framework/plugins`, this proposal suggests to define a shared package to be used for all plugins of the system - `pkg/epp/plugins`.

Additionally, from the same reason, this proposal suggests to have a single Plugin interface across the codebase and all specific plugins should embed this plugin inteface in the extension point. This will allow to register a general `Plugin` instance to EPP and auto-register the plugin in the right extension points (might be very useful when configuration via extrenal configuration file).

### Plugins by code

All extensible structures should be initialized with the plugins in main (under cmd package) and should support setting up extensions by code. Example can be found [here](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/cmd/epp/main.go#L208-L227).

Additionally, most of the code in the current main file should move to a separate file (may be under separate package) to have a very minimal main.go file which only handles the initialization of plugins and calling a `runner` to run the EPP with the given extensions configuration. This will allow to configure plugins with no code duplication and with minimal code.  
By doing the above, we end up with a main file that looks very similar to [kube-scheduler main](https://github.com/kubernetes/kubernetes/blob/master/cmd/kube-scheduler/scheduler.go#L29-L33) 
