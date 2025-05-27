# Scheduling Plugins

This package contains the scheduling plugin interface definitions and
implementations.

Plugins are organized by the following rule. Follow this rule when adding a new
plugin.

```
plugins/
|__ filter/(Plugins that only implement the Filter interface.)
|__ scorer/ (Plugins that only implement the Scorer interface.)
|__ picker/(Plugins that only implement the Picker interface.)
|__ multi/ (Plugins that implement multiple plugin interfaces.)
|____prefix/ (Prefix cache aware scheduling plugin.)
```
