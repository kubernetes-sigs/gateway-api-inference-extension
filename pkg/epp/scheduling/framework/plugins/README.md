# Scheduling Plugins

This package contains the scheduling plugin implementations.

Plugins are organized by the following rule. Follow this rule when adding a new
plugin.

```
plugins/
|__ filter/(Plugins that implement the Filter interface only.)
|__ scorer/ (Plugins that implement the Scorer interface only.)
|__ picker/(Plugins that implement the Picker interface only.)
|__ multi/ (Plugins that implement multiple plugin interfaces.)
|____prefix/ (Prefix cache aware scheduling plugin.)
```
