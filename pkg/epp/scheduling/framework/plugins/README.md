# Scheduling Plugins

This package contains the scheduling plugin implementations.

Plugins are grouped by plugin name to make it easy to find a specific behavior.
Each plugin lives in its own directory (for example `prefixcachescorer/`,
`predictedlatencyscorer/`, `maxscorepicker/`). Shared helpers that are used by
multiple plugins live in supporting packages such as `pickershared/`, and test
helpers stay under `test/`.

When adding a new plugin, create a new directory named after the plugin and keep
all of its code and tests inside that directory.
