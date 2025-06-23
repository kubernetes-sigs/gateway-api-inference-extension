# v0.4.0 Release

We are thrilled to announce the [v0.4.0 release](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/tag/v0.4.0)
—our biggest update yet! This version brings powerful new Endpoint Picker (EPP) scheduler capabilities, performance improvements,
and initial Gateway conformance tests.

## Major Highlights

* **Modular Endpoint Picker (EPP) Scheduler**: A kube-scheduler–style plugin API lets you build custom routing logic,
  filter and score backends, or swap in new picker strategies without touching core code.

* **Prefix-Cache-Aware Routing**: Dramatically lower tail latency by routing requests based on cached network prefixes,
  improving response times under load.

* **Richer Metrics**: Gain deeper insights with new metrics including:

    * NTPOT (Normalized Time Per Output Token)
    * Scheduler latency
    * Per-pod queue depth
    * Build and version info

* **Optional vLLM Simulator Backend**: Spin up a lightweight simulator for local development and testing—no real model
  servers required.

* **Initial Conformance Tests**: Validate your controller’s behavior with end-to-end tests covering InferencePool,
  InferenceModel, HTTPRoute, and more.

## Thank You to Our Contributors

Over 200 PRs landed since v0.3.0 thanks to our amazing community! A special shout-out to our new contributors, who made
their first PRs in this release:

@Conor0Callaghan, @SachinVarghese, @xiaolin593, @cr7258, @maxbrunet, @howardjohn, @nayihz, @mayabar, @shaneutt, @rlakhtakia, @delavet, @SinaChavoshi, @capri-xiyue, @LukeAVanDrie, @GunaKKIBM, @shmuelk, @alexsnaps,
@t3hmrman, @shotarok, @EyalPazz, @wbpcode, @bharathbrat, @zetxqx, @elevran, @waltforme

…and everyone else who reviewed, tested, documented, and refined this release!

## Release Notes

For the complete changelog, visit the [official release page](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/tag/v0.4.0).
