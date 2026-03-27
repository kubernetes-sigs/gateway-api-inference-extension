# Release Process

The Kubernetes Template Project is released on an as-needed basis. The process is as follows:

1. Update `version/version.go` with the new semver tag
1. An issue is proposing a new release with a changelog since the last release
1. All [OWNERS](OWNERS) must LGTM this release
1. An OWNER runs `git tag -s $VERSION` and inserts the changelog and pushes the tag with `git push origin $VERSION`
1. For sub-modules (e.g., `conformance/go.mod`), the OWNER should create and push a prefixed tag: `git tag -s conformance/$VERSION` followed by `git push origin conformance/$VERSION`
1. The release issue is closed
1. An announcement email is sent to `dev@kubernetes.io` with the subject `[ANNOUNCE] kubernetes-template-project $VERSION is released`