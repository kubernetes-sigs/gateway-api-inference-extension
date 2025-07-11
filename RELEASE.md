# Release Process

The Kubernetes Template Project is released on an as-needed basis. The process is as follows:

1. Update `pkg/consts/consts.go` with the new semver tag, for RC release we should use something like `v0.5.0-rc1`, for MAJOR or MINOR release we should use something like `v0.5.0`
1. Run the following command `make generate` which will update generated docs with the correct version info. (Note that you can't test with these YAMLs yet as they contain references to elements which wont exist until the tag is cut and image is promoted to production registry.)
1. An issue is proposing a new release with a changelog since the last release
1. All [OWNERS](OWNERS) must LGTM this release
1. An OWNER runs `git tag -s $VERSION` and inserts the changelog and pushes the tag with `git push $VERSION`
1. The release issue is closed
1. An announcement email is sent to `dev@kubernetes.io` with the subject `[ANNOUNCE] kubernetes-template-project $VERSION is released`
