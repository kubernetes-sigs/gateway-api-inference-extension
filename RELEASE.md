# Release Process

Use [.github/ISSUE_TEMPLATE/new-release.md](.github/ISSUE_TEMPLATE/new-release.md)
as the source of truth when cutting a release.

In short:

1. Open a New Release issue.
1. Run `MAJOR=X MINOR=Y PATCH=Z make release` on the release branch. This
   updates the release artifacts and pins the `conformance` module to the
   matching root-module version.
1. Commit and push the release branch.
1. Run `MAJOR=X MINOR=Y PATCH=Z make release-tags` to create and push both
   signed tags: `$VERSION` and `conformance/$VERSION`.
1. Create the GitHub Release from the root tag only, then follow the remaining
   publication and announcement steps from the issue template.
