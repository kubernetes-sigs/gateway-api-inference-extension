# Contributing to the docs

This doc site is built using MkDocs. It includes a Docker image for you to preview local changes without needing to set up MkDocs and its related plug-ins.

## Preview local changes

1. In the `site-src` directory, make your changes to the Markdown files.

2. If you add a new page, make sure to update the `nav` section in the `mkdocs.yml` file.

3. From the root directory of this project, run the Docker image with the following command from the `Makefile`.
   ```sh
   make live-docs
   ```

4. Open your browser to preview the local build, [http://localhost:3000](http://localhost:3000).

!!! type "One preview at a time"
    For better performance, open one localhost preview at a time. If you have multiple browsers rendering `localhost:3000`, you might notice a lag time in loading pages.

## Style guides

Refer to the following style guides:

* [Gateway API](https://gateway-api.sigs.k8s.io/contributing/style-guide/)
* [Kubernetes](https://kubernetes.io/docs/contribute/style/style-guide/)

If you need guidance on specific words that are not covered in one of those style guides, check a common cloud provider, such as [Google developer docs](https://developers.google.com/style).

## Version the docs

The Material theme uses `mike` to version the docs. The following steps cover common workflows for versioning. For more information, see the following resources:

* [Material theme versioning page](https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/)
* [`mike` readme](https://github.com/jimporter/mike)

Example workflow for using `mike`:

1. List the current versions. Aliases are included in brackets.
   ```sh
   mike list

   # Example output
   0.3 [main]
   0.2 [latest]
   0.1
   ```

2. In the `site-src` directory, make and save your doc changes.

3. Add the changes to the versions that you want to publish them in. The following example adds the changes to both the latest 0.2 and 0.3 main versions.  
   ```sh
   mike deploy --push --update-aliases 0.3 main
   mike deploy --push --update-aliases 0.2 latest
   ```

4. Optional: Add a new version of the docs. The following example adds a new version 0.4 as main based on the current content, renames 0.3 to latest with the current content, removes the latest alias from 0.2 but leaves the version content untouched, and deletes version 0.1.
   ```sh
   mike delete latest
   mike deploy --push --update-aliases 0.4 main
   mike deploy --push --update-aliases 0.3 latest
   mike delete 0.1
   ```

The `mike` commands add each version as a separate commit and directory on the `gh-pages` branch. 

* The versioned directories contain the output of the MkDocs build for each version. 
* The `latest` and `main` aliases are copies of the versioned directories.
* The `versions.json` file has the information for each version and alias that `mike` tracks. You can check this if you use 

Example directory structure:

```plaintext
gh-pages branch
│── 0.1/
│── 0.2/
│── 0.3/
│── 0.4/
│── latest/
│── main/
│── versions.json
```

The doc builds then publish the versioned content from this branch to the website.

## Develop the MkDocs theme

As you contribute to the Kubernetes Gateway API Inference Extension project, you might want to add features to the MkDocs theme or build process.

Helpful resources:
   
* [Customization, extensions, and overrides](https://squidfunk.github.io/mkdocs-material/customization/)
* [Setup features](https://squidfunk.github.io/mkdocs-material/setup/)
* [Plugins](https://squidfunk.github.io/mkdocs-material/plugins/)

General steps:

1. Set up a virtual environment with python, pip, mkdocs, and the plugins that this project uses.
   ```sh
   make virtualenv
   ```

2. Try out the MkDocs Material theme features, plugins, or other customizations that you want to add locally.

3. For plugins, add the plugin to the `/hack/mkdocs/image/requirements.txt` file.

4. From the root directory, run the Docker image of the docs. Make sure that your changes build and works as you expect.
   ```sh
   make live-docs
   ```

## Publish the docs

The project uses Netlify to host the docs. 

You can locally build the files that become the doc site. For example, you might want to check the HTML output of changes that you make to the site.

```sh
make build-docs-netlify
```

The Gateway API Inference Extension team will publish the docs based on the latest changes in the `main` branch.