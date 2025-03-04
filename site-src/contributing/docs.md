# Contributing to the docs

This doc site is built using MkDocs. It includes a Docker image for you to preview local changes without needing to set up MkDocs and its related plug-ins.

## Preview local changes

1. In the `site-src` directory, make your changes to the Markdown files.

2. If you add a new page, make sure to update the `nav` section in the `mkdocs.yml` file.

3. From the root directory of this project, run the Docker image with the following command from the `Makefile`.
   ```sh
   make live-docs
   ```

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

## Add MkDocs features like plug-ins

As you contribute to the Kubernetes Gateway API Inference Extension project, you might want to add features to the MkDocs theme or build process. The following sections explain how to set up a local docs development environment and contribute to the docs build processes.

### Set up your local docs dev environment

To install and use `mkdocs`, refer to the [MkDocs User Guide](https://www.mkdocs.org/user-guide/).

1. Make sure that you have a recent version of `python` and `pip`.

2. You also might want to set up a Python virtual environment for the `gateway-api-inference-extension` project.
   ```sh
   python -m venv ~/.venvs/gateway-api-inference-extension
   source ~/.venvs/gateway-api-inference-extension/bin/activate
   ```

3. Install MkDocs with the theme and plugins that this project uses.
   ```sh
   # Install mkdocs
   pip install mkdocs
   
   # Install the Material theme that this project uses
   pip install mkdocs-material
   
   # Install the plugins that this project uses
   # For a full list, check the /hack/mkdocs/image/requirements.txt
   pip install mkdocs-awesome-pages-plugin
   pip install mkdocs-macros-plugin
   pip install mkdocs-mermaid2-plugin
   pip install mkdocs-material-extensions
   pip install mkdocs-redirects
   pip install mike
   ```

### Add MkDocs plugins

1. Try out the MkDocs plugin that you want to add locally. For more info, see the [MkDocs Plugin docs](https://squidfunk.github.io/mkdocs-material/plugins/).

2. Add the plugin to the `/hack/mkdocs/image/requirements.txt` file.

3. From the root directory, run the Docker image of the docs. Make sure that the plugin builds and works as you expect.
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