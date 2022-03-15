# kakiage samples

TODO: introduce samples

# To use samples as application template

You can use these samples as the template for new application. As the samples references kakiage by relative path, so you have to modify it if you copy a sample to other directory.

## Sample "hello"

This example loads `<PROJECT_ROOT>/webpack/kakiage.js` directly from `index.html`. You have to copy `kakiage.js` (or download from Github's release) to your project directory and modify `<script>` tag in `index.html`.

`kakiage.js` can be built by command `npm run webpack`.

## Samples using npm system

Other samples use npm and webpack for package dependency management. You have to install kakiage as npm package in your application repository by `npm install /path/to/kakiage-<version>.tgz`.

`kakiage-<version>.tgz` can be built by command `npm run build && npm pack`.
