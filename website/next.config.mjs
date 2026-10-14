/** Static export: the whole site is prebuilt HTML + JSON, deployable on any static host.
 *  BASE_PATH=/CleanSlate for GitHub Pages (akhatua2.github.io/CleanSlate); unset for local dev. */
const basePath = process.env.BASE_PATH || "";
const nextConfig = { output: "export", images: { unoptimized: true }, basePath, trailingSlash: true };
export default nextConfig;
