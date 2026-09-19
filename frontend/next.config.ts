import type { NextConfig } from "next";

// Static export served by nginx, which also proxies /api to the FastAPI service.
const config: NextConfig = { output: "export" };

export default config;
