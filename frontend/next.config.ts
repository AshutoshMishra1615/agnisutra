import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";

// Fallback to localhost if env var is missing during build/config load
const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // @ts-expect-error - allowedDevOrigins is valid but types might be outdated
  experimental: {
    allowedDevOrigins: ["172.16.61.162:3000", "localhost:3000"],
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*", // Only rewrite /api requests to avoid conflicts
        destination: `${BACKEND_URL}/:path*`,
      },
    ];
  },
};

const withNextIntl = createNextIntlPlugin();

export default withNextIntl(nextConfig);
