import type { NextConfig } from "next";
import createNextIntlPlugin from 'next-intl/plugin';

const BACKEND_URL = process.env.BACKEND_URL;

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/:path*",
        destination: `${BACKEND_URL}/:path*`,
      },
    ];
  },
};

const withNextIntl = createNextIntlPlugin();

export default withNextIntl(nextConfig);
