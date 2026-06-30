import type { MetadataRoute } from "next";

export const dynamic = "force-static";

const BASE_URL = "https://emoji-generator-69.web.app";

const STATIC_ROUTES = ["/", "/terms"];

export default function sitemap(): MetadataRoute.Sitemap {
  return STATIC_ROUTES.map((route) => ({
    url: `${BASE_URL}${route}`,
    lastModified: new Date(),
  }));
}