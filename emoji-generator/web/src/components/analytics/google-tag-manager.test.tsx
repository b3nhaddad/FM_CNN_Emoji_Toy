import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  GTM_CONTAINER_ID,
  GoogleTagManagerNoscript,
  GoogleTagManagerScript,
} from "./google-tag-manager";

describe("GoogleTagManagerScript", () => {
  it("injects the GTM loader snippet for the configured container", () => {
    render(<GoogleTagManagerScript />);

    const script = document.getElementById("gtm-script");
    expect(script).not.toBeNull();
    expect(script?.tagName).toBe("SCRIPT");
    expect(script?.textContent).toContain("googletagmanager.com/gtm.js");
    expect(script?.textContent).toContain(GTM_CONTAINER_ID);
  });
});

describe("GoogleTagManagerNoscript", () => {
  it("renders a noscript iframe fallback pointing at the GTM container", () => {
    // React discards non-string <noscript> children on the client renderer
    // (it treats noscript like textarea — see shouldSetTextContent in
    // react-dom), so this only renders correctly through the same static
    // SSR markup path Next.js uses to produce the exported HTML.
    const html = renderToStaticMarkup(<GoogleTagManagerNoscript />);

    expect(html).toBe(
      `<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=${GTM_CONTAINER_ID}" height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>`,
    );
  });
});
