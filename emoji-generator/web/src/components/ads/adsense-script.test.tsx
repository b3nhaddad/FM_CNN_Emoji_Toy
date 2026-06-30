import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render } from "@testing-library/react";
import { ADSENSE_CLIENT_ID, AdSenseScript } from "./adsense-script";

afterEach(cleanup);

describe("AdSenseScript", () => {
  it("loads the AdSense loader script for the configured publisher", () => {
    render(<AdSenseScript />);

    const script = document.getElementById("adsbygoogle-script");
    expect(script).not.toBeNull();
    expect(script?.tagName).toBe("SCRIPT");
    expect(script?.getAttribute("src")).toBe(
      `https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=${ADSENSE_CLIENT_ID}`,
    );
  });
});
