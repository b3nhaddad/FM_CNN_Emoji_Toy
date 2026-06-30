import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { cleanup, render } from "@testing-library/react";
import { AdUnit } from "./ad-unit";
import { ADSENSE_CLIENT_ID } from "./adsense-script";

beforeEach(() => {
  delete window.adsbygoogle;
});

afterEach(cleanup);

describe("AdUnit", () => {
  it("renders an adsbygoogle ins unit tagged with its slot id", () => {
    const { container } = render(<AdUnit slot="sidebar-left" />);

    const ins = container.querySelector("ins.adsbygoogle");
    expect(ins).not.toBeNull();
    expect(ins?.getAttribute("data-ad-client")).toBe(ADSENSE_CLIENT_ID);
    expect(ins?.getAttribute("data-ad-slot")).toBe("sidebar-left");
  });

  it("requests an ad by pushing onto window.adsbygoogle", () => {
    render(<AdUnit slot="sidebar-left" />);

    expect(window.adsbygoogle).toHaveLength(1);
  });
});
