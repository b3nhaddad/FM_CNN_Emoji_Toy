"use client";

import { useEffect, useRef } from "react";

declare global {
  interface Window {
    adsbygoogle?: unknown[];
  }
}

const ADSENSE_CLIENT_ID = "ca-pub-3852182085462712";

export function AdUnit({
  slot,
  style,
}: {
  slot: string;
  style?: React.CSSProperties;
}) {
  const insRef = useRef<HTMLModElement>(null);

  useEffect(() => {
    const el = insRef.current;
    if (!el || el.dataset["adsbygoogleStatus"]) return;
    try {
      (window.adsbygoogle = window.adsbygoogle || []).push({});
    } catch {
      // Blocked by an ad blocker — leave the slot empty.
    }
  }, [slot]);

  return (
    <ins
      ref={insRef}
      className="adsbygoogle"
      style={{ display: "block", ...style }}
      data-ad-client={ADSENSE_CLIENT_ID}
      data-ad-slot={slot}
    />
  );
}
