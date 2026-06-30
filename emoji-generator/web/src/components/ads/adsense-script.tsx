// Script is now injected via layout.tsx <head> to avoid Next.js adding
// data-nscript, which AdSense's validator rejects. This file is kept only
// to export the client ID constant and satisfy existing test imports.

export const ADSENSE_CLIENT_ID = "ca-pub-3852182085462712";

export function AdSenseScript() {
  return null;
}
