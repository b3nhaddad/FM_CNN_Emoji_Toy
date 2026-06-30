/** Derives the default username from an email address, matching the legacy behavior exactly. */
export function usernameFromEmail(email: string): string {
  return email.split("@")[0].toLowerCase().replace(/[^a-z0-9_]/g, "_");
}
