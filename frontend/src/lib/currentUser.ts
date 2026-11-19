/** The demo user the app acts as. The API reads it from the `X-User` header on every request. */

const KEY = "groundwork.user";
export const DEFAULT_USER = "alice";

export function currentUser(): string {
  try {
    return localStorage.getItem(KEY) || DEFAULT_USER;
  } catch {
    // Storage can be blocked, for example in some private windows.
    return DEFAULT_USER;
  }
}

export function setCurrentUser(userId: string): void {
  try {
    localStorage.setItem(KEY, userId);
  } catch {
    // Without storage the choice cannot outlive the reload that applies it.
  }
}
