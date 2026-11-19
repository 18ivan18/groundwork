import { useQuery } from "@tanstack/react-query";

import { getUsers } from "@/api/client";
import { currentUser, setCurrentUser } from "@/lib/currentUser";

/**
 * Chooses the demo user the app acts as. Switching reloads the page, so nothing fetched as the
 * previous user stays on screen or in the query cache. M3b replaces this with the shell's switcher.
 */
export function UserPicker({ reload = () => window.location.reload() }: { reload?: () => void }) {
  const users = useQuery({ queryKey: ["users"], queryFn: getUsers });
  const selected = currentUser();
  if (!users.data) return null;
  return (
    <label className="flex items-center gap-2">
      <span>Signed in as</span>
      <select
        aria-label="User"
        className="rounded-md border border-border bg-card px-2 py-1 text-foreground"
        value={selected}
        onChange={(event) => {
          setCurrentUser(event.target.value);
          reload();
        }}
      >
        {users.data.map((user) => (
          <option key={user.user_id} value={user.user_id}>
            {user.display_name}
          </option>
        ))}
      </select>
    </label>
  );
}
