import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { UserPicker } from "./UserPicker";

const USERS = [
  { user_id: "alice", display_name: "Alice", groups: ["everyone"] },
  { user_id: "hana", display_name: "Hana (HR)", groups: ["everyone", "hr"] },
];

function renderPicker(reload: () => void) {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(new Response(JSON.stringify(USERS)))),
  );
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <UserPicker reload={reload} />
    </QueryClientProvider>,
  );
}

describe("UserPicker", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  it("shows the current user", async () => {
    localStorage.setItem("groundwork.user", "hana");
    renderPicker(() => undefined);

    const select = await screen.findByRole<HTMLSelectElement>("combobox", {
      name: "User",
    });
    expect(select.value).toBe("hana");
  });

  it("stores the chosen user and reloads, so nothing from the previous user stays", async () => {
    const reload = vi.fn();
    renderPicker(reload);

    const select = await screen.findByRole("combobox", { name: "User" });
    fireEvent.change(select, { target: { value: "hana" } });

    expect(localStorage.getItem("groundwork.user")).toBe("hana");
    expect(reload).toHaveBeenCalledOnce();
  });
});
