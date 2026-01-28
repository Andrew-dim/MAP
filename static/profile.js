async function logout() {
    try {
      const res = await fetch("/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin"
      });

      if (!res.ok) {
        const txt = await res.text().catch(() => "");
        throw new Error(`Logout failed: ${res.status} ${txt}`);
      }

      window.location.href = "/login";
    } catch (err) {
      console.error(err);
      alert("Logout failed. Please try again.");
    }
  }