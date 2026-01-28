const form = document.getElementById("registerForm");
const msg = document.getElementById("msg");

function setMsg(text, kind) {
  msg.textContent = text || "";
  msg.className = "msg" + (kind ? ` ${kind}` : "");
}


form.addEventListener("submit", async (e) => {
  e.preventDefault();
  setMsg("");

  const name = document.getElementById("name").value.trim();
  const lastname = document.getElementById("lastname").value.trim();
  const email = document.getElementById("email").value.trim();
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;
  const password2 = document.getElementById("password2").value;


  // ---- send to backend ----
  try {
    const res = await fetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({
        username,
        password,
        password2,
        email,
        name,
        lastname
      })
    });

    const out = await res.json().catch(() => ({}));

    if (res.ok && out.ok) {
      setMsg(out.message || "Registration successful. Redirecting…", "ok");
      setTimeout(() => {
        window.location.href = "/verify"; // or /login if you disabled auto-login
      }, 500);
    } else {
      setMsg(out.error || "Registration failed.", "err");
    }

  } catch (err) {
    console.error(err);
    setMsg("Server unreachable. Try again later.", "err");
  }
});
