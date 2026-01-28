const form = document.getElementById("loginForm");
const msg = document.getElementById("msg");
const u = document.getElementById("username");
const p = document.getElementById("password");
const remember = document.getElementById("remember");

function setMsg(text, kind) {
  msg.textContent = text || "";
  msg.className = "msg" + (kind ? ` ${kind}` : "");
}

// Optional: load remembered username
const savedUser = localStorage.getItem("remembered_username");
if (savedUser) {
  u.value = savedUser;
  if (remember) remember.checked = true;
}

const forgotLink = document.getElementById("forgotLink");
if (forgotLink) {
  forgotLink.addEventListener("click", (e) => {
    e.preventDefault();
    setMsg("Contact the administrator to reset your password.", "err");
  });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  setMsg("");

  const username = (u.value || "").trim();
  const password = p.value || "";

  // Remember username (optional)
  if (remember && remember.checked) localStorage.setItem("remembered_username", username);
  else localStorage.removeItem("remembered_username");

  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include", // ensures Flask session cookie is stored
      body: JSON.stringify({ username, password })
    });

    const out = await res.json().catch(() => ({}));

      if (res.ok && out.ok) {
      setMsg(out.message,'ok');
      setTimeout(() => window.location.replace("/"), 300);
    } else {
      // ✅ If backend returns 403 with redirect to OTP page
      if (res.status === 403 && out.redirect) {
        window.location.href = out.redirect;   // "/verify"
        return;
      }
      setMsg(out.error,'err');
    }


  } catch (err) {
    setMsg("Server unreachable. Check backend is running.", "err");
  }
});
