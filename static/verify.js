const form = document.getElementById("verifyForm");
const msg = document.getElementById("msg");
const codeEl = document.getElementById("code");
const resendBtn = document.getElementById("resendBtn");

function setMsg(text, kind) {
  msg.textContent = text || "";
  msg.className = "msg" + (kind ? ` ${kind}` : "");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  setMsg("");

  const code = (codeEl.value || "").trim();

  try {
    const res = await fetch("/api/verify-otp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ code })
    });

    const out = await res.json().catch(() => ({}));

    if (res.ok && out.ok) {
      setMsg(out.message || "Verified. Redirecting…", "ok");
      setTimeout(() => window.location.replace("/"), 400);
    } else {
      setMsg(out.error || "Verification failed.", "err");
    }
  } catch (err) {
    setMsg("Server unreachable. Try again later.", "err");
  }
});

resendBtn.addEventListener("click", async () => {
  setMsg("");

  try {
    const res = await fetch("/api/resend-otp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({})
    });

    const out = await res.json().catch(() => ({}));

    if (res.ok && out.ok) {
      setMsg(out.message || "OTP resent.", "ok");
    } else {
      setMsg(out.error || "Could not resend OTP.", "err");
    }
  } catch (err) {
    setMsg("Server unreachable. Try again later.", "err");
  }
});
