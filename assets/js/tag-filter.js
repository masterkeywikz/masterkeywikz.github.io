(function () {
  const buttons = Array.from(document.querySelectorAll("[data-tag-filter]"));
  const essays = Array.from(document.querySelectorAll("[data-tags]"));

  if (buttons.length === 0 || essays.length === 0) return;

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      applyFilter(button.dataset.tagFilter || "all", true);
    });
  });

  window.addEventListener("hashchange", () => applyHash());
  applyHash();

  function applyHash() {
    const hash = decodeURIComponent(location.hash || "");
    const tag = hash.startsWith("#tag-") ? hash.slice(5) : "all";
    applyFilter(tag, false);
  }

  function applyFilter(tag, updateHash) {
    const activeTag = tag || "all";

    buttons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.tagFilter === activeTag));
    });

    essays.forEach((essay) => {
      const tags = (essay.dataset.tags || "").split(/\s+/).filter(Boolean);
      essay.hidden = activeTag !== "all" && !tags.includes(activeTag);
    });

    if (updateHash) {
      if (activeTag === "all") {
        history.replaceState(null, "", location.pathname);
      } else {
        history.replaceState(null, "", `#tag-${activeTag}`);
      }
    }
  }
})();
