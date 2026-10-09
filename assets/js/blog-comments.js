(function () {
  const article = document.querySelector("[data-commentable]");
  const panel = document.querySelector("[data-comment-panel]");

  if (!article || !panel) return;

  const storageKey = `blog-comments:${article.dataset.postId || location.pathname}`;
  const profileKey = "blog-comments:profile";
  const selector = "p, li, figure, h2, h3, .roofline-table, .bpe-lab";
  const blocks = Array.from(article.querySelectorAll(selector)).filter((block) => {
    return !block.closest(".footnotes") && !block.closest("script");
  });

  let activeAnchor = "";
  let comments = readJson(storageKey, []);
  let profile = readJson(profileKey, { alias: "", avatar: "" });

  blocks.forEach((block, index) => {
    const anchor = block.id || `note-${index + 1}`;
    if (!block.id) block.id = anchor;
    block.dataset.commentAnchor = anchor;
    block.classList.add("comment-anchor");

    const button = document.createElement("button");
    button.className = "comment-button";
    button.type = "button";
    button.textContent = "note";
    button.setAttribute("aria-label", "Add reader note");
    button.addEventListener("click", () => selectAnchor(anchor));
    block.append(button);
  });

  if (blocks.length > 0) {
    activeAnchor = blocks[0].dataset.commentAnchor;
  }

  render();

  function render() {
    updateAnchors();
    panel.replaceChildren();

    const title = document.createElement("h2");
    title.className = "comment-panel__title";
    title.textContent = "Reader notes";

    const profileForm = document.createElement("div");
    profileForm.className = "comment-profile";

    const aliasInput = document.createElement("input");
    aliasInput.type = "text";
    aliasInput.placeholder = "Alias";
    aliasInput.value = profile.alias || "";
    aliasInput.autocomplete = "name";
    aliasInput.addEventListener("input", () => updateProfile(aliasInput, avatarInput));

    const avatarInput = document.createElement("input");
    avatarInput.type = "url";
    avatarInput.placeholder = "Avatar URL";
    avatarInput.value = profile.avatar || "";
    avatarInput.autocomplete = "url";
    avatarInput.addEventListener("input", () => updateProfile(aliasInput, avatarInput));

    profileForm.append(aliasInput, avatarInput);

    const compose = document.createElement("div");
    compose.className = "comment-compose";

    const target = document.createElement("p");
    target.className = "comment-target";
    target.textContent = activeAnchor ? targetLabel(activeAnchor) : "Choose a paragraph";

    const textarea = document.createElement("textarea");
    textarea.placeholder = "Add a note";

    const submit = document.createElement("button");
    submit.type = "button";
    submit.textContent = "Add note";
    submit.disabled = !activeAnchor;
    submit.addEventListener("click", () => {
      const body = textarea.value.trim();
      if (!body || !activeAnchor) return;

      const block = article.querySelector(`[data-comment-anchor="${activeAnchor}"]`);
      comments.push({
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        anchor: activeAnchor,
        alias: (profile.alias || "Guest").trim(),
        avatar: (profile.avatar || "").trim(),
        body,
        quote: block ? compactText(block.textContent || "") : "",
        createdAt: new Date().toISOString(),
      });

      writeJson(storageKey, comments);
      textarea.value = "";
      render();
    });

    compose.append(target, textarea, submit);

    const list = document.createElement("div");
    list.className = "comment-list";

    comments.forEach((comment) => {
      list.append(commentCard(comment));
    });

    if (comments.length === 0) {
      const empty = document.createElement("p");
      empty.className = "comment-target";
      empty.textContent = "No notes yet.";
      list.append(empty);
    }

    panel.append(title, profileForm, compose, list);
  }

  function commentCard(comment) {
    const card = document.createElement("article");
    card.className = "comment-card";
    card.tabIndex = 0;
    card.addEventListener("click", () => selectAnchor(comment.anchor));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectAnchor(comment.anchor);
      }
    });

    const head = document.createElement("div");
    head.className = "comment-card__head";

    const avatar = document.createElement("img");
    avatar.alt = "";
    avatar.src = avatarSrc(comment.alias, comment.avatar);
    avatar.addEventListener("error", () => {
      avatar.src = avatarSrc(comment.alias, "");
    }, { once: true });

    const meta = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = comment.alias || "Guest";
    const time = document.createElement("time");
    time.dateTime = comment.createdAt;
    time.textContent = formatTime(comment.createdAt);
    meta.append(name, time);
    head.append(avatar, meta);

    const body = document.createElement("p");
    body.textContent = comment.body;

    const quote = document.createElement("div");
    quote.className = "comment-card__quote";
    quote.textContent = comment.quote ? `On: ${comment.quote}` : targetLabel(comment.anchor);

    card.append(head, body, quote);
    return card;
  }

  function selectAnchor(anchor) {
    activeAnchor = anchor;
    const block = article.querySelector(`[data-comment-anchor="${anchor}"]`);
    if (block) {
      block.scrollIntoView({ block: "center", behavior: "smooth" });
    }
    render();
    const textarea = panel.querySelector("textarea");
    if (textarea) textarea.focus();
  }

  function updateAnchors() {
    const commentedAnchors = new Set(comments.map((comment) => comment.anchor));
    blocks.forEach((block) => {
      const anchor = block.dataset.commentAnchor;
      block.classList.toggle("is-commented", commentedAnchors.has(anchor));
      block.classList.toggle("is-active-comment", anchor === activeAnchor);
    });
  }

  function updateProfile(aliasInput, avatarInput) {
    profile = {
      alias: aliasInput.value.trim(),
      avatar: avatarInput.value.trim(),
    };
    writeJson(profileKey, profile);
  }

  function targetLabel(anchor) {
    const block = article.querySelector(`[data-comment-anchor="${anchor}"]`);
    return block ? compactText(block.textContent || "Selected note") : "Selected note";
  }

  function compactText(value) {
    return value.replace(/\s+/g, " ").replace(/\bnote\b$/i, "").trim().slice(0, 96);
  }

  function avatarSrc(alias, avatar) {
    if (isUsableImageUrl(avatar)) return avatar;

    const initials = (alias || "Guest")
      .trim()
      .split(/\s+/)
      .map((part) => part[0])
      .join("")
      .slice(0, 2)
      .toUpperCase() || "G";
    const hue = hash(alias || "Guest") % 360;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="32" fill="hsl(${hue} 34% 74%)"/><text x="50%" y="53%" dominant-baseline="middle" text-anchor="middle" fill="#211f1c" font-family="Arial, sans-serif" font-size="24" font-weight="700">${initials}</text></svg>`;

    return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  }

  function isUsableImageUrl(value) {
    if (!value) return false;
    if (/^data:image\//i.test(value)) return true;
    try {
      const url = new URL(value, location.href);
      return url.protocol === "https:" || url.protocol === "http:";
    } catch {
      return false;
    }
  }

  function hash(value) {
    let output = 0;
    for (let index = 0; index < value.length; index += 1) {
      output = (output * 31 + value.charCodeAt(index)) >>> 0;
    }
    return output;
  }

  function formatTime(value) {
    try {
      return new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(new Date(value));
    } catch {
      return "";
    }
  }

  function readJson(key, fallback) {
    try {
      return JSON.parse(localStorage.getItem(key)) || fallback;
    } catch {
      return fallback;
    }
  }

  function writeJson(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // The comment UI should remain usable even in private browsing modes.
    }
  }
})();
