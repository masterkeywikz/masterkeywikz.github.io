(function () {
  const rankRules = [
    { left: "a", right: "n", merged: "an", rank: 256 },
    { left: "an", right: "a", merged: "ana", rank: 257 },
    { left: "b", right: "an", merged: "ban", rank: 258 },
    { left: "ban", right: "ana", merged: "banana", rank: 300 },
  ];

  const rankMap = new Map(
    rankRules.map((rule) => [`${rule.left}\u0000${rule.right}`, rule])
  );

  const initialParts = ["b", "a", "n", "a", "n", "a"];

  function pairKey(left, right) {
    return `${left}\u0000${right}`;
  }

  function formatPart(part) {
    return part === " " ? "space" : part;
  }

  function pairLabel(left, right) {
    return `b"${formatPart(left)}" + b"${formatPart(right)}"`;
  }

  function buildSteps() {
    const steps = [];
    let parts = initialParts.slice();
    const usedRanks = new Set();

    while (true) {
      let chosen = null;
      const pairs = [];

      for (let i = 0; i < parts.length - 1; i += 1) {
        const left = parts[i];
        const right = parts[i + 1];
        const rule = rankMap.get(pairKey(left, right));
        const pair = {
          index: i,
          left,
          right,
          rank: rule ? rule.rank : null,
          merged: rule ? rule.merged : null,
        };
        pairs.push(pair);

        if (rule && (!chosen || rule.rank < chosen.rank)) {
          chosen = pair;
        }
      }

      steps.push({
        parts: parts.slice(),
        pairs,
        chosen,
        usedRanks: new Set(usedRanks),
      });

      if (!chosen) break;

      usedRanks.add(chosen.rank);
      parts = [
        ...parts.slice(0, chosen.index),
        chosen.merged,
        ...parts.slice(chosen.index + 2),
      ];
    }

    return steps;
  }

  function node(name, className, text) {
    const element = document.createElement(name);
    if (className) element.className = className;
    if (text != null) element.textContent = text;
    return element;
  }

  function renderLab(root) {
    const steps = buildSteps();
    let stepIndex = 0;
    let timer = null;

    const status = root.querySelector("[data-bpe-status]");
    const tokenRow = root.querySelector("[data-bpe-tokens]");
    const pairRow = root.querySelector("[data-bpe-pairs]");
    const rankList = root.querySelector("[data-bpe-ranks]");
    const note = root.querySelector("[data-bpe-note]");
    const buttons = {
      prev: root.querySelector('[data-bpe-action="prev"]'),
      next: root.querySelector('[data-bpe-action="next"]'),
      play: root.querySelector('[data-bpe-action="play"]'),
      reset: root.querySelector('[data-bpe-action="reset"]'),
    };

    function stop() {
      if (timer) {
        window.clearInterval(timer);
        timer = null;
      }
      buttons.play.textContent = "Play";
    }

    function currentStep() {
      return steps[stepIndex];
    }

    function render() {
      const step = currentStep();
      const chosen = step.chosen;

      tokenRow.replaceChildren();
      step.parts.forEach((part, index) => {
        const token = node("span", "bpe-token", `b"${formatPart(part)}"`);
        if (chosen && (index === chosen.index || index === chosen.index + 1)) {
          token.classList.add("is-active");
        }
        tokenRow.appendChild(token);
      });

      pairRow.replaceChildren();
      if (step.pairs.length === 0) {
        pairRow.appendChild(node("div", "bpe-pair is-unranked", "No adjacent pairs remain."));
      } else {
        step.pairs.forEach((pair) => {
          const item = node("div", "bpe-pair");
          const label = node("strong", "", pairLabel(pair.left, pair.right));
          const detail = pair.rank == null
            ? "not mergeable"
            : `rank ${pair.rank} -> b"${pair.merged}"`;
          item.append(label, document.createTextNode(detail));
          if (chosen && pair.index === chosen.index) item.classList.add("is-chosen");
          if (pair.rank == null) item.classList.add("is-unranked");
          pairRow.appendChild(item);
        });
      }

      rankList.replaceChildren();
      rankRules.forEach((rule) => {
        const item = node(
          "div",
          "bpe-rank",
          `${rule.rank}: b"${rule.left}" + b"${rule.right}" -> b"${rule.merged}"`
        );
        if (step.usedRanks.has(rule.rank)) item.classList.add("is-used");
        rankList.appendChild(item);
      });

      if (chosen) {
        status.textContent = `Step ${stepIndex + 1} of ${steps.length}: choose the lowest-ranked available pair, rank ${chosen.rank}.`;
        note.textContent = `Next merge: ${pairLabel(chosen.left, chosen.right)} becomes b"${chosen.merged}". If the same ranked pair appears twice, this loop merges the leftmost occurrence first, then repeats.`;
      } else {
        status.textContent = `Step ${stepIndex + 1} of ${steps.length}: no ranked adjacent pair remains. Encoding is done.`;
        note.textContent = `Final token bytes: ${step.parts.map((part) => `b"${formatPart(part)}"`).join(", ")}.`;
        stop();
      }

      buttons.prev.disabled = stepIndex === 0;
      buttons.next.disabled = stepIndex === steps.length - 1;
    }

    buttons.prev.addEventListener("click", () => {
      stop();
      stepIndex = Math.max(0, stepIndex - 1);
      render();
    });

    buttons.next.addEventListener("click", () => {
      stop();
      stepIndex = Math.min(steps.length - 1, stepIndex + 1);
      render();
    });

    buttons.reset.addEventListener("click", () => {
      stop();
      stepIndex = 0;
      render();
    });

    buttons.play.addEventListener("click", () => {
      if (timer) {
        stop();
        return;
      }
      if (stepIndex === steps.length - 1) stepIndex = 0;
      buttons.play.textContent = "Pause";
      render();
      timer = window.setInterval(() => {
        if (stepIndex >= steps.length - 1) {
          stop();
          render();
          return;
        }
        stepIndex += 1;
        render();
      }, 1150);
    });

    render();
  }

  document.querySelectorAll("[data-bpe-lab]").forEach(renderLab);
})();
