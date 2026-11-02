(() => {
  "use strict";
  const data = window.VRLBENCH_PAIR;
  const byId = (id) => document.getElementById(id);
  if (!data) {
    byId("story-explanation").textContent = "The recorded comparison could not be loaded. Open this page with its assets directory.";
    return;
  }

  const tabs = [...document.querySelectorAll("[data-stage]")];
  const explanations = [
    "Both conditions begin with the exact same failed attempt. Neither has cross-trial memory in this first trial.",
    "Reflexion revisits the same listing and inspects its description. Retry chooses another listing and selects grey before buying.",
    "Retry stops after succeeding on trial 2. Reflexion uses all six trials without success in this pair."
  ];

  function actions(container, entries, arrows = true) {
    entries.forEach((entry, index) => {
      if (index && arrows) {
        const arrow = document.createElement("span");
        arrow.className = "action-arrow";
        arrow.textContent = "→";
        arrow.setAttribute("aria-hidden", "true");
        container.append(arrow);
      }
      const code = document.createElement("code");
      code.textContent = entry;
      if (entry === "click[grey]") code.className = "highlight-action";
      container.append(code);
    });
  }

  function fullRecord(parent, entries, memory) {
    const details = document.createElement("details");
    details.className = "raw-actions";
    const summary = document.createElement("summary");
    summary.textContent = `All ${entries.length} recorded actions`;
    const list = document.createElement("ol");
    entries.forEach((action) => {
      const item = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = action;
      item.append(code);
      list.append(item);
    });
    details.append(summary, list);
    if (memory) {
      const memoryLabel = document.createElement("p");
      memoryLabel.textContent = "Context supplied to trial 2, verbatim:";
      const pre = document.createElement("pre");
      pre.textContent = memory;
      details.append(memoryLabel, pre);
    }
    parent.append(details);
  }

  byId("case-task").textContent = data.task;
  actions(byId("shared-actions"), data.shared_t1.actions);
  byId("memory-excerpt").textContent = data.reflexion.memory_excerpt;
  actions(byId("reflexion-actions"), data.reflexion.actions.slice(1, 4));
  actions(byId("retry-actions"), data.retry.actions.slice(1));
  fullRecord(byId("reflexion-actions").parentElement, data.reflexion.actions, data.reflexion.memory);
  fullRecord(byId("retry-actions").parentElement, data.retry.actions);

  for (const method of ["reflexion", "retry"]) {
    const known = data[method].outcomes;
    for (let trial = 0; trial < data.trial_budget; trial++) {
      const mark = document.createElement("span");
      const status = trial >= known.length ? "unrun" : known[trial] ? "solved" : "failed";
      mark.className = `trial-mark ${status}`;
      const symbol = document.createElement("b");
      symbol.textContent = {unrun: "–", solved: "✓", failed: "×"}[status];
      const number = document.createElement("small");
      number.textContent = `T${trial + 1}`;
      mark.setAttribute("aria-label", `Trial ${trial + 1}: ${status === "unrun" ? "not run after success" : status === "solved" ? "success" : "unsuccessful"}`);
      mark.append(symbol, number);
      byId(`${method}-outcomes`).append(mark);
    }
  }

  function selectStage(stage, focus = false) {
    ["shared-scene", "paths-scene", "outcomes-scene"].forEach((id, index) => {
      byId(id).hidden = index !== stage;
    });
    tabs.forEach((tab, index) => {
      tab.setAttribute("aria-selected", String(index === stage));
      tab.tabIndex = index === stage ? 0 : -1;
    });
    byId("story-panel").setAttribute("aria-labelledby", `story-tab-${stage}`);
    byId("story-explanation").textContent = explanations[stage];
    if (focus) tabs[stage].focus();
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => selectStage(index));
    tab.addEventListener("keydown", (event) => {
      let target;
      if (event.key === "ArrowRight") target = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") target = (index + tabs.length - 1) % tabs.length;
      if (event.key === "Home") target = 0;
      if (event.key === "End") target = tabs.length - 1;
      if (target !== undefined) {
        event.preventDefault();
        selectStage(target, true);
      }
    });
  });
  selectStage(1);

  byId("copy-command").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(byId("example-command").textContent);
      byId("copy-command").textContent = "Copied";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(byId("example-command"));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      byId("copy-command").textContent = "Select & copy";
    }
    window.setTimeout(() => { byId("copy-command").textContent = "Copy"; }, 2200);
  });
})();
