(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const recording = window.VRLBENCH_REPLAY;
  const player = byId("recorded-replay");
  if (!player) return;

  const positions = (recording?.trials || []).flatMap((trial) =>
    (trial.frames || []).map((frame, frameIndex) => ({ trial, frame, frameIndex }))
  );
  if (!positions.length) {
    byId("run-task").textContent = "Recording unavailable.";
    byId("run-progress").textContent = "No recorded actions are available.";
    byId("run-image").hidden = true;
    player.querySelectorAll("button, select").forEach((control) => {
      control.disabled = true;
    });
    return;
  }

  let selected = Math.max(0, positions.findIndex(({ trial }) => trial.trial_number === 2));
  let phase = "after";
  const positionSelect = byId("run-position");
  const image = byId("run-image");
  const missingImage = document.createElement("p");
  missingImage.className = "run-image-notice";
  missingImage.textContent = "No page snapshot was recorded at this position.";
  missingImage.hidden = true;
  image.after(missingImage);

  const method = recording.method[0].toUpperCase() + recording.method.slice(1);
  const title = `${recording.environment} · ${method} · ${recording.model}`;
  if (byId("run-title")) byId("run-title").textContent = title;
  if (byId("run-task")) byId("run-task").textContent = recording.task;
  player.setAttribute("aria-label", `${title}. Up to ${recording.trial_budget} trials, ${recording.max_environment_steps} actions per trial.`);

  positions.forEach(({ trial, frameIndex }, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `Trial ${trial.trial_number} · Action ${frameIndex + 1}`;
    positionSelect.append(option);
  });

  image.addEventListener("error", () => {
    image.hidden = true;
    missingImage.hidden = false;
  });

  function render() {
    const { trial, frame, frameIndex } = positions[selected];
    const snapshot = frame[`${phase}_image`];
    image.hidden = !snapshot;
    missingImage.hidden = Boolean(snapshot);
    if (snapshot) {
      image.alt = `${recording.environment}, trial ${trial.trial_number}, ${phase} action ${frameIndex + 1}`;
      image.src = snapshot;
    } else {
      image.removeAttribute("src");
    }

    byId("run-action").textContent = frame.action;
    byId("run-memory").textContent = trial.user_context || trial.system_context || "No cross-trial memory was supplied.";
    byId("run-progress").textContent = `Trial ${trial.trial_number} · Action ${frameIndex + 1} / ${trial.frames.length} · ${phase === "before" ? "Before" : "After"}`;
    byId("run-budget").textContent = `Trial ${trial.trial_number} / ${recording.trial_budget} · Actions used ${frame[`budget_used_${phase}`]} / ${recording.max_environment_steps}`;

    const atTrialEnd = phase === "after" && frameIndex === trial.frames.length - 1;
    const outcome = byId("run-outcome");
    outcome.textContent = atTrialEnd ? (trial.success ? "Success · stopped" : "Unsuccessful") : "Trial in progress";
    outcome.dataset.status = atTrialEnd ? (trial.success ? "success" : "failure") : "in-progress";
    positionSelect.value = String(selected);
    byId("run-prev").disabled = selected === 0;
    byId("run-next").disabled = selected === positions.length - 1;
    byId("run-before").setAttribute("aria-pressed", String(phase === "before"));
    byId("run-after").setAttribute("aria-pressed", String(phase === "after"));
  }

  function select(index) {
    selected = Math.max(0, Math.min(positions.length - 1, index));
    render();
  }

  positionSelect.addEventListener("change", () => select(Number(positionSelect.value)));
  byId("run-prev").addEventListener("click", () => select(selected - 1));
  byId("run-next").addEventListener("click", () => select(selected + 1));
  byId("run-before").addEventListener("click", () => { phase = "before"; render(); });
  byId("run-after").addEventListener("click", () => { phase = "after"; render(); });

  render();
})();
