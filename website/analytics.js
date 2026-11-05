(() => {
  "use strict";

  if (location.origin !== "https://vrl-bench.github.io" ||
      !["/vrlbench/", "/vrlbench/index.html"].includes(location.pathname)) return;

  const measurementId = "G-T91KFJELV5";
  window.dataLayer = window.dataLayer || [];
  function gtag() { window.dataLayer.push(arguments); }
  gtag("js", new Date());
  gtag("config", measurementId, {
    allow_google_signals: false,
    allow_ad_personalization_signals: false,
  });

  const tag = document.createElement("script");
  tag.async = true;
  tag.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`;
  document.head.append(tag);

  // The replay is inline: count the first deliberate visit or interaction.
  let replayOpened = false;
  function recordReplayOpen() {
    if (replayOpened) return;
    replayOpened = true;
    gtag("event", "replay_open");
  }
  document.querySelectorAll('a[href="#inspect"], .run-control button').forEach((control) => {
    control.addEventListener("click", recordReplayOpen);
  });
  document.getElementById("run-position")?.addEventListener("change", recordReplayOpen);
})();
