(() => {
  "use strict";

  const command = document.getElementById("example-command");
  const copyButton = document.getElementById("copy-command");

  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(command.textContent);
      copyButton.textContent = "Copied";
    } catch {
      const range = document.createRange();
      range.selectNodeContents(command);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copyButton.textContent = "Select & copy";
    }
    window.setTimeout(() => { copyButton.textContent = "Copy"; }, 2200);
  });
})();
