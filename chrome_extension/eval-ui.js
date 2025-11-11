// eval-ui.js
document.addEventListener("DOMContentLoaded", async () => {
  const btn = document.getElementById("go");
  const input = document.getElementById("inputText");
  const out = document.getElementById("out");

  const modelSelect = document.getElementById("modelSelect");
  const hfKeyRow = document.getElementById("hfKeyRow");
  const hfKeyInput = document.getElementById("hfKey");
  const saveKeyBtn = document.getElementById("saveKey");

  // restore model + key from storage
  chrome.storage.sync.get(["modelChoice", "hfApiKey"], ({ modelChoice, hfApiKey }) => {
    if (modelChoice) modelSelect.value = modelChoice;
    hfKeyRow.style.display = (modelSelect.value === "bert") ? "block" : "none";
    if (hfApiKey) hfKeyInput.value = hfApiKey;
  });

  modelSelect.addEventListener("change", () => {
    const value = modelSelect.value;
    hfKeyRow.style.display = (value === "bert") ? "block" : "none";
    chrome.storage.sync.set({ modelChoice: value });
  });

  saveKeyBtn.addEventListener("click", () => {
    chrome.storage.sync.set({ hfApiKey: hfKeyInput.value || "" }, () => {
      out.textContent = "Saved Hugging Face API key.";
    });
  });

  btn.addEventListener("click", async () => {
    out.textContent = "Predicting…";
    try {
      const { hfApiKey } = await chrome.storage.sync.get(["hfApiKey"]);
      const res = await predictText(input.value, {
        model: modelSelect.value,
        hfApiKey
      }); // implemented in eval.js
      const classification = (res.prediction === 1) ? "Phishing" : "Legitimate";
      const confidence = res.probabilities[res.prediction].toFixed(3);
      out.textContent = `Prediction: ${classification}\nConfidence: ${(confidence * 100).toFixed(1)}%`;
    } catch (err) {
      out.textContent = `Error: ${err.message}`;
    }
  });
});
