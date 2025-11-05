document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("go");
  const input = document.getElementById("inputText");
  const out = document.getElementById("out");

  btn.addEventListener("click", async () => {
    out.textContent = "Predicting…";
    try {
      const res = await predictText(input.value); // predictText() is in predict.js
      const classification = (res.prediction === 1) ? "Phishing" : "Legitimate";
      const confidence = res.probabilities[res.prediction].toFixed(3);
      out.textContent = `Prediction: ${classification}\nConfidence: ${(confidence * 100) % 100}%`;
    } catch (err) {
      out.textContent = `Error: ${err.message}`;
    }
  });
});
