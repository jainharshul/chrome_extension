// eval.js

// ===== Existing TF-IDF + Logistic Regression path =====
// (kept intact, only exported helpers are reused)
async function loadModel() {
  if (window._cachedModel) return window._cachedModel;
  // NOTE: ensure the path matches your packaging. If your JSON sits at the root,
  // change to chrome.runtime.getURL("tfidf_logreg_model.json")
  const resp = await fetch(chrome.runtime.getURL("assets/tfidf_logreg_model.json"));
  const model = await resp.json();
  if (!model.vectorizer.vocabulary_map || Object.keys(model.vectorizer.vocabulary_map).length === 0) {
    model.vectorizer.vocabulary_map = {};
    for (let i = 0; i < model.vectorizer.vocabulary.length; i++) {
      const tok = model.vectorizer.vocabulary[i];
      if (tok != null) model.vectorizer.vocabulary_map[tok] = i;
    }
  }
  window._cachedModel = model;
  return model;
}

function tokenize(text) {
  const re = /\b\w\w+\b/g;
  const lower = (text || "").toLowerCase();
  const matches = lower.match(re);
  return matches || [];
}
function removeStopwords(tokens, stopWordsSet) {
  if (!stopWordsSet || stopWordsSet.size === 0) return tokens;
  return tokens.filter(t => !stopWordsSet.has(t));
}
function buildNgrams(tokens, nMax = 2) {
  const grams = [];
  for (let i = 0; i < tokens.length; i++) {
    grams.push(tokens[i]);
    if (nMax >= 2 && i + 1 < tokens.length) grams.push(tokens[i] + " " + tokens[i + 1]);
  }
  return grams;
}
function textToTfidfVector(text, vectorizer) {
  const vocabMap = vectorizer.vocabulary_map;
  const indexToToken = vectorizer.vocabulary;
  const idf = vectorizer.idf || [];
  const useIdf = vectorizer.use_idf;
  const sublinearTf = vectorizer.sublinear_tf;
  const norm = vectorizer.norm || "l2";
  const stopSet = new Set(vectorizer.stop_words || []);
  let tokens = tokenize(text);
  tokens = removeStopwords(tokens, stopSet);
  const ngrams = buildNgrams(tokens, vectorizer.ngram_range ? vectorizer.ngram_range[1] : 1);

  const vec = new Array(indexToToken.length).fill(0);
  for (const gram of ngrams) {
    const idx = vocabMap[gram];
    if (idx !== undefined) vec[idx] += 1;
  }
  if (sublinearTf) {
    for (let i = 0; i < vec.length; i++) if (vec[i] > 0) vec[i] = 1 + Math.log(vec[i]);
  }
  if (useIdf && idf && idf.length === vec.length) {
    for (let i = 0; i < vec.length; i++) vec[i] *= idf[i];
  }
  if (norm === "l2") {
    let sumSq = 0;
    for (let i = 0; i < vec.length; i++) sumSq += vec[i] * vec[i];
    const normVal = Math.sqrt(sumSq);
    if (normVal > 0) for (let i = 0; i < vec.length; i++) vec[i] /= normVal;
  }
  return vec;
}
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function softmax(arr) {
  const maxv = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - maxv));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / s);
}
function predictProbaFromVector(model, vec) {
  const clf = model.classifier;
  const coef = clf.coef;
  const intercept = clf.intercept;
  const classes = clf.classes;

  if (!Array.isArray(coef[0])) throw new Error("Unexpected coef shape");

  if (classes.length === 2) {
    const w = coef[0];
    let score = 0;
    for (let i = 0; i < w.length; i++) score += w[i] * (vec[i] || 0);
    const b = Array.isArray(intercept) ? intercept[0] : intercept;
    score += b;
    const p1 = sigmoid(score);
    return classes.map(c => (c === 0 ? 1 - p1 : p1)); // [p(class0), p(class1)]
  } else {
    const scores = coef.map(wRow => {
      let s = 0;
      for (let i = 0; i < wRow.length; i++) s += wRow[i] * (vec[i] || 0);
      return s;
    });
    const b = Array.isArray(intercept) ? intercept : [intercept];
    for (let k = 0; k < scores.length; k++) scores[k] += (b[k] || 0);
    return softmax(scores);
  }
}

async function predictWithTfidf(text) {
  const model = await loadModel();
  const vec = textToTfidfVector(text, model.vectorizer);
  const probs = predictProbaFromVector(model, vec);
  // Assume classes are [0,1] with 1 = phishing
  const prediction = probs[1] >= probs[0] ? 1 : 0;
  return { prediction, probabilities: probs };
}

// ===== New: BERT (zero-shot via Hugging Face Inference API) =====
// Uses a (Distil)BERT MNLI model with candidate labels ["phishing","legitimate"].
const HF_ZSC_MODEL = "typeform/distilbert-base-uncased-mnli"; // lightweight BERT-family MNLI

async function predictWithBertZSC(text, hfApiKey) {
  if (!hfApiKey) {
    throw new Error("Missing Hugging Face API key. Enter it and click ‘Save API Key’.");
  }
  const resp = await fetch("https://api-inference.huggingface.co/models/" + HF_ZSC_MODEL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${hfApiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      inputs: text,
      parameters: {
        candidate_labels: ["phishing", "legitimate"],
        multi_label: false
      }
    })
  });
  if (!resp.ok) {
    const msg = await resp.text();
    throw new Error(`HF API error (${resp.status}): ${msg}`);
  }
  const data = await resp.json();
  // HF zero-shot returns {labels:[...], scores:[...]} or array form
  const out = Array.isArray(data) ? data[0] : data;
  const labels = out.labels || [];
  const scores = out.scores || [];

  // Build probabilities in [p(legitimate), p(phishing)] order to match TF-IDF output shape
  const idxPhish = labels.findIndex(l => l.toLowerCase() === "phishing");
  const idxLegit = labels.findIndex(l => l.toLowerCase() === "legitimate");
  const pPhish = idxPhish >= 0 ? scores[idxPhish] : 0.5;
  const pLegit = idxLegit >= 0 ? scores[idxLegit] : (1 - pPhish);
  const probs = [pLegit, pPhish];
  const prediction = pPhish >= pLegit ? 1 : 0;
  return { prediction, probabilities: probs };
}

// ===== Single entry point used by the UI =====
async function predictText(text, { model = "tfidf", hfApiKey } = {}) {
  if (model === "bert") {
    return predictWithBertZSC(text, hfApiKey);
  }
  return predictWithTfidf(text);
}

// Expose to popup
window.predictText = predictText;
