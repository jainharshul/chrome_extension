async function loadModel() {
  // loads and caches the model JSON for reuse
  if (window._cachedModel) return window._cachedModel;
  const resp = await fetch(chrome.runtime.getURL("assets/tfidf_logreg_model.json"));
  const model = await resp.json();
  // ensure vocabulary_map exists: token -> index
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

// Normalizes input, same as trainging and test data
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

// 4) buildNgrams(tokens, nMax): array of ngrams (strings)
//    For ngram_range=(1,2) produce unigrams and adjacent bigrams
function buildNgrams(tokens, nMax = 2) {
  const grams = [];
  for (let i = 0; i < tokens.length; i++) {
    // unigram
    grams.push(tokens[i]);
    // bigram
    if (nMax >= 2 && i + 1 < tokens.length) grams.push(tokens[i] + " " + tokens[i + 1]);
  }
  return grams;
}

// 5) textToTfidfVector(text, vectorizer): Float32Array or number[]
//    Counts -> optional sublinear_tf -> idf multiply -> l2 normalize
function textToTfidfVector(text, vectorizer) {
  const vocabMap = vectorizer.vocabulary_map;       // token -> index
  const indexToToken = vectorizer.vocabulary;       // index -> token
  const idf = vectorizer.idf || [];                 // idf by index
  const useIdf = vectorizer.use_idf;
  const sublinearTf = vectorizer.sublinear_tf;
  const norm = vectorizer.norm || "l2";
  const stopSet = new Set(vectorizer.stop_words || []);

  // Tokenize -> remove stopwords -> build ngrams
  let tokens = tokenize(text);
  tokens = removeStopwords(tokens, stopSet);        // sklearn removes stopwords first
  const ngrams = buildNgrams(tokens, vectorizer.ngram_range ? vectorizer.ngram_range[1] : 1);

  // Count vector
  const vec = new Array(indexToToken.length).fill(0);
  for (const gram of ngrams) {
    const idx = vocabMap[gram];
    if (idx !== undefined) vec[idx] += 1;
  }

  // sublinear tf
  if (sublinearTf) {
    for (let i = 0; i < vec.length; i++) if (vec[i] > 0) vec[i] = 1 + Math.log(vec[i]);
  }

  // apply idf
  if (useIdf && idf && idf.length === vec.length) {
    for (let i = 0; i < vec.length; i++) vec[i] *= idf[i];
  }

  // l2 normalization
  if (norm === "l2") {
    let sumSq = 0;
    for (let i = 0; i < vec.length; i++) sumSq += vec[i] * vec[i];
    const normVal = Math.sqrt(sumSq);
    if (normVal > 0) {
      for (let i = 0; i < vec.length; i++) vec[i] /= normVal;
    }
  }

  return vec; // dense vector (Array). Consider sparse for performance.
}

// 6) linear scores / predictProbaFromVector(vec, model): number[] (probabilities)
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function softmax(arr) {
  const maxv = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - maxv));
  const s = exps.reduce((a,b) => a + b, 0);
  return exps.map(e => e / s);
}

function predictProbaFromVector(model, vec) {
  const clf = model.classifier;
  const coef = clf.coef;           // array of arrays
  const intercept = clf.intercept;
  const classes = clf.classes;

  if (!Array.isArray(coef[0])) {
    throw new Error("Unexpected coef shape");
  }

  if (classes.length === 2) {
    // binary: coef is [[w...]]
    const w = coef[0];
    let score = 0;
    for (let i = 0; i < w.length; i++) score += w[i] * (vec[i] || 0);
    const b = Array.isArray(intercept) ? intercept[0] : intercept;
    score += b;
    const p1 = sigmoid(score);
    const p0 = 1 - p1;
    // return in same order as classes list
    return classes.map(c => (c === classes[1] ? p1 : p0));
  } else {
    // multiclass: compute scores for each class
    const scores = coef.map((wRow, idx) => {
      let s = 0;
      for (let i = 0; i < wRow.length; i++) s += wRow[i] * (vec[i] || 0);
      s += intercept[idx];
      return s;
    });
    return softmax(scores);
  }
}

// 7) predictText(text): orchestrates loading, vectorizing, predicting
async function predictText(text) {
  const model = await loadModel();
  const vec = textToTfidfVector(text, model.vectorizer);
  const probs = predictProbaFromVector(model, vec);
  // pick top
  const maxIndex = probs.indexOf(Math.max(...probs));
  return { prediction: model.classifier.classes[maxIndex], probabilities: probs };
}