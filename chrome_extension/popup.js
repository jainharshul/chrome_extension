// Helper to run a function in the active tab
async function runInActiveTab(fn, args = []) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;
  await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: fn,
    args
  });
}

document.getElementById('change').addEventListener('click', () => {
  runInActiveTab((color) => {
    // Save previous color on the page itself so we can restore later
    const prev = document.body.style.backgroundColor || "";
    document.body.dataset.prevBg = prev;
    document.body.style.backgroundColor = color;
  }, ['#ffeaa7']); // a soft yellow
});

document.getElementById('restore').addEventListener('click', () => {
  runInActiveTab(() => {
    const prev = document.body.dataset.prevBg || "";
    document.body.style.backgroundColor = prev;
    delete document.body.dataset.prevBg;
  });
});
