// Optional: runs in the background (service worker). Good place for alarms, context menus, etc.
chrome.runtime.onInstalled.addListener(() => {
  console.log('Hello MV3: installed');
});
