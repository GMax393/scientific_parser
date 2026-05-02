const BASE = "http://127.0.0.1:5000";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "sp-open-doi",
    title: "Scientific Parser: открыть выделение",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  const raw = (info.selectionText || "").trim();
  if (!raw) {
    return;
  }
  const q = encodeURIComponent(raw);
  const url = raw.startsWith("http://") || raw.startsWith("https://")
    ? `${BASE}/?url=${q}`
    : `${BASE}/?doi=${q}`;
  chrome.tabs.create({ url });
});
