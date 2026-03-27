// State
var currentSessionId = null;
var sessions = [];
var isStreaming = false;

// DOM Elements
var messagesEl = document.getElementById('messages');
var inputEl = document.getElementById('input');
var chatListEl = document.getElementById('chatList');
var chatTitleEl = document.getElementById('chatTitle');
var modelBadgeEl = document.getElementById('modelBadge');
var searchInputEl = document.getElementById('searchInput');

// Model providers
var DEFAULT_MODEL_PROVIDERS = [
  {id:"ollama", label:"Ollama（本地）"},
  {id:"openai_compatible", label:"OpenAI 兼容 API"},
  {id:"anthropic_compatible", label:"Anthropic 兼容 / 中转"},
  {id:"mock", label:"mock（调试）"}
];

// API helper
async function api(path, method, body) {
  const res = await fetch(path, {
    method: method || "GET",
    headers: {"Content-Type":"application/json"},
    body: body ? JSON.stringify(body) : null
  });
  const text = await res.text();
  if(!res.ok){
    let detail = text;
    try{
      const j = JSON.parse(text);
      if(j && j.error) detail = j.error + (j.code ? " (" + j.code + ")" : "");
    }catch(_){}
    throw new Error(detail || ("HTTP " + res.status));
  }
  return text ? JSON.parse(text) : {};
}

// Generate UUID
function newSessionId(){
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return "s-" + Date.now() + "-" + Math.random().toString(36).slice(2, 9);
}

// Local storage helpers
function getStorage(key, def) {
  try { return JSON.parse(localStorage.getItem(key) || def); } catch(e) { return def; }
}
function setStorage(key, val) { localStorage.setItem(key, JSON.stringify(val)); }

// Escape HTML (for trusted static strings only — prefer textContent where possible)
function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// Render chat list
function renderChatList() {
  chatListEl.innerHTML = '';
  var filter = (searchInputEl.value || "").toLowerCase();

  var ordered = getStorage('chat_order', []);
  var titles = getStorage('chat_titles', {});

  var filtered = sessions.filter(function(s) {
    var title = titles[s.id] || "";
    return title.toLowerCase().includes(filter);
  });

  // Sort by order
  filtered.sort(function(a, b) {
    var ai = ordered.indexOf(a.id);
    var bi = ordered.indexOf(b.id);
    if (ai === -1 && bi === -1) return 0;
    if (ai === -1) return 1;
    if (bi === -1) return -1;
    return ai - bi;
  });

  filtered.forEach(function(s) {
    var title = titles[s.id] || "新对话";
    var item = document.createElement('div');
    item.className = 'chat-list-item' + (s.id === currentSessionId ? ' active' : '');

    var titleSpan = document.createElement('span');
    titleSpan.className = 'chat-title';
    titleSpan.textContent = title; // XSS safe

    var actionsDiv = document.createElement('div');
    actionsDiv.className = 'chat-actions';

    var renameBtn = document.createElement('button');
    renameBtn.className = 'action-btn';
    renameBtn.setAttribute('aria-label', '重命名对话');
    renameBtn.textContent = '✎';
    renameBtn.onclick = function(e) { e.stopPropagation(); renameChat(s.id); };

    var deleteBtn = document.createElement('button');
    deleteBtn.className = 'action-btn';
    deleteBtn.setAttribute('aria-label', '删除对话');
    deleteBtn.textContent = '×';
    deleteBtn.onclick = function(e) { e.stopPropagation(); deleteChat(s.id); };

    actionsDiv.appendChild(renameBtn);
    actionsDiv.appendChild(deleteBtn);
    item.appendChild(titleSpan);
    item.appendChild(actionsDiv);
    item.onclick = function() { selectChat(s.id); };
    chatListEl.appendChild(item);
  });
}

// Select chat
async function selectChat(id) {
  currentSessionId = id;
  localStorage.setItem('current_session', id);
  var titles = getStorage('chat_titles', {});
  chatTitleEl.textContent = titles[id] || "新对话";
  await loadHistory(id);
  renderChatList();
  closeSidebarMobile();
}

// Load history
async function loadHistory(id) {
  messagesEl.innerHTML = '';
  try {
    var data = await api('/api/session/' + encodeURIComponent(id) + '/history', 'GET');
    var msgs = data.messages || [];
    if (msgs.length === 0) {
      addMessage('assistant', '你好！我是 Mini Agent。有什么可以帮助你的吗？');
    } else {
      msgs.forEach(function(m) {
        if (m.role === 'user') addMessage('user', m.content || '');
        else if (m.role === 'assistant') addMessage('assistant', m.content || '');
      });
    }
  } catch(e) {
    addMessage('assistant', '加载历史失败: ' + e.message);
  }
}

// New chat
async function newChat() {
  var id = newSessionId();
  currentSessionId = id;
  var titles = getStorage('chat_titles', {});
  titles[id] = "新对话";
  setStorage('chat_titles', titles);

  var order = getStorage('chat_order', []);
  order.unshift(id);
  setStorage('chat_order', order);

  messagesEl.innerHTML = '';
  addMessage('assistant', '你好！我是 Mini Agent。有什么可以帮助你的吗？');

  chatTitleEl.textContent = "新对话";

  try {
    await api('/api/session', 'POST', { session_id: id });
  } catch(e) {}
  await refreshSessions();
  renderChatList();
  closeSidebarMobile();
}

// Rename chat
function renameChat(id) {
  var titles = getStorage('chat_titles', {});
  var old = titles[id] || "新对话";
  var t = prompt("对话名称:", old);
  if (t === null) return;
  t = t.trim() || "新对话";
  titles[id] = t;
  setStorage('chat_titles', titles);
  if (id === currentSessionId) chatTitleEl.textContent = t;
  renderChatList();
}

// Delete chat
async function deleteChat(id) {
  if (!confirm("删除此对话？")) return;
  try {
    await api('/api/session/' + encodeURIComponent(id), 'DELETE');
  } catch(e) {}
  var titles = getStorage('chat_titles', {});
  delete titles[id];
  setStorage('chat_titles', titles);
  var order = getStorage('chat_order', []);
  order = order.filter(function(x) { return x !== id; });
  setStorage('chat_order', order);
  await refreshSessions();
  if (id === currentSessionId) {
    if (sessions.length > 0) {
      await selectChat(sessions[0].id);
    } else {
      await newChat();
    }
  }
  renderChatList();
}

// Add message to UI (XSS safe — uses textContent)
function addMessage(role, content, time) {
  var time = time || new Date().toLocaleTimeString('zh-CN', {hour: '2-digit', minute:'2-digit'});
  var msgEl = document.createElement('div');
  msgEl.className = 'message ' + role;
  msgEl.setAttribute('role', 'article');

  var avatarEl = document.createElement('div');
  avatarEl.className = 'message-avatar';

  var contentEl = document.createElement('div');
  contentEl.className = 'message-content';

  var bubbleEl = document.createElement('div');
  bubbleEl.className = 'bubble';
  bubbleEl.textContent = content; // XSS safe

  var timeEl = document.createElement('div');
  timeEl.className = 'message-time';
  timeEl.textContent = time;

  contentEl.appendChild(bubbleEl);
  contentEl.appendChild(timeEl);
  msgEl.appendChild(avatarEl);
  msgEl.appendChild(contentEl);
  messagesEl.appendChild(msgEl);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return msgEl;
}

// Streaming message — appends text incrementally
function addStreamingMessage() {
  var msgEl = document.createElement('div');
  msgEl.className = 'message assistant streaming-msg';
  msgEl.setAttribute('role', 'article');

  var avatarEl = document.createElement('div');
  avatarEl.className = 'message-avatar';

  var contentEl = document.createElement('div');
  contentEl.className = 'message-content';

  var bubbleEl = document.createElement('div');
  bubbleEl.className = 'bubble';
  bubbleEl.textContent = '';

  var cursor = document.createElement('span');
  cursor.className = 'stream-cursor';
  cursor.setAttribute('aria-hidden', 'true');
  bubbleEl.appendChild(cursor);

  contentEl.appendChild(bubbleEl);
  msgEl.appendChild(avatarEl);
  msgEl.appendChild(contentEl);
  messagesEl.appendChild(msgEl);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return { msgEl, bubbleEl, cursor };
}

function appendStreamingText(bubbleEl, text) {
  // Remove cursor, append text, re-add cursor
  var cursor = bubbleEl.querySelector('.stream-cursor');
  bubbleEl.textContent = text;
  bubbleEl.appendChild(cursor);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function finalizeStreamingMessage(msgEl) {
  var cursor = msgEl.querySelector('.stream-cursor');
  if (cursor) cursor.remove();
}

// Add thinking indicator
function addThinking() {
  var msgEl = document.createElement('div');
  msgEl.className = 'message assistant thinking-msg';
  msgEl.innerHTML =
    '<div class="message-avatar"></div>' +
    '<div class="message-content">' +
    '<div class="thinking">' +
    '<div class="thinking-dots"><span></span><span></span><span></span></div>' +
    '<span>思考中...</span>' +
    '</div>' +
    '</div>';
  messagesEl.appendChild(msgEl);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return msgEl;
}

// Remove thinking
function removeThinking() {
  var el = messagesEl.querySelector('.thinking-msg');
  if (el) el.remove();
}

// Send message (with SSE streaming)
async function sendMessage() {
  var text = inputEl.value.trim();
  if (!text || isStreaming) return;

  if (!currentSessionId) await newChat();

  inputEl.value = '';
  addMessage('user', text);
  isStreaming = true;
  document.getElementById('btnSend').disabled = true;

  var streamingData = addStreamingMessage();

  try {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: currentSessionId,
        message: text,
        use_rag: true
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      let errMsg = '请求失败';
      try {
        const j = JSON.parse(errText);
        if (j && j.error) errMsg = j.error;
      } catch (_) {}
      throw new Error(errMsg);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    var fullText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      // SSE format: data: {"answer": "..."}\n\n
      const lines = chunk.split('\n');
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i].trim();
        if (!line.startsWith('data:')) continue;
        var dataStr = line.slice(5).trim();
        if (!dataStr || dataStr === '[DONE]') continue;
        try {
          var data = JSON.parse(dataStr);
          if (data.text) {
            fullText += data.text;
            appendStreamingText(streamingData.bubbleEl, fullText);
          }
          if (data.error) {
            finalizeStreamingMessage(streamingData.msgEl);
            streamingData.bubbleEl.textContent = '错误: ' + data.error;
            return;
          }
        } catch (_) {}
      }
    }

    finalizeStreamingMessage(streamingData.msgEl);
    if (!fullText) {
      streamingData.bubbleEl.textContent = '未收到回复';
    }

    // Update title if new
    var titles = getStorage('chat_titles', {});
    if (titles[currentSessionId] === "新对话" && text.length > 0) {
      titles[currentSessionId] = text.slice(0, 30) + (text.length > 30 ? '...' : '');
      setStorage('chat_titles', titles);
      chatTitleEl.textContent = titles[currentSessionId];
      renderChatList();
    }

    await refreshSessions();
  } catch(e) {
    finalizeStreamingMessage(streamingData.msgEl);
    streamingData.bubbleEl.textContent = '请求失败: ' + e.message;
  } finally {
    isStreaming = false;
    document.getElementById('btnSend').disabled = false;
  }
}

// Refresh sessions
async function refreshSessions() {
  try {
    var data = await api('/api/sessions', 'GET');
    sessions = data.sessions || [];
  } catch(e) {
    sessions = [];
  }
}

// Apply model state to modal
function applyModelState(s) {
  s = s || {};
  var sel = document.getElementById('provider');
  var list = s.model_providers || DEFAULT_MODEL_PROVIDERS;
  sel.innerHTML = '';
  var seen = {};
  list.forEach(function(row) {
    var id = row.id || row.value;
    if (!id) return;
    seen[id] = true;
    var opt = document.createElement('option');
    opt.value = id;
    opt.textContent = row.label || id;
    sel.appendChild(opt);
  });
  var cur = (s.model_provider || '').trim();
  if (cur && !seen[cur]) {
    var opt = document.createElement('option');
    opt.value = cur;
    opt.textContent = cur + ' (当前)';
    sel.appendChild(opt);
  }
  sel.value = cur;
  document.getElementById('modelName').value = s.model_name || '';
  modelBadgeEl.textContent = s.model_name || '未设置';
}

// --- Focus trap for modal ---
function trapFocus(el) {
  var focusable = el.querySelectorAll(
    'button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  if (!focusable.length) return;
  var first = focusable[0];
  var last = focusable[focusable.length - 1];

  el.addEventListener('keydown', function handler(e) {
    if (e.key !== 'Tab') return;
    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  });
}

// --- Mobile sidebar ---
function openSidebarMobile() {
  document.getElementById('sidebar').classList.add('open');
  document.getElementById('sidebarOverlay').classList.add('show');
  document.getElementById('btnHamburger').setAttribute('aria-expanded', 'true');
  // Focus first sidebar item
  var first = document.querySelector('.chat-list-item');
  if (first) first.focus();
}

function closeSidebarMobile() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('show');
  document.getElementById('btnHamburger').setAttribute('aria-expanded', 'false');
}

// Init
async function init() {
  // Load sessions
  await refreshSessions();

  // Try to get current session
  var savedSession = localStorage.getItem('current_session');
  if (savedSession && sessions.find(function(s) { return s.id === savedSession; })) {
    await selectChat(savedSession);
  } else if (sessions.length > 0) {
    await selectChat(sessions[0].id);
  } else {
    await newChat();
  }

  // Get model state
  try {
    var state = await api('/api/state', 'GET');
    applyModelState(state);
  } catch(e) {
    console.warn('Failed to load state:', e);
  }

  renderChatList();

  // Setup focus trap on modal
  var modal = document.getElementById('modelModal');
  trapFocus(modal);

  // Open modal with focus trap
  document.getElementById('btnOpenModel').onclick = async function() {
    try {
      var state = await api('/api/state', 'GET');
      applyModelState(state);
    } catch(e) {}
    document.getElementById('modelModal').classList.add('show');
    // Focus first focusable element
    var firstFocusable = modal.querySelector('button, input, select');
    if (firstFocusable) firstFocusable.focus();
  };

  document.getElementById('btnCancelModel').onclick = function() {
    document.getElementById('modelModal').classList.remove('show');
    document.getElementById('btnOpenModel').focus();
  };

  document.getElementById('btnSaveModel').onclick = async function() {
    var provider = document.getElementById('provider').value;
    var modelName = document.getElementById('modelName').value.trim();
    try {
      await api('/api/model', 'POST', {provider, model_name: modelName});
      var state = await api('/api/state', 'GET');
      applyModelState(state);
      document.getElementById('modelModal').classList.remove('show');
      addMessage('system', '模型已更新为 ' + provider + ' / ' + modelName);
      document.getElementById('btnOpenModel').focus();
    } catch(e) {
      alert('更新失败: ' + e.message);
    }
  };

  document.getElementById('modelModal').onclick = function(e) {
    if (e.target === this) {
      this.classList.remove('show');
      document.getElementById('btnOpenModel').focus();
    }
  };
}

// Event listeners
document.getElementById('btnNewChat').onclick = newChat;
document.getElementById('btnSend').onclick = sendMessage;
document.getElementById('input').onkeydown = function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
};

searchInputEl.oninput = renderChatList;

// Mobile sidebar events
document.getElementById('btnHamburger').onclick = openSidebarMobile;
document.getElementById('sidebarOverlay').onclick = closeSidebarMobile;

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  // Esc to close modal or sidebar
  if (e.key === 'Escape') {
    var modal = document.getElementById('modelModal');
    if (modal.classList.contains('show')) {
      modal.classList.remove('show');
      document.getElementById('btnOpenModel').focus();
    } else {
      closeSidebarMobile();
    }
  }
  // "/" to focus search (when not in input)
  if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
    e.preventDefault();
    searchInputEl.focus();
  }
});

// Auto-resize textarea
inputEl.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});

// Start
init();
