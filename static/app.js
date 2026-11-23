/* ==========================================================================
   EduRAG · 前端交互
   协议说明：
   - POST /api/query          先做快速问答（BM25 / 寒暄），返回 is_streaming
   - WS   /api/stream         is_streaming 为真时，走流式检索增强回答
   - GET  /api/history/{id}   读取最近 5 轮会话
   - DEL  /api/history/{id}   清除会话
   - GET  /api/sources        读取资料范围
   - GET  /health             服务探针
   ========================================================================== */

(function () {
    'use strict';

    /* 换一套问答系统时，只需要改这里：接口主机 + 页面文案 */
    const API_HOST = '';

    const PAGE = {
        documentTitle: '高校可研与教务智能问答',
        systemName: '教研问答',
        welcomeEyebrow: 'EDURAG · 校园知识助手',
        welcomeTitle: '让每个问题，\n回到资料本身。',
        welcomeSub: '覆盖理学、工学、医学、文学、农学与教务六类本地资料。先限定资料范围再提问，回答会尽量依据检索到的内容生成。',
        welcomeFoot: '提示：资料范围会直接影响检索结果，不确定时保持“全部资料”。',
        suggestions: [
            { tag: '教务', source: 'jw', q: '学籍异动一般怎么办理？需要哪些材料？' },
            { tag: '工学', source: 'gx', q: '机械可靠性设计包含哪些核心内容？' },
            { tag: '理学', source: 'lx', q: '石墨烯的制备方法主要有哪些？' },
            { tag: '农学', source: 'nx', q: '绿肥作物如何改良盐碱地土壤？' }
        ]
    };

    const SOURCE_LABELS = {
        lx: '理学',
        gx: '工学',
        yx: '医学',
        wx: '文学',
        nx: '农学',
        jw: '教务'
    };

    const STORAGE = {
        theme: 'edurag.theme',
        sessions: 'edurag.sessions.v1'
    };

    const MAX_SESSIONS = 12;
    const IDLE_TIMEOUT = 60000;

    const state = {
        sessionId: null,
        sessions: [],
        thread: [],
        busy: false,
        stop: null,
        theme: 'light',
        turn: 0,
        loadToken: 0
    };

    const els = {};

    /* 1. 基础工具 */

    const $ = (selector, root = document) => root.querySelector(selector);

    function el(tag, className, html) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (html != null) node.innerHTML = html;
        return node;
    }

    const http = (path) => `${API_HOST.replace(/\/$/, '')}${path}`;

    function wsUrl(path) {
        if (!API_HOST) {
            const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
            return `${scheme}//${location.host}${path}`;
        }
        const parsed = new URL(API_HOST);
        const scheme = parsed.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${scheme}//${parsed.host}${path}`;
    }

    function escapeHtml(value) {
        return String(value).replace(/[&<>"']/g, (char) => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }[char]));
    }

    const ALLOWED_TAGS = new Set([
        'P', 'BR', 'STRONG', 'B', 'EM', 'I', 'DEL', 'S', 'CODE', 'PRE',
        'UL', 'OL', 'LI', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
        'BLOCKQUOTE', 'HR', 'A', 'TABLE', 'THEAD', 'TBODY', 'TR', 'TH', 'TD', 'SPAN'
    ]);

    const DROP_TAGS = new Set([
        'SCRIPT', 'STYLE', 'IFRAME', 'OBJECT', 'EMBED', 'IMG', 'SVG', 'MATH',
        'LINK', 'META', 'FORM', 'INPUT', 'BUTTON', 'TEXTAREA', 'SELECT',
        'VIDEO', 'AUDIO', 'SOURCE', 'TRACK', 'CANVAS'
    ]);

    /* 仅保留排版标签，去掉脚本、图片、内联样式与事件属性 */
    function sanitize(html) {
        const template = document.createElement('template');
        template.innerHTML = html;
        const nodes = [...template.content.querySelectorAll('*')];

        nodes.forEach((node) => {
            if (!template.content.contains(node)) return;

            if (DROP_TAGS.has(node.tagName)) {
                node.remove();
                return;
            }

            if (!ALLOWED_TAGS.has(node.tagName)) {
                node.replaceWith(...node.childNodes);
                return;
            }

            [...node.attributes].forEach((attr) => {
                const name = attr.name.toLowerCase();
                if (name.startsWith('on') || name === 'style' || name === 'class') {
                    node.removeAttribute(attr.name);
                }
                if (node.tagName === 'A' && name === 'href' && !/^(https?:|mailto:|#|\/)/i.test(attr.value.trim())) {
                    node.removeAttribute(attr.name);
                }
            });

            if (node.tagName === 'A') {
                node.setAttribute('target', '_blank');
                node.setAttribute('rel', 'noopener noreferrer');
            }
        });

        return template.innerHTML;
    }

    function renderMarkdown(text) {
        if (!text) return '';
        try {
            if (window.marked && typeof window.marked.parse === 'function') {
                return sanitize(window.marked.parse(text, { gfm: true, breaks: true }));
            }
        } catch (error) {
            /* 解析失败时退回纯文本 */
        }
        return `<p>${escapeHtml(text).replace(/\n/g, '<br>')}</p>`;
    }

    function formatTime(timestamp) {
        const date = new Date(timestamp);
        if (Number.isNaN(date.getTime())) return '';
        const now = new Date();
        const clock = `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
        if (date.toDateString() === now.toDateString()) return `今天 ${clock}`;
        return `${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')} ${clock}`;
    }

    function truncate(text, length) {
        const value = String(text).replace(/\s+/g, ' ').trim();
        return value.length > length ? `${value.slice(0, length)}…` : value;
    }

    function toast(message, kind = 'info') {
        if (!els.toastStack) return;
        const node = el('div', 'toast', escapeHtml(message));
        node.dataset.kind = kind;
        els.toastStack.appendChild(node);
        setTimeout(() => {
            node.style.opacity = '0';
            node.style.transform = 'translateY(6px)';
        }, 2400);
        setTimeout(() => node.remove(), 2850);
    }

    async function copyText(text, button) {
        if (!text) return;
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
            } else {
                const helper = document.createElement('textarea');
                helper.value = text;
                helper.setAttribute('readonly', '');
                helper.style.position = 'fixed';
                helper.style.opacity = '0';
                document.body.appendChild(helper);
                helper.select();
                document.execCommand('copy');
                helper.remove();
            }
            toast('已复制到剪贴板');
            if (button) {
                const label = $('span', button);
                if (label) {
                    const original = label.textContent;
                    label.textContent = '已复制';
                    setTimeout(() => { label.textContent = original; }, 1600);
                }
            }
        } catch (error) {
            toast('复制失败，请手动选择文本', 'error');
        }
    }

    /* 2. 会话管理 */

    function loadSessions() {
        try {
            const raw = JSON.parse(localStorage.getItem(STORAGE.sessions) || '[]');
            if (!Array.isArray(raw)) return [];
            return raw
                .filter((item) => item && typeof item.id === 'string' && item.id)
                .slice(0, MAX_SESSIONS)
                .map((item) => ({
                    id: item.id,
                    title: typeof item.title === 'string' && item.title.trim() ? item.title : '新对话',
                    rounds: Number.isFinite(item.rounds) ? item.rounds : 0,
                    ts: Number.isFinite(item.ts) ? item.ts : Date.now()
                }));
        } catch (error) {
            return [];
        }
    }

    function persistSessions() {
        try {
            localStorage.setItem(STORAGE.sessions, JSON.stringify(state.sessions));
        } catch (error) {
            /* 存储不可用时忽略 */
        }
    }

    function activeEntry() {
        return state.sessions.find((item) => item.id === state.sessionId) || null;
    }

    function renderSessions() {
        if (!els.sessionList) return;
        els.sessionCount.textContent = String(state.sessions.length);
        els.sessionList.innerHTML = '';

        if (!state.sessions.length) {
            els.sessionList.appendChild(el('div', 'session-empty', '还没有会话记录<br>提问后会自动保存在这里'));
            return;
        }

        state.sessions.forEach((session) => {
            const button = el('button', 'session-item');
            button.type = 'button';
            button.dataset.id = session.id;
            button.title = session.title;
            if (session.id === state.sessionId) button.classList.add('is-active');
            button.innerHTML = `
                <span class="session-mark"><svg class="icon" aria-hidden="true"><use href="#i-book"/></svg></span>
                <span class="session-copy">
                    <span class="session-title">${escapeHtml(session.title)}</span>
                    <span class="session-sub">${session.rounds ? `${session.rounds} 轮 · ` : ''}${formatTime(session.ts)}</span>
                </span>
                <svg class="icon session-arrow" aria-hidden="true"><use href="#i-arrow-right"/></svg>`;
            button.addEventListener('click', () => openSession(session.id));
            els.sessionList.appendChild(button);
        });
    }

    function updateHeader() {
        const entry = activeEntry();
        const title = entry ? entry.title : '新对话';
        els.convTitle.textContent = title;

        if (state.sessionId) {
            const shortId = state.sessionId.slice(0, 8);
            const rounds = entry ? entry.rounds : 0;
            const stage = rounds ? `已进行 ${rounds} 轮` : '等待提问';
            els.convMeta.textContent = `会话 ${shortId} · ${stage} · 最近 5 轮上下文`;
        } else {
            els.convMeta.textContent = '选择资料范围，开始提问';
        }

        document.title = state.sessionId ? `${title} · ${PAGE.systemName}` : PAGE.documentTitle;
        els.copyAllBtn.disabled = state.thread.length === 0;
        syncComposer();
    }

    function showWelcome() {
        els.thread.innerHTML = '';
        const wrap = el('section', 'welcome');
        const title = PAGE.welcomeTitle.split('\n').map(escapeHtml).join('<br>');
        wrap.innerHTML = `
            <p class="eyebrow">${escapeHtml(PAGE.welcomeEyebrow)}</p>
            <h2>${title}</h2>
            <p class="welcome-sub">${escapeHtml(PAGE.welcomeSub)}</p>
            <div class="suggestions"></div>
            <p class="welcome-foot">${escapeHtml(PAGE.welcomeFoot)}</p>`;
        els.thread.appendChild(wrap);
        els.thread.scrollTop = 0;

        const box = $('.suggestions', wrap);
        PAGE.suggestions.forEach((item) => {
            const button = el('button', 'suggestion');
            button.type = 'button';
            button.innerHTML = `
                <span class="suggestion-tag"><span class="dot"></span>${escapeHtml(item.tag)}</span>
                <span class="suggestion-q">${escapeHtml(item.q)}</span>
                <svg class="icon suggestion-arrow" aria-hidden="true"><use href="#i-arrow-right"/></svg>`;
            button.addEventListener('click', () => runSuggestion(item));
            box.appendChild(button);
        });

        updateHeader();
    }

    function runSuggestion(item) {
        if (state.busy) return;
        if (item.source && [...els.sourceFilter.options].some((option) => option.value === item.source)) {
            els.sourceFilter.value = item.source;
        }
        els.input.value = item.q;
        autosize();
        syncComposer();
        sendMessage();
    }

    function touchTitle(query) {
        const entry = activeEntry();
        if (!entry) return;
        if (!entry.title || entry.title === '新对话') {
            entry.title = truncate(query, 18);
        }
        entry.ts = Date.now();
        persistSessions();
        renderSessions();
        updateHeader();
    }

    function completeTurn(query) {
        const entry = activeEntry();
        if (!entry) return;
        entry.rounds = (entry.rounds || 0) + 1;
        entry.ts = Date.now();
        if (!entry.title || entry.title === '新对话') {
            entry.title = truncate(query, 18);
        }
        const index = state.sessions.findIndex((item) => item.id === entry.id);
        if (index > 0) {
            state.sessions.splice(index, 1);
            state.sessions.unshift(entry);
        }
        persistSessions();
        renderSessions();
        updateHeader();
    }

    /* 3. 消息渲染 */

    function scrollToBottom(force) {
        if (!els.thread) return;
        if (force || isPinned()) {
            els.thread.scrollTop = els.thread.scrollHeight;
        }
    }

    function isPinned() {
        const node = els.thread;
        return node.scrollHeight - node.scrollTop - node.clientHeight < 90;
    }

    function buildUserMessage(text) {
        const wrap = el('article', 'msg msg--user');
        const bubble = el('div', 'bubble');
        bubble.textContent = text;
        wrap.appendChild(bubble);
        return wrap;
    }

    function createBotMessage() {
        const root = el('article', 'msg msg--bot');
        root.innerHTML = `
            <div class="avatar" aria-hidden="true">研</div>
            <div class="msg-main">
                <div class="msg-head">
                    <span class="who">教研助手</span>
                    <span class="mode"></span>
                </div>
                <div class="msg-body">
                    <div class="thinking">
                        <span class="dots"><i></i><i></i><i></i></span>
                        <span class="thinking-text">正在思考…</span>
                    </div>
                    <div class="msg-content md"></div>
                    <span class="caret" hidden></span>
                </div>
                <div class="msg-tools" hidden>
                    <button type="button" class="tool" data-copy>
                        <svg class="icon" aria-hidden="true"><use href="#i-copy"/></svg>
                        <span>复制</span>
                    </button>
                    <span class="stamp"></span>
                </div>
            </div>`;

        const content = $('.msg-content', root);
        const modeEl = $('.mode', root);
        const thinkingEl = $('.thinking', root);
        const thinkingText = $('.thinking-text', root);
        const caret = $('.caret', root);
        const tools = $('.msg-tools', root);
        const stamp = $('.stamp', root);
        const copyBtn = $('[data-copy]', root);

        let text = '';
        let frame = 0;

        /* 把光标放到最后一段文字里，避免它单独占一行 */
        const placeCaret = () => {
            const last = content.lastElementChild;
            if (!last) {
                content.appendChild(caret);
                return;
            }
            if (last.tagName === 'TABLE') {
                content.appendChild(caret);
                return;
            }
            if (last.tagName === 'UL' || last.tagName === 'OL') {
                (last.lastElementChild || last).appendChild(caret);
            } else {
                last.appendChild(caret);
            }
        };

        const api = {
            root,
            content,
            caret,
            getText() { return text; },
            showThinking(label) {
                thinkingEl.hidden = false;
                thinkingText.textContent = label || '正在思考…';
            },
            hideThinking() {
                thinkingEl.hidden = true;
            },
            setMode(label) {
                modeEl.textContent = label || '';
            },
            paint(next) {
                text = next;
                if (frame) return;
                frame = requestAnimationFrame(() => {
                    frame = 0;
                    content.innerHTML = renderMarkdown(text);
                    placeCaret();
                    if (isPinned()) scrollToBottom(true);
                });
            },
            finish(options) {
                const opts = options || {};
                if (frame) {
                    cancelAnimationFrame(frame);
                    frame = 0;
                }
                api.hideThinking();
                caret.hidden = true;

                if (opts.error) {
                    content.innerHTML = `
                        <div class="msg-error">
                            <svg class="icon" aria-hidden="true"><use href="#i-alert"/></svg>
                            <span>${escapeHtml(opts.error)}</span>
                        </div>`;
                    api.setMode(opts.mode || '连接异常');
                    tools.hidden = true;
                    return;
                }

                if (opts.text != null) text = opts.text;
                content.innerHTML = renderMarkdown(text);
                api.setMode(opts.mode || '');
                tools.hidden = !text;

                if (typeof opts.time === 'number' && Number.isFinite(opts.time)) {
                    stamp.textContent = `用时 ${opts.time.toFixed(2)}s`;
                } else {
                    stamp.textContent = '';
                }

                if (!text && opts.stopped) {
                    content.innerHTML = '<p>已停止生成。</p>';
                    tools.hidden = true;
                }
            }
        };

        copyBtn.addEventListener('click', () => copyText(text, copyBtn));
        return api;
    }

    function buildBotMessage(item) {
        const bot = createBotMessage();
        bot.finish(item);
        return bot.root;
    }

    function paintThread() {
        els.thread.innerHTML = '';
        if (!state.thread.length) {
            showWelcome();
            return;
        }
        state.thread.forEach((item) => {
            if (item.role === 'user') {
                els.thread.appendChild(buildUserMessage(item.text));
            } else {
                els.thread.appendChild(buildBotMessage(item));
            }
        });
        els.thread.scrollTop = els.thread.scrollHeight;
        requestAnimationFrame(() => {
            els.thread.scrollTop = els.thread.scrollHeight;
        });
        updateHeader();
    }

    function autosize() {
        const textarea = els.input;
        textarea.style.height = 'auto';
        const next = Math.min(textarea.scrollHeight, 168);
        textarea.style.height = `${next}px`;
        textarea.style.overflowY = textarea.scrollHeight > 168 ? 'auto' : 'hidden';
    }

    function syncComposer() {
        const busy = state.busy;
        const label = $('.send-label', els.sendBtn);
        const iconUse = $('use', els.sendBtn);

        if (label) label.textContent = busy ? '停止' : '发送';
        if (iconUse) iconUse.setAttribute('href', busy ? '#i-stop' : '#i-arrow-up');

        els.sendBtn.classList.toggle('is-stop', busy);
        els.sendBtn.disabled = busy ? false : !els.input.value.trim();
        els.sendBtn.setAttribute('aria-label', busy ? '停止生成' : '发送');
        els.clearBtn.disabled = !state.sessionId || busy;
    }

    /* 4. 网络请求 */

    function fillSourceSelect(codes) {
        const current = els.sourceFilter.value;
        els.sourceFilter.innerHTML = '';

        const all = document.createElement('option');
        all.value = '';
        all.textContent = '全部资料';
        els.sourceFilter.appendChild(all);

        codes.forEach((code) => {
            const option = document.createElement('option');
            option.value = code;
            option.textContent = SOURCE_LABELS[code] || String(code).toUpperCase();
            els.sourceFilter.appendChild(option);
        });

        if (current && [...els.sourceFilter.options].some((option) => option.value === current)) {
            els.sourceFilter.value = current;
        }
    }

    async function loadSources() {
        let codes = Object.keys(SOURCE_LABELS);
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 4000);
        try {
            const response = await fetch(http('/api/sources'), {
                cache: 'no-store',
                signal: controller.signal
            });
            if (response.ok) {
                const data = await response.json();
                if (Array.isArray(data.sources) && data.sources.length) {
                    codes = data.sources.filter((code) => typeof code === 'string' && code);
                }
            }
        } catch (error) {
            /* 接口不可用时使用默认六类资料 */
        } finally {
            clearTimeout(timer);
        }
        fillSourceSelect(codes);
    }

    function setStatus(kind, text) {
        els.statusDot.className = `status-dot is-${kind}`;
        els.statusText.textContent = text;
    }

    async function checkHealth(quiet) {
        if (!quiet) setStatus('checking', '检测中');
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 4000);
        try {
            const response = await fetch(http('/health'), {
                cache: 'no-store',
                signal: controller.signal
            });
            clearTimeout(timer);
            if (!response.ok) throw new Error('offline');
            const data = await response.json();
            if (data.status !== 'healthy') throw new Error('unhealthy');
            setStatus('online', '服务正常');
        } catch (error) {
            clearTimeout(timer);
            setStatus('offline', '服务离线');
        }
    }

    async function newSessionId() {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 3000);
        try {
            const response = await fetch(http('/api/create_session'), {
                method: 'POST',
                signal: controller.signal
            });
            if (response.ok) {
                const data = await response.json();
                if (data && typeof data.session_id === 'string' && data.session_id) {
                    return data.session_id;
                }
            }
        } catch (error) {
            /* 接口不可用时本地生成，保证对话仍可继续 */
        } finally {
            clearTimeout(timer);
        }
        if (window.crypto && typeof window.crypto.randomUUID === 'function') {
            return window.crypto.randomUUID();
        }
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (char) => {
            const value = (Math.random() * 16) | 0;
            const next = char === 'x' ? value : (value & 0x3) | 0x8;
            return next.toString(16);
        });
    }

    async function ensureSession() {
        if (state.sessionId) {
            if (!state.sessions.some((item) => item.id === state.sessionId)) {
                state.sessions.unshift({
                    id: state.sessionId,
                    title: '新对话',
                    rounds: 0,
                    ts: Date.now()
                });
                state.sessions = state.sessions.slice(0, MAX_SESSIONS);
                persistSessions();
                renderSessions();
            }
            return state.sessionId;
        }

        const id = await newSessionId();
        state.sessionId = id;
        state.sessions = [{
            id,
            title: '新对话',
            rounds: 0,
            ts: Date.now()
        }, ...state.sessions.filter((item) => item.id !== id)].slice(0, MAX_SESSIONS);
        persistSessions();
        renderSessions();
        updateHeader();
        return id;
    }

    function startNewSession() {
        if (state.busy) stopGeneration();
        state.turn += 1;
        state.loadToken += 1;
        state.sessionId = null;
        state.thread = [];
        renderSessions();
        showWelcome();
        closeNav();
        els.input.focus();
    }

    async function openSession(id, options) {
        const opts = options || {};
        if (!id) return;

        closeNav();
        if (id === state.sessionId) return;
        if (state.busy) stopGeneration();

        state.turn += 1;
        const token = ++state.loadToken;
        state.sessionId = id;
        state.thread = [];

        const entry = state.sessions.find((item) => item.id === id);
        els.convTitle.textContent = entry ? entry.title : '新对话';
        renderSessions();
        els.thread.innerHTML = `
            <div class="thread-loading">
                <span class="dots"><i></i><i></i><i></i></span>
                <span>正在读取会话记录…</span>
            </div>`;

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 8000);
        try {
            const response = await fetch(http(`/api/history/${encodeURIComponent(id)}`), {
                cache: 'no-store',
                signal: controller.signal
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            if (token !== state.loadToken) return;

            const history = Array.isArray(data.history) ? data.history : [];
            history.forEach((pair) => {
                if (pair && pair.question) state.thread.push({ role: 'user', text: pair.question });
                if (pair && pair.answer) state.thread.push({ role: 'bot', text: pair.answer });
            });

            if (entry) {
                entry.rounds = history.length;
                if (entry.title === '新对话' && history[0] && history[0].question) {
                    entry.title = truncate(history[0].question, 18);
                }
                persistSessions();
            }

            paintThread();
            renderSessions();
        } catch (error) {
            if (token !== state.loadToken) return;
            state.thread = [];
            paintThread();
            if (!opts.silent) toast('读取会话记录失败，请稍后重试', 'error');
        } finally {
            clearTimeout(timer);
        }
    }

    async function clearCurrent() {
        const id = state.sessionId;
        if (!id) return;

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 8000);
        try {
            const response = await fetch(http(`/api/history/${encodeURIComponent(id)}`), {
                method: 'DELETE',
                signal: controller.signal
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
        } catch (error) {
            toast('清除失败，请稍后重试', 'error');
            return;
        } finally {
            clearTimeout(timer);
        }

        state.turn += 1;
        state.loadToken += 1;
        state.sessions = state.sessions.filter((item) => item.id !== id);
        state.sessionId = null;
        state.thread = [];
        persistSessions();
        renderSessions();
        showWelcome();
        toast('对话记录已清除');
    }

    /* 5. 流式回答 */

    function streamAnswer(bot, query, filter, sessionId) {
        return new Promise((resolve) => {
            let text = '';
            let settled = false;
            let firstToken = true;
            let opened = false;
            let idleTimer = null;

            let socket;
            try {
                socket = new WebSocket(wsUrl('/api/stream'));
            } catch (error) {
                const message = '无法建立实时连接，请确认服务是否在线。';
                bot.finish({ error: message });
                resolve({ text: '', error: message, mode: '连接异常' });
                return;
            }

            const settle = (outcome) => {
                if (settled) return;
                settled = true;
                clearTimeout(idleTimer);
                state.stop = null;

                try {
                    socket.onopen = null;
                    socket.onmessage = null;
                    socket.onerror = null;
                    socket.onclose = null;
                    if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
                        socket.close();
                    }
                } catch (error) { /* 忽略关闭异常 */ }

                if (outcome.error) {
                    bot.finish({ error: outcome.error });
                } else {
                    const finalText = outcome.text || text;
                    bot.finish({
                        text: finalText,
                        mode: outcome.mode || '知识库增强',
                        time: outcome.time,
                        stopped: outcome.stopped
                    });
                    outcome.text = finalText;
                }
                resolve(outcome);
            };

            const armIdle = () => {
                clearTimeout(idleTimer);
                idleTimer = setTimeout(() => {
                    settle({ text, error: '响应超时，请稍后重试。', mode: '连接异常' });
                }, IDLE_TIMEOUT);
            };

            state.stop = () => settle({ text, mode: '已停止', stopped: true });
            armIdle();

            socket.onopen = () => {
                opened = true;
                bot.showThinking('正在检索知识库…');
                armIdle();
                try {
                    socket.send(JSON.stringify({
                        query,
                        source_filter: filter,
                        session_id: sessionId
                    }));
                } catch (error) {
                    settle({ text, error: '实时连接已关闭，请重试。', mode: '连接异常' });
                }
            };

            socket.onmessage = (event) => {
                armIdle();
                let data;
                try {
                    data = JSON.parse(event.data);
                } catch (error) {
                    return;
                }

                if (data.type === 'start') {
                    bot.showThinking('正在组织回答…');
                } else if (data.type === 'token') {
                    if (firstToken) {
                        firstToken = false;
                        bot.hideThinking();
                        bot.caret.hidden = false;
                    }
                    text += data.token || '';
                    bot.paint(text);
                    if (isPinned()) scrollToBottom(true);
                } else if (data.type === 'end') {
                    settle({
                        text,
                        mode: '知识库增强',
                        time: typeof data.processing_time === 'number' ? data.processing_time : null
                    });
                } else if (data.type === 'error') {
                    settle({ text, error: data.error || '服务返回错误。', mode: '连接异常' });
                }
            };

            socket.onerror = () => {
                settle({ text, error: '实时连接中断，请确认服务是否在线。', mode: '连接异常' });
            };

            socket.onclose = () => {
                if (settled) return;
                if (opened) {
                    settle({ text, mode: '知识库增强' });
                } else {
                    settle({ text, error: '无法建立实时连接，请确认服务是否在线。', mode: '连接异常' });
                }
            };
        });
    }

    /* 6. 消息发送 */

    async function sendMessage() {
        if (state.busy) return;

        const query = els.input.value.trim();
        if (!query) return;

        const filter = els.sourceFilter.value || null;
        const turn = ++state.turn;
        state.loadToken += 1;

        state.busy = true;
        els.input.value = '';
        autosize();
        syncComposer();
        els.srStatus.textContent = '正在生成回答';

        if (state.thread.length === 0) els.thread.innerHTML = '';
        els.thread.appendChild(buildUserMessage(query));
        state.thread.push({ role: 'user', text: query });
        scrollToBottom(true);

        const bot = createBotMessage();
        els.thread.appendChild(bot.root);
        scrollToBottom(true);

        const controller = new AbortController();
        state.stop = () => controller.abort();
        let timedOut = false;
        const timeout = setTimeout(() => {
            timedOut = true;
            controller.abort();
        }, 20000);

        let sessionId = null;

        try {
            sessionId = await ensureSession();
            if (turn !== state.turn) return;

            touchTitle(query);

            const response = await fetch(http('/api/query'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    source_filter: filter,
                    session_id: sessionId
                }),
                signal: controller.signal
            });
            clearTimeout(timeout);

            if (!response.ok) throw new Error(`服务返回 ${response.status}`);

            const data = await response.json().catch(() => ({}));
            let outcome;

            if (data && data.is_streaming) {
                bot.showThinking('正在检索知识库…');
                outcome = await streamAnswer(bot, query, filter, sessionId);
            } else {
                outcome = {
                    text: (data && typeof data.answer === 'string' && data.answer)
                        ? data.answer
                        : '（服务未返回内容）',
                    mode: '快速应答',
                    time: typeof data.processing_time === 'number' ? data.processing_time : null
                };
                bot.finish(outcome);
            }

            if (turn !== state.turn || state.sessionId !== sessionId) return;

            if (outcome.error) {
                state.thread.push({ role: 'bot', text: '', error: outcome.error, mode: '连接异常' });
                toast(outcome.error, 'error');
                els.srStatus.textContent = '回答失败';
            } else {
                state.thread.push({
                    role: 'bot',
                    text: outcome.text || (outcome.stopped ? '（已停止生成）' : '（无内容）'),
                    mode: outcome.mode || '知识库增强',
                    time: outcome.time
                });
                completeTurn(query);
                els.srStatus.textContent = '回答已完成';
            }
        } catch (error) {
            if (turn !== state.turn) return;

            if (error && error.name === 'AbortError') {
                if (timedOut) {
                    const message = '请求超时，请稍后重试。';
                    bot.finish({ error: message });
                    state.thread.push({ role: 'bot', text: '', error: message, mode: '连接异常' });
                    toast(message, 'error');
                } else {
                    bot.finish({ text: '', mode: '已停止', stopped: true });
                    state.thread.push({ role: 'bot', text: '（已停止生成）', mode: '已停止' });
                }
            } else {
                const message = `请求失败：${(error && error.message) || '网络异常'}`;
                bot.finish({ error: message });
                state.thread.push({ role: 'bot', text: '', error: message, mode: '连接异常' });
                toast(message, 'error');
            }
        } finally {
            clearTimeout(timeout);
            state.busy = false;
            state.stop = null;
            syncComposer();
            updateHeader();
            scrollToBottom();
        }
    }

    function stopGeneration() {
        if (typeof state.stop === 'function') {
            state.stop();
        }
        state.stop = null;
    }

    /* 7. 主题与导航 */

    function applyTheme(theme) {
        state.theme = theme === 'dark' ? 'dark' : 'light';
        document.documentElement.dataset.theme = state.theme;

        els.themeBtn.innerHTML = `<svg class="icon" aria-hidden="true"><use href="#${state.theme === 'dark' ? 'i-sun' : 'i-moon'}"/></svg>`;
        els.themeBtn.setAttribute('aria-label', state.theme === 'dark' ? '切换为浅色主题' : '切换为深色主题');

        const meta = document.getElementById('metaTheme');
        if (meta) meta.setAttribute('content', state.theme === 'dark' ? '#0a0c10' : '#f2ede3');

        try {
            localStorage.setItem(STORAGE.theme, state.theme);
        } catch (error) {
            /* 忽略存储失败 */
        }
    }

    function openNav() {
        document.body.classList.add('nav-open');
    }

    function closeNav() {
        document.body.classList.remove('nav-open');
    }

    /* 8. 事件与初始化 */

    function onInputKeydown(event) {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
            event.preventDefault();
            if (!state.busy) sendMessage();
        }
    }

    function onGlobalKeydown(event) {
        if (els.clearDialog && els.clearDialog.open) return;

        if (event.key === 'Escape') {
            if (document.body.classList.contains('nav-open')) {
                closeNav();
                return;
            }
            if (state.busy) stopGeneration();
        }
        if ((event.ctrlKey || event.metaKey) && (event.key === 'k' || event.key === 'K')) {
            event.preventDefault();
            els.input.focus();
        }
    }

    function copyTranscript() {
        if (!state.thread.length) {
            toast('当前对话还没有内容');
            return;
        }
        const text = state.thread.map((item) => {
            if (item.role === 'user') return `问：${item.text}`;
            return `答：${item.text || item.error || ''}`;
        }).join('\n\n');
        copyText(text, els.copyAllBtn);
    }

    function askClear() {
        if (!state.sessionId) return;
        if (typeof els.clearDialog.showModal === 'function') {
            if (!els.clearDialog.open) els.clearDialog.showModal();
        } else if (window.confirm('清除当前对话？此操作不可恢复。')) {
            clearCurrent();
        }
    }

    function bindEvents() {
        els.sendBtn.addEventListener('click', () => {
            if (state.busy) stopGeneration();
            else sendMessage();
        });

        els.input.addEventListener('input', () => {
            autosize();
            syncComposer();
        });
        els.input.addEventListener('keydown', onInputKeydown);

        els.newChatBtn.addEventListener('click', startNewSession);
        els.copyAllBtn.addEventListener('click', copyTranscript);
        els.clearBtn.addEventListener('click', askClear);
        els.menuBtn.addEventListener('click', openNav);
        els.scrim.addEventListener('click', closeNav);
        els.themeBtn.addEventListener('click', () => {
            applyTheme(state.theme === 'dark' ? 'light' : 'dark');
        });

        els.cancelClear.addEventListener('click', () => els.clearDialog.close());
        els.confirmClear.addEventListener('click', () => {
            els.clearDialog.close();
            clearCurrent();
        });
        els.clearDialog.addEventListener('click', (event) => {
            if (event.target === els.clearDialog) els.clearDialog.close();
        });

        document.addEventListener('keydown', onGlobalKeydown);
        window.addEventListener('resize', () => {
            if (window.innerWidth > 960) closeNav();
        });
    }

    function cacheElements() {
        const ids = [
            'sidebar', 'scrim', 'menuBtn', 'sessionList', 'sessionCount', 'newChatBtn',
            'statusDot', 'statusText', 'convTitle', 'convMeta', 'copyAllBtn', 'clearBtn',
            'themeBtn', 'thread', 'sourceFilter', 'input', 'sendBtn', 'toastStack',
            'clearDialog', 'cancelClear', 'confirmClear', 'srStatus'
        ];
        ids.forEach((id) => {
            els[id] = document.getElementById(id);
        });
    }

    async function init() {
        cacheElements();
        applyTheme(document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light');

        state.sessions = loadSessions();
        bindEvents();
        fillSourceSelect(Object.keys(SOURCE_LABELS));
        renderSessions();
        syncComposer();

        const restore = state.sessions.length
            ? openSession(state.sessions[0].id, { silent: true })
            : Promise.resolve(showWelcome());

        await Promise.allSettled([loadSources(), restore, checkHealth()]);

        setInterval(() => checkHealth(true), 30000);
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) checkHealth(true);
        });

        updateHeader();

        if (window.matchMedia('(pointer: fine)').matches) {
            els.input.focus();
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
