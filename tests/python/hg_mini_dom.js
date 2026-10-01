function makeWorld() {
  const byId = new Map();
  let now = 0, seq = 0;
  const timers = new Map();
  const VOID = new Set(["input", "img", "br", "hr", "meta", "link", "use", "circle", "path", "rect", "line"]);
  class ClassList {
    constructor(el) { this.el = el; }
    get set() { return new Set(String(this.el.attrs.class || "").split(/\s+/).filter(Boolean)); }
    _w(set) { this.el.attrs.class = [...set].join(" "); }
    add(...c) { const s = this.set; c.forEach((x) => s.add(x)); this._w(s); }
    remove(...c) { const s = this.set; c.forEach((x) => s.delete(x)); this._w(s); }
    toggle(c, on) { const s = this.set; const want = on === undefined ? !s.has(c) : !!on; if (want) s.add(c); else s.delete(c); this._w(s); return want; }
    contains(c) { return this.set.has(c); }
  }
  const ENT = { "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'", "&nbsp;": " " };
  const unent = (t) => String(t).replace(/&(amp|lt|gt|quot|#39|nbsp);/g, (m) => ENT[m]);
  const dataKey = (k) => "data-" + String(k).replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
  class El {
    constructor(tag) {
      this.tagName = String(tag).toUpperCase(); this.childNodes = []; this.parentNode = null; this.attrs = {};
      this.style = {
        setProperty(k, v) { this[k] = v; }, cssText: "",
        removeProperty(k) { const v = this[k]; delete this[k]; return v == null ? "" : String(v); },
        getPropertyValue(k) { return this[k] == null ? "" : String(this[k]); },
      };
      this.classList = new ClassList(this);
      this.listeners = {}; this._text = ""; this.value = ""; this.checked = false;
    }
    get nodeType() { return this.tagName === "#TEXT" ? 3 : 1; }
    get children() { return this.childNodes.filter((c) => c.nodeType === 1); }
    get className() { return this.attrs.class || ""; }
    set className(v) { this.attrs.class = String(v); }
    get id() { return this.attrs.id || ""; }
    set id(v) { this.setAttribute("id", v); }
    get type() { return (this.attrs.type || "").toLowerCase(); }
    set type(v) { this.attrs.type = String(v); }
    get hidden() { return "hidden" in this.attrs; }
    set hidden(v) { if (v) this.attrs.hidden = ""; else delete this.attrs.hidden; }
    get disabled() { return "disabled" in this.attrs; }
    set disabled(v) { if (v) this.attrs.disabled = ""; else delete this.attrs.disabled; }
    get title() { return this.attrs.title || ""; }
    set title(v) { this.attrs.title = String(v); }
    get dataset() {
      const el = this;
      return new Proxy({}, {
        get: (_, k) => el.attrs[dataKey(k)],
        set: (_, k, v) => { el.attrs[dataKey(k)] = String(v); return true; },
        has: (_, k) => dataKey(k) in el.attrs,
        deleteProperty: (_, k) => { delete el.attrs[dataKey(k)]; return true; },
      });
    }
    setAttribute(k, v) { this.attrs[k] = String(v); if (k === "id") byId.set(String(v), this); }
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
    hasAttribute(k) { return k in this.attrs; }
    removeAttribute(k) { delete this.attrs[k]; }
    _register() { if (this.attrs.id) byId.set(this.attrs.id, this); this.childNodes.forEach((c) => c._register()); }
    appendChild(c) { if (c.parentNode) c.remove(); c.parentNode = this; this.childNodes.push(c); c._register(); return c; }
    insertBefore(c, ref) {
      if (c.parentNode) c.remove();
      c.parentNode = this;
      const i = ref ? this.childNodes.indexOf(ref) : -1;
      if (i < 0) this.childNodes.push(c); else this.childNodes.splice(i, 0, c);
      c._register();
      return c;
    }
    remove() { const p = this.parentNode; if (p) { p.childNodes = p.childNodes.filter((x) => x !== this); this.parentNode = null; } }
    get firstChild() { return this.childNodes[0] || null; }
    get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
    get firstElementChild() { return this.children[0] || null; }
    get lastElementChild() { const c = this.children; return c[c.length - 1] || null; }
    get childElementCount() { return this.children.length; }
    get isConnected() { let e = this; while (e.parentNode) e = e.parentNode; return e === doc.documentElement; }
    get textContent() { return this.tagName === "#TEXT" ? this._text : this.childNodes.map((c) => c.textContent).join(""); }
    set textContent(v) { this.childNodes = []; if (String(v)) { const t = new El("#text"); t._text = String(v); this.appendChild(t); } }
    get innerText() { return this.textContent; }
    get innerHTML() { return this.childNodes.map((c) => c.outerHTML).join(""); }
    set innerHTML(v) { this.childNodes = []; parseInto(this, String(v)); }
    insertAdjacentHTML(where, html) { const tmp = new El("div"); parseInto(tmp, String(html)); const nodes = tmp.childNodes.slice(); if (where === "beforeend") nodes.forEach((x) => this.appendChild(x)); else if (where === "afterbegin") nodes.reverse().forEach((x) => this.insertBefore(x, this.firstChild)); }
    get outerHTML() {
      if (this.tagName === "#TEXT") return this._text;
      const t = this.tagName.toLowerCase(), a = Object.entries(this.attrs).map(([k, v]) => ` ${k}="${v}"`).join("");
      return VOID.has(t) ? `<${t}${a}/>` : `<${t}${a}>${this.innerHTML}</${t}>`;
    }
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
    removeEventListener(t, f) { this.listeners[t] = (this.listeners[t] || []).filter((x) => x !== f); }
    dispatch(type, init) {
      const ev = Object.assign({ type, target: this, defaultPrevented: false, _stop: false }, init || {});
      ev.preventDefault = () => { ev.defaultPrevented = true; };
      ev.stopPropagation = () => { ev._stop = true; };
      for (let n = this; n && !ev._stop; n = n.parentNode) { ev.currentTarget = n; (n.listeners[type] || []).slice().forEach((f) => f(ev)); }
      if (!ev._stop) { ev.currentTarget = doc; (doc.listeners[type] || []).slice().forEach((f) => f(ev)); }
      return ev;
    }
    dispatchEvent(e) { return this.dispatch(e.type, e); }
    click() {
      if (this.disabled) return;
      const box = this.tagName === "INPUT" ? this : this.tagName === "LABEL" ? this.querySelector("input") : null;
      if (box && box !== this) { box.click(); return; }
      const before = box ? box.checked : null;
      if (box && box.type === "checkbox") box.checked = !box.checked;
      if (box && box.type === "radio") box.checked = true;
      this.dispatch("click");
      // (like a browser: change only when the click changed the box)
      if (box && (box.type === "checkbox" || box.type === "radio") && box.checked !== before) this.dispatch("change");
    }
    querySelectorAll(sel) { const out = []; walk(this, (n) => { if (matches(n, sel, this)) out.push(n); }); return out; }
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
    matches(sel) { return matches(this, sel, null); }
    closest(sel) { for (let n = this; n && n.nodeType === 1; n = n.parentNode) if (matches(n, sel, null)) return n; return null; }
    focus() { doc.activeElement = this; }
    blur() { if (doc.activeElement === this) doc.activeElement = null; }
    contains(x) { for (let n = x; n; n = n.parentNode) if (n === this) return true; return false; }
    getBoundingClientRect() { return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }; }
    get offsetWidth() { return 0; }
    get offsetHeight() { return 0; }
    scrollIntoView() {}
    // (the felt's FLIP slides: recorded, never run)
    animate(frames, opts) { (this._anims = this._anims || []).push({ frames, opts }); return { finished: Promise.resolve(), cancel() {}, finish() {} }; }
    getAnimations() { return []; }
  }
  function walk(n, f) { n.childNodes.forEach((c) => { if (c.nodeType === 1) { f(c); walk(c, f); } }); }
  function parseInto(parent, html) {
    const re = /<!--[\s\S]*?-->|<\/([a-zA-Z][\w-]*)\s*>|<([a-zA-Z][\w-]*)((?:\s+[^\s=>/]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?)*)\s*(\/?)>|([^<]+)/g;
    const stack = [parent];
    let m;
    while ((m = re.exec(html))) {
      const top = stack[stack.length - 1];
      if (m[0].startsWith("<!--")) continue;
      if (m[1]) {
        const t = m[1].toUpperCase();
        for (let i = stack.length - 1; i > 0; i--) if (stack[i].tagName === t) { stack.length = i; break; }
      } else if (m[2]) {
        const e = new El(m[2]);
        const ar = /([^\s=>/]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g;
        let a;
        while ((a = ar.exec(m[3] || ""))) e.attrs[a[1]] = unent(a[2] != null ? a[2] : a[3] != null ? a[3] : a[4] != null ? a[4] : "");
        if (e.tagName === "INPUT") { e.value = e.attrs.value || ""; e.checked = "checked" in e.attrs; }
        if (e.tagName === "TEXTAREA") e.value = "";
        top.appendChild(e);
        if (!m[4] && !VOID.has(m[2].toLowerCase())) stack.push(e);
      } else if (m[5] != null) {
        const t = new El("#text");
        t._text = unent(m[5]);
        top.appendChild(t);
        if (top.tagName === "TEXTAREA") top.value = t._text;
      }
    }
  }
  function compound(n, c) {
    // :not(simple) — negated, then left out of the rest
    const nots = [];
    c = c.replace(/:not\(([^()]*)\)/g, (_, x) => { nots.push(x); return ""; });
    if (nots.some((x) => compound(n, x))) return false;
    if (!c) return nots.length > 0;
    const re = /([a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:([~^$*]?=)["']?([^"'\]]*)["']?)?\]|\*/g;
    let m, any = false;
    while ((m = re.exec(c))) {
      any = true;
      if (m[1] && n.tagName !== m[1].toUpperCase()) return false;
      if (m[2] && n.attrs.id !== m[2]) return false;
      if (m[3] && !n.classList.contains(m[3])) return false;
      if (m[4]) {
        const have = n.attrs[m[4]];
        if (have == null) return false;
        if (m[5] === "=" && have !== m[6]) return false;
        if (m[5] === "^=" && !have.startsWith(m[6])) return false;
        if (m[5] === "*=" && !have.includes(m[6])) return false;
      }
    }
    return any;
  }
  function matches(n, sel, scope) {
    return String(sel).split(",").some((one) => {
      const parts = one.trim().replace(/\s*>\s*/g, " > ").split(/\s+/);
      const rec = (i, node) => {
        if (!node || node.nodeType !== 1 || !compound(node, parts[i])) return false;
        if (i === 0) return true;
        if (parts[i - 1] === ">") return rec(i - 2, node.parentNode);
        for (let p = node.parentNode; p && p !== scope; p = p.parentNode) if (rec(i - 1, p)) return true;
        return false;
      };
      return rec(parts.length - 1, n);
    });
  }
  const doc = {
    body: null, documentElement: new El("html"), activeElement: null, hidden: false, listeners: {},
    createElement: (t) => new El(t), createElementNS: (_, t) => new El(t),
    getElementById: (id) => { const e = byId.get(id); return e && e.isConnected ? e : null; },
    querySelector: (s) => doc.documentElement.querySelector(s),
    querySelectorAll: (s) => doc.documentElement.querySelectorAll(s),
    addEventListener(t, f) { (doc.listeners[t] = doc.listeners[t] || []).push(f); },
    removeEventListener(t, f) { doc.listeners[t] = (doc.listeners[t] || []).filter((x) => x !== f); },
  };
  doc.body = doc.documentElement.appendChild(new El("body"));
  const el = (tag, id, html) => { const e = new El(tag); if (id) e.setAttribute("id", id); if (html) e.innerHTML = html; doc.body.appendChild(e); return e; };
  const setTimeout = (fn, ms) => { const id = ++seq; timers.set(id, { at: now + (ms || 0), fn }); return id; };
  const clearTimeout = (id) => { timers.delete(id); };
  function advance(ms) {
    const end = now + ms;
    for (;;) {
      let next = null;
      for (const [id, t] of timers) if (t.at <= end && (!next || t.at < next[1].at)) next = [id, t];
      if (!next) break;
      timers.delete(next[0]);
      now = next[1].at;
      next[1].fn();
    }
    now = end;
  }
  return { doc, El, el, setTimeout, clearTimeout, advance, now: () => now };
}
