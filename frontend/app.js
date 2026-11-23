/* Tohfa – frontend logic. Plain JS, no build step.
   Talks to the FastAPI backend:  POST /api/recommend   GET /api/status   GET /api/occasions */
(() => {
  "use strict";

  // Same-origin by default (FastAPI serves this folder). Open the file directly or host the
  // frontend elsewhere?  Use  ?api=http://localhost:8000  in the URL.
  const params = new URLSearchParams(location.search);
  const API_BASE = (params.get("api") || (location.protocol === "file:" ? "http://localhost:8000" : "")).replace(/\/$/, "");

  const FALLBACK_OCCASIONS = ["Birthday", "Wedding", "Diwali", "Valentine's Day", "Office Party",
    "Anniversary", "Baby Shower", "Housewarming", "Holi", "Christmas"];
  const TOP_K = 12;

  const $ = (id) => document.getElementById(id);
  const els = {
    form: $("search-form"), input: $("occasion"), btn: $("search-btn"), chips: $("chips"),
    grid: $("grid"), head: $("results-head"), title: $("results-title"), meta: $("results-meta"),
    notices: $("notices"), state: $("state"), live: $("live"), results: $("results"),
    engine: $("engine-status"), engineText: $("engine-text"),
  };
  const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

  let occasions = FALLBACK_OCCASIONS;
  let inflight = null; // AbortController of the running request

  // ---------------------------------------------------------------- tiny DOM helper
  // Everything from the API goes through textContent, never innerHTML, so catalog text can't inject markup.
  function h(tag, props = {}, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }
  const clear = (el) => el.replaceChildren();
  const pct = (x) => Math.round(Math.max(0, Math.min(1, x)) * 100);
  const selectedMode = () => document.querySelector('input[name="mode"]:checked').value;

  // ---------------------------------------------------------------- chips
  function chipButton(label) {
    return h("button", { class: "chip", type: "button", "aria-pressed": "false", onclick: () => runSearch(label) }, label);
  }
  function renderChips(container, list) {
    clear(container);
    list.forEach((o) => container.append(h("li", {}, chipButton(o))));
  }
  function markActiveChip(text) {
    els.chips.querySelectorAll(".chip").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.textContent.toLowerCase() === text.toLowerCase())));
  }

  // ---------------------------------------------------------------- cards
  function imageBlock(product) {
    const fallback = () => h("div", { class: "card__fallback", "aria-hidden": "true" }, (product.title[0] || "?").toUpperCase());
    if (!product.image) return fallback();
    const img = h("img", { class: "card__img", src: product.image, alt: product.title, loading: "lazy", decoding: "async" });
    img.addEventListener("error", () => img.replaceWith(fallback()), { once: true });
    return img;
  }

  function priceBlock(p) {
    if (p.price == null) return null;
    const discounted = p.discount_percentage > 0 && p.price_after_discount != null;
    return h("div", { class: "card__price" },
      h("span", { class: "price" }, inr.format(discounted ? p.price_after_discount : p.price)),
      discounted && h("span", { class: "price price--old" }, inr.format(p.price)),
      discounted && h("span", { class: "price--off" }, `${Math.round(p.discount_percentage)}% off`));
  }

  function matchBlock(p) {
    const s = p.score;
    const label = `${pct(s.final)}% match`;
    const split = [];
    if (s.keyword != null) split.push(`Keywords ${pct(s.keyword)}`);
    if (s.semantic != null) split.push(`Meaning ${pct(s.semantic)}`);
    return h("div", { class: "match" },
      h("div", { class: "match__row" }, h("span", {}, label)),
      h("div", { class: "match__bar", role: "img", "aria-label": label },
        h("span", { class: "match__fill", style: `width:${pct(s.final)}%` })),
      split.length === 2 && h("p", { class: "match__split" }, split.join("  ·  ")),
      p.matched_terms.length > 0 && h("ul", { class: "tags", "aria-label": "Matched on" },
        p.matched_terms.map((t) => h("li", { class: "tag" }, t))));
  }

  function productCard(p) {
    return h("article", { class: "card" },
      h("div", { class: "card__media" }, imageBlock(p)),
      h("div", { class: "card__body" },
        h("p", { class: "card__cat" }, p.vendor_category_desc || "Uncategorised"),
        h("h3", { class: "card__title" }, p.title),
        h("p", { class: "card__desc" }, p.snippet),
        priceBlock(p),
        matchBlock(p)));
  }

  // ---------------------------------------------------------------- states
  function showSkeletons() {
    clear(els.grid);
    for (let i = 0; i < 6; i++) els.grid.append(h("div", { class: "card card--skeleton", "aria-hidden": "true" }));
  }

  function giftIcon() {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 64 64");
    svg.setAttribute("class", "state__icon");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2.5");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.innerHTML = '<rect x="8" y="26" width="48" height="30" rx="3"/><rect x="5" y="18" width="54" height="9" rx="3"/>' +
      '<path d="M32 18v38M32 18c-3-9-14-11-14-5 0 4 8 5 14 5zM32 18c3-9 14-11 14-5 0 4-8 5-14 5z"/>';
    return svg;
  }

  function showEmpty(query) {
    const list = h("ul", { class: "chips", "aria-label": "Try one of these" });
    renderChips(list, occasions.slice(0, 8));
    els.state.replaceChildren(
      giftIcon(),
      h("h2", { class: "state__title" }, `No gifts match “${query}” yet`),
      h("p", { class: "state__text" }, "Try a broader occasion such as a festival, a celebration or a relationship, or pick one below."),
      list);
    els.state.hidden = false;
  }

  function showError(err) {
    const offline = err instanceof TypeError; // fetch() network failure
    els.state.replaceChildren(
      giftIcon(),
      h("h2", { class: "state__title" }, offline ? "Can’t reach the recommendation server" : "Something went wrong"),
      h("p", { class: "state__text" }, offline
        ? ["Start the backend with ", h("code", {}, "uvicorn backend.app.main:app --port 8000"), " and try again."]
        : err.message));
    els.state.hidden = false;
  }

  // ---------------------------------------------------------------- search
  async function runSearch(rawText) {
    const occasion = (rawText ?? els.input.value).trim();
    if (!occasion) { els.input.focus(); return; }
    els.input.value = occasion;
    markActiveChip(occasion);

    inflight?.abort();
    const controller = (inflight = new AbortController());
    els.btn.disabled = true;
    els.grid.setAttribute("aria-busy", "true");
    els.head.hidden = true;
    els.state.hidden = true;
    clear(els.notices);
    showSkeletons();

    try {
      const res = await fetch(`${API_BASE}/api/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ occasion, top_k: TOP_K, mode: selectedMode(), use_llm: true }),
        signal: controller.signal,
      });
      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try { const j = await res.json(); if (typeof j.detail === "string") detail = j.detail; } catch (_) { /* keep default */ }
        throw new Error(detail);
      }
      render(await res.json());
      const url = new URL(location.href);
      url.searchParams.set("q", occasion);
      history.replaceState(null, "", url);
    } catch (err) {
      if (err.name === "AbortError") return; // superseded by a newer search
      clear(els.grid);
      showError(err);
    } finally {
      if (inflight === controller) {
        els.btn.disabled = false;
        els.grid.setAttribute("aria-busy", "false");
        inflight = null;
      }
    }
  }

  const MODE_LABEL = { hybrid: "hybrid search", keyword: "keyword matching", semantic: "semantic search" };

  function render(data) {
    clear(els.grid);
    clear(els.notices);
    data.warnings.forEach((w) => els.notices.append(h("p", { class: "notice", role: "alert" }, w)));

    if (data.results.length === 0) {
      els.head.hidden = true;
      showEmpty(data.occasion);
      els.live.textContent = `No gifts found for ${data.occasion}.`;
      return;
    }

    const n = data.results.length;
    els.title.textContent = `${n} gift${n === 1 ? "" : "s"} for “${data.occasion}”`;
    const bits = [`Ranked with ${MODE_LABEL[data.mode_used]} in ${Math.round(data.elapsed_ms)} ms.`];
    if (data.expanded_terms.length) bits.push(`Also looked for: ${data.expanded_terms.slice(0, 8).join(", ")}.`);
    els.meta.textContent = bits.join(" ");
    els.head.hidden = false;
    els.state.hidden = true;
    data.results.forEach((p) => els.grid.append(productCard(p)));
    els.live.textContent = `${n} gifts found for ${data.occasion}.`;
    els.results.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
  }

  // ---------------------------------------------------------------- engine status pill
  async function loadStatus() {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      const s = await res.json();
      const o = s.ollama;
      if (!o.reachable) {
        setEngine("down", "Ollama is offline – using keyword matching only");
      } else if (!o.embed_model_available || !s.semantic_ready) {
        setEngine("degraded", o.embed_model_available
          ? "Preparing the embedding index…"
          : `Run “ollama pull ${o.embed_model}” to enable semantic search`);
      } else {
        setEngine("ok", `Ollama connected · ${o.embed_model}${o.llm_model_available ? " + " + o.llm_model : ""}`);
      }
    } catch (_) {
      setEngine("down", "Backend not reachable");
    }
  }
  function setEngine(state, text) {
    els.engine.dataset.state = state;
    els.engineText.textContent = text;
  }

  async function loadOccasions() {
    try {
      const res = await fetch(`${API_BASE}/api/occasions`);
      const data = await res.json();
      if (Array.isArray(data.occasions) && data.occasions.length) occasions = data.occasions;
    } catch (_) { /* keep fallback */ }
    renderChips(els.chips, occasions);
  }

  // ---------------------------------------------------------------- boot
  els.form.addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
  document.querySelectorAll('input[name="mode"]').forEach((r) =>
    r.addEventListener("change", () => { if (els.input.value.trim()) runSearch(); }));

  renderChips(els.chips, occasions);
  loadOccasions();
  loadStatus();
  setInterval(loadStatus, 15000);

  const initial = params.get("q");
  if (initial) runSearch(initial);
})();
