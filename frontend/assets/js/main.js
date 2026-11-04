import { capturing, reduced, cursor, initReveals, magnetic, marquee, setRangeFill, splitWords, whenVisible, countTo } from "./fx.js";
import { render, CONFIG } from "./charts.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const params = new URLSearchParams(location.search);
const CAP = capturing ? params.get("capture") : null;
if (capturing) document.documentElement.classList.add("capture");
if (CAP) document.getElementById(CAP)?.classList.add("cap-target");

const COL = ["#3987e5", "#199e70", "#e0621f", "#9085e9", "#c98500"];
const INK = "#f2efe9", INK2 = "#cbc6bb", MUTED = "#8f897d", GRID = "#25221d", AXIS = "#3d3930", REF = "#b0aa9d", SURF = "#141210";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const sign = (v, d = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;
const ax = (title = "", extra = {}) => ({ title: { text: title, font: { color: MUTED, size: 11 } }, gridcolor: GRID, linecolor: AXIS, zerolinecolor: AXIS, tickcolor: AXIS, tickfont: { size: 10.5 }, automargin: true, ...extra });
const lay = (x, y, extra = {}) => ({ paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: "Inter, system-ui, sans-serif", color: INK2, size: 11 }, margin: { l: 38, r: 8, t: 8, b: 30 },
  hoverlabel: { bgcolor: "#221f1a", bordercolor: "#4d483e", font: { color: INK, size: 12 } }, showlegend: false, xaxis: ax(x), yaxis: ax(y), ...extra });
const line = (x, y, col, extra = {}) => ({ type: "scatter", mode: "lines", x, y, line: { color: col, width: 2 }, ...extra });
const vline = (x) => ({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color: REF, width: 1.5, dash: "dash" } });
const plotG = async (id, el, h, extra = {}) => { const g = await fetch(`/api/graphs/${id}`).then((r) => r.json()); await render(el, { data: g.figure.data, layout: { ...g.figure.layout, autosize: true, height: h, ...extra } }); };
const lazy = (el, fn) => whenVisible(el, fn, 0.1);

async function boot() {
  const ov = await fetch("/api/overview").then((r) => r.json());
  const F = ov.factors, L = F.map((f) => f.letter);
  const nameIdx = Object.fromEntries(F.map((f, i) => [f.name, i]));
  const ctx = { ov, F, L, traffic: L[nameIdx["Traffic and combustion"]], dust: L[nameIdx["Dust and smoke"]], coal: L[nameIdx["Coal and heavy industry"]], ozone: L[nameIdx["Sunlit ozone"]], solvent: L[nameIdx["Solvents and petrol vapour"]] };
  buildHero(ctx);
  buildFactors(ctx);
  buildLockdown(ctx);
  buildDiwali(ctx);
  await buildCity(ctx);
  buildTree(ctx);
  buildDays(ctx);
  buildLimits(ctx);
  buildInside(ctx);
  $("#nGraphs").textContent = ov.n_graphs; $("#nGraphs2").textContent = ov.n_graphs;
  splitWords($("#heroTitle"), capturing ? 0 : 1250);
  cursor();
  initReveals();
  $$("[data-magnetic]").forEach((b) => magnetic(b));
  initNav();
  finishLoader();
  $("#heroTitle").classList.add("in");
}

function finishLoader() {
  const l = $("#loader");
  if (capturing) { l.remove(); document.body.classList.remove("is-loading"); return; }
  setTimeout(() => { l.classList.add("done"); document.body.classList.remove("is-loading"); setTimeout(() => l.remove(), 1100); }, reduced ? 0 : 1350);
}

function buildHero({ ov, F, L }) {
  const D = ov.data;
  $("#heroStats").innerHTML = `
    <div class="stat"><div class="stat__v">${ov.pollutants.length}</div><div class="stat__l">pollutants measured</div></div>
    <div class="stat"><div class="stat__v">${F.length}</div><div class="stat__l">hidden factors it found</div></div>
    <div class="stat"><div class="stat__v">${(ov.usable_rows / 1000).toFixed(1)}<small>k</small></div><div class="stat__l">city-days across ${D.cities} cities</div></div>
    <div class="stat"><div class="stat__v">0</div><div class="stat__l">source labels used in training</div></div>`;
  const words = ["No labels", "No map", "Nameless factors", "Rush hour", "Sunday dip", "Diwali", "Lockdown", "Coal towns"];
  const track = $("#marquee");
  track.innerHTML = [...words, ...words].map((w) => `<span>${w}</span>`).join("");
  if (!reduced) marquee(track);
  startFlow(ov, F);
}

/** Pollutants on the left, hidden factors on the right. Each line is a real loading. */
function startFlow(ov, F) {
  const c = $("#flowCanvas"), g = c.getContext("2d"), P = ov.pollutants, W_ = ov.fa.loadings;
  let W, H, visible = true;
  const size = () => { const dpr = Math.min(devicePixelRatio || 1, 2); W = c.clientWidth; H = c.clientHeight; c.width = W * dpr; c.height = H * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0); draw(reduced ? 1500 : performance.now()); };
  const links = [];
  P.forEach((_, i) => F.forEach((_, j) => { if (W_[i][j] >= 0.3) links.push({ i, j, w: W_[i][j], phase: (i * 0.37 + j * 0.61) % 1 }); }));
  function draw(t) {
    if (!W) return;
    g.clearRect(0, 0, W, H);
    const xl = 66, xr = W - 96, yP = (i) => 16 + (i + 0.5) * (H - 32) / P.length, yF = (j) => 34 + (j + 0.5) * (H - 68) / F.length;
    links.forEach((k) => {
      const y1 = yP(k.i), y2 = yF(k.j), mx = (xl + xr) / 2;
      g.beginPath(); g.moveTo(xl, y1); g.bezierCurveTo(mx, y1, mx, y2, xr, y2);
      g.strokeStyle = COL[k.j]; g.globalAlpha = 0.22 + k.w * 0.6; g.lineWidth = 0.6 + k.w * k.w * 6; g.stroke();
      const u = (t / 2600 + k.phase) % 1, v = 1 - u, bx = v * v * v * xl + 3 * v * v * u * mx + 3 * v * u * u * mx + u * u * u * xr, by = v * v * v * y1 + 3 * v * v * u * y1 + 3 * v * u * u * y2 + u * u * u * y2;
      g.globalAlpha = 0.95; g.fillStyle = COL[k.j]; g.beginPath(); g.arc(bx, by, 1.6 + k.w * 2.2, 0, 7); g.fill();
    });
    g.globalAlpha = 1; g.textBaseline = "middle"; g.textAlign = "right"; g.font = "600 12.5px 'IBM Plex Mono', monospace"; g.fillStyle = INK2;
    P.forEach((p, i) => g.fillText(p, xl - 10, yP(i)));
    g.textAlign = "left";
    F.forEach((f, j) => {
      const y = yF(j);
      g.fillStyle = COL[j]; g.beginPath(); g.arc(xr + 14, y, 12, 0, 7); g.fill();
      g.fillStyle = SURF; g.font = "800 13px 'IBM Plex Mono', monospace"; g.textAlign = "center"; g.fillText(f.letter, xr + 14, y + 0.5);
      g.textAlign = "left"; g.fillStyle = MUTED; g.font = "12px Inter, sans-serif"; g.fillText("Factor", xr + 32, y);
    });
    if (!reduced && visible) requestAnimationFrame(draw);
  }
  new IntersectionObserver(([e]) => { const was = visible; visible = e.isIntersecting; if (visible && !was && !reduced) requestAnimationFrame(draw); }).observe(c);
  size();
  addEventListener("resize", size);
  document.fonts?.ready.then(size);
}

function readings(c) {
  const { ov, F, traffic, dust, coal, ozone, solvent } = c;
  const H = ov.hours, W = ov.weekly.sunday_t, D = ov.diwali, Lk = ov.lockdown["2020"], R = ov.city_rank;
  const top = (l, n = 3) => Object.keys(R[l]).slice(0, n).join(", ");
  return {
    [traffic]: `Nitrogen oxides from burning fuel load here. It is lowest at ${H.trough_hour[traffic]}:00, highest at ${H.peak_hour[traffic]}:00, and dips on Sundays (t = ${sign(W[traffic], 1)}). The top cities are ${top(traffic)}. Under lockdown it moved ${sign(Lk.effect[traffic])}, the one factor whose drop clearly stands apart from years with no lockdown.`,
    [dust]: `Both sizes of particle load here. It jumps ${sign(D.day_after[dust])} the day after Diwali, in all five years, and no set of ordinary dates reached that. It peaks in the north in October and November, and AQI follows it more closely than any other factor (r = ${ov.aqi_corr[dust].toFixed(2)}).`,
    [coal]: `Sulphur dioxide, which comes largely from burning sulphur-bearing fuel, dominates this factor. The top cities are ${top(coal)}. Talcher and Jorapokhar sit in coal-mining and thermal power areas, which the model had no way of knowing. Ahmedabad's top rank is one we did not predict.`,
    [ozone]: `Ozone is not emitted directly: sunlight forms it from other gases. This factor is lowest at ${H.trough_hour[ozone]}:00 and peaks at ${H.peak_hour[ozone]}:00, following the sun. It has no Sunday dip (t = ${sign(W[ozone], 1)}), which fits something made in the air and not released from a tailpipe.`,
    [solvent]: `Benzene, toluene and xylene load here, and it dips on Sundays (t = ${sign(W[solvent], 1)}). The top cities are ${top(solvent)}. This is our weakest name: there is no clear Diwali or lockdown signal, and raw benzene rose ${sign(ov.lockdown_raw.Benzene, 0)}% under lockdown, which we cannot explain and may be a data problem.`,
  };
}

function buildFactors(c) {
  const { ov, F, L } = c, P = ov.pollutants, read = readings(c);
  const hrs = [...Array(24).keys()], H = ov.hours, Wk = ov.weekly, D = ov.diwali, Lk = ov.lockdown, R = ov.city_rank;
  $("#factorLead").textContent = `Fitted only on pollutant numbers from before 2020, with no source labels and no locations. Pick a letter, read the evidence, and decide what it is before you press reveal. Choosing ${F.length} factors was done on held-out 2019 data, before any of these checks.`;
  $("#ftabs").innerHTML = F.map((f, i) => `<button class="ftab" type="button" role="tab" data-i="${i}" style="--col:${COL[i]}"><i></i>Factor ${f.letter}</button>`).join("");
  $("#loads").innerHTML = P.map(() => `<div class="load"><span></span><div class="load__t"><i></i></div><b></b></div>`).join("");
  const opts = (i) => ({ animate: !capturing });
  let cur = -1;
  const pick = (i) => {
    if (i === cur) return; cur = i;
    const l = L[i], col = COL[i];
    $$(".ftab").forEach((b, j) => { b.classList.toggle("active", j === i); b.setAttribute("aria-selected", j === i); });
    $("#fTag").textContent = `Factor ${l}: what the model produced`;
    const order = P.map((p, j) => [p, ov.fa.loadings[j][i]]).sort((a, b) => b[1] - a[1]);
    $$("#loads .load").forEach((row, j) => { const [p, v] = order[j]; row.children[0].textContent = p; row.children[2].textContent = v.toFixed(2); const bar = $("i", row); bar.style.setProperty("--w", `${Math.max(0, v) * 100}%`); bar.style.setProperty("--col", v >= 0.3 ? col : "#4d483e"); });
    const names = Object.entries(R[l]);
    $("#fCities").innerHTML = names.slice(0, 5).map(([n, v]) => `<span class="cty cty--hi" style="--col:${col}">${n}<b>${sign(v)}</b></span>`).join("") + names.slice(-3).map(([n, v]) => `<span class="cty">${n}<b>${sign(v)}</b></span>`).join("");
    $("#cHour").textContent = `Lowest at ${H.trough_hour[l]}:00, highest at ${H.peak_hour[l]}:00.`;
    $("#cWeek").textContent = `Sunday ${sign(Wk.sunday_mean[l], 3)}, t = ${sign(Wk.sunday_t[l], 1)}. Beyond about ±2 is unlikely to be chance.`;
    const pv = D.p_value[l];
    $("#cDiwali").textContent = `Day after: ${sign(D.day_after[l])}. Ordinary dates: ${sign(D.placebo_p05[l])} to ${sign(D.placebo_p95[l])}.`;
    const e = Lk["2020"], tooClose = Math.abs(Lk["2018"].effect[l]) >= Math.abs(e.effect[l]) * 0.7 || Math.abs(Lk["2019"].effect[l]) >= Math.abs(e.effect[l]) * 0.7 || Math.abs(e.t[l]) < 2;
    $("#cLock").textContent = `2020: ${sign(e.effect[l])} (t = ${sign(e.t[l], 1)}). ${tooClose ? "Not clearly apart from normal years." : "Clearly apart from normal years."}`;
    render($("#pHour"), { data: [line(hrs, H.profile[l], col, { hovertemplate: "%{x}:00: %{y:.2f}<extra></extra>" })], layout: lay("", "", { xaxis: ax("", { dtick: 6 }) }) }, opts(i));
    render($("#pWeek"), { data: [line(DOW, Wk.means[l], col, { mode: "lines+markers", marker: { size: 7 }, hovertemplate: "%{x}: %{y:.3f}<extra></extra>" })], layout: lay("", "") }, opts(i));
    render($("#pDiwali"), { data: [{ type: "scatter", mode: "lines", x: [-5, 5, 5, -5], y: [D.placebo_p05[l], D.placebo_p05[l], D.placebo_p95[l], D.placebo_p95[l]], fill: "toself", fillcolor: "rgba(176,170,157,.14)", line: { width: 0 }, hoverinfo: "skip" }, line(D.offsets, D.curve[l], col, { mode: "lines+markers", marker: { size: 6 }, hovertemplate: "Day %{x}: %{y:.2f}<extra></extra>" })], layout: lay("Days from Diwali", "", { shapes: [vline(0)], xaxis: ax("Days from Diwali", { dtick: 5 }) }) }, opts(i));
    render($("#pLock"), { data: [{ type: "bar", x: ["2020", "2019", "2018"], y: [Lk["2020"].effect[l], Lk["2019"].effect[l], Lk["2018"].effect[l]], marker: { color: [col, "#4d483e", "#4d483e"], line: { color: SURF, width: 2 }, cornerradius: 4 }, hovertemplate: "%{x}: %{y:.2f}<extra></extra>" }], layout: lay("", "") }, opts(i));
    const box = $("#reading");
    box.hidden = true;
    box.dataset.i = i;
    $("#btnReveal").textContent = `What do we think Factor ${l} is?`;
  };
  $$(".ftab").forEach((b) => b.addEventListener("click", () => pick(+b.dataset.i)));
  $("#btnReveal").addEventListener("click", () => {
    const i = +$("#reading").dataset.i, f = F[i], box = $("#reading");
    box.style.setProperty("--col", COL[i]);
    box.innerHTML = `<div class="tag">Our reading of Factor ${f.letter}</div><h3>${f.name}</h3><p>${read[f.letter]}</p><small>This name is an interpretation. The model only ever produced the letter.</small>`;
    box.hidden = false;
  });
  const start = Math.max(0, L.indexOf(params.get("factor") || L[0]));
  lazy($("#ftabs"), () => { pick(start); if (params.has("reveal")) $("#btnReveal").click(); });
}

function buildLockdown({ ov, F, L, traffic, dust }) {
  const Lk = ov.lockdown, t = F[L.indexOf(traffic)], d = F[L.indexOf(dust)], e = Lk["2020"], H = ov.hours;
  $("#lockLead").textContent = `The model only saw data from before 2020, so this is a real test on unseen events. If "${t.name}" is really about traffic and combustion, it should have fallen, and by more than it moves in a normal year. It moved ${sign(e.effect[traffic])} (t = ${sign(e.t[traffic], 1)}), against ${sign(Lk["2019"].effect[traffic])} and ${sign(Lk["2018"].effect[traffic])} in years with no lockdown.`;
  $("#lockT1").textContent = `${t.name}, 2019 against 2020`;
  $("#lockHow").textContent = `A raw drop proves little: spring air differs from winter air every year, and cities differ. So for each of the ${e.n_cities} cities we take the change from February to the lockdown weeks, then subtract the same change a year earlier. Running the identical sum on 2019 and 2018, when nothing happened, shows how big normal noise is.`;
  $("#lockHonest").textContent = `Not everything passes. "${d.name}" moved ${sign(e.effect[dust])} (t = ${sign(e.t[dust], 1)}), but 2018 moved ${sign(Lk["2018"].effect[dust])} with no lockdown at all, so we do not claim it, even though published studies report that particles fell.`;
  lazy($("#plotTrace"), () => plotG("lockdown-trace", $("#plotTrace"), 340));
  lazy($("#plotDid"), () => plotG("lockdown-did", $("#plotDid"), 340));
  lazy($("#plotLockHours"), () => plotG("lockdown-hours", $("#plotLockHours"), 320));
}

function buildDiwali({ ov, F, L, dust, coal }) {
  const D = ov.diwali, d = F[L.indexOf(dust)], yrs = Object.values(D.by_year[dust]);
  $("#diwaliLead").textContent = `"${d.name}" jumps ${sign(D.day_after[dust])} the day after Diwali, relative to the same city days either side. It is positive in ${yrs.filter((v) => v > 0).length} of ${yrs.length} years. Across ${D.n_placebo.toLocaleString("en-IN")} sets of ordinary October and November dates, the same sum averages ${sign(D.placebo_mean[dust])}, and ${D.p_value[dust] === 0 ? "none reached it" : `${(D.p_value[dust] * 100).toFixed(1)}% reached it`}. The sulphur factor rises too (${sign(D.day_after[coal])}), which fits fireworks.`;
  lazy($("#plotDiwali"), () => plotG("diwali-curve", $("#plotDiwali"), 350));
  lazy($("#plotPlacebo"), () => plotG("diwali-placebo", $("#plotPlacebo"), 350));
}

function fingerprint(box, F, values, extra = () => "") {
  box.innerHTML = F.map((f, i) => {
    const v = values[f.letter], pos = Math.max(-1.6, Math.min(1.6, v)) / 1.6 * 50;
    return `<div class="fp__row"><span>${f.name}</span><div class="fp__t"><i style="--col:${COL[i]};left:${pos >= 0 ? 50 : 50 + pos}%;width:${Math.abs(pos)}%"></i></div><b>${sign(v)}${extra(f)}</b></div>`;
  }).join("");
}

async function buildCity({ ov, F, L }) {
  const sel = $("#citySel");
  sel.innerHTML = ov.cities.map((c) => `<option>${c}</option>`).join("");
  sel.value = "Delhi";
  const show = async () => {
    const r = await fetch(`/api/city/${encodeURIComponent(sel.value)}`).then((x) => x.json());
    fingerprint($("#fp"), F, r.profile);
    $("#cityDays").textContent = `${r.days.toLocaleString("en-IN")} days before 2020. Centre line: average across all monitored days.`;
    $("#twin").textContent = r.nearest[0].city;
    $("#twinNote").textContent = `Then ${r.nearest[1].city} and ${r.nearest[2].city}. Least alike: ${r.farthest.city}. Measured by distance between fingerprints, with no location information.`;
    const x = MONTHS;
    await render($("#plotCity"), { data: F.map((f, i) => ({ type: "scatter", mode: "lines", x, y: r.months[f.letter], name: f.name, line: { color: COL[i], width: 2 } })), layout: { ...lay("", "Score"), autosize: true, height: 370, showlegend: true, legend: { orientation: "h", y: 1.18, x: 0, font: { size: 10.5 } }, margin: { l: 42, r: 8, t: 46, b: 30 } } }, { animate: !capturing });
  };
  sel.addEventListener("change", show);
  lazy(sel, show);
  // score a day
  $("#scorer").innerHTML = ov.pollutants.map((p) => `<label>${p}<input type="number" min="0" step="any" data-p="${p}" placeholder="not measured"></label>`).join("");
  const fill = (ex) => $$("#scorer input").forEach((i) => { i.value = ex.readings[i.dataset.p] ?? ""; });
  $("#examples").innerHTML = ov.examples.map((e, i) => `<button class="chip-btn" type="button" data-i="${i}">${e.label}</button>`).join("");
  const score = async () => {
    const readings = {};
    $$("#scorer input").forEach((i) => { if (i.value !== "") readings[i.dataset.p] = +i.value; });
    const res = await fetch("/api/score", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ readings }) });
    const r = await res.json();
    if (!res.ok) { $("#scoreMsg").textContent = r.detail; return; }
    $("#scoreMsg").textContent = `Scored from ${r.used} of ${ov.pollutants.length} pollutants.`;
    fingerprint($("#scoreOut"), F, r.scores);
    $("#scoreTypes").innerHTML = ov.types.rows.map((t, i) => `<div><b>${Math.round(r.types[i] * 100)}%</b>Type ${t.id + 1}<span>Average AQI ${t.aqi_mean.toFixed(0)}</span></div>`).join("");
  };
  $$("#examples button").forEach((b) => b.addEventListener("click", () => { fill(ov.examples[+b.dataset.i]); score(); }));
  $("#btnScore").addEventListener("click", score);
  fill(ov.examples[0]);
  lazy($("#scorer"), score);
}

function buildTree({ ov }) {
  const T = ov.tree, order = T.order, coal = ov.coal_belt;
  const g4 = T.groups["4"], branch = order.filter((n) => g4[n] === g4[coal[0]]);
  const others = branch.filter((n) => !coal.includes(n));
  $("#treeLead").textContent = `Each city is reduced to its average factor scores, with no coordinates or names. Cut into four, one branch holds ${coal.join(", ")}: all three coal-belt towns in the data${others.length ? `, plus ${others.join(" and ")}, which we cannot explain` : ""}.`;
  $("#treeNote").innerHTML = `Agglomerative clustering, Ward linkage, on ${order.length} cities with at least 300 days. Bold names are coal-belt towns, marked afterwards from outside sources: <a href="${ov.sources.talcher}" target="_blank" rel="noopener" style="border-bottom:1px solid var(--line-2)">Talcher</a>, <a href="${ov.sources.jorapokhar}" target="_blank" rel="noopener" style="border-bottom:1px solid var(--line-2)">Jorapokhar</a>, <a href="${ov.sources.brajrajnagar}" target="_blank" rel="noopener" style="border-bottom:1px solid var(--line-2)">Brajrajnagar</a>. Cophenetic correlation is ${T.cophenetic.ward.toFixed(2)}, so read the tree as a useful sketch, not an exact map of distances.`;
  const svg = $("#treeSvg"), W = 1180, rowH = 26, H = order.length * rowH + 30, left = 150, right = W - 20;
  const maxH = Math.max(...T.segments.map((s) => s[3]));
  const X = (h) => left + (h / maxH) * (right - left), Y = (x) => 15 + x * rowH;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  // Rebuild each merge's leaf interval so a branch can be coloured by its group.
  const nodes = new Map(order.map((n, i) => [`${i},0`, [i, i]]));
  const merges = [];
  for (let i = 0; i < T.segments.length; i += 3) {
    const [xa, ya, , h] = T.segments[i], [xb, yb] = T.segments[i + 1], A = nodes.get(`${xa},${ya}`), B = nodes.get(`${xb},${yb}`);
    const span = [Math.min(A[0], B[0]), Math.max(A[1], B[1])];
    nodes.set(`${(xa + xb) / 2},${h}`, span);
    merges.push({ segs: T.segments.slice(i, i + 3), span, kids: [A, B] });
  }
  svg.innerHTML = merges.flatMap((m, i) => m.segs.map((s, j) => `<path data-m="${i}" data-k="${j}" d="M${X(s[1]).toFixed(1)} ${Y(s[0]).toFixed(1)}L${X(s[3]).toFixed(1)} ${Y(s[2]).toFixed(1)}"/>`)).join("") +
    order.map((n, i) => `<text x="${left - 10}" y="${Y(i) + 4}" text-anchor="end" class="${coal.includes(n) ? "coal" : ""}">${n}</text>`).join("");
  const pal = ["#3987e5", "#199e70", "#e0621f", "#9085e9", "#c98500", "#d55181"];
  const paint = (k) => {
    const g = T.groups[String(k)];
    const ids = [...new Set(order.map((n) => g[n]))];
    const colorOf = (span) => { const set = new Set(order.slice(span[0], span[1] + 1).map((n) => g[n])); return set.size === 1 ? pal[ids.indexOf([...set][0])] : "#57524a"; };
    $$("path", svg).forEach((p) => { const m = merges[+p.dataset.m], k_ = +p.dataset.k; p.style.stroke = k_ === 2 ? colorOf(m.span) : colorOf(m.kids[k_]); });
    $("#cutV").textContent = k;
    setRangeFill($("#cut"));
  };
  $("#cut").addEventListener("input", (e) => paint(+e.target.value));
  paint(4);
}

function buildDays({ ov }) {
  const T = ov.types;
  $("#daysLead").textContent = `A Gaussian mixture and k-means both try to sort days into ${T.k} types using factor scores. Silhouette is only ${T.silhouette_k.toFixed(2)}, the two methods agree weakly (adjusted Rand ${T.ari_kmeans_gmm.toFixed(2)}), and ${(T.uncertain_share * 100).toFixed(0)}% of days have no type above 60%. So the honest output is a blend, such as 70% of one type and 30% of another, and not a single label.`;
  $("#typeCards").innerHTML = T.rows.map((t) => `<div><b>${(t.share * 100).toFixed(0)}%</b>of days are Type ${t.id + 1}<span>Average AQI ${t.aqi_mean.toFixed(0)}. Most often ${Object.keys(t.cities).slice(0, 2).join(", ")}.</span></div>`).join("");
  lazy($("#plotTypes"), () => plotG("types", $("#plotTypes"), 340));
  lazy($("#plotMonths"), () => plotG("type-months", $("#plotMonths"), 340));
}

function buildLimits({ ov, F, L, dust, solvent, coal }) {
  const lm = (t, p) => `<article class="limit" data-reveal><h3>${t}</h3><p>${p}</p></article>`, E = ov.lockdown, S = ov.sources, M = ov.data.missing_pct;
  const R = ov.city_rank[coal], bj = Object.keys(R).indexOf("Brajrajnagar") + 1;
  $("#limitsList").innerHTML = [
    lm("The names are our reading", `The model produces letters. "${F[L.indexOf(solvent)].name}" has the weakest support. Fitting this kind of model cannot prove what caused anything. The checks make the names plausible, not certain.`),
    lm("Weather is in these numbers", `Midday dips partly reflect a deeper mixed layer of air, and winter peaks reflect still air as well as emissions. Stubble burning and winter weather both fit the northern peak, and this data cannot separate them.`),
    lm("A claim we withdrew", `Lockdown appears to cut "${F[L.indexOf(dust)].name}" by ${sign(E["2020"].effect[dust])}, but 2018 moved ${sign(E["2018"].effect[dust])} with nothing happening. So we do not claim it from this test.`),
    lm("Exceptions we cannot explain", `Guwahati sits on the coal branch of the tree. Ahmedabad tops the sulphur factor. Brajrajnagar, also a coal-belt town, ranks only ${bj} of ${Object.keys(R).length} on that factor. Benzene rose ${sign(ov.lockdown_raw.Benzene, 0)}% under lockdown.`),
    lm("The monitors have gaps", `Xylene is missing on ${M.Xylene.toFixed(0)}% of days and PM10 on ${M.PM10.toFixed(0)}%. EM handles gaps properly, but a factor is only as good as the monitors behind it. The network also grew from 2015 to 2020, and the data stops in mid-2020.`),
    lm("Not a substitute for a real study", `Formal source apportionment analyses the chemistry of captured particles, and <a href="${S.variability}" target="_blank" rel="noopener" style="border-bottom:1px solid var(--line-2)">even those studies disagree with each other</a>. This is a cheap first look from data India already collects, in the spirit of <a href="${S.receptor}" target="_blank" rel="noopener" style="border-bottom:1px solid var(--line-2)">PCA-based receptor modelling</a>.`),
  ].join("");
}

function buildInside({ ov, F }) {
  const P = ov.pca, A = ov.adequacy, K = ov.k_grid, T = ov.types;
  const c = (t, p, eq) => `<article class="concept" data-reveal><h3>${t}</h3>${eq ? `<pre>${eq}</pre>` : ""}<p>${p}</p></article>`;
  $("#insideGrid").innerHTML = [
    c("PCA and choosing dimensions", `Eigen-decomposition of the correlation matrix. Kaiser's rule and parallel analysis both say ${P.parallel_k}. KMO is ${A.kmo.toFixed(2)}, so factor analysis is justified.`, "R v = λ v"),
    c("Factor analysis by EM", `Each day is explained by a few hidden factors plus pollutant-specific noise. EM uses only the pollutants each day reported, so gaps need no filling in.`, "x = W z + ε      cov(x) = W Wᵀ + Ψ"),
    c("Choosing the number of factors", `Fit before 2019, scored on 2019. Gains flatten after ${F.length}: the next factor adds under 20% of the previous gain. BIC alone would keep adding factors.`, "log p(x<sub>2019</sub> | model fitted before 2019)"),
    c("Varimax rotation", `Any rotation fits equally well (likelihood ${ov.fa.ll_any_rotation.fitted.toFixed(0)} either way), so we pick the angle where factors are easiest to read. Score: ${ov.fa.varimax_before.toFixed(2)} before, ${ov.fa.varimax_after.toFixed(2)} after.`, "maximise Σ var(squared loadings)"),
    c("K-means", "Lloyd's algorithm with k-means++ starts, on each day's factor scores. The elbow method and silhouette show there is no sharp structure.", "J = Σ ‖x − μ<sub>c(x)</sub>‖²"),
    c("Gaussian mixture by EM", `Soft membership of ${T.k} day types, chosen by BIC. Each day gets a probability for every type, which suits overlapping groups.`, "γ<sub>ik</sub> ∝ π<sub>k</sub> N(x<sub>i</sub> | μ<sub>k</sub>, Σ<sub>k</sub>)"),
    c("Hierarchical clustering", `Bottom-up merging by Lance-Williams updates, with Ward, average, complete and single linkage. Results match SciPy in the tests.`, "d(A∪B, C) from d(A,C), d(B,C), d(A,B)"),
    c("Natural experiments", "Difference-in-differences with placebo years for lockdown, and placebo dates for Diwali. These are checks only, and nothing feeds back into the model.", "effect = Δ<sub>2020</sub> − Δ<sub>2019</sub>"),
  ].join("");
}

function initNav() {
  const links = $$("#navLinks a"), pill = $(".nav__pill"), nav = $("#nav"), progress = $("#progress");
  const setActive = (a) => { links.forEach((l) => l.classList.toggle("active", l === a)); pill.style.opacity = a ? 1 : 0; if (a) { pill.style.width = a.offsetWidth + "px"; pill.style.transform = `translateX(${a.offsetLeft}px)`; } };
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) setActive(links.find((l) => l.getAttribute("href") === `#${e.target.id}`)); }), { rootMargin: "-45% 0px -50% 0px" });
  $$("main > section[id]").forEach((s) => io.observe(s));
  let last = 0;
  addEventListener("scroll", () => {
    const y = scrollY, max = document.documentElement.scrollHeight - innerHeight;
    progress.style.transform = `scaleX(${max > 0 ? y / max : 0})`;
    nav.classList.toggle("hidden", y > last && y > 500);
    last = y;
  }, { passive: true });
}

boot();
