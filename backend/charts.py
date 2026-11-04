"""Every graph from training, drawn from this run's own numbers, with a plain explanation."""
import numpy as np

from . import config

BLUE, AQUA, ORANGE, VIOLET, AMBER, MAGENTA = "#3987e5", "#199e70", "#e0621f", "#9085e9", "#c98500", "#d55181"
PAL = [BLUE, AQUA, ORANGE, VIOLET, AMBER, MAGENTA]    # validated against the #141210 chart surface, in this order
INK, INK2, MUTED, GRID, AXIS, REF, SURFACE = "#f2efe9", "#cbc6bb", "#8f897d", "#25221d", "#3d3930", "#b0aa9d", "#141210"
SEQ = [[0, "#1a1714"], [0.5, "#2c6cba"], [1, "#a8cdfb"]]
DIV = [[0, "#e0621f"], [0.5, "#2b2823"], [1, "#3987e5"]]
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def ax(title="", **k):
    return {"title": {"text": title, "font": {"color": MUTED, "size": 12}}, "gridcolor": GRID, "linecolor": AXIS, "zerolinecolor": AXIS, "tickcolor": AXIS, "ticks": "outside", "ticklen": 4, "automargin": True, **k}


def lay(x="", y="", xa=None, ya=None, **k):
    return {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "font": {"family": FONT, "color": INK2, "size": 12}, "margin": {"l": 60, "r": 18, "t": 26, "b": 48},
            "hoverlabel": {"bgcolor": "#221f1a", "bordercolor": "#4d483e", "font": {"family": FONT, "color": INK, "size": 12}},
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "bgcolor": "rgba(0,0,0,0)"},
            "xaxis": {**ax(x), **(xa or {})}, "yaxis": {**ax(y), **(ya or {})}, **k}


def line(x, y, name="", color=BLUE, width=2, dash=None, **k):
    return {"type": "scatter", "mode": "lines", "x": x, "y": y, "name": name, "line": {"color": color, "width": width, **({"dash": dash} if dash else {})}, **k}


def dots(x, y, name="", color=BLUE, size=8, **k):
    return {"type": "scatter", "mode": "markers", "x": x, "y": y, "name": name, "marker": {"color": color, "size": size, "line": {"color": SURFACE, "width": 1}}, **k}


def bars(x, y, name="", color=BLUE, **k):
    return {"type": "bar", "x": x, "y": y, "name": name, "marker": {"color": color, "line": {"color": SURFACE, "width": 2}, "cornerradius": 4}, **k}


def vline(x, color=REF, dash="dash"):
    return {"type": "line", "x0": x, "x1": x, "yref": "paper", "y0": 0, "y1": 1, "line": {"color": color, "width": 2, "dash": dash}}


def hline(y, color=REF, dash="dash"):
    return {"type": "line", "y0": y, "y1": y, "xref": "paper", "x0": 0, "x1": 1, "line": {"color": color, "width": 2, "dash": dash}}


def heat(z, x, y, hover, scale=SEQ, zmin=None, zmax=None, text=False, **k):
    t = {"type": "heatmap", "z": z, "x": x, "y": y, "colorscale": scale, "hovertemplate": hover + "<extra></extra>", "xgap": 2, "ygap": 2, "colorbar": {"thickness": 10, "outlinewidth": 0, "tickfont": {"color": MUTED}}, **k}
    if zmin is not None:
        t["zmin"], t["zmax"] = zmin, zmax
    if text:
        t["text"] = [[f"{v:.2f}" for v in row] for row in z]
        t["texttemplate"] = "%{text}"
        t["textfont"] = {"color": INK, "size": 11}
    return t


def G(gid, cat, title, insight, what, how, read, where, data, layout, height=380):
    return {"id": gid, "category": cat, "title": title, "insight": insight, "height": height, "explain": {"what": what, "how": how, "read": read, "where": where}, "figure": {"data": data, "layout": layout}}


def build_graphs(ctx):
    rep, df, cols = ctx["rep"], ctx["df"], ctx["cols"]
    X, resp, hard, fit_scores, km_lab = ctx["X"], ctx["resp"], ctx["hard"], ctx["fit_scores"], ctx["km_lab"]
    D, P, FA, F = rep["data"], rep["pca"], rep["fa"], rep["factors"]
    POL = rep["pollutants"]
    K = FA["k"]
    label = [f"{f['letter']} · {f['name']}" for f in F]
    short = [f"Factor {f['letter']}" for f in F]
    by_name = {f["name"]: f["letter"] for f in F}
    idx = {f["letter"]: i for i, f in enumerate(F)}
    traffic, dust, coal, ozone, solvent = (by_name.get(n) for n in ("Traffic and combustion", "Dust and smoke", "Coal and heavy industry", "Sunlit ozone", "Solvents and petrol vapour"))
    rng = np.random.default_rng(11)
    out = []
    add = lambda *a, **k: out.append(G(*a, **k))

    # ------------------------------------------------------------------ DATA
    miss = D["missing_pct"]
    worst = max(miss, key=miss.get)
    add("missing", "Data", "Monitors leave a lot of gaps",
        f"{worst} is missing on {miss[worst]:.0f}% of days. Dropping incomplete days would throw away most of the data, so the model is built to handle gaps.",
        "Share of city-days with no reading for each pollutant.",
        "Count blanks in each column of the daily file.",
        "A tall bar means the pollutant is often not measured. Each day only contributes the pollutants its monitor reported.",
        "Missing sensor data is normal in environmental monitoring, and EM treats it directly.",
        [bars(list(miss.keys()), list(miss.values()), "Missing", BLUE)], lay("", "Days missing (%)"), 330)

    yrs = D["rows_per_year"]
    add("network", "Data", "The monitor network keeps growing",
        f"{D['rows']:,} city-days across {D['cities']} cities, from {D['first']} to {D['last']}. Year-to-year changes can partly reflect which cities joined.",
        "Number of city-days recorded each year.",
        "Count rows per calendar year.",
        "This is why lockdown is measured city by city, never by comparing national totals across years.",
        "A growing panel is a known trap when reading national pollution trends.",
        [bars(list(yrs.keys()), list(yrs.values()), "City-days", BLUE)], lay("", "City-days"), 330)

    pm = df["PM2.5"].dropna().clip(lower=0)
    h, e = np.histogram(pm.clip(upper=400), bins=60)
    add("skew-raw", "Data", "Raw pollution is badly skewed",
        f"PM2.5 skewness is {D['skew_raw']['PM2.5']:.1f}: a few terrible days dominate. Factor analysis assumes roughly bell-shaped data.",
        "Distribution of daily PM2.5 as reported.",
        "A histogram of the raw values, with the top bin collecting everything above 400.",
        "A long right tail means averages and correlations get pulled by extremes.",
        "Log-transforming concentrations is standard practice in air-quality statistics.",
        [bars(np.round((e[:-1] + e[1:]) / 2, 1).tolist(), h.tolist(), "Days", BLUE)], lay("PM2.5 (µg/m³)", "Days", bargap=.05), 330)
    lg = np.log1p(pm)
    h, e = np.histogram(lg, bins=60)
    add("skew-log", "Data", "The log scale fixes it",
        f"After log(1 + x), skewness falls to {D['skew_log']['PM2.5']:.1f}.",
        "The same PM2.5 values after a log transform.",
        "x becomes log(1 + x), then each pollutant is standardised using pre-2020 values only.",
        "A roughly symmetric hump suits the Gaussian assumption far better.",
        "Using only earlier data for the scale stops future information leaking into training.",
        [bars(np.round((e[:-1] + e[1:]) / 2, 3).tolist(), h.tolist(), "Days", BLUE)], lay("log(1 + PM2.5)", "Days", bargap=.05), 330)

    R = P["correlation"]
    add("correlation", "Data", "Which pollutants travel together?",
        "Three blocks are already visible by eye: particles, nitrogen oxides, and solvent vapours.",
        "Correlation between each pair of pollutants, using every day both were measured.",
        "Pearson correlation on the log, standardised values with pairwise deletion.",
        "Blue means two pollutants rise together. Factor analysis tries to explain this whole table using a few hidden causes.",
        "Correlation tables are the starting point of all receptor modelling.",
        [heat(np.round(R, 3).tolist(), POL, POL, "%{y} and %{x}<br>r = %{z:.2f}", DIV, -1, 1, text=True)], lay("", "", ya={"autorange": "reversed"}), 460)

    a = rep["adequacy"]
    ke = a["kmo_each"]
    add("adequacy", "Data", "Is factor analysis even justified?",
        f"KMO is {a['kmo']:.2f} (above 0.6 is usually acceptable). Bartlett's test: chi-square {a['bartlett_chi2']:,.0f} on {a['bartlett_dof']} degrees of freedom, so the correlations are far from zero.",
        "Kaiser-Meyer-Olkin score for each pollutant.",
        "KMO compares correlations with partial correlations. High values mean the correlations are shared across many variables.",
        "A low bar marks a pollutant with little in common with the others.",
        "Standard pre-checks before any factor analysis.",
        [bars(list(ke.keys()), np.round(list(ke.values()), 3).tolist(), "KMO", BLUE)], lay("", "KMO"), 330)

    # ------------------------------------------------------------------ PCA
    ev, pa = P["eigenvalues"], P["parallel_95"]
    n = list(range(1, len(ev) + 1))
    add("scree", "PCA", "Scree plot: how many dimensions are real?",
        f"The Kaiser rule and parallel analysis both say {P['parallel_k']}. Held-out likelihood, used for the final choice, says {K}. Rules of thumb disagree, so we report all three.",
        "Eigenvalues of the correlation matrix, against those from shuffled data.",
        "Eigen-decompose the correlation matrix. Parallel analysis shuffles each pollutant separately and takes the 95th percentile.",
        "Bars above the dashed line carry more than chance would. The Kaiser rule keeps eigenvalues above 1.",
        "Scree plots and parallel analysis guide dimension choice in PCA and factor analysis.",
        [bars(n, np.round(ev, 3).tolist(), "Real eigenvalues", BLUE), line(n, np.round(pa, 3).tolist(), "Shuffled data, 95th percentile", ORANGE, 2, "dash")], lay("Component", "Eigenvalue", xa={"dtick": 1}))
    add("explained", "PCA", "Variance explained",
        f"The first {K} components explain {P['cumulative'][K - 1] * 100:.0f}% of the variance in 12 pollutants.",
        "Cumulative share of variance explained by the leading components.",
        "Running total of eigenvalues divided by their sum.",
        "Where the curve flattens, extra components add little.",
        "A common stopping rule, and also a common way to compress data.",
        [line(n, np.round(np.array(P["cumulative"]) * 100, 1).tolist(), "Cumulative", BLUE, mode="lines+markers")], lay("Components kept", "Variance explained (%)", xa={"dtick": 1}, shapes=[vline(K)]), 330)
    PL = np.array(P["loadings"])
    add("pca-loadings", "PCA", "PCA loadings are hard to name",
        "PC1 loads on nearly everything, so it means 'pollution in general'. This is why rotated factors are used to name sources.",
        "How strongly each pollutant loads on the first four principal components.",
        "Eigenvector x square root of eigenvalue.",
        "Mixed colours within a column mean that component blends several sources.",
        "PCA is a good compressor, but rotation is normally needed before interpreting it.",
        [heat(np.round(PL, 3).tolist(), ["PC1", "PC2", "PC3", "PC4"], POL, "%{y} on %{x}<br>%{z:.2f}", DIV, -1, 1, text=True)], lay("", "", ya={"autorange": "reversed"}), 440)

    # ------------------------------------------------------------------ FACTOR ANALYSIS
    g = rep["k_grid"]
    ks = [r["k"] for r in g]
    add("choose-k", "Factor analysis", "How many hidden sources?",
        f"Fit on data before 2019, scored on 2019. Gains flatten after {K}: the next factor adds under {rep['chain_rule'] * 100:.0f}% of the gain before it. No outside check influenced this.",
        "Log-likelihood per day on held-out 2019 data for each number of factors.",
        "Fit EM on earlier years, then compute the probability of 2019's observed readings under the model.",
        "Higher is better. Look for where it stops rising.",
        "Choosing latent dimensions on later data is the honest test of a factor model.",
        [line(ks, np.round([r["heldout_ll_per_row"] for r in g], 3).tolist(), "Held-out fit", BLUE, mode="lines+markers")], lay("Factors", "Log-likelihood per day", xa={"dtick": 1}, shapes=[vline(K)]))
    add("bic", "Factor analysis", "BIC would keep adding factors",
        "With tens of thousands of days the penalty is small, so BIC keeps falling. That is why held-out data makes the choice.",
        "Bayesian Information Criterion on training data.",
        "BIC = -2 x log-likelihood + parameters x log(rows).",
        "Lower is better, and it keeps improving.",
        "BIC is a common selector, but it can favour large models on large data.",
        [line(ks, np.round([r["bic"] for r in g], 0).tolist(), "BIC", BLUE, mode="lines+markers")], lay("Factors", "BIC (lower is better)", xa={"dtick": 1}), 330)
    H = FA["history"]
    add("em", "Factor analysis", "EM only climbs",
        "Every iteration raises the likelihood of the observed readings, gaps included. A test in the repository enforces this.",
        "Training log-likelihood after each EM iteration.",
        "E-step: posterior of the factors given each day's observed pollutants. M-step: re-solve loadings and noise for each pollutant.",
        "Rises and flattens. Any dip would point to a bug.",
        "EM is the standard way to fit latent-variable models with missing data.",
        [line(list(range(1, len(H) + 1)), np.round(H, 1).tolist(), "Log-likelihood", BLUE)], lay("EM iteration", "Log-likelihood"), 330)
    add("unrotated", "Factor analysis", "Before rotation: true, but unreadable",
        f"Varimax score {FA['varimax_before']:.3f}. Many cells are moderate, so no factor reads as one source.",
        "Loadings straight out of EM.",
        "Any rotation of the loadings gives exactly the same fit, so EM's orientation is arbitrary.",
        "Compare this with the next chart: same model, turned to a readable angle.",
        "This rotational ambiguity is a basic property of factor analysis.",
        [heat(np.round(FA["unrotated"], 3).tolist(), [f"F{i + 1}" for i in range(K)], POL, "%{y} on %{x}<br>%{z:.2f}", DIV, -1, 1, text=True)], lay("", "", ya={"autorange": "reversed"}), 440)
    add("rotated", "Factor analysis", "After varimax: five clean groups",
        f"Varimax score {FA['varimax_after']:.3f}. Likelihood is unchanged: {FA['ll_any_rotation']['fitted']:,.1f} after varimax, {FA['ll_any_rotation']['random_rotation']:,.1f} after a random rotation.",
        "Loadings after varimax rotation, largest factor first.",
        "Varimax turns the axes to maximise the variance of squared loadings, pushing each cell towards 0 or towards large.",
        "Each column should now light up for a few related pollutants. These groups are what the app calls factors A to E.",
        "Rotated factor loadings are how receptor models are interpreted.",
        [heat(np.round(FA["loadings"], 3).tolist(), label, POL, "%{y} on %{x}<br>%{z:.2f}", DIV, -1, 1, text=True)], lay("", "", ya={"autorange": "reversed"}, xa={"tickangle": -20}), 470)
    com = FA["communalities"]
    add("communality", "Factor analysis", "How much of each pollutant is explained?",
        f"{POL[int(np.argmax(com))]} is explained best ({max(com) * 100:.0f}%). {POL[int(np.argmin(com))]} least ({min(com) * 100:.0f}%), so most of its variation is its own.",
        "Split of each pollutant's variance into shared (communality) and unique (noise).",
        "Communality is the sum of squared loadings. Uniqueness is psi, learned by EM.",
        "A big unique bar warns not to read much into that pollutant's loadings.",
        "Low communality often flags noisy sensors or a source the factors miss.",
        [bars(POL, np.round(com, 3).tolist(), "Shared with factors", BLUE), bars(POL, np.round(FA["psi"], 3).tolist(), "Unique to this pollutant", ORANGE)], lay("", "Share of variance", barmode="stack"), 340)
    vs = FA["variance_share"]
    add("factor-sizes", "Factor analysis", "How big is each factor?",
        f"Together they explain {sum(vs) * 100:.0f}% of the variance. Size says how much variation a factor explains, not how harmful it is.",
        "Variance explained by each rotated factor.",
        "Sum of a factor's squared loadings, divided by the number of pollutants.",
        "A larger bar means that factor explains more of the day-to-day variation.",
        "Reported alongside loadings in factor analysis.",
        [bars(np.round(np.array(vs) * 100, 1).tolist(), label, "Variance", BLUE, orientation="h")], lay("Variance explained (%)", "", ya={"autorange": "reversed"}, margin={"l": 210, "r": 18, "t": 26, "b": 48}), 300)
    Rm, Im = np.array(R), np.array(FA["implied_corr"])
    tri = np.triu_indices(len(R), 1)
    add("fit-check", "Factor analysis", "Do five factors reproduce the correlations?",
        f"The model's implied correlations miss the real ones by {np.abs(Rm[tri] - Im[tri]).mean():.3f} on average.",
        "Each pair's real correlation against the one implied by W W^T + Psi.",
        "Compute the model's covariance and compare off-diagonal entries.",
        "Dots on the dashed line are perfectly reproduced.",
        "Residual checks show whether the factor count is sufficient.",
        [dots(np.round(Rm[tri], 3).tolist(), np.round(Im[tri], 3).tolist(), "Pollutant pairs", BLUE, 9), line([-.2, 1], [-.2, 1], "Perfect", REF, 1, "dash")], lay("Real correlation", "Implied by the model"))
    pf = np.array(FA["paf_loadings"])
    em = np.array(FA["loadings"])
    match = [int(np.argmax(np.abs(pf.T @ em[:, j]))) for j in range(K)]
    pf2 = pf[:, match] * np.sign((pf[:, match] * em).sum(0))
    add("paf-vs-em", "Factor analysis", "A second method agrees",
        f"Principal axis factoring, a different algorithm, gives correlation {np.corrcoef(pf2.ravel(), em.ravel())[0, 1]:.2f} with the EM loadings.",
        "Each loading from principal axis factoring against the same loading from EM.",
        "PAF iterates eigen-decomposition with communalities on the diagonal. Both are varimax rotated and matched by column.",
        "Points on the line mean the methods agree, so the factors are not an artefact of one algorithm.",
        "Comparing extraction methods is a routine robustness check.",
        [dots(np.round(em.ravel(), 3).tolist(), np.round(pf2.ravel(), 3).tolist(), "Loadings", BLUE, 8), line([-.3, 1], [-.3, 1], "Perfect", REF, 1, "dash")], lay("EM factor analysis", "Principal axis factoring"), 360)

    # ------------------------------------------------------------------ EVIDENCE
    Hr = rep["hours"]
    hrs = list(range(24))
    add("hours", "Evidence", "Each factor keeps its own clock",
        f"{F[idx[ozone]]['name']} peaks at {Hr['peak_hour'][ozone]}:00 and bottoms at {Hr['trough_hour'][ozone]}:00, following the sun. {F[idx[traffic]]['name']} does the opposite: lowest at {Hr['trough_hour'][traffic]}:00, highest at {Hr['peak_hour'][traffic]}:00.",
        "Average factor score at each hour of the day, before 2020.",
        "The daily model is applied, unchanged, to hourly readings. It never trained on them.",
        "A midday dip partly reflects the atmosphere mixing more in the afternoon, not only lower emissions.",
        "Diurnal profiles are a classic way to identify pollution sources.",
        [line(hrs, Hr["profile"][c], label[i], PAL[i]) for i, c in enumerate(cols)], lay("Hour of day", "Average score", xa={"dtick": 3}), 420)
    wm = Hr["weekday_minus_sunday"]
    add("weekday-gap", "Evidence", "When does a weekday differ from a Sunday?",
        f"At 3am, {F[idx[traffic]]['name']} shows no gap ({wm[traffic][3]:+.2f}). By 9am it is {wm[traffic][9]:+.2f}. A weather explanation would not switch on with the working day.",
        "Weekday score minus Sunday score, hour by hour.",
        "Average hourly scores for Monday to Friday and subtract Sunday's.",
        "A line near zero at night that rises in the day points to human activity.",
        "The weekend effect is widely used to separate traffic from background pollution.",
        [line(hrs, wm[c], label[i], PAL[i]) for i, c in enumerate(cols)], lay("Hour of day", "Weekday minus Sunday", xa={"dtick": 3}), 420)
    Wk = rep["weekly"]
    add("weekly", "Evidence", "The weekly rhythm",
        "City and month averages are removed first, so only the day of the week is left.",
        "Average score by day of week.",
        "Subtract each city-month's mean, then average by weekday.",
        "A dip on Sunday suggests a source tied to the working week.",
        "Used for the same purpose as the previous chart, at daily resolution.",
        [line(DOW, Wk["means"][c], label[i], PAL[i], mode="lines+markers") for i, c in enumerate(cols)], lay("", "Score, city-month removed"), 400)
    st = Wk["sunday_t"]
    add("sunday-t", "Evidence", "How solid is the Sunday dip?",
        f"{F[idx[traffic]]['name']} t = {st[traffic]:.1f}, {F[idx[ozone]]['name']} t = {st[ozone]:.1f}. Ozone is made in the air by sunlight, so its missing dip fits.",
        "t-statistic of each factor's Sunday effect.",
        "Mean of Sunday deviations divided by its standard error.",
        "Beyond about -2 is unlikely to be chance. Days are not fully independent, so treat exact values with care.",
        "A basic significance check.",
        [bars(short, [round(st[c], 2) for c in cols], "t", BLUE)], lay("", "t-statistic", shapes=[hline(-2)]), 320)
    Dw = rep["diwali"]
    add("diwali-curve", "Evidence", "Diwali, in days",
        f"Across {len(config.DIWALI)} Diwalis, {F[idx[dust]]['name']} jumps {Dw['day_after'][dust]:+.2f} the day after. The model was told nothing about the festival.",
        "Each factor's score relative to the same city 4 to 10 days either side, by days from Diwali.",
        "For each date and city, subtract the nearby baseline, then average across cities and years.",
        "Day 0 is Diwali. Daily averages run midnight to midnight, so night-time fireworks land on days 0 and 1.",
        "Festival fireworks are a well-known natural experiment in air-quality research.",
        [line(Dw["offsets"], np.round(Dw["curve"][c], 3).tolist(), label[i], PAL[i], mode="lines+markers") for i, c in enumerate(cols)], lay("Days from Diwali", "Change from nearby days", shapes=[vline(0)]), 420)
    yrs_ = list(Dw["by_year"][dust].keys())
    add("diwali-years", "Evidence", "It happens every year",
        f"The day-after jump in {F[idx[dust]]['name']} is positive in all {len(yrs_)} years.",
        "Day-after effect for each year separately.",
        "The same calculation, not averaged across years.",
        "One good year proves little. Consistency across years is the stronger evidence.",
        "Replication across events.",
        [bars(yrs_, [round(Dw["by_year"][dust][y], 3) for y in yrs_], F[idx[dust]]["name"], BLUE), bars(yrs_, [round(Dw["by_year"][coal][y], 3) for y in yrs_], F[idx[coal]]["name"], ORANGE)], lay("", "Day-after jump", barmode="group"), 340)
    add("diwali-placebo", "Evidence", "Could an ordinary day do that?",
        f"Across {Dw['n_placebo']:,} placebo sets of ordinary October-November dates, the {F[idx[dust]]['name']} jump averaged {Dw['placebo_mean'][dust]:+.2f} (sd {Dw['placebo_sd'][dust]:.2f}). Real: {Dw['day_after'][dust]:+.2f}. None of them reached it." if Dw["p_value"][dust] == 0 else f"Placebo p-value {Dw['p_value'][dust]:.3f}.",
        "Each factor's real day-after jump against the range from ordinary dates.",
        "Repeat the exact calculation on random non-festival dates, one per year, many times.",
        "A dot outside its grey range is hard to explain by chance.",
        "Placebo tests are a standard defence against reading noise as an effect.",
        [{"type": "bar", "x": short, "y": [round(Dw["placebo_p95"][c] - Dw["placebo_p05"][c], 3) for c in cols], "base": [round(Dw["placebo_p05"][c], 3) for c in cols], "name": "Ordinary dates, 5th to 95th percentile", "marker": {"color": "#3a362e"}},
         dots(short, [round(Dw["day_after"][c], 3) for c in cols], "Real Diwali", ORANGE, 12)], lay("", "Day-after change"), 340)
    Lk = rep["lockdown"]
    add("lockdown-did", "Evidence", "Lockdown, with placebo years",
        f"{F[idx[traffic]]['name']}: {Lk[str(2020)]['effect'][traffic]:+.2f} in 2020 (t = {Lk[str(2020)]['t'][traffic]:.1f}), against {Lk[str(2019)]['effect'][traffic]:+.2f} and {Lk[str(2018)]['effect'][traffic]:+.2f} in years with no lockdown. {F[idx[dust]]['name']}'s {Lk[str(2020)]['effect'][dust]:+.2f} is not safe to claim: 2018 moved {Lk[str(2018)]['effect'][dust]:+.2f} by itself.",
        "Difference-in-differences of each factor around 25 March.",
        "For each city: change from 1 Feb-20 Mar to 25 Mar-30 Apr this year, minus the same change a year earlier. Then average across cities.",
        "This removes each city's level and the normal spring change. 2019 and 2018 had no lockdown, so their bars show normal noise.",
        "Difference-in-differences is the standard tool for natural experiments.",
        [bars(short, [round(Lk[str(y)]["effect"][c], 3) for c in cols], f"{y}" + (" (lockdown)" if y == 2020 else " (placebo)"), col) for y, col in ((2020, ORANGE), (2019, BLUE), (2018, AQUA))], lay("", "Change in score", barmode="group"), 400)
    Tr = rep["trace"]
    day = Tr["day"]
    add("lockdown-trace", "Evidence", "The traffic factor, 2019 against 2020",
        f"2020 is already lower in January and falls away from early March, before the national lockdown began. It then stays low through April, near {min(v for v in Tr['2020'][traffic] if v is not None):+.2f}.",
        f"Daily average of {F[idx[traffic]]['name']} across cities, smoothed over a week.",
        "Scores for 2020 come from a model fitted only on earlier data.",
        "The dashed line is 25 March. Because the gap opens before that date, this picture alone cannot credit it all to lockdown, and the mix of cities also differs by year. The previous chart controls for both by comparing each city with itself.",
        "A simple visual check to pair with the formal estimate.",
        [line(day, Tr["2019"][traffic], "2019", REF), line(day, Tr["2020"][traffic], "2020", ORANGE)], lay("Day of year", "Score", shapes=[vline(85)]))
    add("lockdown-hours", "Evidence", "The evening peak disappears",
        f"Evening peak (21:00) falls from {Hr['spring2019'][traffic][21]:+.2f} in spring 2019 to {Hr['lockdown2020'][traffic][21]:+.2f} under lockdown.",
        f"{F[idx[traffic]]['name']} by hour, 25 Mar-30 Apr, in 2019 and 2020.",
        "Hourly readings scored by the same model.",
        "Look at the shape as well as the level. The small spike at 13:00 appears in both years, so it has nothing to do with lockdown and may be a reporting artefact.",
        "Diurnal changes helped separate traffic from other sources in lockdown studies.",
        [line(hrs, Hr["spring2019"][traffic], "Spring 2019", REF), line(hrs, Hr["lockdown2020"][traffic], "Lockdown 2020", ORANGE)], lay("Hour of day", "Score", xa={"dtick": 3}))
    raw = rep["lockdown_raw"]
    add("lockdown-raw", "Evidence", "The raw pollutants tell a messier story",
        f"Unadjusted changes, 2019 to 2020: CO {raw['CO']:+.0f}%, NO2 {raw['NO2']:+.0f}%, but benzene {raw['Benzene']:+.0f}%. We have no good explanation for the benzene rise, which may be a data problem.",
        "Percent change in mean concentration, same dates, 2019 to 2020.",
        "Average across cities reporting in both years, then compare.",
        "No seasonal or city control here, so it is only a rough cross-check.",
        "Published lockdown studies report similar falls in NO2 and particles.",
        [bars(list(raw.keys()), list(raw.values()), "Change", BLUE)], lay("", "Change (%)"), 340)
    Se = rep["seasons"]
    add("season", "Evidence", "The north's winter",
        f"{F[idx[dust]]['name']} peaks in November in the north at {Se['north'][dust][10]:+.2f}, far above the south at {Se['south'][dust][10]:+.2f}. Stubble burning and still winter air both fit, and this data cannot separate them.",
        "Monthly mean score for northern and southern cities.",
        "The region lists are ours, from geography. The model does not know them.",
        "Compare the height of the two lines from October to January.",
        "Seasonal source profiles are central to winter action plans.",
        [line(MONTHS, Se["north"][dust], "North", ORANGE, mode="lines+markers"), line(MONTHS, Se["south"][dust], "South", BLUE, mode="lines+markers")], lay("", "Score"))

    def rank_chart(letter):
        rk = rep["city_rank"][letter]
        names, vals = list(rk.keys()), list(rk.values())
        coalset = set(config.COAL_BELT)
        return [bars([v for n, v in zip(names, vals) if n in coalset], [n for n in names if n in coalset], "Coal-belt towns", ORANGE, orientation="h"),
                bars([v for n, v in zip(names, vals) if n not in coalset], [n for n in names if n not in coalset], "Other cities", BLUE, orientation="h")], names
    tr_, nm = rank_chart(coal)
    rc = rep["city_rank"][coal]
    add("rank-coal", "Evidence", "Who tops the sulphur factor?",
        f"Top three: {', '.join(list(rc.keys())[:3])}. Talcher and Jorapokhar sit in coal-mining and thermal power areas. The model had no map. Brajrajnagar, also in a coal belt, ranks far lower.",
        f"Average pre-2020 score on {F[idx[coal]]['name']}, by city.",
        "Mean daily score per city, for cities with at least 300 days.",
        "Orange marks towns we know to be in coal belts, an outside fact added afterwards.",
        "Checking factors against known source locations is how receptor models are validated.",
        tr_, lay("Average score", "", ya={"categoryorder": "array", "categoryarray": nm, "autorange": "reversed"}, barmode="overlay"), 520)
    tr_, nm = rank_chart(traffic)
    add("rank-traffic", "Evidence", "Who tops the nitrogen factor?",
        f"Top three: {', '.join(list(rep['city_rank'][traffic].keys())[:3])}: three of India's biggest metros.",
        f"Average pre-2020 score on {F[idx[traffic]]['name']}, by city.",
        "Same calculation as the previous chart.",
        "Longer bars mean more nitrogen oxides than the national average.",
        "NOx is widely used as a marker of vehicle and combustion emissions.",
        tr_, lay("Average score", "", ya={"categoryorder": "array", "categoryarray": nm, "autorange": "reversed"}, barmode="overlay"), 520)
    Cp = rep["city_profiles"]
    cn = sorted(Cp, key=lambda c: Cp[c][dust])
    add("city-grid", "Evidence", "Every city's fingerprint",
        "Each row is a city's mix of hidden factors. Cities with similar rows get similar pollution problems.",
        "Average score on each factor, for each city.",
        "Mean daily score, pre-2020.",
        "Blue is above the national average, orange below.",
        "This is the input to the family tree.",
        [heat([[Cp[c][l] for l in cols] for c in cn], short, cn, "%{y}, %{x}<br>%{z:.2f}", DIV, -1.6, 1.6)], lay("", ""), 540)
    aq = rep["aqi_corr"]
    add("aqi", "Evidence", "AQI hides all of this",
        f"The five factors, used together, explain {rep['aqi_r2'] * 100:.0f}% of AQI. Much of what they separate collapses into one AQI number.",
        "Correlation of each factor with the reported AQI, which the model never saw.",
        "Pearson correlation across all pre-2020 days with both values.",
        "AQI follows dust and smoke most closely, so it can miss the other sources.",
        "AQI is built for public warnings, not for finding causes.",
        [bars(label, [round(aq[c], 3) for c in cols], "Correlation", BLUE, orientation="h")], lay("Correlation with AQI", "", ya={"autorange": "reversed"}, margin={"l": 210, "r": 18, "t": 26, "b": 48}), 300)

    # ------------------------------------------------------------------ CLUSTERING
    kg, gg, T = rep["kmeans_grid"], rep["gmm_grid"], rep["types"]
    kk = [r["k"] for r in kg]
    add("elbow", "Clustering", "K-means elbow",
        "No sharp elbow, which is a warning that days do not fall into crisp groups.",
        "Total within-cluster squared distance for each number of clusters.",
        "Run Lloyd's algorithm with k-means++ starts on each day's factor scores.",
        "A clear bend would suggest a natural cluster count.",
        "The elbow method is a simple way to choose k.",
        [line(kk, np.round([r["inertia"] for r in kg], 0).tolist(), "Inertia", BLUE, mode="lines+markers")], lay("Clusters", "Within-cluster distance", xa={"dtick": 1}), 330)
    add("silhouette", "Clustering", "Silhouette: how separated are the clusters?",
        f"Scores stay near {np.mean([r['silhouette'] for r in kg]):.2f}. Values below about 0.25 mean weak structure.",
        "Mean silhouette for each number of clusters.",
        "(distance to the nearest other cluster - distance within own) / the larger of the two, on a sample of days.",
        "Closer to 1 is cleaner. Low values are why soft membership is more honest here.",
        "A standard internal cluster-quality score.",
        [line(kk, np.round([r["silhouette"] for r in kg], 3).tolist(), "Silhouette", BLUE, mode="lines+markers")], lay("Clusters", "Mean silhouette", xa={"dtick": 1}), 330)
    add("gmm-bic", "Clustering", "Gaussian mixture: BIC",
        f"The big drop arrives at {T['k']} components, then gains fall away by the same 20% rule.",
        "BIC of a full-covariance Gaussian mixture on factor scores.",
        "Fit by EM from a k-means start, for each number of components.",
        "Lower is better. Look for the largest useful drop.",
        "BIC is the usual selector for mixtures.",
        [line(kk, np.round([r["bic"] for r in gg], 0).tolist(), "BIC", BLUE, mode="lines+markers")], lay("Components", "BIC", xa={"dtick": 1}, shapes=[vline(T["k"])]), 330)
    GH = T["history"]
    add("gmm-em", "Clustering", "Mixture EM only climbs",
        f"Converged in {len(GH)} iterations.",
        "Log-likelihood per iteration.",
        "E-step: responsibility of each component for each day. M-step: re-estimate weights, means and covariances.",
        "A dip would mean a bug.",
        "The same EM logic as factor analysis, in a different model.",
        [line(list(range(1, len(GH) + 1)), np.round(GH, 1).tolist(), "Log-likelihood", BLUE)], lay("EM iteration", "Log-likelihood"), 320)
    trow = T["rows"]
    tn = [f"Type {t['id'] + 1}" for t in trow]
    add("types", "Clustering", "What each type of day looks like",
        "Types are ordered by average AQI, from cleanest to worst.",
        "Mean factor score of each mixture component.",
        "The learned component means, mu.",
        "Blue is above average, orange below.",
        "Profiling clusters is how they get interpreted.",
        [heat([np.round(t["center"], 2).tolist() for t in trow], label, tn, "%{y}, %{x}<br>%{z:.2f}", DIV, -1.6, 1.6, text=True)], lay("", "", ya={"autorange": "reversed"}, xa={"tickangle": -20}), 380)
    add("type-months", "Clustering", "When each type happens",
        "AQI was not used to form these. Types differ in their season anyway.",
        "Share of each type's days falling in each month.",
        "Assign each day to its most likely component and count by month.",
        "A bright column shows a season where the type is common.",
        "Validates clusters against something they were not trained on.",
        [heat([np.round(t["month_share"], 3).tolist() for t in trow], MONTHS, tn, "%{y}, %{x}<br>%{z:.1%} of its days")], lay("", "", ya={"autorange": "reversed"}), 320)
    add("type-aqi", "Clustering", "Average AQI by type",
        f"Runs from {min(t['aqi_mean'] for t in trow):.0f} to {max(t['aqi_mean'] for t in trow):.0f}, with AQI held out of the fit.",
        "Mean reported AQI of the days assigned to each type.",
        "A check using a value the mixture never saw.",
        "Higher means worse air.",
        "External validation for clustering.",
        [bars(tn, [round(t["aqi_mean"], 1) for t in trow], "Mean AQI", BLUE)], lay("", "Average AQI"), 320)
    maxr = resp.max(1)
    h, e = np.histogram(maxr, bins=30, range=(0, 1))
    add("certainty", "Clustering", "How sure is the mixture?",
        f"{T['uncertain_share'] * 100:.0f}% of days have no component above 60%. A hard label would hide that.",
        "Largest responsibility for each day.",
        "Responsibilities from the final E-step.",
        "A pile near 1 means confident. A spread to the left means in-between days.",
        "Soft clustering is useful when groups overlap.",
        [bars(np.round((e[:-1] + e[1:]) / 2, 3).tolist(), h.tolist(), "Days", BLUE)], lay("Largest responsibility", "Days", bargap=.05), 320)
    conf = np.zeros((T["k"], T["k"]))
    np.add.at(conf, (hard, km_lab), 1)
    conf = conf / conf.sum(1, keepdims=True)
    add("km-vs-gmm", "Clustering", "K-means and the mixture disagree",
        f"Adjusted Rand index {T['ari_kmeans_gmm']:.2f}. K-means favours round clusters while the mixture allows stretched ones, and on overlapping data they split days differently.",
        "For each mixture type, how its days are spread over k-means clusters.",
        "Row-normalised contingency table.",
        "One bright cell per row would mean agreement.",
        "Comparing methods shows how method-dependent a clustering is.",
        [heat(np.round(conf, 3).tolist(), [f"K-means {i + 1}" for i in range(T["k"])], tn, "%{y}, %{x}<br>%{z:.0%}")], lay("", "", ya={"autorange": "reversed"}), 330)
    pick = rng.choice(len(X), min(2500, len(X)), replace=False)
    a_, b_ = idx[dust], idx[traffic]
    add("day-cloud", "Clustering", "Days as points in factor space",
        "A cloud with soft edges, which fits the low silhouette scores.",
        f"A sample of days on two factors, coloured by most likely type.",
        "Posterior factor scores, coloured by responsibility.",
        "Overlap between colours means the types blend into each other.",
        "Scatter plots of latent scores are a common sanity check.",
        [dots(np.round(X[pick][hard[pick] == j, a_], 3).tolist(), np.round(X[pick][hard[pick] == j, b_], 3).tolist(), tn[j], PAL[j], 5, opacity=.7) for j in range(T["k"])], lay(F[a_]["name"], F[b_]["name"]), 420)

    # ------------------------------------------------------------------ TREE
    Tt = rep["tree"]
    order = Tt["order"]
    segs = Tt["segments"]
    xs, ys = [], []
    for x1, y1, x2, y2 in segs:
        xs += [y1, y2, None]
        ys += [x1, x2, None]
    groups = Tt["groups"]["4"]
    coal_group = {n for n in order if groups[n] == groups["Talcher"]}
    add("dendrogram", "Family tree", "Cities, grouped by the air they share",
        f"Cut into four branches, one holds {', '.join(sorted(coal_group))}. Three of those are coal-belt towns. Guwahati is the exception we cannot explain. No map or distance was used.",
        "A family tree of cities from their average factor scores.",
        "Agglomerative clustering with Ward linkage: repeatedly merge the two groups that least increase within-group variance.",
        "Cities that merge near the left are alike. Long horizontal gaps separate distinct groups.",
        "Dendrograms are how biology, linguistics and market research show similarity.",
        [line(xs, ys, "Merges", BLUE, 2, connectgaps=False, hoverinfo="skip")], lay("Merge distance", "", ya={"tickmode": "array", "tickvals": list(range(len(order))), "ticktext": order, "autorange": "reversed"}, showlegend=False), 560)
    cop = Tt["cophenetic"]
    add("linkages", "Family tree", "Which linkage keeps distances best?",
        f"Average linkage scores {cop['average']:.2f}, Ward {cop['ward']:.2f}. Ward is used for its compact, readable groups, at the cost of some distance distortion.",
        "Cophenetic correlation for four linkage rules.",
        "Correlate original city distances with the height at which each pair merges.",
        "Closer to 1 means the tree preserves distances better.",
        "A standard way to compare hierarchical clusterings.",
        [bars(list(cop.keys()), [round(v, 3) for v in cop.values()], "Cophenetic r", BLUE)], lay("", "Cophenetic correlation"), 320)
    H_ = Tt["heights"]
    add("merge-heights", "Family tree", "Where to cut the tree",
        "A big jump in merge height means two very different groups were forced together.",
        "Height of each merge, in order.",
        "The third column of the linkage matrix.",
        "Cut just before a large jump.",
        "The tree's own version of the elbow method.",
        [line(list(range(1, len(H_) + 1)), np.round(H_, 3).tolist(), "Merge height", BLUE, mode="lines+markers")], lay("Merge number", "Merge distance"), 320)
    return out
