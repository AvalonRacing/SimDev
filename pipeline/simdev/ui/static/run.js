// Live force and residual plots on the run page.
//
// Polls every 10 s while the run is live, asking only for iterations it has
// not got yet, so a 5000-iteration solve does not resend its whole history.
// The plateau-gate window is shaded so "is it converging" is visible, not
// only a verdict at the end.
(function () {
  const root = document.getElementById("plots");
  if (!root || typeof uPlot === "undefined") return;
  const run = root.dataset.run;
  const live = root.dataset.live === "true";
  const PALETTE = ["#d0542c", "#2c6fd0", "#1f8a4c", "#8e44ad", "#b26b00", "#16a2b8", "#555"];

  const total = { iteration: [], Cd: [], Cl: [] };
  const components = {};
  const residuals = { iteration: [] };
  let after = 0, afterResiduals = 0, gateWindow = null;
  let forcePlot = null, residualPlot = null, residualNames = [];
  const select = document.getElementById("component");

  const width = () => Math.max(320, Math.min(root.clientWidth - 16, 1100));

  function append(target, chunk) {
    for (const key of Object.keys(chunk)) {
      if (!target[key]) target[key] = [];
      target[key].push(...chunk[key]);
    }
  }

  function shading() {
    return {
      hooks: {
        drawClear: (u) => {
          if (!gateWindow) return;
          const x0 = u.valToPos(gateWindow[0], "x", true);
          const x1 = u.valToPos(gateWindow[1], "x", true);
          u.ctx.save();
          u.ctx.fillStyle = "rgba(44, 111, 208, 0.10)";
          u.ctx.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height);
          u.ctx.restore();
        },
      },
    };
  }

  function shown() {
    const name = select.value;
    return name && components[name] ? components[name] : total;
  }

  function drawForces() {
    const data = shown();
    if (!data.iteration.length) return;
    const rows = [data.iteration, data.Cd, data.Cl];
    if (!forcePlot) {
      forcePlot = new uPlot({
        width: width(), height: 300, plugins: [shading()],
        scales: { x: { time: false } },
        series: [{ label: "iteration" },
                 { label: "Cd", stroke: PALETTE[0] },
                 { label: "Cl", stroke: PALETTE[1] }],
      }, rows, document.getElementById("force-plot"));
    } else {
      forcePlot.setData(rows);
    }
  }

  function drawResiduals() {
    if (!residuals.iteration.length) return;
    const names = Object.keys(residuals).filter((k) => k !== "iteration");
    if (!residualPlot || names.join() !== residualNames.join()) {
      if (residualPlot) residualPlot.destroy();
      residualNames = names;
      residualPlot = new uPlot({
        width: width(), height: 260,
        scales: { x: { time: false }, y: { distr: 3 } },
        series: [{ label: "iteration" },
                 ...names.map((n, i) => ({ label: n, stroke: PALETTE[i % PALETTE.length] }))],
      }, [residuals.iteration, ...names.map((n) => residuals[n])],
         document.getElementById("residual-plot"));
    } else {
      residualPlot.setData([residuals.iteration, ...names.map((n) => residuals[n])]);
    }
  }

  function describeGate(c) {
    const el = document.getElementById("gate");
    if (!c) { el.textContent = ""; return; }
    gateWindow = c.window;
    const window = c.window ? `window ${c.window[0]}–${c.window[1]}` : "";
    el.textContent = `plateau gate: ${c.verdict} · ${window} · drift_tol ${c.drift_tol}` +
      (c.reasons.length ? ` · ${c.reasons.join("; ")}` : "");
  }

  async function poll() {
    try {
      const forces = await (await fetch(`/api/runs/${run}/forces?after=${after}`)).json();
      append(total, forces.total);
      for (const [name, chunk] of Object.entries(forces.components)) {
        if (!components[name]) {
          components[name] = { iteration: [], Cd: [], Cl: [] };
          select.add(new Option(name, name));
        }
        append(components[name], chunk);
      }
      if (total.iteration.length) after = total.iteration[total.iteration.length - 1];
      describeGate(forces.convergence);
      drawForces();

      const res = await (await fetch(`/api/runs/${run}/residuals?after=${afterResiduals}`)).json();
      append(residuals, res);
      if (residuals.iteration.length) afterResiduals = residuals.iteration[residuals.iteration.length - 1];
      drawResiduals();
    } catch (error) {
      console.error(error);
    }
    if (live) setTimeout(poll, 10000);
  }

  select.addEventListener("change", () => {
    if (forcePlot) { forcePlot.destroy(); forcePlot = null; }
    drawForces();
  });
  poll();
})();
