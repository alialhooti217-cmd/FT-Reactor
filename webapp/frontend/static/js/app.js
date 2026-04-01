const form = document.getElementById('sim-form');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');

function card(label, value) {
  return `<article class="card"><h4>${label}</h4><p>${value}</p></article>`;
}

function render(outputs) {
  const feasibility = outputs.feasible ? '✅ FEASIBLE' : `❌ ${outputs.violation_reason}`;
  resultsEl.innerHTML = [
    card('Feasibility', feasibility),
    card('C8–C16 Rate', `${outputs.target_rate_kgph.toFixed(2)} kg/h`),
    card('C8–C16 Fraction', `${(outputs.target_fraction * 100).toFixed(2)} %`),
    card('Specific Energy', `${outputs.specific_energy_kwh_per_kg_target.toFixed(3)} kWh/kg`),
    card('Pressure Drop', `${outputs.delta_p_bar.toFixed(3)} bar`),
    card('Cooling Duty', `${outputs.cooling_duty_mw.toFixed(3)} MW`),
    card('Compressor Power', `${outputs.compressor_power_mw.toFixed(3)} MW`),
    card('Shell Diameter', `${outputs.shell_diameter_m.toFixed(3)} m`),
    card('Tube Length', `${outputs.tube_length_m.toFixed(2)} m`),
    card('L/D', `${outputs.l_over_d.toFixed(2)}`),
    card('Parallel Reactors', `${outputs.n_parallel}`),
    card('Tubes/Reactor', `${outputs.nt_per_reactor}`),
  ].join('');
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const data = Object.fromEntries(new FormData(form).entries());
  statusEl.textContent = 'Running simulation...';

  try {
    const res = await fetch('/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const payload = await res.json();
    if (!res.ok) {
      throw new Error(payload.message || 'Unknown error');
    }
    render(payload.outputs);
    statusEl.textContent = 'Simulation complete.';
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
  }
});
