"""Interactive HTML export for explanation results."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .explainer import ExplanationResult


def export_html(result: ExplanationResult, output_path: str) -> None:
    """Write a self-contained HTML file with filterable, sortable rules.

    Parameters
    ----------
    result : ExplanationResult
        The explanation result to export.
    output_path : str
        Destination file path (e.g. ``"report.html"``).
    """
    rules_data = _extract_rules_data(result.rules.rules)
    report_text = html.escape(str(result.report))

    # Pruned rules
    pruned_data = None
    pruning_summary = ""
    if result.pruned_rules is not None:
        pruned_data = _extract_rules_data(result.pruned_rules.rules)
    if result.pruning_report is not None:
        pr = result.pruning_report
        pruning_summary = (
            f"Original: {pr.original_count} rules | "
            f"Pruned: {pr.pruned_count} rules | "
            f"Removed (low confidence): {pr.removed_low_confidence} | "
            f"Removed (low samples): {pr.removed_low_samples} | "
            f"Conditions simplified: {pr.conditions_simplified}"
        )

    # Monotonicity
    mono_text = ""
    if result.monotonicity_report is not None:
        mono_text = html.escape(str(result.monotonicity_report))

    # Ensemble
    ensemble_text = ""
    stable_rules_data = None
    if result.ensemble_report is not None:
        ensemble_text = html.escape(str(result.ensemble_report))
    if result.stable_rules is not None:
        stable_rules_data = []
        for sr in result.stable_rules:
            r = sr.rule
            conditions = " AND ".join(str(c) for c in r.conditions) or "TRUE"
            stable_rules_data.append({
                "conditions": html.escape(conditions),
                "prediction": html.escape(str(r.prediction)),
                "confidence": r.confidence,
                "samples": r.samples,
                "is_regression": r.prediction_value is not None,
                "prediction_value": r.prediction_value,
                "frequency": sr.frequency,
            })

    page = _build_html(
        rules_data, report_text, result._task,
        pruned_data=pruned_data,
        pruning_summary=pruning_summary,
        mono_text=mono_text,
        ensemble_text=ensemble_text,
        stable_rules_data=stable_rules_data,
    )

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(page)


def _extract_rules_data(rules) -> list[dict]:
    """Convert rules to a list of dicts for HTML rendering."""
    data = []
    for rule in rules:
        conditions = " AND ".join(str(c) for c in rule.conditions) or "TRUE"
        data.append({
            "conditions": html.escape(conditions),
            "prediction": html.escape(str(rule.prediction)),
            "confidence": rule.confidence,
            "samples": rule.samples,
            "is_regression": rule.prediction_value is not None,
            "prediction_value": rule.prediction_value,
        })
    return data


def _build_rules_table(rules_data: list[dict], table_id: str, extra_col: str = "") -> str:
    """Build an HTML table from rules data."""
    rows = []
    for i, r in enumerate(rules_data, 1):
        if r["is_regression"]:
            pred_cell = f"value = {r['prediction_value']:.4f}"
            conf_cell = ""
        else:
            pred_cell = r["prediction"]
            conf_cell = f"{r['confidence']:.2%}"
        freq_cell = f"<td>{r['frequency']:.0%}</td>" if "frequency" in r else ""
        rows.append(
            f"<tr data-class=\"{r['prediction']}\">"
            f"<td>{i}</td>"
            f"<td class=\"conditions\">{r['conditions']}</td>"
            f"<td class=\"prediction\">{pred_cell}</td>"
            f"<td>{conf_cell}</td>"
            f"<td>{r['samples']}</td>"
            f"{freq_cell}"
            f"</tr>"
        )
    freq_header = "<th data-sort=\"num\">Frequency</th>" if extra_col == "frequency" else ""
    return f"""<table id="{table_id}">
  <thead>
    <tr>
      <th data-sort="num">#</th>
      <th>Conditions</th>
      <th>Prediction</th>
      <th data-sort="num">Confidence</th>
      <th data-sort="num">Samples</th>
      {freq_header}
    </tr>
  </thead>
  <tbody>
    {"".join(rows)}
  </tbody>
</table>"""


def _build_html(
    rules_data: list[dict],
    report_text: str,
    task: str,
    *,
    pruned_data: list[dict] | None = None,
    pruning_summary: str = "",
    mono_text: str = "",
    ensemble_text: str = "",
    stable_rules_data: list[dict] | None = None,
) -> str:
    """Construct the full HTML string."""
    # Main rules table
    main_table = _build_rules_table(rules_data, "rules-table")

    # Collect unique classes for filter buttons
    classes = sorted({r["prediction"] for r in rules_data})
    filter_buttons = '<button class="filter-btn active" data-class="all">All</button>\n'
    for cls in classes:
        filter_buttons += f'<button class="filter-btn" data-class="{cls}">{cls}</button>\n'

    # Optional sections
    pruning_section = ""
    if pruned_data is not None:
        pruning_table = _build_rules_table(pruned_data, "pruned-table")
        pruning_section = f"""
<h2>Pruned Rules</h2>
<div class="report">{html.escape(pruning_summary)}</div>
{pruning_table}
"""

    mono_section = ""
    if mono_text:
        mono_section = f"""
<h2>Monotonicity Report</h2>
<div class="report">{mono_text}</div>
"""

    ensemble_section = ""
    if ensemble_text or stable_rules_data:
        stable_table = ""
        if stable_rules_data:
            stable_table = _build_rules_table(stable_rules_data, "stable-table", extra_col="frequency")
        ensemble_section = f"""
<h2>Ensemble Stability</h2>
<div class="report">{ensemble_text}</div>
{stable_table}
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Explainer Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f5; color: #333; padding: 2rem; }}
  h1 {{ margin-bottom: 1rem; }}
  h2 {{ margin-top: 1.5rem; margin-bottom: 0.75rem; }}
  .report {{ background: #fff; padding: 1rem; border-radius: 6px;
             margin-bottom: 1.5rem; white-space: pre-wrap;
             font-family: monospace; font-size: 0.85rem;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .controls {{ margin-bottom: 1rem; display: flex; flex-wrap: wrap; gap: 0.5rem;
               align-items: center; }}
  .filter-btn {{ padding: 0.3rem 0.8rem; border: 1px solid #ccc; border-radius: 4px;
                 background: #fff; cursor: pointer; font-size: 0.85rem; }}
  .filter-btn.active {{ background: #4a90d9; color: #fff; border-color: #4a90d9; }}
  input[type=text] {{ padding: 0.3rem 0.6rem; border: 1px solid #ccc;
                      border-radius: 4px; font-size: 0.85rem; width: 250px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 6px;
           overflow: hidden; margin-bottom: 1.5rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.8rem; border-bottom: 1px solid #eee; }}
  th {{ background: #4a90d9; color: #fff; cursor: pointer; user-select: none; }}
  th:hover {{ background: #3a7bc8; }}
  tr.hidden {{ display: none; }}
  .conditions {{ font-family: monospace; font-size: 0.82rem; }}
</style>
</head>
<body>
<h1>TRACE-XAI &mdash; Explanation Report</h1>

<div class="disclaimer" style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 1rem; margin-bottom: 1.5rem; font-size: 0.85rem;">
  <strong>Interpretability Disclaimer:</strong> The rules below describe the
  <em>behaviour</em> of the surrogate approximation, not causal relationships in the data.
  A rule such as &ldquo;IF age &gt; 45 THEN deny&rdquo; means the surrogate model
  <em>behaves as if</em> this pattern holds &mdash; it does not imply that age causally
  determines the outcome. Fidelity indicates how closely the surrogate matches the
  black-box model, not how well it captures the true decision logic. Two surrogates
  with identical fidelity can produce different rules. Always consider stability
  metrics alongside fidelity when assessing rule reliability.
</div>

<div class="report">{report_text}</div>

<h2>Rules</h2>
<div class="controls">
  {filter_buttons}
  <input type="text" id="search" placeholder="Search features...">
</div>
{main_table}

{pruning_section}
{mono_section}
{ensemble_section}

<script>
(function() {{
  const table = document.getElementById('rules-table');
  const tbody = table.querySelector('tbody');
  const rows = () => Array.from(tbody.querySelectorAll('tr'));
  let activeClass = 'all';

  // Filter buttons
  document.querySelectorAll('.filter-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelector('.filter-btn.active').classList.remove('active');
      btn.classList.add('active');
      activeClass = btn.dataset.class;
      applyFilters();
    }});
  }});

  // Search
  const searchInput = document.getElementById('search');
  searchInput.addEventListener('input', applyFilters);

  function applyFilters() {{
    const q = searchInput.value.toLowerCase();
    rows().forEach(row => {{
      const matchClass = activeClass === 'all' || row.dataset.class === activeClass;
      const matchSearch = !q || row.textContent.toLowerCase().includes(q);
      row.classList.toggle('hidden', !(matchClass && matchSearch));
    }});
  }}

  // Sortable headers for all tables
  document.querySelectorAll('table').forEach(tbl => {{
    tbl.querySelectorAll('th').forEach((th, idx) => {{
      th.addEventListener('click', () => {{
        const isNum = th.dataset.sort === 'num';
        const tb = tbl.querySelector('tbody');
        const sorted = Array.from(tb.querySelectorAll('tr')).sort((a, b) => {{
          let va = a.children[idx].textContent.replace('%','');
          let vb = b.children[idx].textContent.replace('%','');
          if (isNum) return (parseFloat(vb) || 0) - (parseFloat(va) || 0);
          return va.localeCompare(vb);
        }});
        sorted.forEach(r => tb.appendChild(r));
      }});
    }});
  }});
}})();
</script>
</body>
</html>"""
