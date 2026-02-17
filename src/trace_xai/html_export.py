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
    rules_data = []
    for rule in result.rules.rules:
        conditions = " AND ".join(str(c) for c in rule.conditions) or "TRUE"
        rules_data.append({
            "conditions": html.escape(conditions),
            "prediction": html.escape(str(rule.prediction)),
            "confidence": rule.confidence,
            "samples": rule.samples,
            "is_regression": rule.prediction_value is not None,
            "prediction_value": rule.prediction_value,
        })

    report_text = html.escape(str(result.report))

    page = _build_html(rules_data, report_text, result._task)

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(page)


def _build_html(rules_data: list[dict], report_text: str, task: str) -> str:
    """Construct the full HTML string."""
    # Build table rows
    rows = []
    for i, r in enumerate(rules_data, 1):
        if r["is_regression"]:
            pred_cell = f"value = {r['prediction_value']:.4f}"
            conf_cell = ""
        else:
            pred_cell = r["prediction"]
            conf_cell = f"{r['confidence']:.2%}"
        rows.append(
            f"<tr data-class=\"{r['prediction']}\">"
            f"<td>{i}</td>"
            f"<td class=\"conditions\">{r['conditions']}</td>"
            f"<td class=\"prediction\">{pred_cell}</td>"
            f"<td>{conf_cell}</td>"
            f"<td>{r['samples']}</td>"
            f"</tr>"
        )
    table_rows = "\n".join(rows)

    # Collect unique classes for filter buttons
    classes = sorted({r["prediction"] for r in rules_data})
    filter_buttons = '<button class="filter-btn active" data-class="all">All</button>\n'
    for cls in classes:
        filter_buttons += f'<button class="filter-btn" data-class="{cls}">{cls}</button>\n'

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
           overflow: hidden; }}
  th, td {{ text-align: left; padding: 0.5rem 0.8rem; border-bottom: 1px solid #eee; }}
  th {{ background: #4a90d9; color: #fff; cursor: pointer; user-select: none; }}
  th:hover {{ background: #3a7bc8; }}
  tr.hidden {{ display: none; }}
  .conditions {{ font-family: monospace; font-size: 0.82rem; }}
</style>
</head>
<body>
<h1>Explainer General &mdash; Report</h1>
<div class="report">{report_text}</div>

<h2>Rules</h2>
<div class="controls">
  {filter_buttons}
  <input type="text" id="search" placeholder="Search features...">
</div>
<table id="rules-table">
  <thead>
    <tr>
      <th data-sort="num">#</th>
      <th>Conditions</th>
      <th>Prediction</th>
      <th data-sort="num">Confidence</th>
      <th data-sort="num">Samples</th>
    </tr>
  </thead>
  <tbody>
    {table_rows}
  </tbody>
</table>

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

  // Sortable headers
  table.querySelectorAll('th').forEach((th, idx) => {{
    th.addEventListener('click', () => {{
      const isNum = th.dataset.sort === 'num';
      const sorted = rows().sort((a, b) => {{
        let va = a.children[idx].textContent.replace('%','');
        let vb = b.children[idx].textContent.replace('%','');
        if (isNum) return (parseFloat(vb) || 0) - (parseFloat(va) || 0);
        return va.localeCompare(vb);
      }});
      sorted.forEach(r => tbody.appendChild(r));
    }});
  }});
}})();
</script>
</body>
</html>"""
