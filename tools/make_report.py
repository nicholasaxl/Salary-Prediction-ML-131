"""Write salary_analysis_report.ipynb: the executed notebook with every code cell emptied.

Outputs (charts, tables, text) stay, so the file reads as a report on GitHub; the full code lives in
salary_analysis.ipynb. Run after executing the notebook:  python tools/make_report.py
"""
import nbformat as nbf

nb = nbf.read("salary_analysis.ipynb", as_version=4)
for c in nb.cells:
    if c.cell_type == "code":
        c.source = ""
        c.metadata["jupyter"] = {"source_hidden": True}
        c.execution_count = None
for c in nb.cells:
    if c.cell_type == "code":
        for o in c.get("outputs", []):
            if o.get("output_type") == "execute_result":
                o["execution_count"] = None
nbf.write(nb, "salary_analysis_report.ipynb")
print("wrote salary_analysis_report.ipynb")
