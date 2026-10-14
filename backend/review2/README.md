# Review 2 / Lab Assessment 9 report

Generated deliverable: `Pagewise_Review2_Report_24BDS0316.docx` (A4, Times New Roman 12 pt,
numbered headings, table of contents, captioned figures and tables).

## Regenerate

From the repository root (requires `matplotlib` and `python-docx`):

```bash
python backend/review2/diagrams.py      # figures/fig1..fig5 (300 dpi, ~8-inch canvases)
python backend/review2/build_report.py  # the .docx
```

Figures 4 and 5 and every results table are computed from `public/pagewise-dashboard.json`;
nothing numerical is typed by hand. Prose cross-references to tables use keys
(`T("runtime")`), so adding, removing or re-ordering tables cannot leave a stale "Table N".

## Before submitting

1. Fill the yellow placeholders on the title page (School/Department, Course Code – Title,
   Faculty, Slot) and confirm the institution and programme lines.
2. Open in Microsoft Word and accept the prompt to update fields (or select all and press F9)
   so the table of contents and "Page X of Y" are populated.
3. Re-run `python -m pytest -q` from `backend/` and `python scripts/verify_frozen.py`, then
   update the verification sentence at the end of Section 9 if anything changed.
4. Optional cuts if a page limit applies: Appendix A (CLI reference), Appendix B
   (verification checklist), Sections 8.4–8.5. Nothing else refers to them.
