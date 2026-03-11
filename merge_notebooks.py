import json

MAIN = "main.ipynb"
FINAL = "final_notebook.ipynb"

with open(MAIN, "r", encoding="utf-8") as f:
    main_nb = json.load(f)

with open(FINAL, "r", encoding="utf-8") as f:
    final_nb = json.load(f)

# append cells
main_nb["cells"].extend(final_nb.get("cells", []))

# keep main metadata/format as-is (optional: you can also merge metadata if you want)
with open(MAIN, "w", encoding="utf-8") as f:
    json.dump(main_nb, f, ensure_ascii=False, indent=1)

print(f"Appended {len(final_nb.get('cells', []))} cells from {FINAL} into {MAIN}.")