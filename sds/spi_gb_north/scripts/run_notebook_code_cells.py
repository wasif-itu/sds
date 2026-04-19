import json
from pathlib import Path


NOTEBOOK_PATH = Path(
    r"C:\Users\AtifA\Desktop\Wasif SDS Project\sds\sds\spi_gb_north\visualize_spi_roads.ipynb"
)


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    ns: dict[str, object] = {"__name__": "__main__"}

    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        print(f"\n--- Running code cell {idx} ---")
        exec(compile(source, f"{NOTEBOOK_PATH.name}#cell{idx}", "exec"), ns)


if __name__ == "__main__":
    main()
