import sys
from pathlib import Path

from PIL import Image

import build_patch_review_v2 as qa


source = qa.ROOT / "F971" / "OK" / "images" / "THAN_T_D_3_2026-09-14-11-02-26.jpg"
candidates = qa.best_clean_candidates(source, 1024, 128, 20, [(2176, 512, 1024)])
rows = []
labels = []
for index, candidate in enumerate(candidates, 2):
    name = f"clean_replacement_patch120_alt{index}_x{candidate['crop_x']}_y{candidate['crop_y']}.png"
    path = qa.OUT / "candidate_patches_rgb" / name
    with Image.open(source) as image:
        image.crop((candidate["crop_x"], candidate["crop_y"], candidate["crop_x"] + 1024, candidate["crop_y"] + 1024)).save(path, format="PNG", compress_level=1)
    rows.append({"patch_id": str(index), "path": path})
    labels.append((f"P120 alt {index} | x{candidate['crop_x']} y{candidate['crop_y']} | black={candidate['near_black_pct']:.1f}%",
                   source.name, "inspect clean surface"))

qa.tile_sheet(rows, qa.OUT / "contact_sheets" / "clean_patch120_alternatives.png", 512, 2, labels)
print([(r["patch_id"], labels[i][0]) for i, r in enumerate(rows)])
