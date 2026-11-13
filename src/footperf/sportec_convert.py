"""Convert DFL / Sportec raw position XML (IDSSE dataset) to a compact CSV.

Standard library only, streaming: a 400 MB XML file never sits in memory.
Output columns: half, frame, person, team, x, y, speed_kmh (the provider's own speed).
A second file holds the ball per frame: half, frame, x, y, z, possession (1 home / 2 guest),
alive (1 in play / 0 dead ball).

    python src/footperf/sportec_convert.py data/idsse              # every positions file found
    python src/footperf/sportec_convert.py data/idsse --ball-only  # rewrite the ball files only
"""

from __future__ import annotations

import csv
import gzip
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HALF = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}


def convert(xml_path: Path, out_dir: Path, ball_only: bool = False) -> tuple[Path, Path]:
    match_id = xml_path.stem.split("DFL-MAT-")[-1]
    pos_out = out_dir / f"{match_id}_positions.csv.gz"
    ball_out = out_dir / f"{match_id}_ball.csv.gz"
    pos_target = Path("/dev/null") if ball_only else pos_out
    with gzip.open(pos_target, "wt", newline="", compresslevel=6) if not ball_only \
            else open(pos_target, "w", newline="") as fp, \
            gzip.open(ball_out, "wt", newline="", compresslevel=6) as fb:
        wp, wb = csv.writer(fp), csv.writer(fb)
        wp.writerow(["half", "frame", "person", "team", "x", "y", "speed_kmh"])
        wb.writerow(["half", "frame", "x", "y", "z", "possession", "alive"])
        half = person = team = None
        for event, el in ET.iterparse(xml_path, events=("start", "end")):
            if event == "start" and el.tag == "FrameSet":
                half = HALF.get(el.get("GameSection"), 0)
                person, team = el.get("PersonId"), el.get("TeamId")
            elif event == "end" and el.tag == "Frame":
                a = el.attrib
                if team == "BALL":
                    wb.writerow([half, a["N"], a["X"], a["Y"], a.get("Z", ""),
                                 a.get("BallPossession", ""), a.get("BallStatus", "")])
                elif not ball_only:
                    wp.writerow([half, a["N"], person, team, a["X"], a["Y"], a.get("S", "")])
                el.clear()
            elif event == "end" and el.tag == "FrameSet":
                el.clear()
    return pos_out, ball_out


if __name__ == "__main__":
    ball_only = "--ball-only" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = Path(args[0] if args else "data/idsse")
    files = [target] if target.is_file() else sorted(target.glob("DFL_04_03_positions_raw_observed_*.xml"))
    for xml in files:
        folder = xml.parent
        mid = xml.stem.split("DFL-MAT-")[-1]
        if (folder / f"{mid}_positions.csv.gz").exists() and not ball_only:
            print("already converted:", mid)
            continue
        print("converting", mid, "(ball only)" if ball_only else "", flush=True)
        convert(xml, folder, ball_only)
    print("done")
