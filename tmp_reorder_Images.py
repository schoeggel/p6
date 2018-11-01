import shutil
from pathlib import Path

src = Path("E:/trainscannerHackathon/Hackathon")
dst = Path("E:/trainscannerJanReordered")

for d in src.iterdir():
    print(d)
    for f in d.iterdir():
        if f.stem == "OK1L" or f.stem == "OK1R":
            dstname = f.parent.name + "-" + f.name
            print(dstname)
            shutil.copyfile(f, dst/dstname)

