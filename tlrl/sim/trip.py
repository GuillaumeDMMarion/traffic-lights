"""
Source, destination, trip and route file generator & remover.
"""

import os
import warnings
from pathlib import Path
from typing import List, Tuple
from subprocess import DEVNULL, STDOUT, call

SUMO_HOME = Path(
    os.environ.get("SUMO_HOME", Path(__file__).resolve().parent.parent.parent)
)
RANDOMTRIPS_FILEPATH = SUMO_HOME / "tools" / "randomTrips.py"
WORKING_DIR = Path.cwd()


def delete(prefix: str, cwd: Path = WORKING_DIR) -> None:
    """
    Attempts to delete the existing source/destination files, as well as the trips files.

    Args:
        prefix: Prefix of the (to-be-deleted) files' names.
        cwd: Current working directory.
    """
    try:
        # The scr and dst values
        call(
            "del " + prefix + ".src.xml",
            shell=True,
            cwd=cwd,
            stdout=DEVNULL,
            stderr=STDOUT,
        )
        call(
            "del " + prefix + ".dst.xml",
            shell=True,
            cwd=cwd,
            stdout=DEVNULL,
            stderr=STDOUT,
        )
    except Exception as e:
        print(e)
        warnings.warn("SRC/DST Files not found.")
    try:
        # The generated routes
        call("del trips.trips.xml", shell=True, cwd=cwd, stdout=DEVNULL, stderr=STDOUT)
        call(
            "del " + prefix + ".rou.alt.xml",
            shell=True,
            cwd=cwd,
            stdout=DEVNULL,
            stderr=STDOUT,
        )
        call(
            "del " + prefix + ".rou.xml",
            shell=True,
            cwd=cwd,
            stdout=DEVNULL,
            stderr=STDOUT,
        )
    except Exception as e:
        print(e)
        warnings.warn("TRIP/ROUTE Files not found.")


def generate(
    prefix: str,
    src: List[str],
    dst: List[str],
    rng,
    scale: Tuple[float, float] = (0, 0),
    cwd: Path = WORKING_DIR,
) -> None:
    """
    Generates new source/destination files as well as trips files.

    Args:
        prefix: Prefix of the (to-be-generated) files' names.
        src: List of the names of the source edges to be used.
        dst: List of the names of the destination edges to be used.
        rng: Random number generator.
        scale: Tuple of 2 std.dev.
    """
    # SRC GENERATION
    a = rng.normal(loc=100, scale=scale[0], size=len(src))
    a = a / a.sum()
    filepath = cwd / (prefix + ".src.xml")
    file = open(filepath, "w")
    startstring = (
        str("""<edgedata>""")
        + str("""\n""")
        + str(""" <interval begin="0" end="3600"/>""")
    )
    midstring = ""
    for src_id, weight in zip(src, a):
        midstring += str("""\n""") + str(
            '''  <edge id="''' + str(src_id) + '''" value="''' + str(weight) + """"/>"""
        )
    endstring = (
        str("""\n""") + str(""" </interval>""") + str("""\n""") + str("""</edgedata>""")
    )
    file.write(startstring + midstring + endstring)
    file.close()

    # DST GENERATION
    a = rng.normal(loc=100, scale=scale[1], size=len(dst))
    a = a / a.sum()
    filepath = cwd / (prefix + ".dst.xml")
    file = open(filepath, "w")
    startstring = (
        str("""<edgedata>""")
        + str("""\n""")
        + str(""" <interval begin="0" end="3600"/>""")
    )
    midstring = ""
    for dst_id, weight in zip(dst, a):
        midstring += str("""\n""") + str(
            '''  <edge id="''' + str(dst_id) + '''" value="''' + str(weight) + """"/>"""
        )
    endstring = (
        str("""\n""") + str(""" </interval>""") + str("""\n""") + str("""</edgedata>""")
    )
    file.write(startstring + midstring + endstring)
    file.close()

    net_filepath = cwd / (prefix + ".net.xml")
    rou_filepath = cwd / (prefix + ".rou.xml")
    trips_filepath = cwd / (prefix + ".trips.xml")
    randint = str(rng.integers(0, 1e6))
    # ROUTE GENERATION
    # -b: beginning period of route creation in seconds.
    # -e: ending period of route creation in seconds.
    # -p: creation of a route from -b to -e every -p periods.
    # Removed params:
    ## -a vtype.cfg
    ## --trip-attributes type=\'base\'"
    command = f"python \"{RANDOMTRIPS_FILEPATH}\" -n {net_filepath} -r {rou_filepath} -o {trips_filepath} -b 0 -e 3600 -p 1.5 -s {randint} -l --weights-prefix {prefix}  --trip-attributes departSpeed='max'"
    call(command, shell=True, cwd=cwd, stdout=DEVNULL, stderr=STDOUT)
