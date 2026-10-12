"""The tile submitter (``slurm/feed_tiles.sbatch``), run as the scheduler runs it, against
a stand-in ``sbatch`` and ``squeue`` that keep a queue in a file.

The stand-in ``sbatch`` puts one line per array task into the queue and notes how long
the queue is afterwards; the stand-in ``squeue`` lets a few tasks end each time it is
asked, then prints what is left. So the ceiling is observed at every submission, and a
test can refuse a call, or end the submitter in the middle of one.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.regbench import failure_log, stages

REPO = Path(__file__).resolve().parents[3]
SLURM = REPO / "benchmarks" / "regbench" / "slurm"
ALLOC = [
    "{cpus}:{mem_gb}:{time}".format(**stages.allocation("tile", a))
    for a in (1, 2, 3, 4)
]

SBATCH = r"""#!/bin/bash
# sbatch: records the call, queues one line per array task, may refuse or end its caller
name="" spec=""
for a in "$@"; do
    case "$a" in --job-name=*) name="${a#--job-name=}" ;; --array=*) spec="${a#--array=}" ;; esac
done
calls=$(( $(cat "$FAKE/n" 2>/dev/null || echo 0) + 1 ))
echo "$calls" > "$FAKE/n"
if [ -e "$FAKE/refuse_from" ] && [ "$calls" -ge "$(cat "$FAKE/refuse_from")" ]; then
    echo "sbatch: error: QOSMaxSubmitJobPerUserLimit" >&2
    exit 1
fi
id=$((5000 + calls))
count=0
IFS=, read -ra parts <<< "$spec"
for p in "${parts[@]}"; do
    lo="${p%%-*}" hi="${p##*-}"
    for i in $(seq "$lo" "$hi"); do echo "$name" >> "$FAKE/queue"; count=$((count + 1)); done
done
printf '%s\t%s\n' "$id" "$*" >> "$FAKE/calls"
echo "$(wc -l < "$FAKE/queue" | tr -d ' ')" >> "$FAKE/sizes"
if [ -e "$FAKE/kill_at" ] && [ "$calls" = "$(cat "$FAKE/kill_at")" ]; then
    rm "$FAKE/kill_at"
    kill -9 "$(cat "$FAKE/feeder.pid")"
    sleep 1
fi
echo "$id;cluster"
"""

SQUEUE = r"""#!/bin/bash
# squeue: by name (-n) the id of a job that is still queued; else what is in the queue,
# after letting $FAKE/ends tasks end
name=""
while [ $# -gt 0 ]; do case "$1" in -n) name="$2"; shift 2 ;; *) shift ;; esac; done
[ -e "$FAKE/squeue_fails" ] && exit 1
touch "$FAKE/queue"
if [ -n "$name" ]; then
    grep -qx "$name" "$FAKE/queue" && grep "job-name=$name " "$FAKE/calls" | cut -f1 | head -n 1
    exit 0
fi
echo x >> "$FAKE/looks"
ends="$(cat "$FAKE/ends" 2>/dev/null || echo 1000000)"
tail -n +"$((ends + 1))" "$FAKE/queue" > "$FAKE/queue.new"
mv "$FAKE/queue.new" "$FAKE/queue"
cat "$FAKE/queue"
"""


class Site:
    def __init__(self, tmp, due, rounds=3):
        self.tmp, self.fake, self.out = tmp, tmp / "bin", tmp / "out"
        self.fake.mkdir()
        for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE)):
            (self.fake / name).write_text(text)
            (self.fake / name).chmod(0o755)
        self.state = stages.feed_state(self.out, "S-1", "stare", "multiplex")
        self.layout = Path(f"{self.state}.tsv")
        self.n_tiles = stages.write_layout(
            self.layout, [(f"r{i}", 400) for i in range(rounds)]
        )
        self.rounds = rounds
        (self.out / "logs" / "gates").mkdir(parents=True)
        (self.out / "logs" / "gates" / "stare.tsv").write_text("stare\tok\t-\t0\t-\n")
        for attempt, lines in due.items():
            self.due(attempt, lines)

    def due(self, attempt, lines):
        Path(f"{self.state}.due.a{attempt}").write_text(
            "".join(f"{i}\n" for i in lines)
        )

    def env(self, **more):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("REGBENCH_", "SLURM_"))
        }
        env.update(
            PATH=f"{self.fake}:{Path(sys.executable).parent}:{env['PATH']}",
            FAKE=str(self.fake),
            REGBENCH_REPO=str(REPO),
            REGBENCH_CASES=str(self.tmp / "cases"),
            REGBENCH_OUT=str(self.out),
            REGBENCH_SUBMISSION="S-1",
            REGBENCH_STAGE_POLL_S="0",
            SLURM_JOB_ID="77",
            SLURM_JOB_NAME="rb-stare-multiplex-feed-1",
        )
        env.update({k: str(v) for k, v in more.items()})
        return env

    def run(self, link=1, **more):
        """One link of the chain, as the launcher submits it."""
        args = [
            "stare", "stare", "-", "-", "STARE_REGBENCH", "multiplex",
            str(self.layout), str(self.rounds), str(self.n_tiles), str(link), *ALLOC,
        ]  # fmt: skip
        script = f'echo $$ > "$FAKE/feeder.pid"; exec bash {shlex.quote(str(SLURM / "feed_tiles.sbatch"))} "$@"'
        return subprocess.run(
            ["bash", "-c", script, "feed", *args],
            env=self.env(**more),
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=120,
        )

    def calls(self):
        """``[(job name, attempt, [indices], words)]`` of every sbatch call so far."""
        out = []
        path = self.fake / "calls"
        for line in path.read_text().splitlines() if path.exists() else []:
            words = shlex.split(line.split("\t", 1)[1])
            opt = {
                w.split("=", 1)[0]: w.split("=", 1)[1]
                for w in words
                if w.startswith("--") and "=" in w
            }
            script = next(i for i, w in enumerate(words) if w.endswith("stage.sbatch"))
            args = words[script + 1 :]
            attempt = int(args[7])
            base = int(args[11]) if attempt == 1 else 0
            indices = []
            for part in opt["--array"].split(","):
                lo, _, hi = part.partition("-")
                indices += [base + i for i in range(int(lo), int(hi or lo) + 1)]
            out.append((opt["--job-name"], attempt, indices, words))
        return out

    def tiles(self, attempt=1):
        return [i for _, a, idx, _ in self.calls() if a == attempt for i in idx]

    def sizes(self):
        path = self.fake / "sizes"
        return [int(v) for v in path.read_text().split()] if path.exists() else []


def last(done):
    return done.stdout.strip().splitlines()[-1]


# ── the ceiling, and every tile once ──────────────────────────────────────────
@pytest.mark.parametrize(
    "n, most, ceiling, arrays",
    [
        (1, 100, 100, [1]),  # one tile: an array of one task
        (100, 100, 100, [100]),  # exactly the maximum: one array
        (101, 100, 100, [100, 1]),  # one more: a second array
        (250, 100, 100, [100, 100, 50]),
        (30, 7, 20, None),
        (64, 100, 10, None),  # the ceiling below the array maximum
    ],
)
def test_every_tile_is_submitted_once_and_the_ceiling_is_never_exceeded(
    tmp_path, n, most, ceiling, arrays
):
    site = Site(tmp_path, {1: range(n), 2: []})
    done = site.run(
        REGBENCH_STAGE_ARRAY_MAX=most,
        REGBENCH_STAGE_QUEUE_MAX=ceiling,
        REGBENCH_STAGE_FEED_MIN=min(25, ceiling),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert sorted(site.tiles()) == list(range(n))  # each exactly once
    calls = site.calls()
    assert all(len(idx) <= most for _, _, idx, _ in calls)
    assert max(site.sizes()) <= ceiling
    if arrays:
        assert [len(idx) for _, _, idx, _ in calls] == arrays
    # one job per tile, with the pipeline's allocation for that process
    want = stages.allocation("tile", 1)
    for name, _, _, words in calls:
        assert f"--cpus-per-task={want['cpus']}" in words
        assert f"--mem={want['mem_gb']}G" in words and f"--time={want['time']}" in words
        assert name.startswith("rb-stare-multiplex-tile-a1-o")
        assert "--wait" not in words
        assert not [w for w in words if w.startswith("--array=") and "%" in w]
        assert words[-2] == str(site.layout)
    assert "REGBENCH_END kind=submitter" in last(done) and "status=ok" in last(done)
    assert Path(f"{site.state}.done").exists()
    # every job it submitted is with the launcher's: one file to cancel a submission
    ids = (site.out / "logs" / "submitted" / "S-1.ids").read_text().split()
    assert len(ids) == len(calls) == len(set(ids))
    recorded = (site.out / "logs" / "jobs.tsv").read_text().splitlines()
    assert [ln.split("\t")[1:] for ln in recorded] == [
        [name, i] for (name, *_), i in zip(calls, ids)
    ]


def test_it_waits_for_room_and_fills_what_is_free(tmp_path):
    """Three tasks end between two looks: nothing is submitted until FEED_MIN slots are
    free, and then only as many tasks as there is room for."""
    site = Site(tmp_path, {1: range(40), 2: []})
    (site.fake / "ends").write_text("3")
    done = site.run(
        REGBENCH_STAGE_ARRAY_MAX=10,
        REGBENCH_STAGE_QUEUE_MAX=10,
        REGBENCH_STAGE_FEED_MIN=5,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    sizes = [len(idx) for _, _, idx, _ in site.calls()]
    assert sizes[0] == 10 and all(5 <= s <= 10 for s in sizes[:-1])
    assert sorted(site.tiles()) == list(range(40)) and max(site.sizes()) <= 10


def test_a_sparse_due_list_is_fed_by_index_and_an_array_stays_in_its_span(tmp_path):
    """A second submission: only the tiles without a reusable record are due. The array
    names them by their distance from its first one, which stays below the span."""
    due = [3, 4, 5, 9, 1100, 1101, 1199]
    site = Site(tmp_path, {1: due, 2: []})
    done = site.run()
    assert done.returncode == 0, done.stdout + done.stderr
    calls = site.calls()
    specs = [next(w for w in words if w.startswith("--array=")) for *_, words in calls]
    assert specs == ["--array=0-2,6", "--array=0-1,99"]
    assert [words[-1] for *_, words in calls] == ["3", "1100"]
    assert site.tiles() == due


def test_a_retry_job_is_one_task_per_round_with_the_attempts_allocation(tmp_path):
    site = Site(tmp_path, {1: range(5), 2: [0, 2], 3: [2], 4: []})
    done = site.run()
    assert done.returncode == 0, done.stdout + done.stderr
    by = {a: (name, idx, words) for name, a, idx, words in site.calls()}
    assert sorted(by) == [1, 2, 3]  # attempt 4: nothing is due, nothing submitted
    assert by[2][1] == [0, 2] and by[3][1] == [2]
    for attempt in (2, 3):
        want = stages.allocation("tile", attempt)
        name, _, words = by[attempt]
        assert name == f"rb-stare-multiplex-tile-a{attempt}-o0"
        assert f"--mem={want['mem_gb']}G" in words and f"--time={want['time']}" in words
        script = next(i for i, w in enumerate(words) if w.endswith("stage.sbatch"))
        # no layout and no offset: the task's index is the round
        assert words[script + 1 :][6:] == [
            "tile", str(attempt), str(want["mem_gb"]), want["time"]
        ]  # fmt: skip
    assert "attempt 4: nothing is due" in done.stdout


def test_what_is_due_is_asked_of_the_harness_and_everything_is_fed_when_it_cannot_say(
    tmp_path,
):
    site = Site(tmp_path, {}, rounds=2)
    site.n_tiles = 6
    python = site.fake / "python"
    python.write_text('#!/bin/bash\necho "$*" >> "$FAKE/python.log"\nexit 1\n')
    python.chmod(0o755)
    done = site.run()
    assert done.returncode == 0, done.stdout + done.stderr
    asked = (site.fake / "python.log").read_text().splitlines()
    assert len(asked) == 4 and all(
        f"--layout {site.layout} --out {site.out} --due {a}" in line
        for a, line in zip((1, 2, 3, 4), asked)
    )
    assert asked[0].startswith("-m benchmarks.regbench stages --cases ")
    assert done.stdout.count("cannot list what is due") == 4
    # attempt 1: every tile; attempts 2 to 4: every round
    assert sorted(site.tiles(1)) == list(range(6))
    assert [site.tiles(a) for a in (2, 3, 4)] == [[0, 1]] * 3


# ── a blocked run, a refusal, a restart ───────────────────────────────────────
def test_a_blocked_run_submits_no_tile(tmp_path):
    site = Site(tmp_path, {1: range(4), 2: []})
    (site.out / "logs" / "gates" / "stare.tsv").write_text(
        "stare\tinconsistent\t-\t1\t-\n"
    )
    done = site.run()
    assert done.returncode == 0 and site.calls() == []
    assert "status=blocked cause=gate_blocked whose=launcher" in last(done)


def test_a_refusal_is_recorded_as_the_launchers_and_a_later_link_goes_on(tmp_path):
    site = Site(tmp_path, {1: range(25), 2: []})
    (site.fake / "refuse_from").write_text("3")
    env = dict(
        REGBENCH_STAGE_ARRAY_MAX=10,
        REGBENCH_STAGE_FEED_MIN=1,
        REGBENCH_STAGE_SUBMIT_TRIES=2,
    )
    done = site.run(**env)
    assert done.returncode == 1
    assert sorted(site.tiles()) == list(range(20))  # two arrays went through
    end = dict(w.split("=", 1) for w in last(done).split()[1:])
    assert (end["status"], end["cause"], end["whose"]) == (
        "refused", "not_submitted", "launcher"
    )  # fmt: skip
    assert failure_log.classify("not_submitted") == failure_log.LAUNCHER
    refusal = Path(f"{site.state}.refused").read_text()
    assert "QOSMaxSubmitJobPerUserLimit" in refusal
    assert "rb-stare-multiplex-tile-a1-o20, 5 task(s)" in refusal
    assert f'scancel $(cat "{site.out}/logs/submitted/S-1.ids")' in refusal
    assert "tried" not in refusal and done.stdout.count("sbatch refused") == 2
    assert not Path(f"{site.state}.done").exists()
    # what a round whose tiles never reached the queue is told
    said = stages.feed_problem(site.out, "S-1", "stare", "multiplex")
    assert said.startswith("the tile submitter was refused by the scheduler")
    # the next link of the chain: the limit is gone, the rest is fed, nothing twice
    (site.fake / "refuse_from").unlink()
    done = site.run(link=2, **env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert sorted(site.tiles()) == list(range(25))
    assert not Path(f"{site.state}.refused").exists()
    assert stages.feed_problem(site.out, "S-1", "stare", "multiplex") == ""
    # and a third has nothing to do
    before = len(site.calls())
    done = site.run(link=3, **env)
    assert done.returncode == 0 and len(site.calls()) == before
    assert "nothing to do" in done.stdout and "status=ok" in last(done)


@pytest.mark.parametrize("ends", ["1000000", "0"])
def test_a_submitter_that_is_killed_is_continued_from_its_files(tmp_path, ends):
    """The first link is ended inside its second `sbatch`, after the scheduler took the
    array and before its id was written down. The second link finds that array by name
    while it is in the queue (`ends` 0), or submits it again when it is found nowhere
    (it ended meanwhile, and there is no accounting here): never a tile left out."""
    site = Site(tmp_path, {1: range(35), 2: []})
    (site.fake / "kill_at").write_text("2")
    (site.fake / "ends").write_text("0")
    env = dict(REGBENCH_STAGE_ARRAY_MAX=10, REGBENCH_STAGE_QUEUE_MAX=1000)
    done = site.run(**env)
    assert done.returncode == -9
    fed = Path(f"{site.state}.fed.tsv").read_text().splitlines()
    assert fed[-1] == "1\t10\t10\t10\t-" and len(site.calls()) == 2
    assert stages.feed_problem(site.out, "S-1", "stare", "multiplex").startswith(
        "the tile submitter did not finish"
    )
    (site.fake / "ends").write_text(ends)
    if ends == "0":
        # nothing ends: the queue must be let go once the tiles are all in it
        (site.fake / "squeue").write_text(
            SQUEUE.replace(
                'ends="$(cat',
                '[ "$(wc -l < "$FAKE/queue")" -ge 35 ] && echo 1000000 > "$FAKE/ends"\n'
                'ends="$(cat',
            )
        )
    else:
        (site.fake / "queue").write_text("")  # both arrays ended meanwhile
    done = site.run(link=2, **env)
    assert done.returncode == 0, done.stdout + done.stderr
    tiles = site.tiles()
    if ends == "0":
        assert sorted(tiles) == list(range(35))  # found by name: nothing twice
        assert "submitted again" not in done.stdout
    else:
        assert sorted(set(tiles)) == list(range(35))
        assert sorted(i for i in tiles if tiles.count(i) > 1) == sorted(
            [*range(10, 20)] * 2
        )
        assert "that array is submitted again" in done.stdout
    assert Path(f"{site.state}.done").exists()


def test_a_queue_that_cannot_be_read_is_not_an_empty_queue(tmp_path):
    site = Site(tmp_path, {1: range(4), 2: []})
    (site.fake / "squeue_fails").write_text("")
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ["bash", str(SLURM / "feed_tiles.sbatch"), "stare", "stare", "-", "-",
             "STARE_REGBENCH", "multiplex", str(site.layout), "3", "4", "1", *ALLOC],
            env=site.env(REGBENCH_STAGE_POLL_S="1"),
            capture_output=True,
            cwd=REPO,
            timeout=4,
        )  # fmt: skip
    assert site.calls() == []


def test_the_defaults_are_the_ruled_ones():
    text = (SLURM / "feed_tiles.sbatch").read_text()
    assert f': "${{REGBENCH_STAGE_ARRAY_MAX:={stages.ARRAY_MAX}}}"' in text
    assert f': "${{REGBENCH_STAGE_QUEUE_MAX:={stages.QUEUE_MAX}}}"' in text
    assert stages.ARRAY_MAX == stages.QUEUE_MAX == 100
    launcher = (SLURM / "submit_cohort.sh").read_text()
    assert f': "${{REGBENCH_STAGE_ARRAY_MAX:={stages.ARRAY_MAX}}}"' in launcher
    assert f': "${{REGBENCH_STAGE_QUEUE_MAX:={stages.QUEUE_MAX}}}"' in launcher
