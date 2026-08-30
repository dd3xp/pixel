"""Record WHO kills our GPU jobs, and from where.

Everything we launch here has been wiped repeatedly.  Ordinary signal handlers
do not reveal the sender, but sigwaitinfo() returns siginfo_t with si_pid, so
this process holds a little GPU memory (the killer selects victims by GPU use)
and writes down whoever signals it.

Two kills are already on record, both uid 1001 -- the shared account we are on:
  bash -c 'set -eu; echo PIXEL_TREE_BEFORE; ps -eo pid,ppid,user,stat,cmd'
  bash -c 'kill -TERM 2328575 2328582 2328691 ...'
Program-generated, not hand-typed, and absent from ~/.bash_history -- which is
what a non-interactive `ssh host "bash -c ..."` looks like from this side.  So
the remaining question is which machine it comes from, and /proc/<pid>/environ
answers it: a process spawned by sshd inherits SSH_CONNECTION (origin IP, port,
destination).  Capture that, plus the whole ancestor chain, before the sender
exits -- these are short-lived, so everything is read immediately on wake.
"""
import os
import signal
import sys
from datetime import datetime, timezone

import torch

LOG = "/mnt/data/kw/RoundSquisheen/pixel/pixel/logs/sentinel.log"
WATCH = [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGUSR1]
INTERESTING = ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "SSH_ORIGINAL_COMMAND",
               "USER", "LOGNAME", "TERM_PROGRAM", "CLAUDE_CODE_SSE_PORT",
               "TMUX", "STY", "CONDA_PREFIX", "PWD")


def note(msg):
    line = f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}] GPU-SENTINEL {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def read(pid, name):
    try:
        with open(f"/proc/{pid}/{name}", "rb") as f:
            return f.read()
    except OSError:
        return b""


def describe(pid, depth=0):
    """cmdline + the env vars that identify where a process came from, walking
    up the ancestor chain: the immediate sender is often a short-lived `bash -c`
    whose parent is the sshd session that carries SSH_CONNECTION."""
    if pid <= 1 or depth > 6:
        return []
    cmd = read(pid, "cmdline").replace(b"\0", b" ").decode(errors="replace").strip()
    if not cmd:
        return [f"    pid {pid}: (exited or unreadable)"]
    env = {}
    for entry in read(pid, "environ").split(b"\0"):
        if b"=" in entry:
            k, v = entry.split(b"=", 1)
            k = k.decode(errors="replace")
            if k in INTERESTING:
                env[k] = v.decode(errors="replace")[:120]
    ppid = "?"
    for line in read(pid, "status").decode(errors="replace").splitlines():
        if line.startswith("PPid:"):
            ppid = line.split()[1]
            break
    out = [f"    pid {pid} ppid {ppid}: {cmd[:170]!r}"]
    for k, v in env.items():
        out.append(f"        {k}={v}")
    if ppid.isdigit():
        out += describe(int(ppid), depth + 1)
    return out


signal.pthread_sigmask(signal.SIG_BLOCK, WATCH)
buf = torch.zeros(128 * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")
note(f"up pid={os.getpid()} holding {buf.numel() * 2 / 2**20:.0f}MB on "
     f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")

while True:
    info = signal.sigwaitinfo(WATCH)
    note(f"*** GOT {signal.Signals(info.si_signo).name} from pid={info.si_pid} "
         f"uid={info.si_uid} -- ancestor chain follows")
    for line in describe(info.si_pid):
        note(line)
    if info.si_signo == signal.SIGQUIT:
        note("exiting on SIGQUIT")
        sys.exit(0)
