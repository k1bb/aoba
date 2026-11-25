#!/usr/bin/env python3
"""Resilient local-Mac supervisor for multi-day Google Colab A100 AOBA 100-year spin-up.

Features:
- Runs `scripts/run_g5_30yr.py` detached (`nohup`) inside each Colab A100 VM so transient
  WebSocket disconnects in `colab` CLI never interrupt the simulation.
- Periodically syncs `spinup_monitor.csv`, `live_preview.npz`, `mesh.zarr`, completed
  `year_XXXX/` directories (including 3D OGCM Zarr v3 history & restarts), and mid-year
  `restart_mid.pt` back to local Google Drive storage.
- Automatically regenerates `ocean_spinup_dashboard.png` and `spinup_status.json`.
- Cleanly rotates Colab A100 sessions every ~21.5 h via the `STOP_AFTER_MONTH` sentinel
  (well ahead of Colab's 24 h hard limit) with zero data loss and automatic resume.
- Immediately halts and terminates the GPU VM on NaN/Inf detection or completion,
  preventing wasted Compute Units.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

COLAB_BIN = Path("/Users/andrew/.local/bin/colab")
PYTHON_BIN = Path("/Users/andrew/miniforge/envs/myenv/bin/python")
REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPT = REPO_ROOT / "scripts" / "plot_spinup_monitor.py"
EARTH2014_DIR = REPO_ROOT / "data" / "boundary" / "earth2014_5min"
EARTH2014_FILES = [
    "Earth2014.SUR2014.5min.geod.bin",
    "Earth2014.ICE2014.5min.geod.bin",
    "Earth2014.TBI2014.5min.geod.bin",
    "Earth2014.BED2014.5min.geod.bin",
]

REMOTE_REPO = "/content/aoba"
REMOTE_OUT = "/content/run_out"
REMOTE_CUDA_CACHE = "/content/cuda_cache"


class Supervisor:
    def __init__(
        self,
        *,
        session: str,
        gpu: str,
        years: int,
        out_dir: Path,
        poll_sec: int,
        max_session_hours: float,
        min_cu_reserve: float,
        config_path: str = "",
        enable_ocean_bgc: bool = False,
    ) -> None:
        self.session = session
        self.gpu = gpu
        self.years = years
        self.out_dir = out_dir.expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.poll_sec = poll_sec
        self.max_session_hours = max_session_hours
        self.min_cu_reserve = min_cu_reserve
        self.config_path = str(config_path).strip()
        self.enable_ocean_bgc = enable_ocean_bgc
        self.log_path = self.out_dir / "supervisor.log"
        self.status_path = self.out_dir / "spinup_status.json"
        self.stop_requested = False
        self.last_mid_mtime: float = 0.0
        self.mesh_synced = (self.out_dir / "mesh.zarr").exists()

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, _frame) -> None:
        self.log(f"Received signal {signum}; will cleanly stop remote VM and exit.")
        self.stop_requested = True

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def run_colab_exec(self, code: str, *, timeout_s: int = 120, retries: int = 3) -> tuple[int, str]:
        """Run a Python snippet on the active Colab session with WebSocket-disconnect guard."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(code)
            script_path = tf.name
        try:
            for attempt in range(1, retries + 1):
                p = subprocess.Popen(
                    [str(COLAB_BIN), "exec", "--timeout", str(timeout_s), "-s", self.session, "-f", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                lines: list[str] = []
                lost = False
                deadline = time.monotonic() + timeout_s + 25
                assert p.stdout is not None
                for line in p.stdout:
                    lines.append(line)
                    if "Connection was lost" in line:
                        lost = True
                        p.kill()
                        break
                    if time.monotonic() > deadline:
                        lost = True
                        p.kill()
                        break
                try:
                    rc = p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                    rc = 124
                out = "".join(lines)
                if not lost and rc == 0:
                    return 0, out
                self.log(f"colab exec attempt {attempt}/{retries} failed (lost={lost}, rc={rc})")
                if attempt < retries:
                    time.sleep(5 * attempt)
            return rc, out
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass

    def colab_upload(self, local_path: Path, remote_path: str, retries: int = 3) -> None:
        for attempt in range(1, retries + 1):
            r = subprocess.run(
                [str(COLAB_BIN), "upload", "-s", self.session, str(local_path), remote_path],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode == 0:
                return
            self.log(f"upload {local_path.name} attempt {attempt}/{retries} failed: {r.stderr.strip() or r.stdout.strip()}")
            if attempt < retries:
                time.sleep(5 * attempt)
        raise RuntimeError(f"Failed to upload {local_path} -> {remote_path}")

    def colab_download(self, remote_path: str, local_path: Path, retries: int = 3) -> bool:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_local = local_path.with_name(f".{local_path.name}.dl.tmp")
        for attempt in range(1, retries + 1):
            r = subprocess.run(
                [str(COLAB_BIN), "download", "-s", self.session, remote_path, str(tmp_local)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r.returncode == 0 and tmp_local.exists():
                os.replace(tmp_local, local_path)
                return True
            if tmp_local.exists():
                tmp_local.unlink(missing_ok=True)
            self.log(f"download {remote_path} attempt {attempt}/{retries} failed: {r.stderr.strip() or r.stdout.strip()}")
            if attempt < retries:
                time.sleep(5 * attempt)
        return False

    def get_cu_balance(self) -> tuple[float | None, float | None, int | None]:
        r = subprocess.run([str(COLAB_BIN), "usage"], capture_output=True, text=True, timeout=30)
        bal = rate = None
        active = None
        for line in r.stdout.splitlines():
            if "Current balance:" in line:
                try:
                    bal = float(line.split(":")[1].strip().split()[0])
                except Exception:
                    pass
            elif "Usage rate:" in line:
                try:
                    rate = float(line.split(":")[1].strip().split("/")[0])
                except Exception:
                    pass
            elif "Active assignments:" in line:
                try:
                    active = int(line.split(":")[1].strip())
                except Exception:
                    pass
        return bal, rate, active

    def stop_session(self) -> None:
        self.log(f"Stopping Colab session '{self.session}'...")
        subprocess.run([str(COLAB_BIN), "stop", "-s", self.session], capture_output=True, text=True, timeout=60)
        bal, rate, active = self.get_cu_balance()
        self.log(f"Colab usage after stop: balance={bal} CU | rate={rate}/hr | active={active}")

    def provision_session(self) -> None:
        # Ensure any stale session with the same name is stopped first
        subprocess.run([str(COLAB_BIN), "stop", "-s", self.session], capture_output=True, text=True, timeout=30)
        env = os.environ.copy()
        env["COLAB_CLI_RETRIES"] = "3"
        env["COLAB_CLI_RETRY_BACKOFF"] = "15"
        backoff = 30
        while not self.stop_requested:
            self.log(f"Provisioning Colab {self.gpu} session '{self.session}'...")
            r = subprocess.run(
                [str(COLAB_BIN), "new", "-s", self.session, "--gpu", self.gpu],
                env=env,
                capture_output=True,
                text=True,
                timeout=360,
            )
            if r.returncode == 0:
                bal, rate, active = self.get_cu_balance()
                self.log(f"Session '{self.session}' READY! balance={bal} CU | rate={rate}/hr | active={active}")
                return
            self.log(f"Provision failed (rc={r.returncode}): {(r.stderr or r.stdout)[-300:].strip()} — retrying in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
        raise SystemExit(0)

    def local_completed_years(self) -> list[int]:
        yrs = []
        for y in range(1, self.years + 1):
            if (self.out_dir / f"year_{y:04d}" / "_COMPLETE").exists():
                yrs.append(y)
        return yrs

    def is_target_complete(self) -> bool:
        return (self.out_dir / f"year_{self.years:04d}" / "_COMPLETE").exists()

    def setup_and_launch_remote(self) -> None:
        self.log("Setting up AOBA repo & dependencies on remote A100...")
        setup_code = f"""
import subprocess, sys, torch
from pathlib import Path
repo = Path({REMOTE_REPO!r})
if not repo.exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/k1bb/aoba.git", str(repo)], check=True)
else:
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr", "xarray", "ninja"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", str(repo)], check=True)
(repo / "data/boundary/earth2014_5min").mkdir(parents=True, exist_ok=True)
Path({REMOTE_OUT!r}).mkdir(parents=True, exist_ok=True)
Path({REMOTE_CUDA_CACHE!r}).mkdir(parents=True, exist_ok=True)
props = torch.cuda.get_device_properties(0)
print(f"REMOTE_GPU: {{props.name}} | VRAM={{props.total_memory/(1024**3):.1f}}GB | SMs={{props.multi_processor_count}}", flush=True)
"""
        rc, out = self.run_colab_exec(setup_code, timeout_s=240)
        if rc != 0:
            raise RuntimeError(f"Remote setup failed:\n{out}")
        for line in out.splitlines():
            if "REMOTE_GPU:" in line:
                self.log(line.strip())

        self.log("Uploading Earth2014 5-arcmin topography/bathymetry grids...")
        for fname in EARTH2014_FILES:
            src = EARTH2014_DIR / fname
            if not src.exists():
                raise FileNotFoundError(f"Missing local Earth2014 file: {src}")
            self.colab_upload(src, f"{REMOTE_REPO}/data/boundary/earth2014_5min/{fname}")

        # Restore resume state (if any local checkpoint exists)
        self._upload_resume_state()

        # Launch detached run_g5_30yr.py via Popen(start_new_session=True)
        self.log(f"Launching detached {self.years}-year Glevel-5 piControl spin-up on remote A100...")
        launch_code = f"""
import os, subprocess, sys
from pathlib import Path
out_dir = Path({REMOTE_OUT!r})
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "STOP_AFTER_MONTH").unlink(missing_ok=True)
env = os.environ.copy()
env["PYTHONPATH"] = "{REMOTE_REPO}/src"
env["PYTHONUNBUFFERED"] = "1"
env["AOBA_CUDA_CACHE_DIR"] = {REMOTE_CUDA_CACHE!r}
log_f = open(out_dir / "runner.log", "a")
cmd = [
    sys.executable, "-u", "{REMOTE_REPO}/scripts/run_g5_30yr.py",
    "--years", "{self.years}",
    "--out-dir", "{REMOTE_OUT}",
    "--device", "cuda",
]
if {bool(self.config_path)!r}:
    cmd.extend(["--config", "{self.config_path}"])
if {bool(self.enable_ocean_bgc)!r}:
    cmd.append("--enable-ocean-bgc")
p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd="{REMOTE_REPO}", start_new_session=True)
(out_dir / "runner.pid").write_text(str(p.pid))
print(f"LAUNCHED_PID={{p.pid}}", flush=True)
"""
        rc, out = self.run_colab_exec(launch_code, timeout_s=60)
        if rc != 0 or "LAUNCHED_PID=" not in out:
            raise RuntimeError(f"Failed to launch remote spin-up:\n{out}")
        self.log(out.strip().splitlines()[-1])

    def _upload_resume_state(self) -> None:
        # Always upload spinup_monitor.csv if it exists so timeline stays unbroken
        sp_csv = self.out_dir / "spinup_monitor.csv"
        if sp_csv.exists():
            self.colab_upload(sp_csv, f"{REMOTE_OUT}/spinup_monitor.csv")

        # Determine highest completed year & whether a newer mid-year restart exists
        completed = self.local_completed_years()
        last_comp = max(completed) if completed else 0
        mid_year = None
        for y in range(self.years, last_comp, -1):
            if (self.out_dir / f"year_{y:04d}" / "restart_mid.pt").exists():
                mid_year = y
                break

        if mid_year is not None:
            y_dir = self.out_dir / f"year_{mid_year:04d}"
            self.log(f"Packaging mid-year resume bundle for year_{mid_year:04d}...")
            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tf:
                tar_local = Path(tf.name)
            try:
                with tarfile.open(tar_local, "w") as tar:
                    tar.add(y_dir, arcname=f"year_{mid_year:04d}")
                self.colab_upload(tar_local, f"/tmp/resume_state.tar")
            finally:
                tar_local.unlink(missing_ok=True)
            rc, out = self.run_colab_exec(
                f"import tarfile; tarfile.open('/tmp/resume_state.tar').extractall({REMOTE_OUT!r}); print('EXTRACTED_MID')",
                timeout_s=120,
            )
            if rc != 0:
                raise RuntimeError(f"Failed to extract mid-year resume state:\n{out}")
            self.log(f"Uploaded mid-year resume state for year_{mid_year:04d}.")
        elif last_comp > 0:
            y_dir = self.out_dir / f"year_{last_comp:04d}"
            self.log(f"Uploading completed Year {last_comp:04d} restart.pt + _COMPLETE for seamless resume...")
            rc, _ = self.run_colab_exec(
                f"from pathlib import Path; Path('{REMOTE_OUT}/year_{last_comp:04d}').mkdir(parents=True, exist_ok=True)",
                timeout_s=30,
            )
            if rc != 0:
                raise RuntimeError("Failed to create remote resume year dir")
            self.colab_upload(y_dir / "restart.pt", f"{REMOTE_OUT}/year_{last_comp:04d}/restart.pt")
            self.colab_upload(y_dir / "_COMPLETE", f"{REMOTE_OUT}/year_{last_comp:04d}/_COMPLETE")
            self.log(f"Uploaded Year {last_comp:04d} restart state; remote run will resume at Year {last_comp + 1:04d}.")

    def probe_remote(self) -> dict:
        probe_code = f"""
import json, os, subprocess
from pathlib import Path
out = Path({REMOTE_OUT!r})
pid_f = out / "runner.pid"
pid = int(pid_f.read_text().strip()) if pid_f.exists() else -1
alive = False
if pid > 0:
    try:
        os.kill(pid, 0)
        alive = True
    except OSError:
        alive = False
completed = []
mid_info = None
for d in sorted(out.glob("year_*")):
    if not d.is_dir():
        continue
    if (d / "_COMPLETE").exists():
        completed.append(d.name)
    mp = d / "restart_mid.pt"
    if mp.exists():
        mid_info = {{"year_dir": d.name, "mtime": mp.stat().st_mtime, "size_mb": mp.stat().st_size / (1024**2)}}
log_f = out / "runner.log"
tail = log_f.read_text(errors="replace").splitlines()[-12:] if log_f.exists() else []
sp_f = out / "spinup_monitor.csv"
sp_lines = sp_f.read_text(errors="replace").splitlines() if sp_f.exists() else []
gpu_s = ""
try:
    gpu_s = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader"],
        text=True,
    ).strip()
except Exception:
    pass
info = {{
    "pid": pid,
    "alive": alive,
    "has_mesh": (out / "mesh.zarr").exists(),
    "has_preview": (out / "live_preview.npz").exists(),
    "completed_years": completed,
    "mid_info": mid_info,
    "spinup_rows": max(len(sp_lines) - 1, 0),
    "spinup_last": sp_lines[-1] if len(sp_lines) >= 2 else "",
    "log_tail": tail,
    "gpu": gpu_s,
}}
print("PROBE_JSON:" + json.dumps(info), flush=True)
"""
        rc, out = self.run_colab_exec(probe_code, timeout_s=90)
        if rc != 0:
            raise RuntimeError(f"Probe failed (rc={rc})")
        for line in out.splitlines():
            if line.startswith("PROBE_JSON:"):
                return json.loads(line[len("PROBE_JSON:"):])
        raise RuntimeError(f"Missing PROBE_JSON in output:\n{out[-400:]}")

    def sync_from_remote(self, info: dict, *, force_mid_bundle: bool = False) -> None:
        # 1. Sync mesh.zarr once
        if info.get("has_mesh") and not self.mesh_synced:
            rc, _ = self.run_colab_exec(
                f"import tarfile; tarfile.open('/tmp/mesh.tar', 'w').add('{REMOTE_OUT}/mesh.zarr', arcname='mesh.zarr'); print('OK')",
                timeout_s=120,
            )
            if rc == 0:
                tmp_tar = self.out_dir / ".mesh.tar"
                if self.colab_download("/tmp/mesh.tar", tmp_tar):
                    with tarfile.open(tmp_tar) as tf:
                        tf.extractall(self.out_dir)
                    tmp_tar.unlink(missing_ok=True)
                    self.mesh_synced = True
                    self.log("Synced mesh.zarr to local Drive.")

        # 2. Always sync spinup_monitor.csv & live_preview.npz & runner.log
        if info.get("spinup_rows", 0) > 0:
            self.colab_download(f"{REMOTE_OUT}/spinup_monitor.csv", self.out_dir / "spinup_monitor.csv")
        if info.get("has_preview"):
            self.colab_download(f"{REMOTE_OUT}/live_preview.npz", self.out_dir / "live_preview.npz")
        self.colab_download(f"{REMOTE_OUT}/runner.log", self.out_dir / "remote_runner.log")

        # 3. Sync newly completed years and free remote disk space for older years
        remote_comp: list[str] = info.get("completed_years", [])
        for yname in remote_comp:
            local_y = self.out_dir / yname
            if not (local_y / "_COMPLETE").exists():
                self.log(f"Downloading completed {yname} archive from remote A100...")
                tar_code = f"""
import tarfile
from pathlib import Path
src = Path('{REMOTE_OUT}') / {yname!r}
with tarfile.open('/tmp/{yname}.tar', 'w') as tf:
    tf.add(src, arcname={yname!r})
print('TAR_OK')
"""
                rc, _ = self.run_colab_exec(tar_code, timeout_s=180)
                if rc == 0:
                    tmp_tar = self.out_dir / f".{yname}.tar"
                    if self.colab_download(f"/tmp/{yname}.tar", tmp_tar):
                        with tarfile.open(tmp_tar) as tf:
                            tf.extractall(self.out_dir)
                        tmp_tar.unlink(missing_ok=True)
                        # Remove stale mid-year restart locally if completed
                        (local_y / "restart_mid.pt").unlink(missing_ok=True)
                        self.log(f"Synced & verified {yname}/_COMPLETE locally!")
                # Delete temporary tarball on remote
                self.run_colab_exec(f"import os; os.path.exists('/tmp/{yname}.tar') and os.unlink('/tmp/{yname}.tar')", timeout_s=30)

        # Clean older completed years on remote (keep latest completed year on remote just in case)
        if len(remote_comp) > 1:
            to_prune = [y for y in remote_comp[:-1] if (self.out_dir / y / "_COMPLETE").exists()]
            if to_prune:
                self.run_colab_exec(
                    f"import shutil; from pathlib import Path\n"
                    f"for y in {to_prune!r}: shutil.rmtree(Path('{REMOTE_OUT}') / y, ignore_errors=True)",
                    timeout_s=60,
                )

        # 4. Sync current mid-year checkpoint & partial Zarr when updated (or during session rotation)
        mid = info.get("mid_info")
        if mid and (force_mid_bundle or float(mid["mtime"]) > self.last_mid_mtime + 1.0):
            yname = mid["year_dir"]
            self.log(f"Syncing mid-year checkpoint bundle for {yname} ({mid['size_mb']:.0f} MB)...")
            tar_code = f"""
import tarfile
from pathlib import Path
src = Path('{REMOTE_OUT}') / {yname!r}
with tarfile.open('/tmp/mid_sync.tar', 'w') as tf:
    tf.add(src, arcname={yname!r})
print('MID_TAR_OK')
"""
            rc, _ = self.run_colab_exec(tar_code, timeout_s=180)
            if rc == 0:
                tmp_tar = self.out_dir / ".mid_sync.tar"
                if self.colab_download("/tmp/mid_sync.tar", tmp_tar):
                    with tarfile.open(tmp_tar) as tf:
                        tf.extractall(self.out_dir)
                    tmp_tar.unlink(missing_ok=True)
                    self.last_mid_mtime = float(mid["mtime"])
                    self.log(f"Synced mid-year checkpoint for {yname}.")
            self.run_colab_exec("import os; os.path.exists('/tmp/mid_sync.tar') and os.unlink('/tmp/mid_sync.tar')", timeout_s=30)

        # 5. Regenerate ocean spin-up dashboard PNG & status JSON
        sp_csv = self.out_dir / "spinup_monitor.csv"
        if sp_csv.exists():
            try:
                subprocess.run(
                    [str(PYTHON_BIN), str(PLOT_SCRIPT), "--csv", str(sp_csv)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except Exception as e:
                self.log(f"Dashboard plot warning: {e}")

    def check_health_nan(self) -> tuple[bool, str]:
        sp_csv = self.out_dir / "spinup_monitor.csv"
        if not sp_csv.exists():
            return True, ""
        with open(sp_csv, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < 2:
            return True, ""
        last = rows[-1]
        for key in ("ts_k", "t_ocn_global_k", "s_ocn_global_psu", "ohc_zj"):
            val_s = last.get(key, "")
            try:
                v = float(val_s)
                if not math.isfinite(v):
                    return False, f"Non-finite {key}={val_s} at Y{last.get('year')} M{last.get('month')} ({last.get('date')})"
            except Exception:
                return False, f"Invalid {key}={val_s}"
        return True, ""

    def write_status(self, info: dict, session_start: float, state: str) -> None:
        bal, rate, active = self.get_cu_balance()
        payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state": state,
            "target_years": self.years,
            "completed_years": len(self.local_completed_years()),
            "session_uptime_hr": round((time.monotonic() - session_start) / 3600.0, 2),
            "cu_balance": bal,
            "cu_rate_per_hr": rate,
            "active_assignments": active,
            "remote_gpu": info.get("gpu", ""),
            "remote_alive": info.get("alive", False),
            "spinup_rows": info.get("spinup_rows", 0),
            "spinup_last": info.get("spinup_last", ""),
            "log_tail": info.get("log_tail", [])[-5:],
        }
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.status_path)

    def run_forever(self) -> None:
        self.log(
            f"=== Starting AOBA {self.years}-yr Colab {self.gpu} Supervisor -> {self.out_dir} "
            f"(poll={self.poll_sec}s, rotate={self.max_session_hours}h) ==="
        )
        while not self.stop_requested and not self.is_target_complete():
            bal, _, _ = self.get_cu_balance()
            if bal is not None and bal < self.min_cu_reserve:
                self.log(f"CU balance ({bal:.2f}) below safety reserve ({self.min_cu_reserve:.2f}); stopping.")
                break

            self.provision_session()
            session_start = time.monotonic()
            rotation_triggered = False
            consecutive_probe_err = 0

            try:
                self.setup_and_launch_remote()
                while not self.stop_requested:
                    time.sleep(self.poll_sec)
                    try:
                        info = self.probe_remote()
                        consecutive_probe_err = 0
                    except Exception as e:
                        consecutive_probe_err += 1
                        self.log(f"Probe error ({consecutive_probe_err}/3): {e}")
                        if consecutive_probe_err >= 3:
                            self.log("Remote VM appears unreachable; rotating to a fresh A100 session...")
                            break
                        continue

                    uptime_hr = (time.monotonic() - session_start) / 3600.0
                    self.sync_from_remote(info, force_mid_bundle=False)
                    self.write_status(info, session_start, "ROTATING" if rotation_triggered else "RUNNING")

                    ok, reason = self.check_health_nan()
                    if not ok:
                        self.log(f"CRITICAL HEALTH ALERT: {reason}! Halting run and stopping GPU VM immediately!")
                        self.write_status(info, session_start, f"ABORTED: {reason}")
                        self.stop_requested = True
                        break

                    if self.is_target_complete():
                        self.log(f"TARGET REACHED: Year {self.years:04d} completed and verified locally!")
                        self.write_status(info, session_start, "COMPLETED")
                        break

                    if not info.get("alive", False):
                        self.log(f"Remote runner exited (tail: {info.get('log_tail', [])[-3:]}). Syncing final state...")
                        self.sync_from_remote(info, force_mid_bundle=True)
                        if self.is_target_complete():
                            self.log(f"TARGET REACHED: Year {self.years:04d} completed!")
                            self.write_status(info, session_start, "COMPLETED")
                            break
                        if rotation_triggered:
                            self.log("Clean STOP_AFTER_MONTH handoff completed! Rotating to fresh A100 session...")
                        else:
                            tail_txt = "\n".join(info.get("log_tail", []))
                            if "Traceback" in tail_txt or "Error:" in tail_txt or "Exception" in tail_txt:
                                self.log(f"FATAL REMOTE ERROR detected — stopping VM immediately:\n{tail_txt}")
                                self.write_status(info, session_start, "FAILED")
                                self.stop_requested = True
                            else:
                                self.log("Runner exited unexpectedly before target year; will check CU & restart on fresh session.")
                        break

                    # Trigger graceful monthly handoff ahead of Colab's 24h limit
                    if not rotation_triggered and uptime_hr >= self.max_session_hours:
                        self.log(f"Session uptime {uptime_hr:.2f}h >= {self.max_session_hours}h -> requesting graceful STOP_AFTER_MONTH...")
                        self.run_colab_exec(f"from pathlib import Path; (Path({REMOTE_OUT!r}) / 'STOP_AFTER_MONTH').touch()", timeout_s=30)
                        rotation_triggered = True
                    elif rotation_triggered and uptime_hr >= self.max_session_hours + 0.75:
                        self.log("Graceful stop timeout reached; forcing final sync and rotating session...")
                        self.sync_from_remote(info, force_mid_bundle=True)
                        break

                    # Check CU safety floor
                    bal, _, _ = self.get_cu_balance()
                    if bal is not None and bal < self.min_cu_reserve and not rotation_triggered:
                        self.log(f"CU balance ({bal:.2f}) below reserve ({self.min_cu_reserve}); requesting STOP_AFTER_MONTH.")
                        self.run_colab_exec(f"from pathlib import Path; (Path({REMOTE_OUT!r}) / 'STOP_AFTER_MONTH').touch()", timeout_s=30)
                        rotation_triggered = True
                        self.stop_requested = True
            except Exception as e:
                self.log(f"Session error: {e}")
            finally:
                self.stop_session()

            if not self.stop_requested and not self.is_target_complete():
                time.sleep(15)

        self.log("=== Supervisor finished cleanly. ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resilient local-Mac supervisor for Google Colab A100 AOBA spin-up")
    parser.add_argument("--session", type=str, default="aoba-spinup-100yr")
    parser.add_argument("--gpu", type=str, default="A100")
    parser.add_argument("--years", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/Users/andrew/Library/CloudStorage/GoogleDrive-k1bridgebook@gmail.com/マイドライブ/aoba_colab/outputs/glevel5_picontrol_100yr",
    )
    parser.add_argument("--poll-sec", type=int, default=900, help="Sync & monitor interval in seconds (default 15 min)")
    parser.add_argument("--max-session-hours", type=float, default=21.5, help="Graceful VM rotation threshold [hours]")
    parser.add_argument("--min-cu-reserve", type=float, default=100.0, help="Minimum CU balance reserve before halting")
    parser.add_argument("--config", type=str, default="", help="Relative path inside repo to TOML preset (e.g. config/presets/earth_g5_3d_8x8_g7ocn.toml)")
    parser.add_argument("--enable-ocean-bgc", action="store_true", default=False, help="Force-enable FlexPFT marine BGC & carbonate chemistry (overrides TOML)")
    args = parser.parse_args()

    sup = Supervisor(
        session=args.session,
        gpu=args.gpu,
        years=args.years,
        out_dir=Path(args.out_dir),
        poll_sec=args.poll_sec,
        max_session_hours=args.max_session_hours,
        min_cu_reserve=args.min_cu_reserve,
        config_path=args.config,
        enable_ocean_bgc=bool(args.enable_ocean_bgc),
    )
    sup.run_forever()


if __name__ == "__main__":
    main()
