# SimDev web UI

Queue and watch runs, and manage the CAD library, from a browser on any of
your machines. The UI runs on the Z8 and is reachable over Tailscale only.

## On the Z8 (once)

1. Install the UI dependencies into the venv:

   ```bash
   cd ~/SimDev && .venv/bin/pip install -e ".[ui]"
   ```

2. Tailscale is already installed (`/snap/bin/tailscale`). Check that it is up
   and note the machine name:

   ```bash
   tailscale ip -4        # e.g. 100.70.121.119
   tailscale status       # first line: this machine's MagicDNS name
   ```

3. Install and start the service:

   ```bash
   mkdir -p ~/.config/systemd/user
   cp scripts/simdev-ui.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now simdev-ui
   sudo loginctl enable-linger $USER
   journalctl --user -u simdev-ui -f     # "SimDev UI on http://100.x.y.z:8000"
   ```

## On each Windows laptop or workstation (once)

1. Install Tailscale from tailscale.com/download and sign in with the same
   account as the Z8.
2. Open `http://<z8-magicdns-name>:8000` (or `http://100.x.y.z:8000`) and
   bookmark it.

## Security

There are no logins: anyone on your tailnet can start and delete runs. The
server binds to the Tailscale address only and refuses `--host 0.0.0.0`. Do
not share the Z8 with other Tailscale users, and do not put the port behind
Tailscale Funnel.

## Day to day

- **CAD library:** upload a driving state (13 attitude parts + speed and
  corner) or a design (Body + Wing for one state). Filed Body/Wing under the
  wrong state? Use *Move* on the design's row.
- **New run:** pick design · state and a profile, change any of the five
  overrides, *Preview*, *Queue*.
- **Queue:** runs start one at a time. Reorder with ↑/↓, edit or remove queued
  runs, cancel a running one (stops mpirun and every rank).
- **Run page:** stages, live Cd/Cl with the plateau window shaded, residuals,
  images, logs. *Resume* re-queues a failed run; finished stages are skipped.
  *Strip mesh* frees the disk space and keeps results.
- **Results:** one row per finished run (the old Excel sheet): coefficients
  with their noise, a change note and a reference run per row, TSV export.
- **Compare:** pictures of several runs side by side with sync, blink, swipe
  and fade; field deltas against a reference; charts.

## Updating

```bash
cd ~/SimDev && git pull && .venv/bin/pip install -e ".[ui]"
systemctl --user restart simdev-ui      # running solves keep going
```

## Without the service

```bash
source /usr/lib/openfoam/openfoam2412/etc/bashrc
simdev ui                       # on the Tailscale address
simdev ui --host 127.0.0.1      # local only
```
