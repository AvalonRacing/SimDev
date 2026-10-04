# Motor RPM Logger: Plan

Status: plan only, nothing built yet (2026-10-03).
Goal: log the motor RPM of the RC car during test runs and line it up with the
data we already record (GPS/IMU logger and transmitter log), so that
`scripts/extract_testing_data.py` can use it.

## 1. Approach in one paragraph

Brushless sensored RC motors (Hobbywing Xerun and nearly every other brand)
send three hall sensor signals to the ESC over a standard 6-pin cable. We put
a passive Y-splitter into that cable and read the hall lines with a small
microcontroller. The ESC keeps working as before. The microcontroller writes a
timestamp to a microSD card on every hall edge. RPM is computed afterwards in
Python from the time between edges. This works with any ESC brand, gives the
full time resolution of the motor, and costs well under 100 EUR.

Not covered here: sensorless motors (they have no hall cable; see section 10
for alternatives).

## 2. Requirements

| # | Requirement | Target |
|---|-------------|--------|
| R1 | The ESC and motor run exactly as before, including sensored start-up from standstill | no change you can feel or measure |
| R2 | RPM range | 0 to 80 000 mechanical RPM |
| R3 | Time resolution | each hall edge timestamped to 1 µs or better |
| R4 | Run length | at least 30 min per session, several sessions per card |
| R5 | Sync with the IMU/GPS log | better than 20 ms (IMU runs at 25 Hz) |
| R6 | A power cut (battery unplugged) loses at most the last second of data | yes |
| R7 | Weight and size | < 30 g, fits in the car |

## 3. Background the next person needs

### Sensor cable pinout (ROAR / Novak standard, 6-pin JST-ZH 1.5 mm)

| Pin | Signal | Usual wire colour |
|-----|--------|-------------------|
| 1 | GND | black |
| 2 | Hall C | orange |
| 3 | Hall B | white |
| 4 | Hall A | green |
| 5 | Motor temperature (thermistor) | blue |
| 6 | +5 V (supplied by the ESC) | red |

**Check the pinout with a multimeter on the actual motor and ESC before you
solder anything.** Colours are not always the same across brands.

### Electrical behaviour

- The hall outputs are open-collector. The ESC pulls them up to 5 V, so each
  line is a 0/5 V square wave.
- The ESC's 5 V rail is weak. **Do not power the logger from pin 6.**
- Only one hall line is high/low in each 60° step. Together the three lines go
  through 6 valid states per electrical revolution (states 000 and 111 never
  occur). This lets the logger spot glitches and the direction of rotation.

### Converting edges to RPM

```
edges per electrical rev  = 6   (all three lines, both edges)
                          = 2   (one line, both edges)
mechanical RPM            = electrical RPM / pole_pairs
pole_pairs                = motor poles / 2
```

- Most 1/10 540-size motors are 2-pole, so `pole_pairs = 1`.
- Most 1/8 motors (for example Xerun 4268/4274) are 4-pole, so `pole_pairs = 2`.
- **Check the pole count on the motor datasheet.** It is the most common way to
  get RPM wrong by a factor of 2.

Expected edge rates (all 3 lines):

| Motor | 80 000 RPM gives |
|-------|------------------|
| 2-pole | 1333 erev/s, 8 000 edges/s |
| 4-pole | 2667 erev/s, 16 000 edges/s |

Both rates are easy for a hardware timer capture or GPIO interrupt on any
modern MCU.

## 4. Hardware

### Recommended parts

| Part | Suggestion | Why |
|------|------------|-----|
| MCU | Teensy 4.1 | 600 MHz, built-in microSD slot (fast SDIO), cycle counter for sub-µs timestamps, good Arduino libraries. Alternatives: RP2040 (PIO gives exact edge timing) or ESP32-S3 (MCPWM capture); either needs an SPI microSD breakout. |
| Level shifter / buffer | SN74LVC3G17 (triple Schmitt buffer) powered from 3.3 V | Inputs tolerate 5 V, input current is tiny (does not load the ESC's hall lines), Schmitt trigger cleans slow or noisy edges. The Teensy 4.x is **not** 5 V tolerant, so this part is required. |
| Sensor Y-splitter | 6-pin JST-ZH male to 2x female, about 10 cm | Sold ready-made for sensor cable extensions, or build one from JST-ZH pigtails |
| Power | Small 5 V buck from the receiver bus, or a separate 1S LiPo with a 5 V boost | The receiver BEC is often set to 6 to 7.4 V, which is above the Teensy VIN limit (6 V). Use a regulator. |
| microSD | 8 to 32 GB, A1 or better, FAT32/exFAT | |
| Optional | Push button + LED | start/stop a session, show status |
| Optional | 3-pin servo Y-cable on the throttle channel | for sync, see section 6 |

### Wiring

```
Motor sensor cable ──► Y-splitter ──► ESC sensor port      (unchanged path)
                          │
                          └─► logger tap:
                               pin 1 GND    ──► logger GND (common ground!)
                               pin 4 Hall A ──► 74LVC3G17 in A ──► Teensy pin (capture)
                               pin 3 Hall B ──► 74LVC3G17 in B ──► Teensy pin
                               pin 2 Hall C ──► 74LVC3G17 in C ──► Teensy pin
                               pin 5, pin 6 ──► not connected

74LVC3G17: VCC = Teensy 3.3 V, 100 nF decoupling cap next to the chip.
Optional: 100 Ω series resistor in each hall line before the buffer
(limits current if something is miswired).
```

Notes:

- Ground must be shared between the ESC and the logger, otherwise the hall
  levels mean nothing.
- Keep the tap wires short and away from the three motor phase wires, which
  carry large switching currents.
- Do not add RC filters on the hall lines. They delay the edges, and the
  Schmitt buffer is enough.
- If only one line can be wired at first, use Hall A with both edges. This
  gives 2 edges per erev, which is still enough resolution.

## 5. Firmware

### Design

1. **Interrupt / capture:** every edge on A, B or C triggers a capture. The
   handler reads the cycle counter (`ARM_DWT_CYCCNT`, 600 MHz) and the current
   3-bit hall state, and pushes both into a lock-free ring buffer. The handler
   does nothing else.
2. **Main loop:** takes records from the ring buffer and writes them in
   512-byte blocks to the SD card. Use a large ring buffer (for example 64 kB),
   because SD cards sometimes stall for 100 ms or more.
3. **File handling:** create a new file for each session
   (`log_rpm_NNNN.bin`). Preallocate the file and call `sync()` about once per
   second (R6). Close the file on stop or after a timeout with no edges (for
   example 10 s).
4. **Start/stop:** start on boot or on a button press, and stop on a button
   press or the timeout. The LED shows: blinking = waiting, steady = logging,
   fast blink = error (no card, buffer overflow).
5. **Timestamps:** extend the 32-bit cycle counter to 64 bits in software (it
   wraps every 7.2 s at 600 MHz). Store microseconds as `uint64` or ticks as
   `uint64`. Keep it simple and avoid wrap handling in post-processing.

### Binary file format (suggested)

Header (once, fixed size, for example 64 bytes):

```
magic        char[8]   "RPMLOG01"
version      uint16
tick_hz      uint32    timer frequency of the timestamps (e.g. 600000000)
pole_pairs   uint8     as configured (0 = unknown, set in post-processing)
channels     uint8     bitmask of which hall lines are wired
start_millis uint32    millis() at session start
reserved     ...       pad to 64 bytes
```

Records (repeated):

```
t_ticks      uint64    edge timestamp
state        uint8     hall state CBA as bits 2..0
flags        uint8     bit0 = ring buffer overflowed before this record
                       bit1 = throttle pulse record (optional sync channel, see section 6)
value        uint16    throttle pulse width in µs when bit1 is set, else 0
pad          uint32    keeps records 16 bytes, aligned
```

At 16 000 edges/s that is 256 kB/s, which the Teensy 4.1 SD slot handles
easily. If the data rate ever becomes a problem, drop to 8-byte records
(32-bit delta times).

Write a small `README` or comment block in the firmware describing the format
and bump `version` when the format changes.

### Suggested repository layout

```
hardware/rpm_logger/
  firmware/           PlatformIO project (Teensy 4.1)
  wiring.md           photo + final pin assignment
scripts/rpm_log_to_csv.py   binary -> CSV/Parquet + RPM
tests/test_rpm_log.py       decoder tests on synthetic files
```

## 6. Time sync with the existing logs

Today's test data (see `scripts/extract_testing_data.py`) has two loggers:

- `log_imu*.csv`: 25 Hz GPS + IMU (RaceBox-style), the master clock
- `log_trx_*.csv`: 100 Hz transmitter log with steering and throttle in %.
  It is aligned to the IMU by cross-correlating steering with yaw curvature
  (`find_offset`).

The RPM logger has its own clock, so it must be aligned the same way. There
are two options and both are worth implementing:

1. **Without extra hardware:** compute wheel speed from RPM
   (`v = rpm / gear_ratio * pi * tyre_diameter / 60`) and cross-correlate it
   with GPS speed to find the offset. Low-pass filter the wheel speed first,
   and fit on sections without heavy wheelspin or braking. This mirrors the
   existing `find_offset` approach.
2. **With the throttle channel (more robust, recommended):** tap the
   receiver's throttle output with a servo Y-cable and capture the pulse width
   (1000 to 2000 µs at 50 to 400 Hz) on a fourth input pin. Log it as records
   with `flags.bit1` set. Throttle pulse width correlates directly with the
   transmitter log's throttle %, which is already aligned to the IMU. As a
   bonus this gives the true ESC input, which is useful for checking the ESC
   response.

Also check clock drift: compare the offset fitted on the first and last
quarters of a long run. A crystal-based MCU should drift less than 50 ppm
(0.09 s over 30 min). If drift is visible, fit offset + scale, not just an
offset.

## 7. Post-processing

New script `scripts/rpm_log_to_csv.py`:

1. Read the header and records, convert ticks to seconds.
2. **Glitch filter:** drop edges whose hall state is not a valid next state
   in the 6-step sequence, and edges whose interval is physically impossible
   (shorter than the interval at maximum RPM, divided by 2).
3. **RPM per edge:**
   - all 3 lines: `rpm_mech = 60 / (6 * dt_edge * pole_pairs)`, ideally
     computed over a full electrical revolution (6 edges back) to remove
     unevenness from sensor placement
   - one line: use same-type edges (rising to rising), giving one value per erev
4. **Direction:** taken from the order of the hall state sequence (forward or
   reverse).
5. **Zero speed:** if no edge for more than a timeout (for example 50 ms),
   set RPM to 0.
6. **Resample:** to 100 Hz and to the 25 Hz IMU clock (after the sync
   offset), using interpolation of per-edge values, with no averaging across
   gaps.
7. **Derived channels:** wheel speed, slip ratio vs GPS speed
   (`(v_wheel - v_gps) / v_gps`), and motor acceleration.

Then extend `scripts/extract_testing_data.py`: if a `log_rpm*.bin` (or the
converted CSV) is present in the run folder, align it and add `rpm`,
`v_wheel` and `slip` columns to `run_merged.csv`. Keep it optional so runs
without an RPM log still process.

## 8. Test and validation plan

Do these in order. Do not mount the logger in the car before step 3 passes.

1. **Decoder unit tests (no hardware):** write synthetic binary files with
   known RPM ramps, glitches, overflow flags and reverse direction, and check
   `rpm_log_to_csv.py` returns the known values. Put these in `tests/`.
2. **Bench test with a fake motor:** a second microcontroller (any Arduino)
   generates the 6-step hall sequence at known frequencies (sweep 1 to
   3000 erev/s), passed through the real buffer circuit. Check the logged RPM
   against the set frequency to within 0.1 %, with no lost edges and no buffer
   overflow during a 30 min run. Unplug power mid-run and check the file is
   readable up to about the last second.
3. **Bench test on the real ESC and motor, car on a stand, wheels off the
   ground:**
   - Sensored start-up and low-speed crawl feel the same with and without the
     tap connected (R1). Compare with a scope if one is available: the hall
     edges at the ESC should look the same with the tap attached.
   - Logged RPM matches the ESC's own reading (HW Link / OTA Programmer) and,
     if available, an optical tachometer on the spur gear (convert using the
     pinion/spur ratio).
   - Check the pole count is right: a factor-of-2 error shows up here.
4. **Track test:** one run together with the IMU and transmitter loggers.
   Check the sync offset from section 6 (both methods should agree within one
   IMU sample), check that wheel speed matches GPS speed on straights within a
   few %, and look at the plots for gaps and spikes.

## 9. Milestones

| # | Deliverable | Done when |
|---|-------------|-----------|
| M1 | Parts ordered, pinout and pole count confirmed on our motor | photo + notes in `hardware/rpm_logger/wiring.md` |
| M2 | Firmware logs edges from the fake motor to SD | bench test 2 passes |
| M3 | Decoder script + unit tests | test step 1 passes in CI / `pytest` |
| M4 | Works on the real ESC and motor | bench test 3 passes |
| M5 | Integrated in `extract_testing_data.py`, first track run processed | track test 4 passes, plots in `TestingData_Processed/` |

## 10. Risks and alternatives

| Risk | Mitigation |
|------|------------|
| The tap disturbs ESC commutation | Schmitt buffer with high-impedance inputs, short wires, test step 3. If it still happens, use an optocoupler or a hall sensor on the spur gear instead. |
| Electrical noise from the phase wires causes false edges | Schmitt buffer, wire routing, valid-state glitch filter in post-processing |
| SD card stalls cause lost edges | Big ring buffer, overflow flag in the records, good-quality card |
| Pole count or gear ratio wrong | Confirm in M1, validate in bench test 3 |
| Sensorless motor used later | Hall sensor (A3144 / DRV5023) plus a magnet on the spur gear or driveshaft, logged on the same firmware as a single-line input with 1 edge per rev. Gives drivetrain RPM instead of motor RPM. |

## 11. Open questions (answer before or during M1)

- Which car and which exact motor + ESC are used? Pole count (2 or 4)?
- Gear ratio (pinion/spur and internal ratio) and loaded tyre diameter, for
  wheel speed?
- Does the RaceBox-style logger give absolute (GPS/UTC) time? If yes, a
  cheap GPS module with a PPS output on the RPM logger would give direct
  absolute sync instead of correlation.
- What is the receiver BEC voltage? This decides the power supply for the
  logger.
- Is motor temperature (pin 5, thermistor) also wanted? It could be logged
  with an ADC and a pull-up, but the curve of the thermistor has to be found
  first.
