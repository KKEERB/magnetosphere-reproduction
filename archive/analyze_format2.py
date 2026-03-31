#!/usr/bin/env python3
"""Analyze JHU/APL DMSP SSJ5 binary format - coordinate interpretation."""
import struct, sys, math
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Lenovo\magnetosphere-reproduction\data\dmsp\2015apr10.f17', 'rb') as f:
    data = f.read()
n_records = len(data) // 128

# Check latitude changes across the file to determine time resolution
print("=== LATITUDE CHANGES AT DIFFERENT POINTS ===")
check_recs = [0, 100, 200, 300, 400, 500, 1000, 2000, 5000, 10000, 20000, 40000, 60000, 80000]
prev_lat = None
prev_rec = None
for rec in check_recs:
    if rec >= n_records:
        continue
    offset = rec * 128
    tlat = struct.unpack_from('<H', data, offset + 98)[0]
    if prev_lat is not None:
        dlat = abs(tlat - prev_lat) / 100.0
        drec = rec - prev_rec
        print(f"  Rec {rec:6d}: lat={tlat/100:.2f}, delta={dlat:.2f} deg over {drec} records")
    else:
        print(f"  Rec {rec:6d}: lat={tlat/100:.2f}")
    prev_lat = tlat
    prev_rec = rec

print()

# Check consecutive record latitude changes
print("=== CONSECUTIVE LAT CHANGES (first 20 records) ===")
for rec in range(20):
    offset = rec * 128
    tlat = struct.unpack_from('<H', data, offset + 98)[0]
    if rec > 0:
        prev_offset = (rec - 1) * 128
        prev_tlat = struct.unpack_from('<H', data, prev_offset + 98)[0]
        dlat = tlat - prev_tlat
        print(f"  Rec {rec}: lat={tlat/100:.2f}, delta={dlat/100:.2f} deg")
    else:
        print(f"  Rec {rec}: lat={tlat/100:.2f}")

print()

# Check if latitude jumps suggest orbit boundaries
print("=== LARGE LATITUDE JUMPS (orbit boundaries?) ===")
prev_lat = None
jumps = []
for rec in range(min(90000, n_records)):
    offset = rec * 128
    tlat = struct.unpack_from('<H', data, offset + 98)[0]
    if prev_lat is not None:
        dlat = abs(tlat - prev_lat)
        if dlat > 5000:  # more than 50 degrees jump
            jumps.append((rec, prev_lat, tlat, dlat))
    prev_lat = tlat

print(f"  Found {len(jumps)} large jumps:")
for rec, p, c, d in jumps[:20]:
    print(f"    Rec {rec}: {p/100:.2f} -> {c/100:.2f} (jump={d/100:.2f} deg)")

print()

# Determine number of orbits
if jumps:
    n_orbits = len(jumps) + 1
    recs_per_orbit = [jumps[0][0]]
    for i in range(1, len(jumps)):
        recs_per_orbit.append(jumps[i][0] - jumps[i-1][0])
    recs_per_orbit.append(n_records - jumps[-1][0])
    print(f"  Approximate orbits: {n_orbits}")
    print(f"  Records per orbit: min={min(recs_per_orbit)}, max={max(recs_per_orbit)}, avg={sum(recs_per_orbit)/len(recs_per_orbit):.0f}")
    print(f"  Total records: {n_records}, seconds per day: 86400")
    print(f"  If 1 record/sec: {n_records/86400:.2f} days of data")
    print(f"  DMSP orbits/day ~14, orbital period ~101 min = 6060 sec")
    avg_recs_per_orbit = sum(recs_per_orbit) / len(recs_per_orbit)
    print(f"  Avg recs/orbit: {avg_recs_per_orbit:.0f}")
    print(f"  If 1 rec/sec: orbital period = {avg_recs_per_orbit:.0f} sec ({avg_recs_per_orbit/60:.1f} min)")
    print(f"  DMSP orbital period ~101 min = 6060 sec")
    print(f"  Ratio: {avg_recs_per_orbit / 6060:.2f}")

print()

# Now check the header time field more carefully
print("=== HEADER TIME FIELD OVER ORBIT BOUNDARY ===")
if jumps:
    boundary = jumps[0][0]
    for rec in range(max(0, boundary-3), min(n_records, boundary+3)):
        offset = rec * 128
        w45 = struct.unpack_from('<H', data, offset + 4)[0]
        w67 = struct.unpack_from('<H', data, offset + 6)[0]
        tlat = struct.unpack_from('<H', data, offset + 98)[0]
        h = w45 // 3600
        m = (w45 % 3600) // 60
        s = w45 % 60
        print(f"  Rec {rec}: w45={w45:5d} ({h:02d}:{m:02d}:{s:02d}), w67={w67}, lat={tlat/100:.2f}")

print()

# Check what header byte1 (value 15) really means
print("=== BYTE 1 VERIFICATION ACROSS FILE ===")
byte1_vals = set()
for rec in range(n_records):
    byte1 = data[rec * 128 + 1]
    byte1_vals.add(byte1)
print(f"  Unique byte1 values: {byte1_vals}")

# Check F18 file
with open(r'C:\Users\Lenovo\magnetosphere-reproduction\data\dmsp\2015apr10.f18', 'rb') as f:
    data18 = f.read()
byte1_vals18 = set()
for rec in range(len(data18) // 128):
    byte1 = data18[rec * 128 + 1]
    byte1_vals18.add(byte1)
print(f"  F18 unique byte1 values: {byte1_vals18}")

print()

# Try to decode the trailer more carefully using AFRL-like fields
# The trailer has pairs of (constant, varying) fields:
# (0, lat*100), (-2, varying), (13, varying), (129, varying), (-5, varying), (24, varying), (8, 2000), (1, 0)
# Could be: (pad, geo_lat), (alt_flag, alt?), (hour, mlt_field?), ...
#
# Actually, what if we read the trailer as a series of fields with different sizes?
# Let me try reading as mixed byte and word fields:

print("=== TRAILER AS MIXED BYTE AND WORD FIELDS ===")
for rec in range(3):
    offset = rec * 128
    b = data[offset + 96: offset + 128]
    print(f"  Rec {rec}:")
    print(f"    Bytes 96-127: {' '.join(f'{x:02x}' for x in b)}")

    # Try: each ephemeris value is a 32-bit int (4 bytes)
    # 32 bytes / 4 = 8 values
    vals32 = struct.unpack_from('<8i', data, offset + 96)
    print(f"    As 8x32-bit signed LE: {[f'{v:10d}' for v in vals32]}")

    # What if the format is:
    # bytes 96-97: 0 (padding)
    # bytes 98-101: 32-bit value for lat
    # bytes 102-105: 32-bit value
    # etc.
    lat32 = struct.unpack_from('<i', data, offset + 98)[0]
    print(f"    bytes 98-101 as 32-bit signed: {lat32} -> {lat32/100.0:.2f} deg")

print()

# What if bytes 96-127 are stored in BIG ENDIAN?
print("=== TRAILER AS BIG-ENDIAN 16-BIT WORDS ===")
for rec in range(3):
    offset = rec * 128
    words_be = struct.unpack_from('>16h', data, offset + 96)
    print(f"  Rec {rec}: {[f'{v:6d}' for v in words_be]}")

print()

# Final comprehensive summary
print("=" * 70)
print("COMPREHENSIVE FORMAT SUMMARY")
print("=" * 70)
print("""
FILE: <YYYY><MON><DD>.f<NN> (e.g., 2015apr10.f17)
RECORD SIZE: 128 bytes per record
BYTE ORDER: Little-endian (Intel)
DATA TYPE: 16-bit unsigned integers for particle counts, mixed for ephemeris

RECORD LAYOUT:
  Bytes  0-3:   HEADER (4 bytes)
    Byte 0:     Satellite number (17=F17, 18=F18, etc.)
    Byte 1:     Instrument/format ID (15 = SSJ5 data)
    Byte 2:     Day of Year (1-366)
    Byte 3:     Reserved (always 0)

  Bytes  4-7:   POSITION/TIME (4 bytes) - interpretation uncertain
    Bytes 4-5:   LE uint16 - varies (possibly second-of-day or position index)
    Bytes 6-7:   LE uint16 - geographic longitude x 10

  Bytes  8-87:  PARTICLE COUNTS (80 bytes = 40 x 16-bit LE words)
    Words 0-19:  Electron channel counts (20 channels, log-compressed)
    Words 20-39: Ion channel counts (20 channels, log-compressed)

    Channel order (same as AFRL):
      Electron: ch4(9450eV) ch3(13900eV) ch2(20400eV) ch1(30000eV)
                ch8(2040eV) ch7(3000eV) ch6(4400eV) ch5(6460eV)
                ch12(646eV) ch11/status1 ch10(949eV) ch9(1392eV)
                ch16(139eV) ch15(204eV) ch14(300eV) ch13(440eV)
                ch20(30eV) ch19(44eV) ch18(65eV) ch17(95eV)
      Ion:      (same channel energy order)

    Log decompression: counts = (I%32) + 32*(2^((I-I%32)/32) - 1) - 1
      I=0: missing (-1), I=1: 0 counts, I=2: 1 count, etc.
      Max (I=255 for 8-bit): 4094 counts

  Bytes 88-95:  RECORD METADATA (8 bytes = 4 x 16-bit LE words)
    Bytes 88-89: Record counter (sequential, 1-based from 2)
    Bytes 90-91: Unused (always 0)
    Bytes 92-93: Orbit/mission flag (constant 2000 = 0x07D0)
    Bytes 94-95: Data quality flag (always 1)

  Bytes 96-127: EPHEMERIS (32 bytes = 16 x 16-bit LE words)
    Word 0 (96-97):   Zero/padding (always 0)
    Word 1 (98-99):   Geographic latitude x 100 (e.g., 7538 = 75.38 deg)
    Word 2 (100-101): Constant 0xFFFE (-2 signed) - purpose uncertain
    Word 3 (102-103): Varies - possibly ECI x-coordinate or footpoint data
    Word 4 (104-105): Constant 13 - possibly hour of day or altitude unit
    Word 5 (106-107): Varies - possibly ECI y-coordinate or footpoint longitude
    Word 6 (108-109): Constant 129 - possibly MLT or altitude reference
    Word 7 (110-111): Varies - possibly ECI z-coordinate or altitude
    Word 8 (112-113): Constant 0xFFFB (-5 signed)
    Word 9 (114-115): Varies
    Word 10 (116-117): Constant 24
    Word 11 (118-119): Varies
    Word 12 (120-121): Constant 8
    Word 13 (122-123): Constant 2000 (0x07D0) - orbit number
    Word 14 (124-125): Constant 1 - data validity
    Word 15 (126-127): Zero/padding (always 0)

NOTE: The trailer ephemeris interpretation is PARTIALLY confirmed.
      Latitude (word 1) is definitively geo_lat * 100.
      The remaining fields likely contain longitude, altitude, MLT,
      and footpoint coordinates in a yet-to-be-fully-decoded format.
      The JHU/APL site notes that position used NORAD routines.

REFERENCES:
  1. AFRL-RV-PS-TR-2014-0174: DMSP Space Weather Sensors Data Archive
     File Format Descriptions (authoritative SSJ format reference)
     https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/dmsp/doc/
     AFRL%20ASCII%20and%20Binary%20File%20Format%20Descriptions.pdf
  2. JHU/APL Auroral Particles and Imagery Page:
     http://sd-www.jhuapl.edu/Aurora/
  3. DMSP SSJ5 instrument design (Boston College):
     https://dmsp.bc.edu/html2/ssj5_inst.html
  4. NASA GSFC DMSP raw binary format:
     https://core2.gsfc.nasa.gov/research/mag_field/conrad/report/node51.html
""")
