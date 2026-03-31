#!/usr/bin/env python3
"""Analyze the JHU/APL DMSP SSJ5 binary format."""
import struct, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\Lenovo\magnetosphere-reproduction\data\dmsp\2015apr10.f17', 'rb') as f:
    data = f.read()

n_records = len(data) // 128

def decompress(i):
    """AFRL log decompression for 9-bit (or 8-bit) packed counts."""
    if i == 0:
        return -1  # missing
    x = i % 32   # 5-bit mantissa
    y = (i - x) // 32  # 3 or 4-bit exponent
    return x + 32 * (2**y - 1) - 1

print("=" * 70)
print("DMSP SSJ5 JHU/APL BINARY FORMAT ANALYSIS")
print("=" * 70)
print(f"File: 2015apr10.f17 (F17, DOY 100, 2015-04-10)")
print(f"File size: {len(data)} bytes, Records: {n_records}")
print()

# === SECTION 1: HEADER (bytes 0-7) ===
print("=" * 70)
print("SECTION 1: HEADER (bytes 0-7)")
print("=" * 70)
print("Byte 0: Satellite number (17 for F17, 18 for F18)")
print("Byte 1: Instrument/Format ID (15 = 0x0F, constant)")
print("Byte 2: Day of Year (100)")
print("Byte 3: Reserved (0)")
print()
print("Bytes 4-5: LE uint16 - varies (65407, 65406, ...)")
print("Bytes 6-7: LE uint16 - geographic longitude x10 (902 -> 90.2 deg)")
print()

# Check byte 4-5 meaning over wider range
print("=== Header bytes 4-5 vs trailer latitude (wider range) ===")
check_points = [0, 100, 500, 1000, 5000, 10000, 20000, 40000, 60000, 80000]
for rec in check_points:
    if rec >= n_records:
        continue
    offset = rec * 128
    w45 = struct.unpack_from('<H', data, offset+4)[0]
    w67 = struct.unpack_from('<H', data, offset+6)[0]
    tlat = struct.unpack_from('<H', data, offset+98)[0]
    b4 = data[offset+4]
    b5 = data[offset+5]
    tlon = struct.unpack_from('<H', data, offset+106)[0]
    print(f"  Rec {rec:6d}: w45={w45:5d} w67={w67:5d} | b4={b4:4d} b5={b5:4d} | trailer_lat={tlat:5d} ({tlat/100:.2f}) trailer_lon={tlon:5d}")

print()
print("=== Checking if w45 = second of day ===")
for rec in [0, 1000, 5000, 10000, 20000, 40000, 60000, 80000]:
    if rec >= n_records:
        continue
    offset = rec * 128
    w45 = struct.unpack_from('<H', data, offset+4)[0]
    h = w45 // 3600
    m = (w45 % 3600) // 60
    s = w45 % 60
    tlat = struct.unpack_from('<H', data, offset+98)[0]
    print(f"  Rec {rec:6d}: w45={w45:5d} -> {h:02d}:{m:02d}:{s:02d} | lat={tlat/100:.2f}")

print()
print("=== Checking if w67 = longitude ===")
for rec in [0, 1000, 5000, 10000, 20000, 40000, 60000, 80000]:
    if rec >= n_records:
        continue
    offset = rec * 128
    w67 = struct.unpack_from('<H', data, offset+6)[0]
    tlat = struct.unpack_from('<H', data, offset+98)[0]
    tlon = struct.unpack_from('<H', data, offset+106)[0]
    print(f"  Rec {rec:6d}: w67={w67:5d} ({w67/10:.1f}) | trailer_lat={tlat/100:.2f} trailer_lon={tlon/100:.2f}")

print()

# === SECTION 2: PARTICLE DATA (bytes 8-87) ===
print("=" * 70)
print("SECTION 2: PARTICLE DATA (bytes 8-87) = 40 x 16-bit LE words")
print("=" * 70)
print("Nonzero values only appear at even byte offsets (every other byte is 0)")
print("This confirms 40 channels x 2 bytes each = 80 bytes")
print()
print("Channel mapping (following AFRL SSJ5 format):")
print("  Channels  0-19: Electron channels (20 channels)")
print("    Ch  0: e- 9450 eV    Ch  1: e- 13900 eV    Ch  2: e- 20400 eV")
print("    Ch  3: e- 30000 eV    Ch  4: e-  2040 eV    Ch  5: e-  3000 eV")
print("    Ch  6: e-  4400 eV    Ch  7: e-  6460 eV    Ch  8: e-   646 eV")
print("    Ch  9: e- status1      Ch 10: e-   949 eV    Ch 11: e-  1392 eV")
print("    Ch 12: e-   139 eV    Ch 13: e-   204 eV    Ch 14: e-   300 eV")
print("    Ch 15: e-   440 eV    Ch 16: e-    30 eV    Ch 17: e-    44 eV")
print("    Ch 18: e-    65 eV    Ch 19: e-    95 eV")
print("  Channels 20-39: Ion channels (20 channels)")
print("    Ch 20: i+ 9450 eV    Ch 21: i+ 13900 eV    Ch 22: i+ 20400 eV")
print("    Ch 23: i+ 30000 eV    Ch 24: i+  2040 eV    Ch 25: i+  3000 eV")
print("    Ch 26: i+  4400 eV    Ch 27: i+  6460 eV    Ch 28: i+   646 eV")
print("    Ch 29: i+ status2      Ch 30: i+   949 eV    Ch 31: i+  1392 eV")
print("    Ch 32: i+   139 eV    Ch 33: i+   204 eV    Ch 34: i+   300 eV")
print("    Ch 35: i+   440 eV    Ch 36: i+    30 eV    Ch 37: i+    44 eV")
print("    Ch 38: i+    65 eV    Ch 39: i+    95 eV")
print()
print("Log decompression: counts = (I mod 32) + 32*(2^((I-(I mod 32))/32) - 1) - 1")
print("  I=0 -> -1 (missing), I=1 -> 0, I=2 -> 1, I=4 -> 3, I=34 -> 33")
print()

# Show sample data
print("=== Sample particle data (first 5 records) ===")
for rec in range(5):
    offset = rec * 128
    words = struct.unpack_from('<40H', data, offset+8)
    nonzero = [(i, words[i], decompress(words[i])) for i in range(40) if words[i] > 0]
    print(f"  Record {rec}: {len(nonzero)} channels with data:")
    for ch, raw, dec in nonzero:
        print(f"    Ch{ch:2d}: raw={raw:4d} -> {dec:5d} counts")

print()

# === SECTION 3: RECORD INFO (bytes 88-95) ===
print("=" * 70)
print("SECTION 3: RECORD INFO (bytes 88-95) = 4 x 16-bit LE words")
print("=" * 70)
for rec in range(5):
    offset = rec * 128
    w44 = struct.unpack_from('<H', data, offset+88)[0]
    w45 = struct.unpack_from('<H', data, offset+90)[0]
    w46 = struct.unpack_from('<H', data, offset+92)[0]
    w47 = struct.unpack_from('<H', data, offset+94)[0]
    print(f"  Rec {rec}: word44={w44}, word45={w45}, word46={w46} (0x{w46:04x}), word47={w47}")

print()
print("  word44 (bytes 88-89): Record counter (sequential from 2)")
print("  word45 (bytes 90-91): Zero/unused (always 0)")
print("  word46 (bytes 92-93): Constant 2000 (0x07D0) - possibly orbit number or flag")
print("  word47 (bytes 94-95): Constant 1 - data quality/validity flag")
print()

# === SECTION 4: EPHEMERIS TRAILER (bytes 96-127) ===
print("=" * 70)
print("SECTION 4: EPHEMERIS TRAILER (bytes 96-127) = 16 x 16-bit LE words")
print("=" * 70)
print()
print("Trailer word mapping analysis:")
print()

# Full trailer dump for first 5 records
for rec in range(5):
    offset = rec * 128
    words = struct.unpack_from('<16H', data, offset+96)
    words_s = struct.unpack_from('<16h', data, offset+96)
    print(f"  Record {rec}:")
    for i, (wu, ws) in enumerate(zip(words, words_s)):
        print(f"    T-word {i:2d} (bytes {96+i*2:3d}-{97+i*2:3d}): unsigned={wu:6d}  signed={ws:7d}")

print()

# Try to interpret trailer fields
print("=== TRAILER FIELD INTERPRETATION ===")
for rec in range(5):
    offset = rec * 128
    tw = struct.unpack_from('<16H', data, offset+96)

    t0 = tw[0]   # always 0
    t1 = tw[1]   # varies, looks like lat*100
    t2 = tw[2]   # 65534 = 0xFFFE
    t3 = tw[3]   # varies, decreases
    t4 = tw[4]   # always 13
    t5 = tw[5]   # varies, decreases
    t6 = tw[6]   # always 129
    t7 = tw[7]   # varies, decreases (altitude?)
    t8 = tw[8]   # 65531 = 0xFFFB
    t9 = tw[9]   # varies, decreases
    t10 = tw[10]  # always 24
    t11 = tw[11]  # varies, decreases
    t12 = tw[12]  # always 8
    t13 = tw[13]  # 2000 = 0x07D0
    t14 = tw[14]  # always 1
    t15 = tw[15]  # always 0

    lat = t1 / 100.0
    print(f"  Rec {rec}: lat={lat:.2f}, t2={t2}(0x{t2:04x}), t3={t3}, t4={t4}, t5={t5}, t6={t6}, t7={t7}, t8={t8}(0x{t8:04x}), t9={t9}, t10={t10}, t11={t11}, t12={t12}, t13={t13}, t14={t14}, t15={t15}")

print()

# Try MLT interpretation
print("=== TRYING MAGNETIC LOCAL TIME INTERPRETATION ===")
for rec in range(5):
    offset = rec * 128
    tw = struct.unpack_from('<16H', data, offset+96)
    # t4=13 could be MLT hour
    # t3 could be MLT encoded as seconds
    mlt_sec = tw[3]
    if mlt_sec > 43200:
        mlt_sec -= 65536
    mlt_h = mlt_sec // 3600
    mlt_m = (mlt_sec % 3600) // 60
    mlt_s = mlt_sec % 60
    print(f"  Rec {rec}: t3={tw[3]} (signed={mlt_sec}) -> MLT {abs(mlt_h):02d}:{abs(mlt_m):02d}:{abs(mlt_s):02d}, t4={tw[4]}")

print()

# Try altitude interpretation
print("=== TRYING ALTITUDE INTERPRETATION ===")
for rec in range(5):
    offset = rec * 128
    tw = struct.unpack_from('<16H', data, offset+96)
    # t7 varies and could be altitude
    # DMSP altitude ~850 km
    # t7 for rec 0 = 30400. 30400/10 = 3040? No.
    # Maybe t7 = altitude in some other unit?
    # Or maybe t7 is NOT altitude but another coordinate
    alt = tw[7]
    print(f"  Rec {rec}: t7={alt}, t7/100={alt/100:.2f}, t7/10={alt/10:.1f}")

# Check if t7 and t3/t5/t9/t11 could be Cartesian coordinates
print()
print("=== TRYING CARTESIAN COORDINATES ===")
for rec in range(5):
    offset = rec * 128
    tw = struct.unpack_from('<16H', data, offset+96)
    ts = struct.unpack_from('<16h', data, offset+96)
    x = ts[3]
    y = ts[5]
    z = ts[7]
    r = (x**2 + y**2 + z**2)**0.5
    print(f"  Rec {rec}: x={x:7d}, y={y:7d}, z={z:7d}, |r|={r:.0f}")
    # If these are ECI coordinates * some scale factor...
    # Earth radius ~6371 km, altitude ~850 km, r ~7221 km
    # If scale = 10000: r/10000 = too small
    # If scale = 0.1: r*0.1 = ~30000? Too big

print()
print("=== TRAILER WORD5 AS LONGITUDE? ===")
for rec in range(5):
    offset = rec * 128
    tw_s = struct.unpack_from('<16h', data, offset+96)
    t5 = tw_s[5]
    lon = t5 / 100.0
    print(f"  Rec {rec}: t5={t5} ({t5/100:.2f} deg)")
    # For a southbound pass in the Indian Ocean sector, longitude should
    # be around 60-120E. -13262 -> -132.62? That is 132W. Possible but need to check.

# Check if any trailer fields match header fields
print()
print("=== HEADER vs TRAILER COMPARISON ===")
for rec in range(5):
    offset = rec * 128
    hdr_w45 = struct.unpack_from('<H', data, offset+4)[0]
    hdr_w67 = struct.unpack_from('<H', data, offset+6)[0]
    tw = struct.unpack_from('<16H', data, offset+96)
    print(f"  Rec {rec}: hdr_w45={hdr_w45:5d}, hdr_w67={hdr_w67:5d} | t3={tw[3]:5d}, t5={tw[5]:5d}, t9={tw[9]:5d}, t11={tw[11]:5d}")
PYEOF