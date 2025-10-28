import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.edenhofer2023 import Edenhofer2023Query

# Query extinction DENSITY (E per pc) at arbitrary (l,b,d)
q_den = Edenhofer2023Query(integrated=False, load_samples=False)  # density mode. :contentReference[oaicite:7]{index=7}

# Grid definition (pc, heliocentric coords; Sun at (0,0,0))
L = 1000.0                         # half-size of XY box (pc)
NX = NY = 101                      # XY resolution
Z_max = 200.0                      # vertical half-thickness to integrate (pc)
dZ = 10.0                          # step (pc)

x = np.linspace(-L, L, NX)
y = np.linspace(-L, L, NY)
z = np.arange(-Z_max, Z_max + dZ, dZ)

XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')   # shapes (NX,NY,NZ)

# Convert (X,Y,Z) → (l,b,d)
d = np.sqrt(XX**2 + YY**2 + ZZ**2)   # pc
l = np.degrees(np.arctan2(YY, XX)) % 360.0
b = np.degrees(np.arcsin(ZZ / np.clip(d, 1e-6, None)))

# Mask points outside the map’s radial limit
mask = (d > 1250.0)
# Build SkyCoord for all points (vectorized)
coords = SkyCoord(l.flatten()*u.deg, b.flatten()*u.deg, (d.flatten()*u.pc), frame='galactic')

# Query extinction density (E per pc), reshape, and convert to A_V per pc
E_per_pc = q_den.query(coords, mode='mean').reshape(XX.shape)      # E per pc
E_per_pc = np.where(mask, 0.0, E_per_pc)                           # zero outside map
A_V_per_pc = 2.8 * E_per_pc                                        # to V band. :contentReference[oaicite:8]{index=8}

# Integrate over Z (Σ A_V in mag): simple Riemann sum
Sigma_Av = np.sum(A_V_per_pc, axis=2) * dZ                         # mag

# Convert to Σ_dust (Msun/pc^2)
Sigma_dust_XY = 0.213 * Sigma_Av
# Now Sigma_dust_XY has shape (NX, NY), a face-on dust surface-density map.