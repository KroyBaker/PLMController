from TIPLMSuiteFourPhaseDualOrder import RecombinedFourPhaseCGHGenerator, DeviceLibrary
import cv2
import numpy as np
from pathlib import Path

# One image from several superpixel orders folded onto the axis and added
# coherently (see RecombinedFourPhaseCGHGenerator for the physics and bench
# layouts). Default fold: B + i C -- every pixel of the 2x2 superpixel gets
# its own weight, both orders are dimmed equally by the mirror aperture, and
# the on-axis order (with the mirror-gap and window light) is not used.
#
# Measured, 632.8 nm theoretical LUT, wave-optics check with exact square mirrors:
#                                           harper     bear1
#   paper encoding, one on-axis pinhole     23.2 dB    28.4 dB
#   older S + i D fold                      24.8 dB    29.4 dB
#   B + i C, square pinholes                28.8 dB    34.2 dB
#   B + i C, diamond pinholes 0.8 order     30.4 dB    36.7 dB
#   B + i C, diamond pinholes 0.9 (default) 30.5 dB    37.1 dB   (~2 min)
# The diamond is a square pinhole rotated by 45 degrees (0.9 order: 1.582 mm
# side, 2.237 mm tip to tip at f1 = 60 mm).

# Define input parameters
TargetImage = "./teejay.png"   # define target image
ColorChannel = "gray"         # any PNG/JPG/BMP/TIFF: 'gray' -> luminance; 0/1/2 -> R/G/B
PreserveAspect = True         # letterbox instead of stretching to the PLM aspect
Orders = [("B", 1.0, 0.9, "diamond"), ("C", 1j, 0.9, "diamond")]
# (order, complex weight, pinhole size in orders, shape: 'square' 'circle' 'diamond' 'soft')
F1 = 0.060                    # first 4f lens, metres

D = DeviceLibrary()
DeviceDict = D.defineDevice('0.67_632.8nm', measuredLUT=False)
print("Phase LUT scheme: " + DeviceDict["discScheme"])

G = RecombinedFourPhaseCGHGenerator()
G.createCGH(DeviceDictionary=DeviceDict,
            filename=TargetImage, colorChannel=ColorChannel,
            FlipUD=False, FlipLR=True, preserveAspect=PreserveAspect,
            orders=Orders)
stem = (Path(TargetImage).stem + "_RECOMBINED_" + "".join(o[0] for o in Orders)
        + "_" + "_".join(sorted({o[3] if len(o) > 3 else "square" for o in Orders})))
G.writeCGHToFile(stem + "_CGH.bmp")
print("Wrote " + stem + "_CGH.bmp")

# Bench: one illumination beam per folded order, tilted so that order leaves
# on axis, and a single on-axis pinhole (or: a pinhole per order in the
# filter plane, each band translated onto the axis and recombined).
lam, p = G.lambda_m, G.pitchW
print("Fourier plane of L1 (f = %.0f mm):" % (F1 * 1e3))
for order, weight, aperture, *shape in Orders:
    cx, cy = G.FOLD_CENTERS[order]
    shape = shape[0] if shape else "square"
    print("  order %s (weight %s): band centre (%+.3f, %+.3f) mm, %s of a %.3f mm square%s;"
          " beam tilt (%.2f, %.2f) deg"
          % (order, np.round(complex(weight), 3), lam * F1 * cx / p * 1e3, lam * F1 * cy / p * 1e3,
             shape, aperture * lam * F1 / (2 * p) * 1e3,
             " (the square rotated 45 deg)" if shape == "diamond" else "",
             np.degrees(lam * cx / p), np.degrees(lam * cy / p)))
print("  relative beam phases: tune on the bench for the best image (the model's pixel-centred"
      " carriers differ from the physical tilts by constants)")

def display_normalize(image, high_percentile=99.5):
        image = np.asarray(image, dtype=np.float32)
        lo = float(np.percentile(image, 1.0))
        hi = float(np.percentile(image, high_percentile))
        if hi <= lo:
                return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        image = np.clip(image, lo, hi)
        return ((image - lo) / (hi - lo)).astype(np.float32)

cv2.destroyAllWindows()
cv2.imshow("Recombined", display_normalize(G.imRecovered_disc))
cv2.waitKey(0)
