from TIPLMSuiteFourPhaseDualOrder import DualOrderFourPhaseCGHGenerator, DeviceLibrary
import cv2
import numpy as np
from pathlib import Path

# Two images from one four-phase hologram. Image 1 is the paper's on-axis
# four-phase image; image 2 lives in a superpixel order the paper's filter
# discards, written with the ~94% redundant reorderings of each superpixel
# plus an epsilon tolerance on image 1 (see TIPLMSuiteFourPhaseDualOrder).
#
# Measured with the defaults below (632.8 nm theoretical LUT, band-limited 4f):
#   paper encoding, same pinhole 1 : image 1 23.0 dB | pinhole 2 8.8 dB (no image)
#   this                           : image 1 23.7 dB | image 2 26.4 dB   (~2.5 min)
# Epsilon 0 (image-1 field untouched) still gives 23.0 / 24.2 dB.

# Define input parameters
TargetImage = "./teejay.png"   # pinhole 1, on axis
SecondImage = "./aryb.jpg"                # pinhole 2
ColorChannel = "gray"        # any PNG/JPG/BMP/TIFF: 'gray' -> luminance; 0/1/2 -> R/G/B only
PreserveAspect = True        # letterbox (black bars) instead of stretching to the 1358x800 PLM
SecondOrder = "D"            # 'D' diagonal (1/2p, 1/2p) [best], 'B' x-order, 'C' y-order
Epsilon = 0.05               # extra image-1 field error allowed per superpixel (0 = paper's multisets)
AmplitudeScale = 0.8
ApertureScale = 0.6          # pinhole 1 width, fraction of one superpixel order
SecondApertureScale = 0.4    # pinhole 2 width
SecondWeight = 0.3           # image-2 priority (0.1 -> 24.4 / 25.8 dB)
F1 = 0.100                   # first 4f lens, metres
F2 = 0.050                   # second 4f lens, metres

D = DeviceLibrary()
DeviceDict = D.defineDevice('0.67_632.8nm', measuredLUT=False)
print("Phase LUT scheme: " + DeviceDict["discScheme"])

G = DualOrderFourPhaseCGHGenerator()
G.createCGH(DeviceDictionary=DeviceDict,
            filename=TargetImage, secondFilename=SecondImage, colorChannel=ColorChannel,
            FlipUD=False, FlipLR=True, preserveAspect=PreserveAspect,
            secondOrder=SecondOrder, epsilon=Epsilon,
            amplitudeScale=AmplitudeScale, apertureScale=ApertureScale,
            secondApertureScale=SecondApertureScale, secondWeight=SecondWeight)
stem = Path(TargetImage).stem + "_DUALORDER_" + SecondOrder
G.writeCGHToFile(stem + "_CGH.bmp")
print("Wrote " + stem + "_CGH.bmp")

f1 = G.filterPlaneGeometry(F1)
f2 = G.secondFilterGeometry(F1)
print("Fourier plane of L1 (f = %.0f mm):" % (F1*1e3))
print("  pinhole 1: centre (0, 0), aperture %.3f x %.3f mm" % (f1["aperture"][0]*1e3, f1["aperture"][1]*1e3))
print("  pinhole 2: centre (+-%.3f, +-%.3f) mm (any sign combination), aperture %.3f x %.3f mm"
      % (f2["center"][0]*1e3, f2["center"][1]*1e3, f2["aperture"][0]*1e3, f2["aperture"][1]*1e3))
print("Both beams image onto the same camera plane: view one pinhole at a time, or pick\n"
      "pinhole 2 off right behind the filter plane into its own L2 + camera.")

def display_normalize(image, high_percentile=99.5):
        image = np.asarray(image, dtype=np.float32)
        lo = float(np.percentile(image, 1.0))
        hi = float(np.percentile(image, high_percentile))
        if hi <= lo:
                return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        image = np.clip(image, lo, hi)
        return ((image - lo) / (hi - lo)).astype(np.float32)

cv2.destroyAllWindows()
cv2.imshow("Pinhole 1 (on axis)", display_normalize(G.imRecovered_disc))
cv2.imshow("Pinhole 2 (%s order)" % SecondOrder, display_normalize(G.imRecovered2))
cv2.waitKey(0)
