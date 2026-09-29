from TIPLMSuiteFourPhaseDualOrder import DualOrderFourPhaseCGHGenerator, DeviceLibrary
from TIPLMSuiteFourPhaseCurvedBeam import CurvedBeamLayout, CurvedBeamSimulator, plm067WindowReflections
import cv2
import numpy as np

# What a converging or diverging input beam does to the four-phase 4f,
# checked with wave optics (TIPLMSuiteFourPhaseCurvedBeam). Short version:
#   * image quality does not change -- IF the pinholes move to where the
#     curved beam focuses the spectrum: s = f1 (1 - f1 / D) behind L1 when the
#     PLM sits at L1's front focus (same transverse positions and sizes);
#   * pinholes left at the collimated plane lose a lot (dual-order image 2:
#     -0.4 dB at 0.1 mm off, -2.5 dB at 0.5 mm, i.e. |D| < ~36 m already hurts);
#   * the package window's reflections turn from a uniform offset into
#     Newton-ring fringes: -0.3 dB collimated vs -3.6 dB at |D| = 0.5 m and
#     -6.6 dB at 0.2 m for 0.5% AR faces (pinhole 2 is immune);
#   * with the PLM NOT at L1's front focus the curvature also zooms the
#     filter plane (pinhole spacing and size scale together).
# Keep the beam collimated unless you need the zoom.

# Define input parameters
TargetImage = "./bear1.png"     # pinhole 1, on axis
SecondImage = "./bear2.png"     # pinhole 2
ColorChannel = "gray"
PreserveAspect = True
SourceDistance = -0.5           # metres: + converges to a point this far in front of
                                # the PLM, - diverges from a point this far behind it,
                                # np.inf = collimated
F1 = 0.060                      # first 4f lens, metres
F2 = 0.050                      # second 4f lens, metres
D1 = None                       # PLM -> L1 distance; None = F1 (PLM at L1's front focus)
WindowReflectance = 0.005       # per window face vs the mirrors (AR ~0.005, bare 0.04); None = ignore

D = DeviceLibrary()
DeviceDict = D.defineDevice('0.67_632.8nm', measuredLUT=False)

G = DualOrderFourPhaseCGHGenerator()
G.createCGH(DeviceDictionary=DeviceDict,
            filename=TargetImage, secondFilename=SecondImage, colorChannel=ColorChannel,
            FlipUD=False, FlipLR=True, preserveAspect=PreserveAspect)

windows = None if WindowReflectance is None else plm067WindowReflections(WindowReflectance)
L = CurvedBeamLayout(F1, F2, SourceDistance, d1=D1)
rep = L.report(G.apertureScale, G.secondApertureScale, G.secondOrder, windows=windows)
print("\nLayout for a source at %s:" % ("infinity (collimated)" if np.isinf(SourceDistance)
                                          else "%+.0f mm" % (SourceDistance * 1e3)))
print("  filter plane %.2f mm after L1 (%+.2f mm vs collimated); spectral zoom x%.3f"
      % (rep["filterDistanceAfterL1"] * 1e3, rep["filterShiftFromCollimated"] * 1e3, rep["zoomVsCollimated"]))
for name, ph in rep["pinholes"].items():
    print("  %s: centre (+-%.3f, +-%.3f) mm, %.3f mm square" % (name, ph["center"][0] * 1e3,
          ph["center"][1] * 1e3, ph["width"] * 1e3))
print("  camera %.2f mm after L2, magnification %.3f" % (rep["cameraDistanceAfterL2"] * 1e3, rep["magnification"]))
print("  max incidence on the array %.2f deg (phase depth -%.3f%%)"
      % (rep["maxIncidenceDeg"], rep["maxPhaseDepthError"] * 100))
print("  beam footprint: L1 %.1f mm, L2 %.1f mm%s" % (rep["beamFootprint"]["L1"] * 1e3, rep["beamFootprint"]["L2"] * 1e3,
      "  <-- exceeds a 1-inch lens" if any(rep["vignetting"].values()) else ""))

S = CurvedBeamSimulator(G, L, windows=windows)
rays = S.rayCheck()
if not np.isinf(SourceDistance):
    print("  measured illumination wavevectors (%s) meet %.3f..%.3f mm from the PLM; after L1 they cross"
          " the filter plane within %.1e m of the axis" % (rays["kind"], rays["convergesAt"].min() * 1e3,
          rays["convergesAt"].max() * 1e3, rays["filterPlaneSpread"].max()))

print("\nWave-optics PSNR (image 1 / image 2):")
ref = CurvedBeamSimulator(G, CurvedBeamLayout(F1, F2, np.inf, d1=D1), windows=windows).psnr()[0]
print("  collimated reference            %.2f / %.2f dB" % (ref["pinhole1"], ref["pinhole2"]))
here, imgs = S.psnr()
print("  pinholes at the new filter plane %.2f / %.2f dB" % (here["pinhole1"], here["pinhole2"]))
off = F1 - L.filterDistance
if abs(off) > 1e-9 and (D1 is None or D1 == F1):
    stay = S.psnr(off)[0]
    print("  pinholes left at the collimated plane (%.2f mm off) %.2f / %.2f dB"
          % (off * 1e3, stay["pinhole1"], stay["pinhole2"]))

def display_normalize(image, high_percentile=99.5):
        image = np.asarray(image, dtype=np.float32)
        lo = float(np.percentile(image, 1.0))
        hi = float(np.percentile(image, high_percentile))
        if hi <= lo:
                return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        image = np.clip(image, lo, hi)
        return ((image - lo) / (hi - lo)).astype(np.float32)

cv2.destroyAllWindows()
cv2.imshow("Pinhole 1, curved beam", display_normalize(np.fliplr(imgs["pinhole1"])))
cv2.imshow("Pinhole 2, curved beam", display_normalize(np.fliplr(imgs["pinhole2"])))
cv2.waitKey(0)
