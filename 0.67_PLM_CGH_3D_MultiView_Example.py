from TIPLMSuiteHologram3D import MultiViewHologramGenerator, DeviceLibrary
import cv2
import numpy as np
from pathlib import Path

# Full-parallax 3D hologram of a textured FBX model over the PLM's whole
# diffraction cone (see MultiViewHologramGenerator for the physics).
#
# The Fourier plane of L1 is cut into Views x Views sub-pupils that tile the
# entire first diffraction zone (lambda f1 / p square, 5.86 mm at f1 = 100 mm);
# looking through one sub-pupil shows the model from that direction. The
# four corner sub-pupils are blocked on the bench: a phase-only PLM is
# uniformly bright, so the light that the dark background must not show has
# to leave through them.
#
# Measured, macdonald cat, 632.8 nm theoretical LUT, magnification 0.1,
# PSNR per view with the camera pixel = view pixel (~2.3 min):
#   3 x 3 (5 views, 453 x 267 points each)   23.0 dB    (80 iterations)
#   4 x 4 (12 views, 340 x 200)              21.7 dB    (80 iterations)
#   5 x 5 (21 views, 272 x 160)  default     21.8 dB    (20.1 .. 22.8, full run)
#   7 x 7 (45 views, 194 x 115)              20.4 dB    (80 iterations)
# Wave-optics check (Collins, exact square mirrors, physical rectangular
# pinholes): 49-52 dB agreement with the model for interior views, 30 dB at
# the zone edge. Window reflections (0.5% faces) only reach the centre view.

# Define input parameters
Model = "./source/macdonald cat.fbx"   # binary FBX; texture embedded or in ./textures
Views = (5, 5)                # sub-pupils across x, y (both tile the whole zone)
Blocked = "corners"           # the light dump; or a list of (column, row) sub-pupils
F1 = 0.100                    # first 4f lens, metres
F2 = 0.050                    # second 4f lens, metres. Viewing cone = 3.36 deg * F1 / F2
                              # (etendue: 0.5 -> 6.7 deg, 7.3 x 4.3 mm image;
                              #           0.1 -> 33.6 deg, 1.5 x 0.9 mm image)
AngleScale = 1.0              # 1 = true 3D at that magnification; > 1 exaggerates the turn
ObjectHeight = 0.8            # fraction of the frame height
Yaw, Pitch = 0.0, 0.0         # turn the model first (degrees)

D = DeviceLibrary()
DeviceDict = D.defineDevice('0.67_632.8nm', measuredLUT=False)
print("Phase LUT scheme: " + DeviceDict["discScheme"])

G = MultiViewHologramGenerator()
G.createCGH(DeviceDictionary=DeviceDict, filename=Model, views=Views, blocked=Blocked,
            magnification=F2 / F1, angleScale=AngleScale, objectHeight=ObjectHeight,
            yaw=Yaw, pitch=Pitch, FlipLR=True, FlipUD=False)
stem = Path(Model).stem.replace(" ", "_") + "_3D_%dx%d" % tuple(np.broadcast_to(Views, 2))
G.writeCGHToFile(stem + "_CGH.bmp")
G.saveViewSheet(stem + "_views.png")
G.saveViewSweep(stem + "_sweep.gif")
print("Wrote %s_CGH.bmp, %s_views.png (targets | simulated views), %s_sweep.gif" % (stem, stem, stem))

geo = G.viewGeometry(F1, F2)
print("Fourier plane of L1 (f = %.0f mm): zone aperture %.3f x %.3f mm (blocks the other zones)"
      % (F1 * 1e3, geo["zoneAperture"][0] * 1e3, geo["zoneAperture"][1] * 1e3))
print("  sub-pupil (column, row): centre (x, y) mm, size mm, view direction at the image (deg)")
for g in geo["views"]:
    print("  (%d, %d): (%+.3f, %+.3f)  %.3f x %.3f  (%+.2f, %+.2f)%s"
          % (g["view"] + tuple(1e3 * np.array(g["center"])) + tuple(1e3 * np.array(g["size"]))
             + g["angle_deg"] + ("   BLOCKED (light dump)" if g["blocked"] else "",)))
print("Image %.2f x %.2f mm, viewing cone %.1f x %.1f deg. On the bench: a square pinhole one sub-pupil\n"
      "wide on an XY stage in the Fourier plane (camera at the image plane) shows one view per\n"
      "position; or open the zone aperture (corners blocked) and look at the image plane from\n"
      "a distance with a small camera aperture, moving it sideways."
      % (geo["imageSize"][0] * 1e3, geo["imageSize"][1] * 1e3, geo["cone_deg"][0], geo["cone_deg"][1]))

cv2.destroyAllWindows()
cv2.imshow("Views: targets | simulated", cv2.imread(stem + "_views.png"))
cv2.waitKey(0)
