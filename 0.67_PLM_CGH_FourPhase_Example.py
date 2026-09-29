from TIPLMSuiteFourPhase import FourPhaseCGHGenerator, DeviceLibrary
import cv2
import numpy as np
from pathlib import Path

# Four-phase encoding (Bass et al., SPIE 13918, 2026): 2x2 superpixels, every
# pixel free to take any of the 16 device states, spatial filter ON AXIS one
# superpixel order wide. The four phasors are averaged, giving 3876 distinct
# complex values, mapped with a KD tree.
#
#   Alg = "DIRECT" -- image at the 4f exit (paper Fig. 2a-c)
#   Alg = "GS"     -- Gerchberg-Saxton to a plane PropDistance beyond the 4f
#                     exit (paper Fig. 2d-h; speckled, as in the paper)
#
# Measured, harper grayscale, 632.8 nm theoretical LUT, DIRECT:
#   ApertureScale 1.0 : 21.6 dB band-limited 4f (26.9 dB ideal superpixel average)
#   ApertureScale 0.6 : 23.0 dB -- a pinhole narrower than one order passes
#                       less of the odd 2x2 modes, which leak through their
#                       gradients (see the class docstring)
#   GS at 0.29 m, 20 iterations: 11.5 dB band-limited 4f (19.2 dB ideal)

# Define input parameters
TargetImage = "./harper_sketch_1358x800_grayscale_tiled.bmp"   # define target image
ColorChannel = "gray"                            # any PNG/JPG/BMP/TIFF: 'gray' -> luminance; 0/1/2 -> R/G/B
PreserveAspect = True                            # letterbox instead of stretching to the PLM aspect
Alg = "DIRECT"                                   # "DIRECT" or "GS"
AmplitudeScale = "auto"                          # radius in the gamut; "auto" scores the optics
ApertureScale = 1.0                              # pinhole width, fraction of one superpixel order
PropDistance = 0.29                              # GS only: metres beyond the 4f exit (PLM-referred)
NumIter = 20                                     # GS only
F1 = 0.060                                       # first 4f lens, metres
F2 = 0.050                                       # second 4f lens, metres
RandomSeed = 123
if(RandomSeed is not None):
        np.random.seed(int(RandomSeed))

D = DeviceLibrary()
DeviceDict = D.defineDevice('0.67_632.8nm', measuredLUT=False)
print("Phase LUT scheme: " + DeviceDict["discScheme"])

G = FourPhaseCGHGenerator()
G.createCGH(DeviceDictionary=DeviceDict,
            filename=TargetImage, colorChannel=ColorChannel,
            FlipUD=False, FlipLR=True, preserveAspect=PreserveAspect,
            alg=Alg, numIter=NumIter,
            amplitudeScale=AmplitudeScale, apertureScale=ApertureScale,
            propDistance=PropDistance if Alg == "GS" else 0.0)
stem = Path(TargetImage).stem + "_FOURPHASE_" + Alg
G.writeCGHToFile(stem + "_CGH.bmp")
print("Wrote " + stem + "_CGH.bmp")

# The filter sits ON the optical axis, one superpixel order wide. NB the
# passband therefore contains the zero order: unmodulated mirror-gap and
# cover-glass light lands inside it.
flt = G.filterPlaneGeometry(F1)
img = G.imagePlaneGeometry(F1, F2)
print("Layout: PLM -%.0fmm- L1 -%.0fmm- FILTER -%.0fmm- L2 -%.0fmm- camera (track %.0fmm)"
      % (F1*1e3, F1*1e3, F2*1e3, F2*1e3, 2*(F1+F2)*1e3))
print("Filter center:  (%.3f, %.3f) mm - ON AXIS" % (flt["center"][0]*1e3, flt["center"][1]*1e3))
print("Filter aperture: %.3f x %.3f mm" % (flt["aperture"][0]*1e3, flt["aperture"][1]*1e3))
print("Image at 4f exit: %.2f x %.2f mm, superpixel %.2f um, resolution %.2f um, DOF +/-%.2f mm"
      % (img["imageSize"][0]*1e3, img["imageSize"][1]*1e3, img["superpixelPitch"][0]*1e6,
         img["resolution"]*1e6, img["depthOfFocus"]*1e3))
if Alg == "GS":
        print("GS image plane: %.0f mm beyond the 4f exit (x magnification^2 = %.0f mm physical)"
              % (PropDistance*1e3, PropDistance*(F2/F1)**2*1e3))

def display_normalize(image, high_percentile=99.5):
        image = np.asarray(image, dtype=np.float32)
        lo = float(np.percentile(image, 1.0))
        hi = float(np.percentile(image, high_percentile))
        if hi <= lo:
                return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        image = np.clip(image, lo, hi)
        return ((image - lo) / (hi - lo)).astype(np.float32)

cv2.destroyAllWindows()
cv2.imshow("Four-phase reconstruction (%s)" % Alg, display_normalize(G.imRecovered_disc))
cv2.waitKey(0)
