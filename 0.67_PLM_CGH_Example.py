from TIPLMSuite import CGHGenerator, DeviceLibrary
import cv2
import numpy as np
from pathlib import Path

try:
        import torch
except ImportError:
        torch = None

# create instance of TI CGHGenerator and DeviceLibrary
G = CGHGenerator()
D = DeviceLibrary()

# Define input parameters
TargetImage = "./harper_sketch_1358x800_grayscale_tiled.bmp"                   # define target image
RandomSeed = 123

# Measured illumination profile (camera in the loop, plm_illumination_profile.py).
# The laser illumination on the PLM is not uniform, so optimizing against the
# measured profile improves what the camera actually sees at the 4f image plane.
#   False  = assume uniform illumination (legacy behavior)
#   "auto" = use measured_illumination/illumination_profile.npy if it exists
#   True   = require the measured profile (error if missing)
#   or a path to a specific intensity .npy/.png profile
UseIlluminationProfile = "auto"
if(RandomSeed is not None):
        np.random.seed(int(RandomSeed)) 
        if(torch is not None):
                torch.manual_seed(int(RandomSeed))

# Call create CGH
G.createCGH(DeviceDictionary = D.defineDevice('0.67'), # Device Dictionary
            filename=TargetImage, bitPlanes =1, colorChannel = 0, # Image based parameters
            FlipUD = True, # 0.67 EVM requires an image flip.
            binarizeTarget=False, targetThreshold=0.5, # Treat JPEG compression shades as black/white.
            preserveAspect=False, # Fill the full 1358x800 PLM frame with the target image.
            illumination=UseIlluminationProfile, # Optimize against the measured beam profile.
            alg = 'ADAM', propMethod='Fourier', numIter=30000, showImages=False) # Algorithm based parameters

G.writeCGHToFile(Path(TargetImage).stem + "_CGH.bmp") # Write the TI PLM mapped CGH to a file. This can also be extracted through G.CGH_mapped

# Optional: Display the CGH using OpenCV. This displays the continuous hologram.
def display_normalize(image, high_percentile=99.5):
        image = np.asarray(image, dtype=np.float32)
        lo = float(np.percentile(image, 1.0))
        hi = float(np.percentile(image, high_percentile))
        if hi <= lo:
                return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        image = np.clip(image, lo, hi)
        return ((image - lo) / (hi - lo)).astype(np.float32)

cv2.destroyAllWindows()
cv2.imshow("Final Image", display_normalize(G.imRecovered_disc))
cv2.waitKey(0)
