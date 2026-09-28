import numpy as np
import cv2
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tqdm import tqdm
from PIL import Image, ImageOps

try:
    import torch
    from torch import nn
except ImportError:
    # Only the ADAM / ADAMwGS optimisers need PyTorch; the four-phase
    # encoders (TIPLMSuiteFourPhase*) run without it.
    torch = None

    class nn:
        class Module:
            def __init__(self, *args, **kwargs):
                raise ImportError("This optimiser needs PyTorch: pip install torch")


class DeviceLibrary:
    """
    Device library trimmed to the 0.67 TI PLM only.
    """

    PHASE_LUT_REFERENCE_WAVELENGTH_M = 532e-9
    PHASE_LUT_WAVELENGTHS_M = {
        "EMPIRICAL": 532e-9,
        "EMPIRICAL_532NM": 532e-9,
        "EMPIRICAL_632.8NM": 632.8e-9,
        "EMPIRICAL_6328NM": 632.8e-9,
        "EMPIRICAL_633NM": 632.8e-9,
        "EMPIRICAL_650NM": 650e-9,
    }
    PHASE_LUT_SCHEME_LABELS = {
        "EMPIRICAL": "Empirical_532nm",
        "EMPIRICAL_532NM": "Empirical_532nm",
        "EMPIRICAL_632.8NM": "Empirical_632.8nm",
        "EMPIRICAL_6328NM": "Empirical_632.8nm",
        "EMPIRICAL_633NM": "Empirical_632.8nm",
        "EMPIRICAL_650NM": "Empirical_650nm",
    }
    DEVICE_ALIASES = {
        "0.67": None,
        "0.67_532NM": "EMPIRICAL_532NM",
        "0.67_632.8NM": "EMPIRICAL_632.8NM",
        "0.67_6328NM": "EMPIRICAL_6328NM",
        "0.67_633NM": "EMPIRICAL_633NM",
        "0.67_650NM": "EMPIRICAL_650NM",
    }
    BASE_PHASE_LUT_532NM = np.array([
        0.0000,
        0.0118,
        0.0242,
        0.0464,
        0.0665,
        0.0823,
        0.1295,
        0.2018,
        0.3069,
        0.3384,
        0.3941,
        0.4730,
        0.5546,
        0.6309,
        0.7738,
        0.9375,
        1.0000
    ], dtype=np.float64)

    # Measured phase LUTs (from plm_lut_calibration.py) live here as
    # measured_lut_<tag>nm.txt, one file per wavelength (tag e.g. "632p8").
    # Measured LUTs are opt-in: defineDevice defaults to the standard
    # theoretical rescale of the empirical 532 nm table because the tuned
    # LUT measurement is suspected to be inaccurate (camera saturation).
    # Pass measuredLUT="auto", True, or a path to use a measured table.
    MEASURED_LUT_DIR = Path(__file__).resolve().parent / "measured_luts"

    @staticmethod
    def _wavelength_tag(wavelength_m):
        return f"{wavelength_m * 1e9:g}".replace(".", "p")

    @classmethod
    def measured_lut_path(cls, wavelength_m):
        """Conventional path of the measured LUT for a given wavelength."""
        return cls.MEASURED_LUT_DIR / f"measured_lut_{cls._wavelength_tag(wavelength_m)}nm.txt"

    @classmethod
    def _read_lut_file(cls, path, nLevel):
        """Load a normalized phase LUT (dphi/2pi) from a text file.

        Accepts nLevel values (levels 0..nLevel-1) or nLevel+1 values (trailing
        1.0 sentinel, as written by plm_lut_calibration.py). '#' comment lines
        are ignored. Returns an (nLevel+1,) array with a 1.0 sentinel appended.
        """
        values = np.loadtxt(path, dtype=np.float64).reshape(-1)
        if values.size == nLevel:
            values = np.append(values, 1.0)
        if values.size != nLevel + 1:
            raise ValueError(
                f"Measured LUT {path} must have {nLevel} or {nLevel + 1} values, "
                f"got {values.size}."
            )
        if np.any(values < -1e-6) or np.any(values > 1 + 1e-6):
            raise ValueError(f"Measured LUT {path} values must be normalized to [0, 1].")
        values = np.clip(values, 0.0, 1.0)
        values[-1] = 1.0
        return values

    @classmethod
    def _resolve_measured_lut(cls, wavelength_m, nLevel, measuredLUT):
        """Resolve the measuredLUT option to a pLevel array (or None).

        measuredLUT:
            False / None (default) -- force the theoretical rescale (skip measured LUT)
            "auto"           -- use the conventional file if it exists, else None
            True             -- require the conventional file (error if missing)
            path-like        -- load that specific LUT file
        Returns (pLevel_or_None, path_or_None).
        """
        if measuredLUT is False or measuredLUT is None:
            return None, None
        if isinstance(measuredLUT, str) and measuredLUT.lower() == "auto":
            path = cls.measured_lut_path(wavelength_m)
            if not path.exists():
                return None, None
            return cls._read_lut_file(path, nLevel), path
        if measuredLUT is True:
            path = cls.measured_lut_path(wavelength_m)
            if not path.exists():
                raise FileNotFoundError(
                    f"measuredLUT=True but no measured LUT found at {path}. "
                    "Run plm_lut_calibration.py on hardware first."
                )
            return cls._read_lut_file(path, nLevel), path
        path = Path(measuredLUT)
        if not path.exists():
            raise FileNotFoundError(f"Measured LUT not found: {path}")
        return cls._read_lut_file(path, nLevel), path

    @classmethod
    def _phase_lut_for_wavelength(cls, wavelength_m):
        pLevel = cls.BASE_PHASE_LUT_532NM.copy()
        if not np.isclose(wavelength_m, cls.PHASE_LUT_REFERENCE_WAVELENGTH_M):
            pLevel[:-1] = np.remainder(
                pLevel[:-1] * cls.PHASE_LUT_REFERENCE_WAVELENGTH_M / wavelength_m,
                1
            )
            pLevel[-1] = 1.0
        return pLevel

    def defineDevice(self, device, discScheme="Empirical", lambdaGlobal=None, lambdaM=None,
                     measuredLUT=False):
        device = str(device).upper()
        discScheme = str(discScheme).upper()

        if device not in self.DEVICE_ALIASES:
            raise ValueError("TIPLMSuite only supports the 0.67 device.")
        alias_disc_scheme = self.DEVICE_ALIASES[device]
        if alias_disc_scheme is not None and discScheme == "EMPIRICAL":
            discScheme = alias_disc_scheme
        if discScheme not in self.PHASE_LUT_WAVELENGTHS_M:
            raise ValueError("TIPLMSuite includes empirical 0.67 phase tables for 532 nm, 632.8 nm, and 650 nm.")

        phase_lut_wavelength_m = self.PHASE_LUT_WAVELENGTHS_M[discScheme]
        phase_lut_scheme_label = self.PHASE_LUT_SCHEME_LABELS[discScheme]

        # Prefer a hardware-measured LUT for this wavelength when available;
        # otherwise fall back to rescaling the empirical 532 nm table.
        measured_pLevel, measured_path = self._resolve_measured_lut(
            phase_lut_wavelength_m, 16, measuredLUT)
        if measured_pLevel is not None:
            pLevel = measured_pLevel
            phase_lut_scheme_label = f"{phase_lut_scheme_label}_MeasuredLUT"
            print(f"TIPLMSuite: using measured phase LUT {measured_path}")
        else:
            pLevel = self._phase_lut_for_wavelength(phase_lut_wavelength_m)

        if lambdaGlobal is not None and lambdaGlobal > 0:
            reference_wavelength_m = phase_lut_wavelength_m if lambdaM is None else lambdaM
            pLevel[:-1] = np.remainder(pLevel[:-1] * lambdaGlobal / reference_wavelength_m, 1)
            pLevel[-1] = 1.0

        return {
            "device": "0.67",
            "discScheme": phase_lut_scheme_label,
            "lambda_m": phase_lut_wavelength_m,
            "phase_lut_wavelength_m": phase_lut_wavelength_m,
            "pitchW": 10.8e-6,
            "pitchH": 10.8e-6,
            "w": 1358,
            "h": 800,
            "nLevel": 16,
            "pLevel": pLevel,
            "memory_layout": [[1, 3], [0, 2]],
            "memory_lut": [3, 2, 1, 7, 0, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12],
            "evm_padding": 0,
            "evm_fliplr": True,
            "evm_flipud": False
        }

    def formatPLM(self, DeviceDictionary, CGH_state_map):
        if str(DeviceDictionary["device"]).upper() != "0.67":
            raise ValueError("TIPLMSuite only supports 0.67 PLM formatting.")
        return self.formatPLM_p67(CGH_state_map, DeviceDictionary["nLevel"])

    def formatPLM_p67(self, CGH_state_map, nLevel):
        if nLevel != 16:
            raise ValueError("0.67 PLM formatting expects 16 phase levels.")

        electrode_bits = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 1, 0, 0],
        ], dtype=np.float64)

        stateq = np.clip(CGH_state_map.astype(int), 0, nLevel - 1)
        rows, cols = stateq.shape
        pslm_logical = np.zeros((rows * 2, cols * 2), dtype=np.float64)

        pslm_logical[0::2, 1::2] = electrode_bits[stateq, 0]  # E3
        pslm_logical[1::2, 1::2] = electrode_bits[stateq, 1]  # E2
        pslm_logical[0::2, 0::2] = electrode_bits[stateq, 2]  # E1
        pslm_logical[1::2, 0::2] = electrode_bits[stateq, 3]  # E0

        return np.fliplr(pslm_logical) * 255


class CGHGenerator:
    """
    CGH generator trimmed to the 0.67 device and Fourier ADAM/ADAMwGS.
    """
    SCALE_EPS = 1e-20

    # Measured illumination profile (from plm_illumination_profile.py) lives at
    # measured_illumination/illumination_profile.npy (intensity, PLM logical
    # grid). Opt-in via createCGH(illumination=...): "auto" uses the published
    # profile when present, True requires it, a path/array supplies one
    # directly; False/None (default) assumes uniform illumination.
    MEASURED_ILLUMINATION_FILE = (Path(__file__).resolve().parent
                                  / "measured_illumination" / "illumination_profile.npy")

    def __init__(self):
        self.CGH_mapped = None
        self.CGH_phase = None
        self.CGH_output_cont = None
        self.CGH_output_phase_disc = None
        self.CGH_output_state_disc = None
        self.CGH_output_disc = None
        self.imRecovered_cont = None
        self.imRecovered_disc = None
        self.seedPhase = None
        self.imTarget = None
        self.imTarget_orig = None
        self.illumAmplitude = None
        self.deviceLibary = DeviceLibrary()
        self.lambda_m = 632.8e-9
        self.pitchW = None
        self.pitchH = None
        self.bitPlanes = 1
        self.show_images = False
        self.lossMode = "auto"
        self.MLDevice = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                         if torch is not None else "cpu")

    def loadTarget(self, filename="", colorChannel=0):
        if filename != "":
            fn = filename
        else:
            Tk().withdraw()
            fn = askopenfilename(title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])

        grayscale_keys = {"gray", "grey", "l", "luma", "luminance"}
        use_grayscale = colorChannel is None or (
            isinstance(colorChannel, str) and colorChannel.lower() in grayscale_keys
        )
        if not use_grayscale:
            if isinstance(colorChannel, str):
                raise ValueError("colorChannel must be 0, 1, 2, or 'gray'.")
            if colorChannel < 0 or colorChannel > 2:
                raise ValueError("colorChannel must be 0, 1, 2, or 'gray'.")

        try:
            with Image.open(fn) as pil_image:
                pil_image = ImageOps.exif_transpose(pil_image)
                if use_grayscale or pil_image.mode in ("1", "L", "I", "I;16", "F"):
                    image = np.asarray(pil_image.convert("L"), dtype=np.float64) / 255.0
                    self.imTarget_orig = image
                    self.imTarget = self.imTarget_orig
                    return

                image = np.asarray(pil_image.convert("RGB"), dtype=np.float64) / 255.0
        except Exception as exc:
            raise FileNotFoundError(f"Could not load image: {fn}") from exc

        self.imTarget_orig = image[:, :, colorChannel]
        self.imTarget = self.imTarget_orig

    def resizeTarget(self, lambda_m, h, w, pitchW, pitchH, ShiftFOV=True, preserveAspect=True):
        image = self.imTarget
        source_shape = np.shape(image)

        if preserveAspect:
            fov_aspect_ratio = np.arcsin(lambda_m / pitchW) / np.arcsin(lambda_m / pitchH)
            source_h, source_w = source_shape[:2]
            source_aspect_ratio = source_w / source_h

            def split_padding(delta):
                before = int(delta // 2)
                return before, int(delta - before)

            if source_aspect_ratio < fov_aspect_ratio:
                target_w = int(np.round(source_h * fov_aspect_ratio))
                pad_left, pad_right = split_padding(max(target_w - source_w, 0))
                image = np.pad(image, ((0, 0), (pad_left, pad_right)), "constant")
            elif source_aspect_ratio > fov_aspect_ratio:
                target_h = int(np.round(source_w / fov_aspect_ratio))
                pad_top, pad_bottom = split_padding(max(target_h - source_h, 0))
                image = np.pad(image, ((pad_top, pad_bottom), (0, 0)), "constant")

        if ShiftFOV:
            image = np.roll(image, shift=(int(np.round(np.shape(image)[0] / 2)), int(np.round(np.shape(image)[1] / 2))), axis=(0, 1))

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        den = np.max(image) - np.min(image)
        if den < 1e-12:
            image = np.zeros_like(image)
        else:
            image = (image - np.min(image)) / den
        self.imTarget = image

    def binarizeTarget(self, threshold=0.5):
        if self.imTarget is None:
            return
        self.imTarget = (self.imTarget >= threshold).astype(np.float64)

    def _targetIsBinary(self):
        if self.imTarget is None:
            return False
        return np.all((self.imTarget == 0) | (self.imTarget == 1))

    def _resolveLossMode(self, lossMode):
        mode = str(lossMode).replace("-", "_").replace(" ", "_").lower()
        if mode == "auto":
            return "balanced" if self._targetIsBinary() else "mse"
        if mode not in ("mse", "balanced"):
            raise ValueError("lossMode must be 'auto', 'mse', or 'balanced'.")
        return mode

    @staticmethod
    def _torchLossWeights(target, lossMode, eps=1e-8):
        if lossMode != "balanced":
            return None

        foreground = target > 0.5
        background = ~foreground
        fg_fraction = torch.clamp(foreground.float().mean(), min=eps)
        bg_fraction = torch.clamp(background.float().mean(), min=eps)

        if bool((foreground.any() & background.any()).detach().cpu()):
            weights = torch.where(foreground, 0.5 / fg_fraction, 0.5 / bg_fraction)
            return weights / torch.clamp(weights.mean(), min=eps)

        return None

    @staticmethod
    def _torchWeightedGain(prediction, target, weights=None, eps=1e-20):
        if weights is None:
            denom = torch.clamp((prediction ** 2).mean(dim=(-1, -2), keepdim=True), min=eps)
            return (prediction * target).mean(dim=(-1, -2), keepdim=True) / denom

        denom = torch.clamp((weights * prediction ** 2).mean(dim=(-1, -2), keepdim=True), min=eps)
        return (weights * prediction * target).mean(dim=(-1, -2), keepdim=True) / denom

    @staticmethod
    def _torchWeightedMSE(prediction, target, weights=None):
        error = (prediction - target) ** 2
        if weights is not None:
            error = weights * error
        return error.mean()

    def _resolveIllumination(self, illumination, shape):
        """Resolve the illumination option to an amplitude array (or None).

        Delegates to plm_illumination_profile.load_illumination_amplitude:
        intensity is loaded, resized to the hologram grid, square-rooted, and
        normalized to unit mean power (mean(amp^2) = 1), so reconstructions
        keep the same total power as the uniform-illumination case.
        """
        if illumination is False or illumination is None:
            return None
        from plm_illumination_profile import load_illumination_amplitude
        amplitude, path = load_illumination_amplitude(
            illumination, shape=shape, normalize="power",
            published_path=self.MEASURED_ILLUMINATION_FILE)
        if amplitude is not None and path is not None:
            print(f"TIPLMSuite: using measured illumination profile {path}")
        return amplitude

    def updateTarget(self, FlipUD=False, FlipLR=False, invertTarget=False):
        if FlipUD:
            self.imTarget = np.flipud(self.imTarget).copy()
        if FlipLR:
            self.imTarget = np.fliplr(self.imTarget).copy()
        if invertTarget:
            self.imTarget = 1 - self.imTarget.copy()

    def createCGH(self, DeviceDictionary,
                  filename="", bitPlanes=1, colorChannel=0,
                  FlipUD=False, FlipLR=False, InvertTarget=False,
                  alg="ADAMWGS", numIter=1, initialPhase="Random", propMethod="Fourier",
                  ShiftFOV=True, showImages=False, binarizeTarget=False, targetThreshold=0.5,
                  preserveAspect=True,
                  lossMode="auto",
                  illumination=False):
        if str(DeviceDictionary["device"]).upper() != "0.67":
            raise ValueError("TIPLMSuite only supports the 0.67 device.")
        alg_key = str(alg).upper()
        if alg_key not in ("ADAM", "ADAMWGS"):
            raise ValueError("TIPLMSuite only supports alg='ADAM' or alg='ADAMWGS'.")
        if torch is None:
            raise ImportError("alg='%s' needs PyTorch: pip install torch" % alg_key)
        if bitPlanes != 1:
            raise ValueError("TIPLMSuite only supports bitPlanes=1.")
        if str(propMethod).upper() != "FOURIER":
            raise ValueError("TIPLMSuite only supports propMethod='Fourier'.")

        self.show_images = showImages
        self.bitPlanes = bitPlanes
        device_lambda_m = DeviceDictionary.get("lambda_m", DeviceDictionary.get("phase_lut_wavelength_m"))
        if device_lambda_m is not None and float(device_lambda_m) > 0:
            self.lambda_m = float(device_lambda_m)
        self.loadTarget(filename, colorChannel)
        self.resizeTarget(
            self.lambda_m,
            DeviceDictionary["h"],
            DeviceDictionary["w"],
            DeviceDictionary["pitchW"],
            DeviceDictionary["pitchH"],
            ShiftFOV,
            preserveAspect,
        )
        self.pitchW = DeviceDictionary["pitchW"]
        self.pitchH = DeviceDictionary["pitchH"]
        if binarizeTarget:
            self.binarizeTarget(targetThreshold)
        self.updateTarget(FlipUD, FlipLR, InvertTarget)
        self.lossMode = self._resolveLossMode(lossMode)
        self.illumAmplitude = self._resolveIllumination(
            illumination, (DeviceDictionary["h"], DeviceDictionary["w"]))

        if alg_key == "ADAM":
            self.runADAM(initialPhase, numIter, propMethod, self.lossMode)
        else:
            self.runADAMwGS(DeviceDictionary["nLevel"], DeviceDictionary["pLevel"], initialPhase, numIter, propMethod, self.lossMode)

        self.CGH_output_cont = np.mod(self.CGH_output_cont, 2 * np.pi)
        self.CGH_output_phase_disc, self.CGH_output_state_disc = self.discretePhase(
            self.CGH_output_cont,
            DeviceDictionary["nLevel"],
            DeviceDictionary["pLevel"]
        )
        self.CGH_output_phase_disc = 2 * np.pi * self.CGH_output_phase_disc
        self.CGH_output_disc = self.CGH_output_state_disc

        self.recoverImg(ShiftFOV, propMethod)
        self.CGH_mapped = self.deviceLibary.formatPLM(DeviceDictionary, self.CGH_output_state_disc)
        self.CGH_phase = self.CGH_output_phase_disc

    def recoverImg(self, ShiftFOV=True, propMethod="FOURIER"):
        if self.imTarget is None:
            return

        I_cont, _ = self.reconstruct(self.CGH_output_cont, propMethod)
        I_disc, _ = self.reconstruct(self.CGH_output_phase_disc, propMethod)

        if ShiftFOV:
            shift = (int(self.imTarget.shape[0] / 2), int(self.imTarget.shape[1] / 2))
            I_cont = np.roll(I_cont, shift=shift, axis=(0, 1))
            I_disc = np.roll(I_disc, shift=shift, axis=(0, 1))

        self.imRecovered_cont = cv2.resize(I_cont, np.flip(self.imTarget.shape), interpolation=cv2.INTER_CUBIC)
        self.imRecovered_disc = cv2.resize(I_disc, np.flip(self.imTarget.shape), interpolation=cv2.INTER_CUBIC)

    def runADAM(self, initialPhase="Random", numIter=1, propMethod="Fourier", lossMode="mse"):
        if numIter < 1:
            raise ValueError("numIter must be >= 1.")

        if initialPhase == "Random":
            phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])
        elif initialPhase == "Unity":
            phase_init = 2 * np.pi * np.ones(self.imTarget.shape)
        elif initialPhase == "Custom":
            if self.seedPhase is None:
                print("Error: No seed phase defined. Using random seed")
                phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])
            elif self.seedPhase.shape != self.imTarget.shape:
                raise ValueError("Seed phase must be the same size as target image.")
            else:
                phase_init = self.seedPhase
        else:
            print("Error: Invalid initial phase seed. Using random seed")
            phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])

        msel = np.zeros((numIter, 1))
        psnrl = np.zeros((numIter, 1))

        phase_hp = torch.tensor(phase_init, dtype=torch.float32, device=self.MLDevice)
        phase_hp.requires_grad_(True)

        optimizer = torch.optim.Adam([{"params": phase_hp}], lr=0.32)
        target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)
        loss_weights = self._torchLossWeights(target_intensity, lossMode)
        illum_amp = None
        if self.illumAmplitude is not None:
            illum_amp = torch.tensor(self.illumAmplitude, dtype=torch.float32, device=self.MLDevice)

        best_loss = 1e10
        best_phase = None

        for i in tqdm(range(0, numIter)):
            optimizer.zero_grad()

            E_ip_inp = torch.exp(1j * phase_hp)
            if illum_amp is not None:
                E_ip_inp = illum_amp * E_ip_inp
            E_hp = self.torchProp("forward", E_ip_inp, propMethod)
            E_hp_intensity = torch.abs(E_hp) ** 2

            with torch.no_grad():
                s = self._torchWeightedGain(E_hp_intensity, target_intensity, loss_weights, self.SCALE_EPS)

            loss_val = self._torchWeightedMSE(s * E_hp_intensity, target_intensity, loss_weights)
            msel[i] = loss_val.item()
            psnrl[i] = 20 * np.log10(1 / np.sqrt(msel[i] + 1e-20))

            if self.show_images:
                image_recon = (s * E_hp_intensity).detach().cpu().numpy()
                cv2.imshow("ADAM Image", image_recon)
                cv2.setWindowTitle("ADAM Image", "Iteration: " + str(i + 1) + " MSE: " + str(msel[i]) + " PSNR: " + str(psnrl[i]))
                cv2.waitKey(0)

            loss_val.backward()
            optimizer.step()

            with torch.no_grad():
                if loss_val.item() < best_loss:
                    best_loss = loss_val.item()
                    best_phase = phase_hp.detach().clone()

        metric_name = "Balanced MSEL" if lossMode == "balanced" else "MSEL"
        psnr_name = "Balanced PSNR" if lossMode == "balanced" else "PSNR"
        print("Best " + metric_name + ": " + str(np.min(msel)))
        print("Best " + psnr_name + ": " + str(np.max(psnrl)))

        if best_phase is None:
            best_phase = phase_hp.detach().clone()

        self.CGH_output_cont = np.mod(best_phase.detach().cpu().numpy(), 2 * np.pi)

    def runADAMwGS(self, nLevel, pLevel, initialPhase="Random", numIter=1, propMethod="Fourier", lossMode="mse"):
        if numIter < 1:
            raise ValueError("numIter must be >= 1.")
        if np.asarray(pLevel).ndim != 1:
            raise ValueError("TIPLMSuite only supports the 1D 0.67 phase table.")

        if initialPhase == "Random":
            phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])
        elif initialPhase == "Unity":
            phase_init = 2 * np.pi * np.ones(self.imTarget.shape)
        elif initialPhase == "Custom":
            if self.seedPhase is None:
                print("Error: No seed phase defined. Using random seed")
                phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])
            elif self.seedPhase.shape != self.imTarget.shape:
                raise ValueError("Seed phase must be the same size as target image.")
            else:
                phase_init = self.seedPhase
        else:
            print("Error: Invalid initial phase seed. Using random seed")
            phase_init = 2 * np.pi * np.random.rand(self.imTarget.shape[0], self.imTarget.shape[1])

        msel = np.zeros((numIter, 1))
        deterministic_msel = np.zeros((numIter, 1))
        psnrl = np.zeros((numIter, 1))
        deterministic_psnrl = np.zeros((numIter, 1))

        phase_hp = torch.tensor(phase_init, dtype=torch.float32, device=self.MLDevice)
        phase_hp.requires_grad_(True)

        levels = np.asarray(pLevel[:nLevel]).reshape(1, 1, -1)
        logits = np.tile(levels, (self.imTarget.shape[0], self.imTarget.shape[1], 1))
        logits = torch.tensor(logits, device=self.MLDevice, dtype=torch.float32)

        tauInitial = torch.tensor(6.5, dtype=torch.float32, device=self.MLDevice)
        tauMin = torch.tensor(3.1, dtype=torch.float32, device=self.MLDevice)
        quantizeMethod = self.torchquantize(lut=logits, levels=nLevel, tau=6.5, quantizeMethod="gumbelsoftmax")
        optimizer = torch.optim.Adam([{"params": phase_hp}], lr=0.17)
        
        target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)
        loss_weights = self._torchLossWeights(target_intensity, lossMode)
        illum_amp = None
        if self.illumAmplitude is not None:
            illum_amp = torch.tensor(self.illumAmplitude, dtype=torch.float32, device=self.MLDevice)

        best_loss = 1e10
        best_quantized_phase = None

        for i in tqdm(range(0, numIter)):
            anneal_progress = 1.0 if numIter <= 1 else i / (numIter - 1)
            annealRate = torch.tensor(anneal_progress, dtype=torch.float32, device=self.MLDevice)
            quantizeMethod.anneal_temperature(annealRate, tauInitial=tauInitial, tauMin=tauMin)

            optimizer.zero_grad()
            quantized_phase = quantizeMethod(phase_hp)
            E_ip_inp = torch.exp(1j * quantized_phase)
            if illum_amp is not None:
                E_ip_inp = illum_amp * E_ip_inp
            E_hp = self.torchProp("forward", E_ip_inp, propMethod)

            #E_hp_amp = torch.abs(E_hp)
            intensity = torch.abs(E_hp) ** 2

            with torch.no_grad():
                s = self._torchWeightedGain(intensity, target_intensity, loss_weights, self.SCALE_EPS)

            loss_val = self._torchWeightedMSE(s * intensity, target_intensity, loss_weights)
            msel[i] = loss_val.item()
            psnrl[i] = 20 * np.log10(1 / np.sqrt(msel[i] + 1e-20))

            with torch.no_grad():
                deterministic_quantized_phase = quantizeMethod.deterministic(phase_hp)
                deterministic_E_ip_inp = torch.exp(1j * deterministic_quantized_phase)
                if illum_amp is not None:
                    deterministic_E_ip_inp = illum_amp * deterministic_E_ip_inp
                deterministic_E_out = self.torchProp("forward", deterministic_E_ip_inp, propMethod)
                deterministic_intensity = torch.abs(deterministic_E_out) ** 2
                deterministic_gain = self._torchWeightedGain(deterministic_intensity, target_intensity, loss_weights, self.SCALE_EPS)
                deterministic_loss_val = self._torchWeightedMSE(deterministic_gain * deterministic_intensity, target_intensity, loss_weights)
                deterministic_msel[i] = deterministic_loss_val.item()
                deterministic_psnrl[i] = 20 * np.log10(1 / np.sqrt(deterministic_msel[i] + 1e-20))

                if deterministic_loss_val.item() < best_loss:
                    best_loss = deterministic_loss_val.item()
                    best_quantized_phase = deterministic_quantized_phase.detach().clone()

            if self.show_images:
                image_recon = self.reconstruct(deterministic_quantized_phase.cpu().detach().numpy(), propMethod)[0]
                cv2.imshow("ADAMwGS Image", image_recon)
                cv2.setWindowTitle("ADAMwGS Image", "Iteration: " + str(i + 1) + " MSE: " + str(deterministic_msel[i]) + " PSNR: " + str(deterministic_psnrl[i]))
                cv2.waitKey(0)

            loss_val.backward()
            optimizer.step()

        metric_name = "Balanced MSEL" if lossMode == "balanced" else "MSEL"
        psnr_name = "Balanced PSNR" if lossMode == "balanced" else "PSNR"
        print("Best " + metric_name + ": " + str(np.min(deterministic_msel)))
        print("Best " + psnr_name + ": " + str(np.max(deterministic_psnrl)))
        print("Best Stochastic Training " + metric_name + ": " + str(np.min(msel)))
        print("Best Stochastic Training " + psnr_name + ": " + str(np.max(psnrl)))

        if best_quantized_phase is None:
            best_quantized_phase = quantizeMethod.deterministic(phase_hp).detach().clone()

        self.CGH_output_cont = np.mod(best_quantized_phase.detach().cpu().numpy(), 2 * np.pi)

    def prop(self, direction, field_in, algorithm="FOURIER"):
        direction = str(direction).upper()
        algorithm = str(algorithm).upper()

        if direction not in ("FORWARD", "BACKWARD"):
            raise ValueError("Propagation direction must be 'forward' or 'backward'.")
        if algorithm != "FOURIER":
            raise ValueError("TIPLMSuite only supports Fourier propagation.")

        if direction == "FORWARD":
            return np.fft.ifft2(field_in)
        return np.fft.fft2(field_in)

    def torchProp(self, direction, field_in, algorithm="FOURIER"):
        direction = str(direction).upper()
        algorithm = str(algorithm).upper()

        if direction not in ("FORWARD", "BACKWARD"):
            raise ValueError("Propagation direction must be 'forward' or 'backward'.")
        if algorithm != "FOURIER":
            raise ValueError("TIPLMSuite only supports Fourier propagation.")

        if direction == "FORWARD":
            return torch.fft.ifft2(field_in)
        return torch.fft.fft2(field_in)

    @staticmethod
    def score_phase(phase, lut, s=5.0, func="prelu"):
        func_key = str(func).replace("-", "_").replace(" ", "_").lower()
        wrapped_phase = ((phase + np.pi) % (2 * np.pi)) - np.pi
        wrapped_phase = wrapped_phase.unsqueeze(-1).repeat(1, 1, lut.shape[2])
        diff = wrapped_phase - 2 * np.pi * lut
        diff = ((diff + np.pi) % (2 * np.pi)) - np.pi
        diff /= np.pi

        if func_key == "sigmoid":
            z = s * diff
            scores = torch.sigmoid(z) * (1 - torch.sigmoid(z)) * 4
        elif func_key == "prelu" or func_key == "leaky_relu" or func_key == "leakyrelu":
            margin = 1 - diff.abs() * s
            scores = torch.where(margin < 0, margin, 0.100 * margin)
        elif func_key == "relu":
            scores = torch.relu(1 - diff.abs() * s)
        else:
            raise ValueError(f"Unknown scoring function: {func}")

        return scores * 350 * s

    class GumbelSoftmaxQuantization(nn.Module):
        def __init__(self, lut=None, num_levels=16, tau=1.0, hard=True, score_func="prelu"):
            super(CGHGenerator.GumbelSoftmaxQuantization, self).__init__()
            self.num_levels = num_levels
            self.tau = tau
            self.tauInit = tau
            self.hard = hard
            self.score_func = score_func
            if lut is not None:
                self.quant_levels = lut
            else:
                self.quant_levels = nn.Parameter(torch.linspace(0, 1, num_levels), requires_grad=False)

        def anneal_temperature(self, annealRate, tauInitial, tauMin):
            self.tau = tauInitial * torch.exp(-torch.log(tauInitial / tauMin) * annealRate)

        def forward(self, phase):
            scores = CGHGenerator.score_phase(phase, self.quant_levels, s=self.tauInit / self.tau, func=self.score_func)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.quant_levels) + 1e-20) + 1e-20)
            up_scores = (scores + gumbel_noise) / self.tau
            up_scores = up_scores - torch.max(up_scores, dim=2, keepdim=True)[0]
            soft_assignments = torch.nn.functional.softmax(up_scores, dim=2)

            if self.hard:
                indices = torch.argmax(soft_assignments, dim=2, keepdim=True)
                hard_assignments = torch.zeros_like(soft_assignments).scatter_(2, indices, 1.0)
                probs = hard_assignments.detach() + soft_assignments - soft_assignments.detach()
            else:
                probs = soft_assignments

            return torch.sum(2 * np.pi * probs * self.quant_levels, dim=2)

        def deterministic(self, phase):
            scores = CGHGenerator.score_phase(phase, self.quant_levels, s=self.tauInit / self.tau, func=self.score_func)
            indices = torch.argmax(scores, dim=2, keepdim=True)
            hard_assignments = torch.zeros_like(scores).scatter_(2, indices, 1.0)
            return torch.sum(2 * np.pi * hard_assignments * self.quant_levels, dim=2)

    def torchquantize(self, lut=None, levels=None, quantizeMethod="GUMBELSOFTMAX", tau=1.0):
        if str(quantizeMethod).upper() != "GUMBELSOFTMAX":
            raise ValueError("TIPLMSuite only supports Gumbel-softmax quantization.")
        return CGHGenerator.GumbelSoftmaxQuantization(lut=lut, num_levels=levels, tau=tau, hard=True, score_func="relu").to(self.MLDevice)

    def reconstruct(self, phase, propMethod="FOURIER"):
        if self.illumAmplitude is not None and self.illumAmplitude.shape == phase.shape:
            amplitude = self.illumAmplitude
        else:
            amplitude = np.ones(phase.shape)
        E_hp = amplitude * np.exp(1j * phase)
        E_ip = self.prop("forward", E_hp, propMethod)
        I = np.real(np.abs(E_ip) ** 2)

        if self.imTarget is not None:
            scale = np.mean(I * self.imTarget) / (np.mean(I ** 2) + 1e-10)
            I = scale * I

        return I, E_ip

    @staticmethod
    def quantizeCircularPhase(sig, state_levels):
        sig = np.asarray(sig, dtype=np.float64).reshape(-1)
        state_levels = np.mod(np.asarray(state_levels, dtype=np.float64).reshape(-1), 1.0)
        if state_levels.size == 0:
            raise ValueError("state_levels must contain at least one phase level.")

        diff = np.remainder(sig[:, None] - state_levels[None, :] + 0.5, 1.0) - 0.5
        indices = np.argmin(np.abs(diff), axis=1).astype(np.float64)
        quants = state_levels[indices.astype(np.int64)]
        return indices, quants

    def discretePhase(self, phase, nLevel, pLevel):
        phase_level = np.asarray(pLevel, dtype=np.float64)
        phaseq = np.zeros(np.shape(phase), dtype=np.float64)
        stateq = np.zeros(np.shape(phase), dtype=np.float64)
        phase = np.mod(phase, 2 * np.pi) / (2 * np.pi)

        if phase_level.ndim > 1:
            num_luts = phase_level.shape[0]
            for i in range(0, np.size(phase, 1)):
                lut_idx = 0 if num_luts == 1 else min(i % num_luts, num_luts - 1)
                state_levels = phase_level[lut_idx, :nLevel]
                stateq[:, i], phaseq[:, i] = self.quantizeCircularPhase(phase[:, i], state_levels)
        else:
            state_levels = phase_level[:nLevel]
            stateq_flat, phaseq_flat = self.quantizeCircularPhase(phase.reshape(-1), state_levels)
            stateq = stateq_flat.reshape(phase.shape)
            phaseq = phaseq_flat.reshape(phase.shape)

        phaseq = np.mod(phaseq, 1.0)
        stateq = np.mod(stateq, nLevel)
        return phaseq, stateq

    def writeCGHToFile(self, filename, binary=False):
        if self.CGH_mapped is None:
            raise ValueError("No CGH has been generated.")

        outputCGH = cv2.normalize(self.CGH_mapped, None, 0, 255, cv2.NORM_MINMAX)
        _, binaryCGH = cv2.threshold(outputCGH, 127, 255, cv2.THRESH_BINARY)
        binaryCGH = binaryCGH.astype(np.uint8)

        if not binary:
            cv2.imwrite(filename, binaryCGH, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        else:
            pilImage = Image.fromarray(binaryCGH)
            pilImage = pilImage.convert("1")
            pilImage.save(filename, "BMP")

    def writeVarToFile(self, filename, Var):
        outputCGH = cv2.normalize(Var, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(filename, outputCGH.astype(np.uint8))

    @staticmethod
    def quantiz(sig, partition, codebook):
        index = []
        quants = []
        for s in sig:
            idx = 0
            while idx < len(partition) and s > partition[idx]:
                idx += 1
            index.append(idx)
            quants.append(codebook[idx])
        return np.array(index), np.array(quants)

    @staticmethod
    def normalize(x):
        den = np.max(x) - np.min(x)
        if den < 1e-12:
            return np.zeros_like(x)
        return (x - np.min(x)) / den

