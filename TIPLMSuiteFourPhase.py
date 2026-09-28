import numpy as np
import cv2
from itertools import combinations_with_replacement

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

from TIPLMSuite import CGHGenerator, DeviceLibrary

__all__ = ["FourPhaseCGHGenerator", "DeviceLibrary"]


class FourPhaseCGHGenerator(CGHGenerator):
    """
    Four-phase superpixel CGH generator for the 0.67 TI PLM.

    Reference:
        J. M. Bass, J. Wang, M. M. Balaji, F. Willomitzer,
        "Four-phase holography for high-quality light field modulation with
        MEMS phase light modulators", Proc. SPIE 13918, 1391808 (2026).

    Principle (paper Sec. 2, Fig. 1d-e)
    -----------------------------------
    The PLM is tiled into 2 x 2 superpixels and every pixel may take any of
    the 16 device states. A spatial filter centred ON AXIS in the Fourier
    plane of a 4f system, one superpixel order wide, averages the four pixel
    phasors, so each superpixel emits

        U = (1/4) * (e^{i phi_1} + e^{i phi_2} + e^{i phi_3} + e^{i phi_4})

    The average ignores which pixel holds which state, so the reachable
    phasors are the multisets of 4 states from 16: C(19, 4) = 3876 unique
    complex values (vs 136 for two-phase encoding and 16 without encoding).
    Of the 16^4 = 65536 raw pixel assignments ~94% are therefore redundant
    reorderings; this class writes each multiset in ascending state order.
    The gamut is built from the ACTUAL LUT phases, so irregular measured
    tables are handled exactly.

    Algorithms
    ----------
    alg='DIRECT' (paper Fig. 2a-c): the target field sqrt(I) * e^{i*phase}
        is mapped per superpixel onto the nearest gamut phasor with a KD tree
        built once over the 3876 phasors (log-time per query, the paper's
        mapping). The image forms at the exit of the 4f.
    alg='GS' (paper Fig. 2d-h): Gerchberg-Saxton between the 4f exit and a
        plane propDistance beyond it (angular-spectrum propagation), with the
        modulator-plane constraint being the same KD-tree projection onto the
        four-phase gamut instead of a phase-only one. propDistance is referred
        to the PLM (a 1:1 relay); with 4f magnification M the physical
        distance behind the image plane is M^2 * propDistance.

    The global amplitude scale (the radius of the circle inside the gamut
    that the normalised target is mapped to) is the one free parameter the
    paper leaves open; cf. Fig. 3c, where the circle is shrunk to trade a
    constant intensity loss for uniform phase coverage. That same knob is
    what extends the PLM past its design wavelength: build the device LUT
    for the longer wavelength and use an amplitude scale inside the covered
    disc. amplitudeScale='auto' picks it by scoring the optical model below.

    Optical model
    -------------
    Quality numbers come from a band-limited 4f model: the pixel field is
    Fourier transformed, multiplied by a rect aperture of half-width
    apertureScale / (4 p) centred on axis, transformed back, and the camera
    intensity is integrated over each superpixel footprint. A real rect
    aperture does NOT perform the ideal superpixel average -- the passband
    also admits the pixel-scale (odd) modes of each 2 x 2 block, weighted by
    sin(pi * f * p) -- so the optical PSNR sits below the ideal
    superpixel-model PSNR, which is printed alongside as a cross-check.

    Experimental note
    -----------------
    The filter sits ON the optical axis, so the zero order is inside the
    passband: unmodulated light (mirror gaps, cover-glass reflection) adds a
    coherent pedestal to the image. That is inherent to the paper's geometry.
    """

    SUPERPIXEL = 2
    N_STATES = 16
    AMPLITUDE_FRACTIONS = np.linspace(0.50, 1.00, 11)
    NN_CHUNK = 4096

    def __init__(self):
        super().__init__()
        self.superpixelSize = self.SUPERPIXEL
        self.usable_h = None
        self.usable_w = None
        self.pLevels = None
        self.alg = "DIRECT"
        self.amplitudeScale = None
        self.apertureScale = 1.0
        self.propDistance = 0.0
        self._gamut = None
        self._gamutStates = None
        self._gamutTree = None
        self._mask = None
        self._padShape = None
        self.psnr_disc = None
        self.psnr_superpixel = None

    # ------------------------------------------------------------------
    # Gamut and the KD-tree mapping
    # ------------------------------------------------------------------
    def fourPhaseGamut(self):
        """
        The 3876 phasors reachable by one superpixel: every multiset of 4
        states from the 16-level LUT, averaged with the actual device phases.
        Returns (gamut, states): gamut is complex (3876,), states[m] holds the
        four device states of multiset m in ascending order.
        """
        if self._gamut is None:
            combos = np.array(list(combinations_with_replacement(range(self.N_STATES), 4)),
                              dtype=np.int64)
            self._gamut = np.exp(2j * np.pi * self.pLevels[combos]).mean(axis=1)
            self._gamutStates = combos
            self._gamutTree = None
        return self._gamut, self._gamutStates

    def _nearestPhasor(self, points):
        """Nearest gamut phasor for (N, 2) (Re, Im) points -> (squared distances,
        indices). KD tree when scipy is present, chunked brute force otherwise."""
        gamut, _ = self.fourPhaseGamut()
        if cKDTree is not None:
            if self._gamutTree is None:
                self._gamutTree = cKDTree(np.column_stack([gamut.real, gamut.imag]))
            dist, idx = self._gamutTree.query(points, workers=-1)
            return dist ** 2, idx.astype(np.int64)

        g = np.column_stack([gamut.real, gamut.imag])
        g_power = (g ** 2).sum(axis=1)
        d2 = np.empty(points.shape[0])
        idx = np.empty(points.shape[0], dtype=np.int64)
        for start in range(0, points.shape[0], self.NN_CHUNK):
            chunk = points[start:start + self.NN_CHUNK]
            d = g_power[None, :] - 2.0 * chunk @ g.T
            idx[start:start + self.NN_CHUNK] = d.argmin(axis=1)
            d2[start:start + self.NN_CHUNK] = (d.min(axis=1) + (chunk ** 2).sum(axis=1))
        return d2, idx

    def encodeField(self, U):
        """
        Quantize a complex superpixel field U (hs, ws), already scaled into the
        unit disc, onto the four-phase gamut. Returns the (2hs, 2ws) map of
        device states, each multiset written row-major in ascending order.
        """
        hs, ws = U.shape
        points = np.column_stack([U.real.ravel(), U.imag.ravel()])
        _, idx = self._nearestPhasor(points)
        c = self._gamutStates[idx].reshape(hs, ws, 2, 2)
        return c.transpose(0, 2, 1, 3).reshape(2 * hs, 2 * ws)

    # ------------------------------------------------------------------
    # Optics
    # ------------------------------------------------------------------
    @staticmethod
    def _nextPow2(n):
        p = 1
        while p < n:
            p <<= 1
        return p

    def _buildApertureMask(self):
        """Rect aperture of one superpixel order, centred on axis."""
        GH, GW = self._nextPow2(self.usable_h), self._nextPow2(self.usable_w)
        self._padShape = (GH, GW)
        FX, FY = np.meshgrid(np.fft.fftfreq(GW, d=self.pitchW), np.fft.fftfreq(GH, d=self.pitchH))
        halfx = self.apertureScale / (2.0 * self.SUPERPIXEL * self.pitchW)
        halfy = self.apertureScale / (2.0 * self.SUPERPIXEL * self.pitchH)
        self._mask = (np.abs(FX) <= halfx) & (np.abs(FY) <= halfy)

    def filteredField(self, phase):
        """Complex field at the 4f exit (PLM-referred, pixel resolution) for a
        (usable_h, usable_w) map of pixel phases."""
        uh, uw = self.usable_h, self.usable_w
        GH, GW = self._padShape
        oy, ox = (GH - uh) // 2, (GW - uw) // 2
        field = np.zeros((GH, GW), dtype=np.complex128)
        field[oy:oy + uh, ox:ox + uw] = np.exp(1j * np.asarray(phase, dtype=np.float64))
        return np.fft.ifft2(np.fft.fft2(field) * self._mask)[oy:oy + uh, ox:ox + uw]

    def _binIntensity(self, field):
        n = self.SUPERPIXEL
        hs, ws = field.shape[0] // n, field.shape[1] // n
        return (np.abs(field) ** 2).reshape(hs, n, ws, n).mean(axis=(1, 3))

    def superpixelField(self, phase):
        """The paper's idealised model: the plain 2 x 2 average of the phasors."""
        n = self.SUPERPIXEL
        E = np.exp(1j * np.asarray(phase, dtype=np.float64))
        return E.reshape(E.shape[0] // n, n, E.shape[1] // n, n).mean(axis=(1, 3))

    def _asmTransfer(self, shape, pitch, z):
        ny, nx = shape
        fx = np.fft.fftfreq(nx, d=pitch)[None, :]
        fy = np.fft.fftfreq(ny, d=pitch)[:, None]
        arg = 1.0 / self.lambda_m ** 2 - fx ** 2 - fy ** 2
        kz = 2.0 * np.pi * np.sqrt(np.maximum(arg, 0.0))
        return np.exp(1j * kz * z) * (arg > 0)

    def _asmPadShape(self, shape, pitch, z):
        """Grid large enough that the band-limited field does not wrap over z:
        the 4f passband reaches sin(theta) = lambda / (4 p)."""
        spread = int(np.ceil(abs(z) * self.lambda_m / (4.0 * self.pitchW) / pitch)) + 8
        return (self._nextPow2(shape[0] + 2 * spread), self._nextPow2(shape[1] + 2 * spread))

    def propagate(self, field, pitch, z, padShape=None):
        """Angular-spectrum propagation over z (metres) on a zero-padded grid;
        returns the window matching the input."""
        if z == 0:
            return field
        ny, nx = field.shape
        GH, GW = padShape if padShape is not None else self._asmPadShape(field.shape, pitch, z)
        oy, ox = (GH - ny) // 2, (GW - nx) // 2
        buf = np.zeros((GH, GW), dtype=np.complex128)
        buf[oy:oy + ny, ox:ox + nx] = field
        out = np.fft.ifft2(np.fft.fft2(buf) * self._asmTransfer((GH, GW), pitch, z))
        return out[oy:oy + ny, ox:ox + nx]

    def opticalIntensity(self, phase):
        """Camera intensity per superpixel: band-limited 4f, then (alg='GS')
        propagation over propDistance, then superpixel-footprint integration."""
        field = self.filteredField(phase)
        if self.propDistance:
            field = self.propagate(field, self.pitchW, self.propDistance)
        return self._binIntensity(field)

    def filterPlaneGeometry(self, f1):
        """Spatial filter in the Fourier plane of the first 4f lens (focal
        length f1, metres): centre (x, y) and aperture (w, h), in metres."""
        n = self.SUPERPIXEL
        return {
            "center": (0.0, 0.0),
            "radius": 0.0,
            "aperture": (self.apertureScale * self.lambda_m * f1 / (n * self.pitchW),
                         self.apertureScale * self.lambda_m * f1 / (n * self.pitchH)),
        }

    def imagePlaneGeometry(self, f1, f2):
        """Geometry of the synthesised field at the 4f exit (metres)."""
        n = self.SUPERPIXEL
        if self.usable_w is None:
            raise ValueError("Usable area is unknown; run createCGH first.")
        mag = f2 / f1
        na = (self.filterPlaneGeometry(f1)["aperture"][0] / 2.0) / f2
        return {
            "magnification": -mag,
            "superpixelPitch": (n * self.pitchW * mag, n * self.pitchH * mag),
            "imageSize": (self.usable_w * self.pitchW * mag, self.usable_h * self.pitchH * mag),
            "resolution": self.lambda_m / (2.0 * na),
            "depthOfFocus": self.lambda_m / (2.0 * na ** 2),
        }

    # ------------------------------------------------------------------
    # Target preparation
    # ------------------------------------------------------------------
    GRAYSCALE_KEYS = ("gray", "grey", "l", "luma", "luminance")

    def loadTarget(self, filename="", colorChannel="gray"):
        """
        Load any image Pillow reads (PNG, JPG, BMP, TIFF, GIF, WebP, ...) into
        self.imTarget as floats in [0, 1].

        colorChannel 'gray' (default) converts colour to luminance,
        0.299 R + 0.587 G + 0.114 B; 0, 1 or 2 keeps only R, G or B instead.
        Grayscale files are used as they are. Transparent pixels count as
        black (no light), and 16-bit / float images keep their precision.
        """
        if isinstance(colorChannel, str):
            if colorChannel.lower() not in self.GRAYSCALE_KEYS:
                raise ValueError("colorChannel must be 'gray', 0, 1 or 2.")
            channel = None
        elif colorChannel is None:
            channel = None
        elif int(colorChannel) in (0, 1, 2):
            channel = int(colorChannel)
        else:
            raise ValueError("colorChannel must be 'gray', 0, 1 or 2.")

        if filename == "":
            from tkinter import Tk
            from tkinter.filedialog import askopenfilename
            Tk().withdraw()
            filename = askopenfilename(
                title="Select an image file",
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif *.webp")])

        from PIL import Image, ImageOps
        try:
            with Image.open(filename) as im:
                im = ImageOps.exif_transpose(im)
                im.load()
        except Exception as exc:
            raise FileNotFoundError(f"Could not load image: {filename}") from exc

        alpha = None
        if im.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in im.info:
            im = im.convert("RGBA")
            alpha = np.asarray(im.getchannel("A"), dtype=np.float64) / 255.0

        if im.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
            image = np.asarray(im, dtype=np.float64)
            image = image / (65535.0 if image.max() <= 65535 else image.max())
        elif im.mode == "F":
            image = np.asarray(im, dtype=np.float64)
        elif im.mode in ("1", "L"):
            image = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
        else:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
            if channel is None:
                image = rgb @ np.array([0.299, 0.587, 0.114])
            else:
                image = rgb[:, :, channel]

        if alpha is not None:
            image = image * alpha
        self.imTarget_orig = image
        self.imTarget = image

    def _resizeTargetImagePlane(self, hs, ws, preserveAspect=True):
        """Sample the target onto the superpixel grid (one superpixel = one
        output pixel), letterboxed to the grid's physical aspect."""
        image = self.imTarget
        if preserveAspect:
            n = self.SUPERPIXEL
            out_aspect = (ws * n * self.pitchW) / (hs * n * self.pitchH)
            source_h, source_w = np.shape(image)[:2]
            source_aspect = source_w / source_h

            def split_padding(delta):
                before = int(delta // 2)
                return before, int(delta - before)

            if source_aspect < out_aspect:
                pad = split_padding(max(int(np.round(source_h * out_aspect)) - source_w, 0))
                image = np.pad(image, ((0, 0), pad), "constant")
            elif source_aspect > out_aspect:
                pad = split_padding(max(int(np.round(source_w / out_aspect)) - source_h, 0))
                image = np.pad(image, (pad, (0, 0)), "constant")

        image = cv2.resize(image, (ws, hs), interpolation=cv2.INTER_AREA)
        self.imTarget = self.normalize(image)

    def prepareTarget(self, filename, colorChannel="gray",
                      FlipUD=False, FlipLR=False, InvertTarget=False,
                      preserveAspect=True, binarizeTarget=False, targetThreshold=0.5):
        """Load and resample a target onto this device's superpixel grid.
        Returns it (also left in self.imTarget)."""
        self.loadTarget(filename, colorChannel)
        self._resizeTargetImagePlane(self.usable_h // self.SUPERPIXEL,
                                     self.usable_w // self.SUPERPIXEL, preserveAspect)
        if binarizeTarget:
            self.binarizeTarget(targetThreshold)
        self.updateTarget(FlipUD, FlipLR, InvertTarget)
        return self.imTarget

    def _setupDevice(self, DeviceDictionary):
        if str(DeviceDictionary["device"]).upper() != "0.67":
            raise ValueError("TIPLMSuiteFourPhase only supports the 0.67 device.")
        pLevel = np.asarray(DeviceDictionary["pLevel"], dtype=np.float64)
        if pLevel.ndim != 1 or DeviceDictionary["nLevel"] != self.N_STATES:
            raise ValueError("Four-phase encoding expects the 16-level 1D 0.67 LUT.")
        device_lambda_m = DeviceDictionary.get("lambda_m", DeviceDictionary.get("phase_lut_wavelength_m"))
        if device_lambda_m is not None and float(device_lambda_m) > 0:
            self.lambda_m = float(device_lambda_m)
        n = self.SUPERPIXEL
        self.usable_h = (DeviceDictionary["h"] // n) * n
        self.usable_w = (DeviceDictionary["w"] // n) * n
        self.pitchW = DeviceDictionary["pitchW"]
        self.pitchH = DeviceDictionary["pitchH"]
        self.pLevels = np.mod(pLevel[:self.N_STATES], 1.0)
        self._gamut = self._gamutStates = self._gamutTree = None
        self._buildApertureMask()

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------
    @staticmethod
    def _psnr(I, target):
        """PSNR after the least-squares gain (the camera exposure is free)."""
        gain = np.mean(I * target) / (np.mean(I ** 2) + 1e-12)
        return 10.0 * np.log10(1.0 / (np.mean((gain * I - target) ** 2) + 1e-20))

    def _scaleCandidates(self, amplitudeScale):
        if isinstance(amplitudeScale, str):
            if amplitudeScale.lower() != "auto":
                raise ValueError("amplitudeScale must be a number in (0, 1] or 'auto'.")
            return self.AMPLITUDE_FRACTIONS
        return [float(amplitudeScale)]

    def runDirect(self, amplitudeScale="auto", targetPhase=0.0):
        """Paper Fig. 2(a-c): map sqrt(target) * e^{i targetPhase} straight onto
        the gamut. Returns the (usable_h, usable_w) state map."""
        amp = np.sqrt(np.clip(self.imTarget, 0.0, None))
        U = amp / max(float(amp.max()), 1e-12) * np.exp(1j * targetPhase)
        best = None
        for scale in self._scaleCandidates(amplitudeScale):
            states = self.encodeField(scale * U)
            score = self._psnr(self.opticalIntensity(2 * np.pi * self.pLevels[states]), self.imTarget)
            if best is None or score > best[0]:
                best = (score, scale, states)
        self.amplitudeScale = best[1]
        return best[2]

    def runGS(self, numIter=20, amplitudeScale=1.0, initialPhase="Random"):
        """
        Paper Fig. 2(d-h): Gerchberg-Saxton between the 4f exit and a plane
        propDistance beyond it, on the superpixel grid. The modulator-plane
        step scales the back-propagated field into the unit disc and projects
        it onto the four-phase gamut (complex, not phase-only).

        The scale maps the 99th-percentile amplitude to amplitudeScale; the
        brightest 1% land on the gamut rim. Normalising by the peak instead
        leaves most superpixels near |U| = 0 (mean |U|^2 ~ 0.04 on harper at
        29 cm), i.e. almost all light in the three odd 2 x 2 modes, which the
        rect aperture partly passes: 8.9 dB band-limited / 17.0 dB ideal,
        against 11.4 / 17.7 dB with the percentile (10 iterations each).
        """
        if isinstance(amplitudeScale, str):
            raise ValueError("alg='GS' needs a numeric amplitudeScale.")
        self.amplitudeScale = float(amplitudeScale)
        n = self.SUPERPIXEL
        pitch = n * self.pitchW
        amp = np.sqrt(np.clip(self.imTarget, 0.0, None))
        pad = self._asmPadShape(amp.shape, pitch, self.propDistance)
        if initialPhase == "Random":
            U = amp * np.exp(2j * np.pi * np.random.rand(*amp.shape))
        else:
            U = amp.astype(np.complex128)

        best = None
        for _ in range(max(1, int(numIter))):
            U0 = self.propagate(U, pitch, -self.propDistance, pad)
            U0 = self.amplitudeScale * U0 / max(float(np.percentile(np.abs(U0), 99.0)), 1e-12)
            states = self.encodeField(U0)
            Uq = self.superpixelField(2 * np.pi * self.pLevels[states])
            U = self.propagate(Uq, pitch, self.propDistance, pad)
            score = self._psnr(np.abs(U) ** 2, self.imTarget)
            if best is None or score > best[0]:
                best = (score, states)
            U = amp * np.exp(1j * np.angle(U))
        return best[1]

    # ------------------------------------------------------------------
    def createCGH(self, DeviceDictionary,
                  filename="", colorChannel="gray",
                  FlipUD=False, FlipLR=False, InvertTarget=False,
                  alg="DIRECT", numIter=20, initialPhase="Random",
                  preserveAspect=True, binarizeTarget=False, targetThreshold=0.5,
                  amplitudeScale="auto", apertureScale=1.0,
                  propDistance=0.0, targetPhase=0.0):
        alg = str(alg).upper()
        if alg not in ("DIRECT", "GS"):
            raise ValueError("alg must be 'DIRECT' or 'GS'.")
        if alg == "GS" and not propDistance:
            raise ValueError("alg='GS' projects to a plane propDistance (m) beyond the 4f exit.")
        if alg == "DIRECT" and propDistance:
            raise ValueError("alg='DIRECT' forms the image at the 4f exit; use propDistance=0.")
        self.alg = alg
        self.apertureScale = float(apertureScale)
        self.propDistance = float(propDistance)
        self._setupDevice(DeviceDictionary)
        self.prepareTarget(filename, colorChannel, FlipUD, FlipLR,
                           InvertTarget, preserveAspect, binarizeTarget, targetThreshold)

        if alg == "DIRECT":
            stateq = self.runDirect(amplitudeScale, targetPhase)
        else:
            if isinstance(amplitudeScale, str):
                amplitudeScale = 1.0
            stateq = self.runGS(numIter, amplitudeScale, initialPhase)
        self._storeResult(DeviceDictionary, stateq)

    def _storeResult(self, DeviceDictionary, stateq):
        h, w = DeviceDictionary["h"], DeviceDictionary["w"]
        # Pixels outside the usable superpixel grid are parked at state 0.
        state_full = np.zeros((h, w), dtype=np.float64)
        phase_full = np.full((h, w), 2 * np.pi * self.pLevels[0], dtype=np.float64)
        state_full[:self.usable_h, :self.usable_w] = stateq
        phase_full[:self.usable_h, :self.usable_w] = 2 * np.pi * self.pLevels[stateq]

        self.CGH_output_state_disc = state_full
        self.CGH_output_disc = state_full
        self.CGH_output_phase_disc = phase_full
        self.CGH_output_cont = phase_full[:self.usable_h, :self.usable_w].copy()
        self.CGH_phase = phase_full
        self.CGH_mapped = self.deviceLibary.formatPLM(DeviceDictionary, state_full)
        self.recoverImg()

    def recoverImg(self, ShiftFOV=False, propMethod="FOURIER"):
        if self.imTarget is None:
            return
        phase = self.CGH_output_phase_disc[:self.usable_h, :self.usable_w]
        I = self.opticalIntensity(phase)
        self.psnr_disc = self._psnr(I, self.imTarget)

        # The paper's idealised superpixel average, for comparison.
        Usp = self.superpixelField(phase)
        if self.propDistance:
            Usp = self.propagate(Usp, self.SUPERPIXEL * self.pitchW, self.propDistance)
        self.psnr_superpixel = self._psnr(np.abs(Usp) ** 2, self.imTarget)

        print("Four-phase %s (amplitude scale %.2f): PSNR %.1f dB band-limited 4f | "
              "%.1f dB ideal superpixel average"
              % (self.alg, self.amplitudeScale, self.psnr_disc, self.psnr_superpixel))

        gain = np.mean(I * self.imTarget) / (np.mean(I ** 2) + 1e-12)
        self.imRecovered_disc = gain * I
        self.imRecovered_cont = self.imRecovered_disc
