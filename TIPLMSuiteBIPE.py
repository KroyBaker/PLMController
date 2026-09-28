import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch import nn

from TIPLMSuite import CGHGenerator, DeviceLibrary

__all__ = ["BIPECGHGenerator", "DeviceLibrary"]


class BIPECGHGenerator(CGHGenerator):
    """
    Binary-Phase-Engraved (BiPE) superpixel CGH generator for the 0.67 TI PLM.

    Reference:
        "Superpixel-based binary-phase complex field modulation"
        Light: Advanced Manufacturing (2025), doi: 10.37188/lam.2025.017

    Principle
    ---------
    The PLM is partitioned into n x n superpixels (n = 2 or 4). Every pixel is
    driven to one of only TWO device states whose phases are ~pi apart. An
    off-axis spatial filter in the Fourier plane of a 4f imaging system imposes
    a linear carrier ramp across the PLM of

        pi / n^2 radians per pixel horizontally
        pi / n   radians per pixel vertically

    so the k-th pixel of a superpixel (k = row * n + col, row-major) carries the
    phase pre-factor exp(i * pi * k / n^2). The filter low-passes the field to
    superpixel resolution, so each superpixel emits

        U(c) = (1 / n^2) * sum_k exp[i * pi * (k / n^2 + c_k)],  c_k in {0, 1}

    which spans a gamut of 2^(n^2) complex values (65536 for n = 4): full
    complex-field modulation from a binary-phase device.

    This class models that chain differentiably (binary phase + carrier ramp ->
    superpixel average -> Fourier propagation) and supports both suite
    algorithms:

      ADAM    - optimizes an unconstrained per-pixel phase; the resulting
                continuous superpixel field is then encoded onto the binary
                gamut with the paper's per-superpixel argmin search
                C(u,v) = argmin_c |U(c) - R * exp(-i*pi*(u/n + v)) * T(u,v)|^2,
                including an amplitude-scale (R) search.
      ADAMWGS - quantization-aware optimization with a binary Gumbel-softmax
                (cosine phase scores, so gradients survive with only two
                levels ~pi apart) plus straight-through estimation.

    The two device states are picked from the empirical 16-level LUT as the
    state closest to phase 0 and the state closest to pi away from it, and the
    ACTUAL LUT phases are used in the forward model, so the simulation is
    faithful to the hardware.

    Experimental note
    -----------------
    The carrier ramp is NOT written to the PLM (the PLM only shows the binary
    pattern); it must be produced optically by centering the spatial filter at
    frequencies (fx, fy) = (1 / (2 n^2 p), 1 / (2 n p)) where p is the pixel
    pitch, i.e. at (lambda * f * fx, lambda * f * fy) from the optical axis in
    the Fourier plane of a lens with focal length f. Use carrierFrequencies()
    to get the numbers. The filter aperture should pass roughly one superpixel
    order (width ~ lambda * f / (n * p)).

    An additional off-axis carrier for the reconstruction itself (to steer the
    image away from residual zero-order light) is available through the
    offAxisCarrier argument of createCGH as a fractional field-of-view shift.
    """

    GAMUT_AMPLITUDE_FRACTIONS = np.linspace(0.5, 1.1, 13)
    GAMUT_CHUNK = 2048

    class BinaryGumbelQuantization(nn.Module):
        """
        Two-level Gumbel-softmax quantizer for BiPE.

        The base suite scores levels with a sharp ReLU kernel scaled by ~350,
        which saturates the softmax when only two levels ~pi apart exist and
        kills all gradients. Here the level scores are smooth bounded cosines,
        scores_k = gain * (tauInit / tau) * cos(phase - 2*pi*level_k),
        so every pixel keeps a usable gradient while the annealing schedule
        (tau falling, score gain rising) still hardens the assignments.
        """

        def __init__(self, lut, tau=6.5, hard=True, scoreGain=6.5):
            super().__init__()
            self.quant_levels = lut
            self.tau = tau
            self.tauInit = tau
            self.hard = hard
            self.scoreGain = scoreGain

        def anneal_temperature(self, annealRate, tauInitial, tauMin):
            self.tau = tauInitial * torch.exp(-torch.log(tauInitial / tauMin) * annealRate)

        def _scores(self, phase):
            diff = phase.unsqueeze(-1) - 2 * np.pi * self.quant_levels
            return self.scoreGain * (self.tauInit / self.tau) * torch.cos(diff)

        def forward(self, phase):
            scores = self._scores(phase)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-20) + 1e-20)
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
            scores = self._scores(phase)
            indices = torch.argmax(scores, dim=2, keepdim=True)
            hard_assignments = torch.zeros_like(scores).scatter_(2, indices, 1.0)
            return torch.sum(2 * np.pi * hard_assignments * self.quant_levels, dim=2)

    def __init__(self, superpixelSize=4):
        super().__init__()
        self.superpixelSize = self._validateSuperpixelSize(superpixelSize)
        self.reconstructionPlane = "FOURIER"
        self.offAxisCarrier = (0.0, 0.0)
        self.binaryStateIndices = None
        self.binaryPLevels = None
        self.usable_h = None
        self.usable_w = None
        self._ramp = None
        # forward model: "SUPERPIXEL" = ideal n x n box-average (the classic
        # superpixel idealization); "OPTICAL" = band-limited 4f with the real
        # rect aperture. See _opticalIntensity for why these differ so much.
        self.forwardModel = "SUPERPIXEL"
        self.apertureScale = 1.0
        self._maskNumpy = None
        self._maskTorch = None
        self._padShape = None
        # Measured illumination (plm_illumination_profile.py). illumAmplitude is
        # the beam amplitude over the usable PLM area; illumSuperpixel is its
        # per-superpixel mean, which is what the encoder pre-compensates with.
        self.illumSuperpixel = None
        self.illumCompensationFloor = 0.15
        self.illumCompensation = 1.0
        self._illumTorch = None

    # ------------------------------------------------------------------
    # Illumination
    # ------------------------------------------------------------------
    # A superpixel spans n * 10.8 um (21.6 or 43.2 um) while the beam varies over
    # millimetres, so the illumination is constant to ~1% across one superpixel
    # and factors straight out of the superpixel sum:
    #     U_sp = (1/n^2) sum_k a_k e^{i phi_k}  ~=  a_sp * U(c).
    # At the 4f IMAGE plane one superpixel is one camera pixel, so the beam
    # multiplies the reconstruction pixel-for-pixel -- the Gaussian is printed
    # straight onto the image. That also makes it directly correctable: ask each
    # superpixel for U_desired / a_sp and the emitted field comes out flat.
    def _gamutSize(self):
        """Number of distinct complex values one superpixel can emit."""
        return 1 << (self.superpixelSize ** 2)

    # A dense gamut can absorb pre-emphasis; a coarse one cannot. Dividing the
    # beam out widens the spread of requested amplitudes, and since the encoder's
    # amplitude scale is global, the bright-region requests shrink toward the
    # middle of the gamut. With BiPE n=2's 16 phasors that quantizes badly enough
    # to cost ~5 dB, which is why compensation is off by default there. Measured
    # on this repo (simulated beam, amplitude 0.26-1.00 of peak, dot-grid target):
    #   four-phase 2x2 (3876) : 13.6 -> 22.5 dB, shading spread 0.41 -> 0.05
    #   BiPE 4x4    (65536)   : 12.6 -> 12.2 dB, shading spread 0.41 -> 0.13
    #   BiPE 2x2    (16)      : 13.6 -> 13.1 dB, and -5 dB on dlp_logo
    GAMUT_COMPENSATION_THRESHOLD = 256

    def _autoCompensation(self):
        return 1.0 if self._gamutSize() >= self.GAMUT_COMPENSATION_THRESHOLD else 0.0

    def _prepareIllumination(self, illumination, illuminationFloor,
                             illuminationCompensation="auto"):
        """Resolve the illumination option onto the usable area + superpixel grid."""
        self.illumCompensationFloor = float(np.clip(illuminationFloor, 1e-3, 1.0))
        if isinstance(illuminationCompensation, str):
            if illuminationCompensation.lower() != "auto":
                raise ValueError("illuminationCompensation must be a number in "
                                 "[0, 1] or 'auto'.")
            illuminationCompensation = self._autoCompensation()
        self.illumCompensation = float(np.clip(illuminationCompensation, 0.0, 1.0))
        amplitude = self._resolveIllumination(
            illumination, (self.usable_h, self.usable_w))
        self.illumAmplitude = amplitude
        if amplitude is None:
            self.illumSuperpixel = None
            self._illumTorch = None
            return

        n = self.superpixelSize
        hs, ws = self.usable_h // n, self.usable_w // n
        self.illumSuperpixel = amplitude.reshape(hs, n, ws, n).mean(axis=(1, 3))
        self._illumTorch = torch.tensor(amplitude, dtype=torch.float32,
                                        device=self.MLDevice)
        rel = self.illumSuperpixel / max(float(self.illumSuperpixel.max()), 1e-12)
        clipped = float(np.mean(rel < self.illumCompensationFloor)) * 100.0
        boost = 1.0 / max(float(self._illuminationDivisor().min()), 1e-12)
        print("Illumination: superpixel amplitude spans %.3f-%.3f of peak; gamut %d "
              "phasors -> compensation %.2f (max boost %.2fx), floor %.2f "
              "(%.1f%% boost-limited)"
              % (rel.min(), rel.max(), self._gamutSize(), self.illumCompensation,
                 boost, self.illumCompensationFloor, clipped))
        if self.illumCompensation == 0.0:
            print("   compensation 0: beam modelled but NOT divided out (gamut too "
                  "coarse to pre-emphasise); pass illuminationCompensation>0 to "
                  "trade PSNR for a flatter image.")

    def _illuminationDivisor(self):
        """Relative beam amplitude per superpixel, floored and raised to gamma.

        Normalized to peak 1 so the brightest superpixel's request is unchanged
        and dimmer ones are boosted. Two limiters, because full compensation is
        not always the best trade:

        floor -- the beam wings (a -> 0) would demand unbounded amplitude, and
            since the encoder's amplitude scale is global, that would squash the
            whole image to fit. Clamping caps the boost at 1 / floor.
        gamma (illumCompensation) -- divide by a^gamma. gamma = 1 fully flattens
            the delivered image; gamma = 0 leaves the request alone and only uses
            the beam in the forward model. Intermediate values trade flatness for
            gamut headroom, which matters because pre-emphasis widens the range of
            requested amplitudes and a coarse gamut (BiPE n=2 has just 16 phasors)
            quantizes the shrunken bright-region requests badly. Dense gamuts
            (four-phase 3876, BiPE n=4 65536) tolerate gamma = 1.
        """
        if self.illumSuperpixel is None:
            return None
        peak = max(float(self.illumSuperpixel.max()), 1e-12)
        rel = np.maximum(self.illumSuperpixel / peak, self.illumCompensationFloor)
        if self.illumCompensation >= 1.0:
            return rel
        return rel ** self.illumCompensation

    def _preEmphasise(self, U_local_2d):
        """Divide the requested superpixel field by the local beam amplitude.

        The gamut value the encoder picks is what the PLM synthesises before the
        beam scales it, so requesting U/a makes the EMITTED field U.
        """
        divisor = self._illuminationDivisor()
        if divisor is None:
            return U_local_2d
        return U_local_2d / divisor

    # ------------------------------------------------------------------
    # True optical forward model
    # ------------------------------------------------------------------
    # The classic superpixel model says a superpixel emits
    #     U = (1/n^2) sum_k exp(i(phi_k + pi k/n^2)),
    # i.e. a perfect n x n box-average, and takes |U|^2 sampled on the superpixel
    # grid as the image. A real 4f does something measurably different: it band-
    # limits with a RECT aperture (a brick wall at the superpixel Nyquist
    # frequency, whose impulse response is a sinc with ringing tails that reach
    # into neighbouring superpixels). The box-average's transfer function is a
    # Dirichlet kernel that passes energy well beyond that cut-off, which then
    # ALIASES on the naive downsample -- so the box model responds to sub-
    # superpixel structure that a real filter removes.
    #
    # That gap is not academic. Measured on this repo's dlp_logo target, n=4:
    #   solution optimized for the box model : 93 dB (box) but  8.7 dB (optical)
    #   solution optimized for the optics    :  8.4 dB (box) but 54.8 dB (optical)
    # The two models are effectively orthogonal, and hardware sees the optical
    # one. Use forwardModel="OPTICAL" with alg="ADAMWGS" (quantization in the
    # loop) so the binary states are optimized against what the bench actually
    # does.
    @staticmethod
    def _nextPow2(n):
        p = 1
        while p < n:
            p <<= 1
        return p

    def _buildApertureMask(self):
        """Rect aperture of one superpixel order, centred on the BiPE carrier."""
        n = self.superpixelSize
        uh, uw = self.usable_h, self.usable_w
        GH = self._nextPow2(uh)
        GW = self._nextPow2(uw)
        self._padShape = (GH, GW)

        pw, ph = self.pitchW, self.pitchH
        FX, FY = np.meshgrid(np.fft.fftfreq(GW, d=pw), np.fft.fftfreq(GH, d=ph))
        cx, cy = self.carrierFrequencies(pw, ph)
        halfx = self.apertureScale / (2.0 * n * pw)
        halfy = self.apertureScale / (2.0 * n * ph)
        mask = ((np.abs(FX - cx) <= halfx) & (np.abs(FY - cy) <= halfy))
        self._maskNumpy = mask
        self._maskTorch = torch.tensor(mask.astype(np.float32), device=self.MLDevice)

    def _opticalIntensity(self, modulation_phase, torchMode):
        """
        Band-limited 4f: pad the PLM field, FFT, multiply by the rect aperture at
        the BiPE carrier, inverse FFT, then integrate INTENSITY over each
        superpixel footprint (a camera integrates intensity, and the carrier
        phase drops out of the modulus -- no demodulation is needed).

        The carrier ramp is deliberately NOT added to the phase here: off-axis
        filter placement is what supplies it, which is the whole point of BiPE.
        """
        n = self.superpixelSize
        uh, uw = self.usable_h, self.usable_w
        hs, ws = uh // n, uw // n
        GH, GW = self._padShape
        oy, ox = (GH - uh) // 2, (GW - uw) // 2

        if torchMode:
            field = torch.zeros((GH, GW), dtype=torch.complex64, device=self.MLDevice)
            E = torch.exp(1j * modulation_phase)
            if self._illumTorch is not None:
                E = self._illumTorch * E
            field[oy:oy + uh, ox:ox + uw] = E
            spec = torch.fft.fft2(field) * self._maskTorch
            win = torch.fft.ifft2(spec)[oy:oy + uh, ox:ox + uw]
            I = torch.abs(win) ** 2
            return I.reshape(hs, n, ws, n).mean(dim=(1, 3))

        field = np.zeros((GH, GW), dtype=np.complex128)
        E = np.exp(1j * np.asarray(modulation_phase, dtype=np.float64))
        if self.illumAmplitude is not None:
            E = self.illumAmplitude * E
        field[oy:oy + uh, ox:ox + uw] = E
        spec = np.fft.fft2(field) * self._maskNumpy
        win = np.fft.ifft2(spec)[oy:oy + uh, ox:ox + uw]
        I = np.abs(win) ** 2
        return I.reshape(hs, n, ws, n).mean(axis=(1, 3))

    @staticmethod
    def _validateSuperpixelSize(superpixelSize):
        superpixelSize = int(superpixelSize)
        if superpixelSize not in (2, 4):
            raise ValueError("BiPE superpixelSize must be 2 or 4.")
        return superpixelSize

    @staticmethod
    def selectBinaryStates(nLevel, pLevel):
        """
        Pick the two device states used by BiPE from the phase LUT: the PAIR of
        states whose circular phase separation is closest to half a wave. With
        two unit-amplitude states the superpixel gamut depends only on their
        separation (a common phase just rotates it), so this is the optimal
        criterion. A greedy pick anchored at the state nearest phase 0 fails on
        irregular measured tables (e.g. the tuned 632.8 nm LUT, where it finds
        (0, 12) at 0.70 pi but the table contains (9, 15) at 1.00 pi).
        Returns (state_index_0, state_index_pi), the first being the pair
        member whose phase is closer to 0 (mod 2 pi).
        """
        levels = np.mod(np.asarray(pLevel, dtype=np.float64).reshape(-1)[:nLevel], 1.0)
        if levels.size < 2:
            raise ValueError("Phase LUT must contain at least two states.")
        separation = np.abs(np.remainder(levels[None, :] - levels[:, None] + 0.5, 1.0) - 0.5)
        cost = np.abs(separation - 0.5)
        np.fill_diagonal(cost, np.inf)
        i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
        d0_i = min(levels[i], 1.0 - levels[i])
        d0_j = min(levels[j], 1.0 - levels[j])
        return (int(i), int(j)) if d0_i <= d0_j else (int(j), int(i))

    def carrierFrequencies(self, pitchW=None, pitchH=None):
        """
        Spatial frequencies (fx, fy) of the BiPE carrier ramp in cycles/meter.
        The spatial filter must be centered at (lambda * f * fx, lambda * f * fy)
        in the Fourier plane of a lens with focal length f.
        """
        n = self.superpixelSize
        pw = self.pitchW if pitchW is None else pitchW
        ph = self.pitchH if pitchH is None else pitchH
        if pw is None or ph is None:
            raise ValueError("Pixel pitch is unknown; run createCGH first or pass pitchW/pitchH.")
        return 1.0 / (2.0 * n * n * pw), 1.0 / (2.0 * n * ph)

    def filterPlaneGeometry(self, f1, pitchW=None, pitchH=None):
        """
        Physical geometry of the BiPE spatial filter in the Fourier plane of the
        first 4f lens (focal length f1, metres). All returned values are metres.

            center      (x, y) offset of the filter centre from the optical axis
            radius      radial distance of that centre from the axis
            aperture    (width, height) of the aperture, i.e. one superpixel order

        The aperture height is always exactly twice the y-offset, so the inner
        edge grazes the zero order for every n and f1; only a longer f1 scales
        the whole pattern up.
        """
        n = self.superpixelSize
        pw = self.pitchW if pitchW is None else pitchW
        ph = self.pitchH if pitchH is None else pitchH
        fx, fy = self.carrierFrequencies(pw, ph)
        cx = self.lambda_m * f1 * fx
        cy = self.lambda_m * f1 * fy
        return {
            "center": (cx, cy),
            "radius": float(np.hypot(cx, cy)),
            "aperture": (self.lambda_m * f1 / (n * pw), self.lambda_m * f1 / (n * ph)),
        }

    def imagePlaneGeometry(self, f1, f2, pitchW=None, pitchH=None):
        """
        Geometry of the synthesised complex field at the exit of a 4f system
        built from lenses f1 then f2 (metres). Magnification is -f2/f1; the sign
        means the 4f inverts the field, which createCGH's FlipUD/FlipLR
        pre-compensate. All returned values are metres except magnification.

            magnification       -f2 / f1
            superpixelPitch     (w, h) superpixel pitch at the output plane
            imageSize           (w, h) extent of the usable superpixel grid
            resolution          diffraction limit set by the filter aperture
            depthOfFocus        +/- axial tolerance on camera placement
        """
        n = self.superpixelSize
        pw = self.pitchW if pitchW is None else pitchW
        ph = self.pitchH if pitchH is None else pitchH
        if self.usable_w is None or self.usable_h is None:
            raise ValueError("Usable area is unknown; run createCGH first.")

        M = -f2 / f1
        mag = abs(M)
        out_pw, out_ph = n * pw * mag, n * ph * mag
        na = (self.filterPlaneGeometry(f1, pw, ph)["aperture"][0] / 2.0) / f2
        return {
            "magnification": M,
            "superpixelPitch": (out_pw, out_ph),
            "imageSize": (self.usable_w * pw * mag, self.usable_h * ph * mag),
            "resolution": self.lambda_m / (2.0 * na),
            "depthOfFocus": self.lambda_m / (2.0 * na ** 2),
        }

    @staticmethod
    def _buildRamp(rows, cols, superpixelSize):
        n = superpixelSize
        X = np.arange(cols, dtype=np.float64)[None, :]
        Y = np.arange(rows, dtype=np.float64)[:, None]
        return np.mod(np.pi * (X / (n * n) + Y / n), 2 * np.pi)

    def bipeGamut(self):
        """
        Complex gamut of one superpixel: the 2^(n^2) field values
        U(c) = (1/n^2) * sum_k exp[i * (pi * k / n^2 + phi_{c_k})]
        built from the two ACTUAL device phases. Returns (gamut, bits) where
        gamut is complex (2^(n^2),) and bits[m, k] is the binary state of
        pixel k (row-major within the superpixel) for configuration m.
        """
        n = self.superpixelSize
        nsq = n * n
        phi = 2 * np.pi * self.binaryPLevels
        k = np.arange(nsq)
        carrier = np.pi * k / nsq
        m = np.arange(1 << nsq, dtype=np.int64)
        bits = ((m[:, None] >> k[None, :]) & 1).astype(np.uint8)
        phases = carrier[None, :] + np.where(bits > 0, phi[1], phi[0])
        gamut = np.exp(1j * phases).mean(axis=1)
        return gamut, bits

    def _projectToGamut(self, cont_phase):
        """
        Encode a continuous solution onto binary BiPE states: take the
        continuous superpixel field it produces, derotate the inter-superpixel
        carrier term exp(i*pi*(u/n + v)), then pick per superpixel the binary
        configuration whose gamut value is closest (paper's argmin encoding),
        after searching for the amplitude scale R that minimizes the relative
        projection error. Returns a (rows, cols) array of 0/1 pixel states.
        """
        n = self.superpixelSize
        rows, cols = cont_phase.shape
        hs, ws = rows // n, cols // n

        E = np.exp(1j * (cont_phase + self._ramp))
        if self.illumAmplitude is not None:
            # Match the forward model ADAM optimized against, so the field handed
            # to the encoder is the one the beam actually emits; _projectLocalField
            # then divides the beam back out to get the required gamut value.
            E = self.illumAmplitude * E
        U = E.reshape(hs, n, ws, n).mean(axis=(1, 3))
        return self._projectLocalField(self._derotate(U))

    def _derotate(self, U):
        """Remove the inter-superpixel carrier exp(+i*pi*(u/n + v)).

        This term is physical: demodulating the pixel-level carrier at superpixel
        centres leaves exactly this residual, so derotating here is what makes the
        emitted field match the requested one. (Verified optically: keeping it
        scores 25.1 dB at the nominal aperture vs 15.7 dB without.)
        """
        hs, ws = U.shape
        n = self.superpixelSize
        v = np.arange(hs, dtype=np.float64)[:, None]
        u = np.arange(ws, dtype=np.float64)[None, :]
        return U * np.exp(-1j * np.pi * (u / n + v))

    def _projectLocalField(self, U_local_2d):
        """Per-superpixel argmin onto the binary gamut, with amplitude-scale search."""
        n = self.superpixelSize
        hs, ws = U_local_2d.shape
        rows, cols = hs * n, ws * n
        U_local = self._preEmphasise(U_local_2d).reshape(-1)

        gamut, bits = self.bipeGamut()

        a = torch.tensor(
            np.stack([U_local.real, U_local.imag], axis=1),
            dtype=torch.float32, device=self.MLDevice
        )
        g = torch.tensor(
            np.stack([gamut.real, gamut.imag], axis=0),
            dtype=torch.float32, device=self.MLDevice
        )
        g_power = (g ** 2).sum(dim=0)

        # Search the amplitude scale R on a subsample of superpixels.
        amp_max = float(np.max(np.abs(U_local)))
        if amp_max < 1e-12:
            amp_max = 1e-12
        gamut_radius = float(np.max(np.abs(gamut)))
        num = a.shape[0]
        sub = torch.randperm(num, device=self.MLDevice)[:min(4096, num)]
        a_sub = a[sub]
        a_sub_power = (a_sub ** 2).sum(dim=1)

        best_scale = gamut_radius / amp_max
        best_err = None
        for fraction in self.GAMUT_AMPLITUDE_FRACTIONS:
            scale = fraction * gamut_radius / amp_max
            cross = (scale * a_sub) @ g
            dmin = (g_power[None, :] - 2 * cross).min(dim=1)[0]
            err = ((scale ** 2 * a_sub_power + dmin) / scale ** 2).sum().item()
            if best_err is None or err < best_err:
                best_err = err
                best_scale = scale

        best_idx = torch.empty(num, dtype=torch.int64, device=self.MLDevice)
        for start in range(0, num, self.GAMUT_CHUNK):
            stop = min(start + self.GAMUT_CHUNK, num)
            cross = (best_scale * a[start:stop]) @ g
            best_idx[start:stop] = (g_power[None, :] - 2 * cross).argmin(dim=1)

        c = bits[best_idx.cpu().numpy()].reshape(hs, ws, n, n)
        return c.transpose(0, 2, 1, 3).reshape(rows, cols)

    def runDirect(self, targetPhase=0.0, feedbackIters=0, feedbackBeta=0.15):
        """
        The paper's encoding with no gradient pre-optimization: ask each superpixel
        for sqrt(target) directly and project onto the binary gamut.

        This is the right choice at the IMAGE plane. You are imaging, not
        diffracting to a far field, so the superpixel field needs NO diffuser
        phase -- a smooth field is band-limited and therefore survives the
        aperture intact. ADAM, optimizing the box-average model, has no incentive
        to stay smooth: it converges on a diffuser-like solution with sub-
        superpixel structure that scores brilliantly in that model and is then
        destroyed by the real filter. Measured on dlp_logo, n=4, at the nominal
        aperture: this method 25.1 dB optically, ADAM + projection 11.8 dB.

        feedbackIters > 0 adds an adaptive weighted correction in the spirit of
        AWCGS, but closed through the TRUE optical model: ask for more amplitude
        where the optics under-delivers, re-encode, keep the best iterate. Note a
        plain Gerchberg-Saxton loop does NOT apply at the image plane -- there is
        no propagation between the superpixel field and the camera to alternate
        across, and the image-plane phase GS would randomize is exactly what must
        stay smooth. Only the re-weighting idea carries over.

        Measured effect: n=2 gains ~+3.7 dB (22.2 -> 25.9), because its 16-value
        gamut is coarse enough that quantization bias is the bottleneck and
        pre-emphasis corrects it. n=4 gains nothing (18.7 dB at iteration 0): its
        65536-value gamut already hits the requested field, so the residual is
        aperture crosstalk, which an amplitude correction cannot undo. The loop
        oscillates, hence best-iterate tracking rather than a fixed count.
        """
        target = np.clip(self.imTarget, 0.0, None)
        amp = np.sqrt(target)
        peak = float(amp.max()) if amp.size else 1.0
        phase_term = np.exp(1j * (targetPhase if np.ndim(targetPhase) else float(targetPhase)))

        A = amp.copy()
        best_state, best_psnr = None, -np.inf
        iters = max(0, int(feedbackIters))
        for k in range(iters + 1):
            stateq = self._projectLocalField(self._derotate(A * phase_term))
            binary_phase = 2 * np.pi * self.binaryPLevels[stateq.astype(np.int64)]
            I = self._opticalIntensity(binary_phase, torchMode=False)
            gain = np.mean(I * target) / (np.mean(I ** 2) + 1e-12)
            mse = float(np.mean((gain * I - target) ** 2))
            cur = 20 * np.log10(1 / np.sqrt(mse + 1e-20))
            if cur > best_psnr:
                best_psnr, best_state = cur, stateq
            if k < iters:
                achieved = np.sqrt(np.clip(gain * I, 0.0, None))
                A = np.clip(A + feedbackBeta * (amp - achieved), 0.0, None)
                scale = A.max()
                if scale > 1e-12:
                    A *= peak / scale
        if iters:
            print("DIRECT + adaptive feedback: best optical PSNR %.1f dB over %d iterations"
                  % (best_psnr, iters + 1))
        # a "continuous" reference for recoverImg is filled in by createCGH
        self.CGH_output_cont = np.zeros((self.usable_h, self.usable_w), dtype=np.float64)
        return best_state

    def _superpixelIntensityTorch(self, modulation_phase, ramp):
        n = self.superpixelSize
        rows, cols = modulation_phase.shape
        if self.forwardModel == "OPTICAL":
            I = self._opticalIntensity(modulation_phase, torchMode=True)
            if self.reconstructionPlane == "IMAGE":
                return I
            # far field of the band-limited complex field is not available from
            # intensity alone; OPTICAL is an image-plane model by construction
            raise ValueError("forwardModel='OPTICAL' requires reconstructionPlane='IMAGE'.")
        E = torch.exp(1j * (modulation_phase + ramp))
        if self._illumTorch is not None:
            E = self._illumTorch * E
        U = E.reshape(rows // n, n, cols // n, n).mean(dim=(1, 3))
        if self.reconstructionPlane == "IMAGE":
            return torch.abs(U) ** 2
        E_ip = self.torchProp("forward", U, "FOURIER")
        return torch.abs(E_ip) ** 2

    def reconstructBIPE(self, modulation_phase):
        """
        Reconstruct the observed intensity from a full-resolution modulation
        phase map (the phase written to the PLM, without the carrier ramp).
        Returns (intensity, field) at superpixel resolution.

        With reconstructionPlane == "IMAGE" the field U synthesised by the 4f is
        itself what the camera sees at the 4f exit, so no further propagation is
        applied. With "FOURIER" the far field of U is formed, which assumes a
        further transform lens downstream of the 4f.
        """
        n = self.superpixelSize
        rows, cols = np.shape(modulation_phase)
        if self.forwardModel == "OPTICAL":
            I = self._opticalIntensity(modulation_phase, torchMode=False)
            E_ip = None
        else:
            E = np.exp(1j * (np.asarray(modulation_phase, dtype=np.float64) + self._ramp))
            if self.illumAmplitude is not None:
                E = self.illumAmplitude * E
            U = E.reshape(rows // n, n, cols // n, n).mean(axis=(1, 3))
            E_ip = U if self.reconstructionPlane == "IMAGE" else self.prop("forward", U, "FOURIER")
            I = np.real(np.abs(E_ip) ** 2)

        if self.imTarget is not None:
            scale = np.mean(I * self.imTarget) / (np.mean(I ** 2) + 1e-10)
            I = scale * I

        return I, E_ip

    def _resizeTargetImagePlane(self, hs, ws, pitchW, pitchH, preserveAspect=True):
        """
        Sample the target directly onto the superpixel grid for a camera at the
        4f exit. The base resizeTarget scales by an arcsin(lambda/pitch)
        diffraction FOV and fftshifts, both of which only apply to a far-field
        reconstruction; here one superpixel maps to one output pixel, so the
        target just needs letterboxing to the physical aspect of the grid.
        """
        image = self.imTarget

        if preserveAspect:
            out_aspect = (ws * pitchW) / (hs * pitchH)
            source_h, source_w = np.shape(image)[:2]
            source_aspect = source_w / source_h

            def split_padding(delta):
                before = int(delta // 2)
                return before, int(delta - before)

            if source_aspect < out_aspect:
                pad_left, pad_right = split_padding(max(int(np.round(source_h * out_aspect)) - source_w, 0))
                image = np.pad(image, ((0, 0), (pad_left, pad_right)), "constant")
            elif source_aspect > out_aspect:
                pad_top, pad_bottom = split_padding(max(int(np.round(source_w / out_aspect)) - source_h, 0))
                image = np.pad(image, ((pad_top, pad_bottom), (0, 0)), "constant")

        # INTER_AREA rather than the base class's INTER_NEAREST: this is a large
        # downscale onto a coarse grid, so averaging preserves detail that
        # point-sampling drops.
        image = cv2.resize(image, (ws, hs), interpolation=cv2.INTER_AREA)
        den = np.max(image) - np.min(image)
        if den < 1e-12:
            image = np.zeros_like(image)
        else:
            image = (image - np.min(image)) / den
        self.imTarget = image

    def _initialModulationPhase(self, initialPhase):
        shape = (self.usable_h, self.usable_w)
        if initialPhase == "Random":
            return 2 * np.pi * np.random.rand(*shape)
        if initialPhase == "Unity":
            return 2 * np.pi * np.ones(shape)
        if initialPhase == "Custom":
            if self.seedPhase is None:
                print("Error: No seed phase defined. Using random seed")
                return 2 * np.pi * np.random.rand(*shape)
            if self.seedPhase.shape != shape:
                raise ValueError("Seed phase must match the usable PLM area "
                                 f"{shape} for BiPE.")
            return self.seedPhase
        print("Error: Invalid initial phase seed. Using random seed")
        return 2 * np.pi * np.random.rand(*shape)

    def createCGH(self, DeviceDictionary,
                  filename="", bitPlanes=1, colorChannel=0,
                  FlipUD=False, FlipLR=False, InvertTarget=False,
                  alg="ADAMWGS", numIter=1, initialPhase="Random", propMethod="Fourier",
                  ShiftFOV=True, showImages=False, binarizeTarget=False, targetThreshold=0.5,
                  preserveAspect=True,
                  lossMode="auto",
                  superpixelSize=None, offAxisCarrier=(0.0, 0.0),
                  reconstructionPlane="FOURIER",
                  forwardModel="SUPERPIXEL", apertureScale=1.0,
                  feedbackIters=0, feedbackBeta=0.15,
                  illumination=False, illuminationFloor=0.15,
                  illuminationCompensation="auto"):
        if str(DeviceDictionary["device"]).upper() != "0.67":
            raise ValueError("TIPLMSuiteBIPE only supports the 0.67 device.")
        alg_key = str(alg).upper()
        if alg_key not in ("ADAM", "ADAMWGS", "DIRECT"):
            raise ValueError("TIPLMSuiteBIPE supports alg='DIRECT', 'ADAM' or 'ADAMWGS'.")
        if bitPlanes != 1:
            raise ValueError("TIPLMSuiteBIPE only supports bitPlanes=1.")
        if str(propMethod).upper() != "FOURIER":
            raise ValueError("TIPLMSuiteBIPE only supports propMethod='Fourier'.")

        plane_key = str(reconstructionPlane).upper()
        if plane_key not in ("IMAGE", "FOURIER"):
            raise ValueError("reconstructionPlane must be 'IMAGE' or 'FOURIER'.")
        self.reconstructionPlane = plane_key

        model_key = str(forwardModel).upper()
        if model_key not in ("SUPERPIXEL", "OPTICAL"):
            raise ValueError("forwardModel must be 'SUPERPIXEL' or 'OPTICAL'.")
        if model_key == "OPTICAL":
            if plane_key != "IMAGE":
                raise ValueError("forwardModel='OPTICAL' models the 4f image plane; "
                                 "use reconstructionPlane='IMAGE'.")
            if alg_key == "ADAM":
                # ADAM's gamut projection assumes the box-average model, so an
                # optically-optimized continuous phase would be re-encoded under
                # the wrong assumption and the gain would be thrown away.
                raise ValueError("forwardModel='OPTICAL' requires alg='DIRECT' or "
                                 "alg='ADAMWGS' (quantization must be inside the loop).")
        self.forwardModel = model_key
        self.apertureScale = float(apertureScale)

        if superpixelSize is not None:
            self.superpixelSize = self._validateSuperpixelSize(superpixelSize)
        n = self.superpixelSize
        self.offAxisCarrier = (float(offAxisCarrier[0]), float(offAxisCarrier[1]))

        # ShiftFOV and offAxisCarrier are far-field constructs: the first is an
        # fftshift convention and the second steers the reconstruction away from
        # residual zero order by tilting the field. Neither means anything for a
        # camera at the 4f exit, where the synthesised field is imaged directly.
        if plane_key == "IMAGE":
            if self.offAxisCarrier != (0.0, 0.0):
                raise ValueError("offAxisCarrier only applies to a far-field "
                                 "reconstruction; use (0.0, 0.0) with "
                                 "reconstructionPlane='IMAGE'.")
            ShiftFOV = False

        self.show_images = showImages
        self.bitPlanes = bitPlanes
        device_lambda_m = DeviceDictionary.get("lambda_m", DeviceDictionary.get("phase_lut_wavelength_m"))
        if device_lambda_m is not None and float(device_lambda_m) > 0:
            self.lambda_m = float(device_lambda_m)

        h = DeviceDictionary["h"]
        w = DeviceDictionary["w"]
        self.usable_h = (h // n) * n
        self.usable_w = (w // n) * n
        hs = self.usable_h // n
        ws = self.usable_w // n

        self.pitchW = DeviceDictionary["pitchW"]
        self.pitchH = DeviceDictionary["pitchH"]

        self.loadTarget(filename, colorChannel)
        # The reconstruction lives at superpixel resolution: the effective
        # aperture is the superpixel grid with pitch n * pixel pitch.
        if plane_key == "IMAGE":
            self._resizeTargetImagePlane(
                hs,
                ws,
                n * DeviceDictionary["pitchW"],
                n * DeviceDictionary["pitchH"],
                preserveAspect,
            )
        else:
            self.resizeTarget(
                self.lambda_m,
                hs,
                ws,
                n * DeviceDictionary["pitchW"],
                n * DeviceDictionary["pitchH"],
                ShiftFOV,
                preserveAspect,
            )
        if binarizeTarget:
            self.binarizeTarget(targetThreshold)
        self.updateTarget(FlipUD, FlipLR, InvertTarget)

        carrier_y, carrier_x = self.offAxisCarrier
        if carrier_y != 0.0 or carrier_x != 0.0:
            self.imTarget = np.roll(
                self.imTarget,
                shift=(int(np.round(carrier_y * hs)), int(np.round(carrier_x * ws))),
                axis=(0, 1),
            )

        self.lossMode = self._resolveLossMode(lossMode)

        nLevel = DeviceDictionary["nLevel"]
        pLevel = np.asarray(DeviceDictionary["pLevel"], dtype=np.float64)
        if pLevel.ndim != 1:
            raise ValueError("TIPLMSuiteBIPE only supports the 1D 0.67 phase table.")
        i0, i_pi = self.selectBinaryStates(nLevel, pLevel)
        self.binaryStateIndices = (i0, i_pi)
        levels = np.mod(pLevel[:nLevel], 1.0)
        self.binaryPLevels = np.array([levels[i0], levels[i_pi]], dtype=np.float64)

        self._ramp = self._buildRamp(self.usable_h, self.usable_w, n)
        self._buildApertureMask()
        self._prepareIllumination(illumination, illuminationFloor,
                                  illuminationCompensation)

        if alg_key == "DIRECT":
            stateq_direct = self.runDirect(feedbackIters=feedbackIters,
                                           feedbackBeta=feedbackBeta)
        elif alg_key == "ADAM":
            self.runADAM(initialPhase, numIter, propMethod, self.lossMode)
        else:
            self.runADAMwGS(nLevel, pLevel, initialPhase, numIter, propMethod, self.lossMode)

        self.CGH_output_cont = np.mod(self.CGH_output_cont, 2 * np.pi)

        if alg_key == "DIRECT":
            stateq_bin = stateq_direct.astype(np.int64)
        elif alg_key == "ADAM":
            # Encode the continuous field onto the binary gamut (paper's
            # per-superpixel argmin search).
            stateq_bin = self._projectToGamut(self.CGH_output_cont).astype(np.int64)
        else:
            # ADAMwGS already returns two-level phases; map them back exactly.
            stateq_bin, _ = self.quantizeCircularPhase(
                self.CGH_output_cont.reshape(-1) / (2 * np.pi),
                self.binaryPLevels
            )
            stateq_bin = stateq_bin.reshape(self.CGH_output_cont.shape).astype(np.int64)
        phaseq = self.binaryPLevels[stateq_bin]

        # Pixels outside the usable superpixel grid are parked at the 0-state.
        state_full = np.full((h, w), i0, dtype=np.float64)
        phase_full = np.full((h, w), 2 * np.pi * levels[i0], dtype=np.float64)
        state_full[:self.usable_h, :self.usable_w] = np.where(stateq_bin == 0, i0, i_pi)
        phase_full[:self.usable_h, :self.usable_w] = 2 * np.pi * phaseq

        self.CGH_output_phase_disc = phase_full
        self.CGH_output_state_disc = state_full
        self.CGH_output_disc = state_full
        if alg_key == "DIRECT":
            # no continuous stage exists; report the binary result for both
            self.CGH_output_cont = phase_full[:self.usable_h, :self.usable_w].copy()

        self.recoverImg(ShiftFOV, propMethod)
        self.CGH_mapped = self.deviceLibary.formatPLM(DeviceDictionary, self.CGH_output_state_disc)
        self.CGH_phase = self.CGH_output_phase_disc

    def recoverImg(self, ShiftFOV=True, propMethod="FOURIER"):
        if self.imTarget is None:
            return

        I_cont, _ = self.reconstructBIPE(self.CGH_output_cont)
        I_disc, _ = self.reconstructBIPE(self.CGH_output_phase_disc[:self.usable_h, :self.usable_w])

        # Honest quality numbers, computed BEFORE the display roll: PSNR of the
        # forward model against the target at superpixel resolution. The PSNR
        # printed during ADAM optimization is the CONTINUOUS (pre-encoding)
        # metric — with reconstructionPlane='IMAGE' the unconstrained field can
        # match the target almost exactly, so that number runs absurdly high
        # (>100 dB) and says nothing about the binary hologram. The encoded
        # figure below is the one that predicts hardware.
        def _psnr(I):
            mse = float(np.mean((I - self.imTarget) ** 2))
            return 20 * np.log10(1 / np.sqrt(mse + 1e-20))
        self.psnr_cont = _psnr(I_cont)
        self.psnr_disc = _psnr(I_disc)
        print("PSNR vs target [%s model]: continuous %.1f dB | ENCODED BINARY %.1f dB"
              % (self.forwardModel, self.psnr_cont, self.psnr_disc))

        # Always cross-check the encoded hologram against the OTHER model. The
        # box-average idealization and the real band-limited 4f disagree by tens
        # of dB, and only the optical number predicts what a camera will see.
        try:
            binary_phase = self.CGH_output_phase_disc[:self.usable_h, :self.usable_w]
            if self.forwardModel == "OPTICAL":
                n = self.superpixelSize
                hs, ws = self.usable_h // n, self.usable_w // n
                E = np.exp(1j * (binary_phase + self._ramp))
                if self.illumAmplitude is not None:
                    E = self.illumAmplitude * E
                U = E.reshape(hs, n, ws, n).mean(axis=(1, 3))
                other, label = np.abs(U) ** 2, "box-average"
            else:
                other, label = self._opticalIntensity(binary_phase, torchMode=False), "band-limited 4f"
            s = np.mean(other * self.imTarget) / (np.mean(other ** 2) + 1e-12)
            self.psnr_cross = _psnr(s * other)
            print("   cross-check, same hologram in the %s model: %.1f dB"
                  % (label, self.psnr_cross))
        except Exception as err:
            print("   cross-check unavailable: %s" % err)

        if ShiftFOV and self.reconstructionPlane != "IMAGE":
            shift = (int(self.imTarget.shape[0] / 2), int(self.imTarget.shape[1] / 2))
            I_cont = np.roll(I_cont, shift=shift, axis=(0, 1))
            I_disc = np.roll(I_disc, shift=shift, axis=(0, 1))

        self.imRecovered_cont = cv2.resize(I_cont, np.flip(self.imTarget.shape), interpolation=cv2.INTER_CUBIC)
        self.imRecovered_disc = cv2.resize(I_disc, np.flip(self.imTarget.shape), interpolation=cv2.INTER_CUBIC)

    def runADAM(self, initialPhase="Random", numIter=1, propMethod="Fourier", lossMode="mse"):
        if numIter < 1:
            raise ValueError("numIter must be >= 1.")

        phase_init = self._initialModulationPhase(initialPhase)

        msel = np.zeros((numIter, 1))
        psnrl = np.zeros((numIter, 1))

        phase_hp = torch.tensor(phase_init, dtype=torch.float32, device=self.MLDevice)
        phase_hp.requires_grad_(True)

        ramp = torch.tensor(self._ramp, dtype=torch.float32, device=self.MLDevice)

        optimizer = torch.optim.Adam([{"params": phase_hp}], lr=0.32)
        target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)
        loss_weights = self._torchLossWeights(target_intensity, lossMode)

        best_loss = 1e10
        best_phase = None

        for i in tqdm(range(0, numIter)):
            optimizer.zero_grad()

            intensity = self._superpixelIntensityTorch(phase_hp, ramp)

            with torch.no_grad():
                s = self._torchWeightedGain(intensity, target_intensity, loss_weights, self.SCALE_EPS)

            loss_val = self._torchWeightedMSE(s * intensity, target_intensity, loss_weights)
            msel[i] = loss_val.item()
            psnrl[i] = 20 * np.log10(1 / np.sqrt(msel[i] + 1e-20))

            if self.show_images:
                image_recon = (s * intensity).detach().cpu().numpy()
                cv2.imshow("BiPE ADAM Image", image_recon)
                cv2.setWindowTitle("BiPE ADAM Image", "Iteration: " + str(i + 1) + " MSE: " + str(msel[i]) + " PSNR: " + str(psnrl[i]))
                cv2.waitKey(0)

            loss_val.backward()
            optimizer.step()

            with torch.no_grad():
                if loss_val.item() < best_loss:
                    best_loss = loss_val.item()
                    best_phase = phase_hp.detach().clone()

        metric_name = "Balanced MSEL" if lossMode == "balanced" else "MSEL"
        psnr_name = "Balanced PSNR" if lossMode == "balanced" else "PSNR"
        print("Best " + metric_name + " (continuous, pre-encoding): " + str(np.min(msel)))
        print("Best " + psnr_name + " (continuous, pre-encoding — see encoded PSNR below): " + str(np.max(psnrl)))

        if best_phase is None:
            best_phase = phase_hp.detach().clone()

        self.CGH_output_cont = np.mod(best_phase.detach().cpu().numpy(), 2 * np.pi)

    def runADAMwGS(self, nLevel, pLevel, initialPhase="Random", numIter=1, propMethod="Fourier", lossMode="mse"):
        if numIter < 1:
            raise ValueError("numIter must be >= 1.")

        phase_init = self._initialModulationPhase(initialPhase)

        msel = np.zeros((numIter, 1))
        deterministic_msel = np.zeros((numIter, 1))
        psnrl = np.zeros((numIter, 1))
        deterministic_psnrl = np.zeros((numIter, 1))

        phase_hp = torch.tensor(phase_init, dtype=torch.float32, device=self.MLDevice)
        phase_hp.requires_grad_(True)

        ramp = torch.tensor(self._ramp, dtype=torch.float32, device=self.MLDevice)

        # BiPE only ever addresses the two ~pi-separated device states.
        levels = self.binaryPLevels.reshape(1, 1, -1)
        logits = np.tile(levels, (self.usable_h, self.usable_w, 1))
        logits = torch.tensor(logits, device=self.MLDevice, dtype=torch.float32)

        tauInitial = torch.tensor(6.5, dtype=torch.float32, device=self.MLDevice)
        tauMin = torch.tensor(3.1, dtype=torch.float32, device=self.MLDevice)
        quantizeMethod = self.BinaryGumbelQuantization(lut=logits, tau=6.5, hard=True).to(self.MLDevice)
        optimizer = torch.optim.Adam([{"params": phase_hp}], lr=0.17)

        target_intensity = torch.tensor(self.imTarget, dtype=torch.float32, device=self.MLDevice)
        loss_weights = self._torchLossWeights(target_intensity, lossMode)

        best_loss = 1e10
        best_quantized_phase = None

        for i in tqdm(range(0, numIter)):
            anneal_progress = 1.0 if numIter <= 1 else i / (numIter - 1)
            annealRate = torch.tensor(anneal_progress, dtype=torch.float32, device=self.MLDevice)
            quantizeMethod.anneal_temperature(annealRate, tauInitial=tauInitial, tauMin=tauMin)

            optimizer.zero_grad()
            quantized_phase = quantizeMethod(phase_hp)
            intensity = self._superpixelIntensityTorch(quantized_phase, ramp)

            with torch.no_grad():
                s = self._torchWeightedGain(intensity, target_intensity, loss_weights, self.SCALE_EPS)

            loss_val = self._torchWeightedMSE(s * intensity, target_intensity, loss_weights)
            msel[i] = loss_val.item()
            psnrl[i] = 20 * np.log10(1 / np.sqrt(msel[i] + 1e-20))

            with torch.no_grad():
                deterministic_quantized_phase = quantizeMethod.deterministic(phase_hp)
                deterministic_intensity = self._superpixelIntensityTorch(deterministic_quantized_phase, ramp)
                deterministic_gain = self._torchWeightedGain(deterministic_intensity, target_intensity, loss_weights, self.SCALE_EPS)
                deterministic_loss_val = self._torchWeightedMSE(deterministic_gain * deterministic_intensity, target_intensity, loss_weights)
                deterministic_msel[i] = deterministic_loss_val.item()
                deterministic_psnrl[i] = 20 * np.log10(1 / np.sqrt(deterministic_msel[i] + 1e-20))

                if deterministic_loss_val.item() < best_loss:
                    best_loss = deterministic_loss_val.item()
                    best_quantized_phase = deterministic_quantized_phase.detach().clone()

            if self.show_images:
                image_recon = self.reconstructBIPE(deterministic_quantized_phase.cpu().detach().numpy())[0]
                cv2.imshow("BiPE ADAMwGS Image", image_recon)
                cv2.setWindowTitle("BiPE ADAMwGS Image", "Iteration: " + str(i + 1) + " MSE: " + str(deterministic_msel[i]) + " PSNR: " + str(deterministic_psnrl[i]))
                cv2.waitKey(0)

            loss_val.backward()
            optimizer.step()

        metric_name = "Balanced MSEL" if lossMode == "balanced" else "MSEL"
        psnr_name = "Balanced PSNR" if lossMode == "balanced" else "PSNR"
        print("Best " + metric_name + " (quantized forward, ideal filter): " + str(np.min(deterministic_msel)))
        print("Best " + psnr_name + " (quantized forward, ideal filter): " + str(np.max(deterministic_psnrl)))
        print("Best Stochastic Training " + metric_name + ": " + str(np.min(msel)))
        print("Best Stochastic Training " + psnr_name + ": " + str(np.max(psnrl)))

        if best_quantized_phase is None:
            best_quantized_phase = quantizeMethod.deterministic(phase_hp).detach().clone()

        self.CGH_output_cont = np.mod(best_quantized_phase.detach().cpu().numpy(), 2 * np.pi)
