import numpy as np
from itertools import permutations

try:
    import scipy.fft as _fft

    def _fft2(x):
        return _fft.fft2(x, workers=-1)

    def _ifft2(x):
        return _fft.ifft2(x, workers=-1)
except ImportError:
    _fft2, _ifft2 = np.fft.fft2, np.fft.ifft2

from TIPLMSuiteFourPhase import FourPhaseCGHGenerator, DeviceLibrary

__all__ = ["DualOrderFourPhaseCGHGenerator", "RecombinedFourPhaseCGHGenerator", "DeviceLibrary"]


class DualOrderFourPhaseCGHGenerator(FourPhaseCGHGenerator):
    """
    Two images from one four-phase hologram: image 1 through the paper's
    on-axis pinhole, image 2 through a second pinhole on one of the
    superpixel orders that the paper's filter throws away.

    The redundancy
    --------------
    A 2 x 2 superpixel with pixel phasors p00 p01 / p10 p11 is exactly the
    sum of four spatial modes

        S = (p00 + p01 + p10 + p11) / 4     on axis            (image 1)
        B = (p00 - p01 + p10 - p11) / 4     order (1/2p, 0)
        C = (p00 + p01 - p10 - p11) / 4     order (0, 1/2p)
        D = (p00 - p01 - p10 + p11) / 4     order (1/2p, 1/2p) (image 2 default)

    with |S|^2 + |B|^2 + |C|^2 + |D|^2 = 1. The paper's encoding fixes only
    S, which depends on the multiset of the four states, so 65536 - 3876
    (94%) of the raw assignments are reorderings that leave image 1 alone.
    They are not free choices of B, C, D, though: B, C and D are the three
    ways of splitting the four phasors into two pairs, so a multiset owns
    three fixed pair-difference magnitudes {r1, r2, r3} and its (up to) 24
    arrangements only decide which of them goes to which order, with signs.
    Image 2 can therefore only choose between three brightnesses per
    superpixel -- typically ~0.35..0.5, i.e. no true black. That is why the
    epsilon tolerance matters: a neighbouring multiset whose S is within
    epsilon of the paper's choice can have a pair split with r ~ 0 (e.g.
    {a, a, b, b}) and so give image 2 its dark regions.

    Candidates for a superpixel are all arrangements of every multiset whose
    phasor satisfies |S - A1| <= |S_paper - A1| + epsilon, with A1 the
    requested image-1 field and S_paper the paper's nearest-phasor choice.
    epsilon = 0 keeps the paper's multiset exactly (S untouched).

    What really limits it: pinhole cross-talk
    -----------------------------------------
    A rect pinhole does not perform the ideal superpixel average. Along each
    axis the passband weights the matching-parity mode by cos(pi f p) and the
    other one by sin(pi f p), so every mode leaks into every pinhole through
    its spatial GRADIENTS. S and D talk to each other only at second order
    (sin * sin); B and C leak first order into both. With the leftover energy
    1 - |S|^2 - |D|^2 forced into B and C, keeping those two fields smooth is
    what decides both images, and it is why the arrangement must be chosen
    against the real optics rather than per superpixel in isolation. Even
    with no second image, choosing arrangements this way lifts image 1 above
    the paper's ascending arrangement (the image-1 pre-pass below).

    Optimiser
    ---------
    Loss (field domain, both pinholes, targets with a gain fitted each pass):
        L = w1 ||A1 e - g1 sqrt(t1)||^2 + w2 ||A2 e - g2 sqrt(t2) e^{i psi}||^2
    with A_k = F^-1 W_k F the pinhole operators. Changing one superpixel
    changes L by an exactly quadratic amount; the same-superpixel block of
    A_k^H A_k is diagonal in the S, B, C, D basis, so the best move for a
    superpixel is a weighted nearest neighbour in mode space,
        argmin_c sum_M lam_M |M_c - (M_now - r_M / lam_M)|^2,
    over its candidates (r = gradient in mode basis, lam_M = curvature).
    Superpixels are updated on an L x L sub-lattice, one colour at a time, so
    simultaneously moved superpixels are far enough apart for the per-site
    curvature to hold; L = 3 for pinholes narrower than 0.8 of an order (the
    optical kernel is wider). The image-2 phase psi starts fixed at +i on
    the order's carrier (for bright image-1 regions the pair differences are
    ~ perpendicular to S) and is then left free, GS style.

    Measured (harper grayscale + dlp_logo, 632.8 nm theoretical LUT,
    amplitude scale 0.8, D order, band-limited 4f, apertures as fractions of
    one superpixel order, secondWeight 0.3 unless noted):

        pinholes 1 / 2   epsilon   paper image 1   this: image 1 / image 2
        1.0 / 0.5        0.05         21.5 dB         18.4 / 24.0 dB
        0.8 / 0.4        0.05         22.5 dB         22.7 / 25.9 dB
        0.6 / 0.4        0            23.0 dB         23.0 / 24.2 dB
        0.6 / 0.4        0.01         23.0 dB         23.5 / 25.8 dB
        0.6 / 0.4        0.02         23.0 dB         23.7 / 26.3 dB
        0.6 / 0.4        0.05         23.0 dB         23.7 / 26.4 dB  (default)
        0.6 / 0.4        0.05, w 0.1  23.0 dB         24.4 / 25.8 dB
    The paper's arrangement seen through pinhole 2 scores 8.8 dB (no image);
    image 1 after the image-1-only pre-pass is 24.6 dB at 0.6. The mirror
    sinc envelope (pixelFillFactor 0.9-1.0) moves these by < 0.2 dB, and the
    residual ghost of either target in the other image has |corr| < 0.05.

    Independent wave-optics check (TIPLMSuiteFourPhaseCurvedBeam, collimated,
    image 1 / image 2; paper encoding at the same pinhole 1 in brackets):
        harper + logo    23.80 / 26.38 dB, 4 samples per mirror 23.82 / 26.35,
                         true square mirrors 23.82 / 26.33          (23.16)
        bear1 + bear2    27.72 / 26.42, true square mirrors 27.75 / 26.33   (28.36)
    so the numbers are not a model artefact. Robustness (harper+logo, bears):
        LUT random error 0.01 wave rms   image 2 -0.4 / -2.1 dB
                         0.02 wave rms   image 2 -2.4 / -6.4 dB  (paper image 1 -1.2 / -1.6)
        LUT phase depth +-5%             image 2 -0.3 .. -1.4 dB
        mirror gaps (fill 0.9) reflecting fully   image 1 -2 / -6 dB, as the paper;
                                                  image 2 untouched
        Gaussian beam (1/e^2 radius 1.5x half-diagonal)   every encoding -5 .. -7 dB
    Image 2 needs a well-calibrated LUT (errors <~ 0.01 wave); every
    encoding here needs a flat-top beam (no illumination compensation).

    Narrowing pinhole 1 below one order trades resolution for much less
    first-order leakage; it helps the paper's own encoding too (21.5 dB at
    1.0 -> 23.0 dB at 0.6), which is why the table compares at equal
    aperture. Pinhole 2 can be narrower still since the logo carries little
    detail, but below ~0.4 the logo blurs (0.3: 23.5 dB).

    Experimental notes
    ------------------
    * Both pinhole beams are imaged onto the same camera plane (pinhole 2's
      with a tilt), so they overlap there. Look at one pinhole at a time, or
      split them right behind the filter plane (a small pick-off mirror or
      wedge on pinhole 2) into two imaging arms.
    * The D order appears identically at all four diagonal positions
      (+-1/2p, +-1/2p) (the pixel spectrum is periodic); use any one. It
      contains no zero order, so mirror-gap / cover-glass light, which lands
      at DC, does not reach pinhole 2.
    * pixelFillFactor applies the mirror-aperture sinc envelope, which tilts
      the passband at the off-axis order (sinc(0.4..0.6) for the default
      pinhole 2). None models ideal point pixels, as the base class does.

    RecombinedFourPhaseCGHGenerator (below) uses the same machinery to put
    BOTH pinholes into one image instead.
    """

    ORDERS = {"B": (0.5, 0.0), "C": (0.0, 0.5), "D": (0.5, 0.5)}
    ORDER_SIGNS = {"B": ((1, -1), (1, -1)), "C": ((1, 1), (-1, -1)), "D": ((1, -1), (-1, 1))}
    SELECT_CHUNK = 8192

    def __init__(self):
        super().__init__()
        self.secondOrder = "D"
        self.secondApertureScale = 0.4
        self.epsilon = 0.05
        self.pixelFillFactor = None
        self.imTarget2 = None
        self.imRecovered2 = None
        self.history = []
        self._cfgTables = None

    # ------------------------------------------------------------------
    # The 65536 raw assignments and their redundancy
    # ------------------------------------------------------------------
    def configTables(self):
        """
        Every ordered assignment of 16 states to the four pixels (k row-major:
        p00, p01, p10, p11; config index m = s00 + 16 s01 + 256 s10 + 4096 s11).
        Returns a dict with
            states  (65536, 4)   device state per pixel
            modes   (65536, 4)   complex S, B, C, D
            perm    (3876, 24)   config indices of each multiset's arrangements
                                 (multiset order = fourPhaseGamut(); padded
                                 by repetition where states repeat)
            paper   (3876,)      the ascending arrangement FourPhaseCGHGenerator writes
        """
        if self._cfgTables is None:
            gamut, combos = self.fourPhaseGamut()
            m = np.arange(1 << 16)
            states = np.stack([(m >> (4 * k)) & 15 for k in range(4)], axis=1)
            z = np.exp(2j * np.pi * self.pLevels[states])
            modes = np.stack([
                z.sum(axis=1),
                z[:, 0] - z[:, 1] + z[:, 2] - z[:, 3],
                z[:, 0] + z[:, 1] - z[:, 2] - z[:, 3],
                z[:, 0] - z[:, 1] - z[:, 2] + z[:, 3],
            ], axis=1) / 4.0

            weights = np.array([1, 16, 256, 4096])
            perm = np.empty((len(combos), 24), dtype=np.int64)
            for a, combo in enumerate(combos):
                cfgs = sorted({int(np.dot(p, weights)) for p in permutations(combo)})
                perm[a] = np.resize(cfgs, 24)

            self._cfgTables = {
                "states": states,
                "modes": modes,
                "perm": perm,
                "paper": combos @ weights,
            }
        return self._cfgTables

    def configsToStates(self, cfg):
        """(hs, ws) config indices -> (2hs, 2ws) device-state map."""
        hs, ws = cfg.shape
        st = self.configTables()["states"][cfg.ravel()].reshape(hs, ws, 2, 2)
        return st.transpose(0, 2, 1, 3).reshape(2 * hs, 2 * ws)

    def _candidateAtoms(self, A1, d_paper, atom_paper, epsilon, maxAtoms):
        """(N, K) multisets whose phasor lies within d_paper + epsilon of the
        request; slots beyond the tolerance repeat the paper's multiset."""
        if epsilon <= 0:
            return atom_paper.reshape(-1, 1).astype(np.int32)
        gamut, _ = self.fourPhaseGamut()
        pts = np.column_stack([A1.real.ravel(), A1.imag.ravel()])
        K = min(int(maxAtoms), gamut.size)
        if self._gamutTree is not None:
            dist, atoms = self._gamutTree.query(pts, k=K, workers=-1)
        else:
            d2 = np.abs(pts[:, :1] + 1j * pts[:, 1:] - gamut[None, :]) ** 2
            atoms = np.argsort(d2, axis=1)[:, :K]
            dist = np.sqrt(np.take_along_axis(d2, atoms, axis=1))
        ok = dist <= d_paper.ravel()[:, None] + epsilon + 1e-12
        atoms = np.where(ok, atoms, atom_paper.reshape(-1, 1))
        inside = ok.sum(axis=1)
        print("epsilon %.3f: %.1f multisets per superpixel on average (max %d, cap %d)"
              % (epsilon, inside.mean(), inside.max(), K))
        return atoms.astype(np.int32)

    # ------------------------------------------------------------------
    # Pinholes
    # ------------------------------------------------------------------
    @staticmethod
    def pinholeShape(dx, dy, half, shape="square"):
        """Pinhole (field) transmission at offsets (dx, dy) from its centre,
        in the same units as `half` (the square's half-width). Non-square
        shapes keep the square's area; 'soft' is the square with raised-cosine
        edges (30% of the half-width, 50% transmission at the square's edge).

        shape = ("super", n, s) is a continuous family: the superellipse
        |x|^n + |y|^n <= r^n of the square's area (n = 1 diamond, 2 circle,
        -> inf square, < 1 four-pointed star) with a raised-cosine taper of
        relative width s across its edge (s = 0 binary; s = 2 grades from
        full transmission at the centre to zero at twice the radius)."""
        if isinstance(shape, (tuple, list)):
            from math import gamma
            _, n, s = shape
            n, s = float(n), float(s)
            r = half * np.sqrt(gamma(1 + 2 / n)) / gamma(1 + 1 / n)
            rho = (np.abs(dx) ** n + np.abs(dy) ** n) ** (1 / n) / r
            if s <= 0:
                return (rho <= 1.0).astype(np.float64)
            t = np.clip((rho - (1 - s / 2)) / s, 0.0, 1.0)
            return 0.5 * (1 + np.cos(np.pi * t))
        if shape == "square":
            return ((np.abs(dx) <= half) & (np.abs(dy) <= half)).astype(np.float64)
        if shape == "circle":
            return (dx ** 2 + dy ** 2 <= (2 * half / np.sqrt(np.pi)) ** 2).astype(np.float64)
        if shape == "diamond":
            return (np.abs(dx) + np.abs(dy) <= np.sqrt(2) * half).astype(np.float64)
        if shape == "soft":
            w = 0.3 * half

            def edge(u):
                t = np.clip((np.abs(u) - (half - w / 2)) / w, 0.0, 1.0)
                return 0.5 * (1 + np.cos(np.pi * t))
            return edge(dx) * edge(dy)
        raise ValueError("shape must be 'square', 'circle', 'diamond' or 'soft'.")

    def _pinholeWeight(self, center, apertureScale, shape="square"):
        """Pinhole (square half-width apertureScale / 4 cycles per pixel, or
        an equal-area shape) centred at `center` cycles/pixel, wrap-aware
        (the pixel spectrum is periodic), optionally times the mirror-aperture
        sinc at the physical frequency."""
        GH, GW = self._padShape
        fx = np.fft.fftfreq(GW)[None, :]
        fy = np.fft.fftfreq(GH)[:, None]
        cx, cy = center
        dx = np.mod(fx - cx + 0.5, 1.0) - 0.5
        dy = np.mod(fy - cy + 0.5, 1.0) - 0.5
        half = apertureScale / (2.0 * self.SUPERPIXEL)
        W = self.pinholeShape(dx, dy, half, shape)
        if self.pixelFillFactor:
            W = W * np.sinc(self.pixelFillFactor * (cx + dx)) * np.sinc(self.pixelFillFactor * (cy + dy))
        return W

    def _buildPinholes(self):
        self._W1 = self._pinholeWeight((0.0, 0.0), self.apertureScale)
        self._W2 = self._pinholeWeight(self.ORDERS[self.secondOrder], self.secondApertureScale)
        self._mask = self._W1  # the base class's filteredField uses pinhole 1
        self._curv1 = self._modeCurvature(self._W1 ** 2)
        self._curv2 = self._modeCurvature(self._W2 ** 2)

    def _modeCurvature(self, W2):
        """Diagonal of the same-superpixel block of A^H A in the S, B, C, D
        basis (lam_M = v_M^T Q v_M / 4, v_M the +-1 mode pattern)."""
        GH, GW = self._padShape
        h = np.fft.ifft2(W2)
        pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
        Q = np.array([[h[(b[0] - a[0]) % GH, (b[1] - a[1]) % GW] for b in pos] for a in pos])
        V = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=np.float64)
        return np.real(np.diag(V @ Q @ V.T)) / 4.0

    def secondFilterGeometry(self, f1):
        """Pinhole 2 in the Fourier plane of the first 4f lens (metres). The
        same image sits at every sign combination of `center`."""
        cx, cy = self.ORDERS[self.secondOrder]
        n = self.SUPERPIXEL
        x, y = self.lambda_m * f1 * cx / self.pitchW, self.lambda_m * f1 * cy / self.pitchH
        return {
            "center": (x, y),
            "radius": float(np.hypot(x, y)),
            "aperture": (self.secondApertureScale * self.lambda_m * f1 / (n * self.pitchW),
                         self.secondApertureScale * self.lambda_m * f1 / (n * self.pitchH)),
        }

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------
    def _pixelSpectrum(self, cfg):
        GH, GW = self._padShape
        uh, uw = self.usable_h, self.usable_w
        oy, ox = (GH - uh) // 2, (GW - uw) // 2
        e = np.exp(2j * np.pi * self.pLevels[self.configsToStates(cfg)])
        buf = np.zeros((GH, GW), dtype=np.complex128)
        buf[oy:oy + uh, ox:ox + uw] = e
        return e, _fft2(buf)

    def _crop(self, X):
        GH, GW = self._padShape
        oy, ox = (GH - self.usable_h) // 2, (GW - self.usable_w) // 2
        return X[oy:oy + self.usable_h, ox:ox + self.usable_w]

    def _pad(self, x):
        GH, GW = self._padShape
        oy, ox = (GH - self.usable_h) // 2, (GW - self.usable_w) // 2
        buf = np.zeros((GH, GW), dtype=np.complex128)
        buf[oy:oy + self.usable_h, ox:ox + self.usable_w] = x
        return buf

    def images(self, cfg):
        """Camera intensities (per superpixel) behind pinhole 1 and pinhole 2."""
        _, E = self._pixelSpectrum(cfg)
        I1 = self._binIntensity(self._crop(_ifft2(E * self._W1)))
        I2 = self._binIntensity(self._crop(_ifft2(E * self._W2)))
        return I1, I2

    def _score(self, cfg):
        I1, I2 = self.images(cfg)
        return self._psnr(I1, self.imTarget), self._psnr(I2, self.imTarget2)

    @staticmethod
    def _modes(e):
        hs, ws = e.shape[0] // 2, e.shape[1] // 2
        q = e.reshape(hs, 2, ws, 2).transpose(0, 2, 1, 3).reshape(hs, ws, 4)
        return np.stack([
            q[..., 0] + q[..., 1] + q[..., 2] + q[..., 3],
            q[..., 0] - q[..., 1] + q[..., 2] - q[..., 3],
            q[..., 0] + q[..., 1] - q[..., 2] - q[..., 3],
            q[..., 0] - q[..., 1] - q[..., 2] + q[..., 3],
        ], axis=-1).reshape(-1, 4) / 4.0

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    def _selectionMetric(self, Q):
        """For a Hermitian mode-basis curvature Q = L L^H, argmin_c
        (m_c - m*)^H Q (m_c - m*) is a Euclidean nearest neighbour between
        L^H m_c and L^H m*. Returns (L, the 65536 configs mapped, their norms)."""
        Q = 0.5 * (Q + Q.conj().T)
        L = np.linalg.cholesky(Q + 1e-6 * np.real(np.trace(Q)) * np.eye(4))
        Y = self.configTables()["modes"] @ L.conj()
        Y = np.concatenate([Y.real, Y.imag], axis=1).astype(np.float32)
        return L, Y, (Y ** 2).sum(axis=1)

    def _select(self, sites, candidates, Mstar, metric):
        """Best candidate per site under the metric. candidates = (ids, table):
        site i may take any config in table[ids[i]]."""
        ids, table = candidates
        L, Y, norm = metric
        y = Mstar[sites] @ L.conj()
        y = np.concatenate([y.real, y.imag], axis=1).astype(np.float32)
        out = np.empty(len(sites), dtype=np.int64)
        for s in range(0, len(sites), self.SELECT_CHUNK):
            chunk = sites[s:s + self.SELECT_CHUNK]
            cand = table[ids[chunk]].reshape(len(chunk), -1)
            J = norm[cand] - 2.0 * np.einsum("nkd,nd->nk", Y[cand], y[s:s + self.SELECT_CHUNK])
            out[s:s + self.SELECT_CHUNK] = cand[np.arange(len(chunk)), J.argmin(axis=1)]
        return out

    def _sweep(self, cfg, atoms, w2n, fixedPhase):
        """One pass over every sub-lattice colour. Returns (cfg, sites changed)."""
        hs, ws = cfg.shape
        st1 = np.repeat(np.repeat(np.sqrt(self.imTarget), 2, 0), 2, 1)
        st2 = np.repeat(np.repeat(np.sqrt(self.imTarget2), 2, 0), 2, 1)
        e, E = self._pixelSpectrum(cfg)
        G1, G2 = self._crop(_ifft2(E * self._W1)), self._crop(_ifft2(E * self._W2))
        g1 = np.sum(np.abs(G1) * st1) / max(np.sum(st1 ** 2), 1e-12)
        g2 = np.sum(np.abs(G2) * st2) / max(np.sum(st2 ** 2), 1e-12)
        # relative-error weights: both images count in units of their own level
        w1, w2 = 1.0 / max(g1, 1e-12) ** 2, w2n / max(g2, 1e-12) ** 2
        lam = w1 * self._curv1 + w2 * self._curv2
        if fixedPhase:
            psi = 1j * np.tile(np.array(self.ORDER_SIGNS[self.secondOrder], dtype=np.float64),
                               (hs, ws))
        else:
            psi = np.exp(1j * np.angle(G2))
        AT = self._crop(_ifft2(w1 * self._W1 * _fft2(self._pad(g1 * st1))
                               + w2 * self._W2 * _fft2(self._pad(g2 * st2 * psi))))
        WW = w1 * self._W1 ** 2 + w2 * self._W2 ** 2
        metric = self._selectionMetric(np.diag(lam))
        candidates = (atoms, self.configTables()["perm"])

        flat = cfg.ravel().copy()
        changed = 0
        for k, sites in enumerate(self._groups):
            if k:
                e, E = self._pixelSpectrum(flat.reshape(hs, ws))
            r = self._modes(self._crop(_ifft2(WW * E)) - AT)
            targets = self._modes(e) - r / lam[None, :]
            new = self._select(sites, candidates, targets, metric)
            changed += int(np.count_nonzero(new != flat[sites]))
            flat[sites] = new
        return flat.reshape(hs, ws), changed

    def _subLattice(self, spacing):
        hs, ws = self.usable_h // 2, self.usable_w // 2
        uu, vv = np.meshgrid(np.arange(ws), np.arange(hs))
        return [np.flatnonzero(((vv % spacing == i) & (uu % spacing == j)).ravel())
                for i in range(spacing) for j in range(spacing)]

    # ------------------------------------------------------------------
    def createCGH(self, DeviceDictionary,
                  filename="", secondFilename="", colorChannel="gray",
                  FlipUD=False, FlipLR=False, InvertTarget=False, InvertSecond=False,
                  preserveAspect=True,
                  secondOrder="D", epsilon=0.05,
                  amplitudeScale=0.8, apertureScale=0.6, secondApertureScale=0.4,
                  secondWeight=0.3, image1Iters=4, dualIters=12, fixedPhaseIters=3,
                  updateSpacing="auto", maxAtoms=12, pixelFillFactor=None):
        """
        filename / secondFilename  targets for pinhole 1 (on axis) and pinhole 2
        secondOrder     'D' (diagonal, default), 'B' (x) or 'C' (y)
        epsilon         extra image-1 field error allowed per superpixel
                        (gamut radius = 1); 0 keeps the paper's multisets
        amplitudeScale  image-1 amplitude scale as in the paper encoder
        apertureScale, secondApertureScale  pinhole widths, fractions of one
                        superpixel order (lambda f / 2p)
        secondWeight    image-2 weight relative to image 1 (relative errors)
        image1Iters     arrangement-only passes for image 1 before image 2
        dualIters       passes with both pinholes
        fixedPhaseIters dual passes with the image-2 phase held at +i
        updateSpacing   sub-lattice spacing; 'auto' -> 2 when pinhole 1 is at
                        least 0.8 order wide, else 3 (wider optical kernel)
        maxAtoms        cap on multisets considered per superpixel
        """
        secondOrder = str(secondOrder).upper()
        if secondOrder not in self.ORDERS:
            raise ValueError("secondOrder must be 'B', 'C' or 'D'.")
        if isinstance(amplitudeScale, str):
            raise ValueError("DualOrderFourPhaseCGHGenerator needs a numeric amplitudeScale.")
        self.alg = "DUAL"
        self.secondOrder = secondOrder
        self.epsilon = float(epsilon)
        self.apertureScale = float(apertureScale)
        self.secondApertureScale = float(secondApertureScale)
        self.pixelFillFactor = pixelFillFactor
        self.propDistance = 0.0
        self._setupDevice(DeviceDictionary)
        self._cfgTables = None
        self._buildPinholes()
        if updateSpacing == "auto":
            updateSpacing = 2 if self.apertureScale >= 0.8 else 3
        self._groups = self._subLattice(int(updateSpacing))

        self.imTarget2 = self.prepareTarget(secondFilename, colorChannel, FlipUD, FlipLR,
                                            InvertSecond, preserveAspect).copy()
        self.prepareTarget(filename, colorChannel, FlipUD, FlipLR, InvertTarget, preserveAspect)

        # 1) the paper: nearest multiset for image 1, ascending arrangement
        tables = self.configTables()
        amp = np.sqrt(np.clip(self.imTarget, 0.0, None))
        A1 = float(amplitudeScale) * amp / max(float(amp.max()), 1e-12)
        self.amplitudeScale = float(amplitudeScale)
        d2, atom_paper = self._nearestPhasor(np.column_stack([A1.ravel(), np.zeros(A1.size)]))
        atom_paper = atom_paper.reshape(A1.shape)
        cfg = tables["paper"][atom_paper]
        self.psnr_paper = self._score(cfg)
        self.history = [("paper",) + self.psnr_paper]
        print("paper encoding (ascending arrangement): image 1 %.2f dB | pinhole 2 %.2f dB"
              % self.psnr_paper)

        # 2) image 1 alone, re-arranging the paper's multisets (S untouched)
        same = atom_paper.reshape(-1, 1).astype(np.int32)
        for it in range(int(image1Iters)):
            cfg, changed = self._sweep(cfg, same, 0.0, True)
            self.history.append(("image1",) + self._score(cfg))
            print("   image-1 pass %d: %.2f dB (%d superpixels re-arranged)"
                  % (it + 1, self.history[-1][1], changed))
            if not changed:
                break
        self.psnr_image1Only = self.history[-1][1]

        # 3) both pinholes, over the epsilon-feasible multisets
        atoms = self._candidateAtoms(A1, np.sqrt(d2).reshape(A1.shape), atom_paper,
                                     self.epsilon, maxAtoms)
        for it in range(int(dualIters)):
            cfg, changed = self._sweep(cfg, atoms, float(secondWeight), it < fixedPhaseIters)
            self.history.append(("dual",) + self._score(cfg))
            print("   dual pass %d: image 1 %.2f dB | image 2 %.2f dB (%d changed)"
                  % ((it + 1,) + self.history[-1][1:] + (changed,)))
            if not changed:
                break

        self.CGH_output_config = cfg
        self._storeResult(DeviceDictionary, self.configsToStates(cfg))

    def recoverImg(self, ShiftFOV=False, propMethod="FOURIER"):
        if self.imTarget is None or self.imTarget2 is None:
            return
        cfg = self.CGH_output_config
        I1, I2 = self.images(cfg)
        self.psnr_disc = self._psnr(I1, self.imTarget)
        self.psnr2 = self._psnr(I2, self.imTarget2)
        S = self.configTables()["modes"][cfg.ravel(), 0].reshape(cfg.shape)
        self.psnr_superpixel = self._psnr(np.abs(S) ** 2, self.imTarget)
        print("Dual-order four-phase (%s order, epsilon %.3f): image 1 %.2f dB (paper %.2f) | "
              "image 2 %.2f dB (paper %.2f)"
              % (self.secondOrder, self.epsilon, self.psnr_disc, self.psnr_paper[0],
                 self.psnr2, self.psnr_paper[1]))
        g1 = np.mean(I1 * self.imTarget) / (np.mean(I1 ** 2) + 1e-12)
        g2 = np.mean(I2 * self.imTarget2) / (np.mean(I2 ** 2) + 1e-12)
        self.imRecovered_disc = g1 * I1
        self.imRecovered_cont = self.imRecovered_disc
        self.imRecovered2 = g2 * I2


class RecombinedFourPhaseCGHGenerator(DualOrderFourPhaseCGHGenerator):
    """
    ONE image from several superpixel orders: each folded order's band is
    brought onto the axis (its carrier tilt removed) and all are overlapped
    coherently with complex weights c_M, so the camera sees

        G = sum_M c_M * carrier_M * A_M e,    M in {S, B, C, D}

    Per superpixel that is sum_M c_M M = sum_k w_k p_k, a fixed linear
    combination of the four pixels with weights w = H c / 4 (pixelWeights()).
    The weights decide the gamut: pixels that share a weight are
    interchangeable, so only DISTINCT weights give distinct phasors.

        fold (default first)   pixel weights               phasors   radius
        B + i C                (1+i, -1+i, 1-i, -1-i)/4     58081    sqrt 2
        S + i D                (1+i, 1-i, 1-i, 1+i)/4       18496    sqrt 2
        C (1+i) + D (1-i)      (1, i, -1, -i)/2             58081    2
        S + D                  (1, 0, 0, 1)/2                 136    1
    (four-phase alone: 3876.) B + i C wins twice over: every pixel gets its
    own weight, and B and C sit at the same distance from the axis, so the
    mirror-aperture envelope dims both equally (0.64 each, where S + D is
    1.0 vs 0.4). It also uses no on-axis order, so mirror-gap and window
    light (which lands on axis) never reaches the image.

    Encoding: the KD tree of the paper, over the 65536 combined phasors,
    then the same exact local descent as DualOrderFourPhaseCGHGenerator on
    the one-image loss ||A e - g sqrt(t)||^2. The same-superpixel curvature
    couples the modes, so the per-site metric is the full 4 x 4 block
    (Cholesky, _selectionMetric). Each superpixel may take any of the
    maxCandidates configs nearest to its request in the combined gamut --
    48 was a real restriction (+1.4 dB at 192 on S + i D).

    Mirror aperture -- why pixelFillFactor defaults to 1.0 here
    ------------------------------------------------------------
    Each mirror is a finite square, so the PLM spectrum carries a sinc
    envelope: the D order arrives at ~0.4x the DC order's amplitude, and
    tilted across pinhole 2 (sinc 0.86 -> 0.37 per axis at 0.8 order). A
    separate image in pinhole 2 barely notices (the camera gain absorbs it),
    but a COHERENT sum does: a hologram optimised without the envelope
    scores 26.9 dB in its own model but only 17.1 dB in the wave-optics check
    (TIPLMSuiteFourPhaseCurvedBeam) with an equal-weight fold, recovering to
    26.0 dB only if the DC arm is attenuated ~4x in intensity (|c| ~ 2).
    Optimising S + i D with the envelope in the model (pixelFillFactor=1.0,
    default) needs no attenuator: 24.4 dB in this model, 24.8 dB in the wave check
    with exact square mirrors (25.8 dB if mirrors are drawn as 2-sample
    blocks, whose envelope is too kind to the D order: 0.71 vs 0.64/axis).

    Measured (632.8 nm theoretical LUT, wave check = TIPLMSuiteFourPhaseCurvedBeam
    with exact square mirrors; model / wave check):
                                                harper           bear1
        paper, one on-axis pinhole (0.6)        23.0 / 23.2      28.3 / 28.4
        S + i D, 48 cand, scale 0.8 (old)       24.4 / 24.8      29.3 / 29.4
        S + i D, 192 candidates                 26.1 / 26.2
        C (1+i) + D (1-i)                       25.3 / 25.3
        B + i C, 48 cand, scale 0.8             27.5 / 27.5      34.4 / 34.5
        B + i C, 192 cand, scale 0.95, square   28.8 / 28.8      34.2 / 34.2
        B + i C, 384 cand, scale 0.95, square   29.0 / 28.9
    Square pinholes of 0.8 order beat 0.7 and 0.9 (26.9 dB each); 20 passes
    add nothing over 12.

    Pinhole SHAPE (equal open area, 0.8 order, wave check):
                        harper   bear1    testimag  Siemens star
        square          28.8     34.2     36.4      26.6
        circle          29.7     35.6     37.5      27.0
        diamond         30.4     36.7     38.4      26.8
    Diamond SIZE (orders), harper / bear1:
        binary   0.6 27.6 / 36.1   0.7 29.3 / 35.9   0.8 30.4 / 36.7
                 0.9 30.5 / 37.1 (default)           1.0 29.8 / 36.8
        graded   0.7 30.9          0.8 31.3 / 36.9   0.9 31.2 / 37.6   1.0 30.8
    Folding adds control, so it can spend some on resolution: the fold's best
    size (0.9) is larger than a single pinhole's (0.6-0.7).
    also on harper: soft-edged square 29.3 (29.8 at 0.9), circle at 0.9
    29.7 -> 30.3, B square + C circle 29.7, B circle 0.9 + C square 0.7 30.2.
    The diamond is the square rotated 45 deg: it keeps the frequencies along
    the x and y axes, where natural images put most of their spectrum, and
    drops the diagonal corners (least gain on the isotropic star).

    Graded (analog) pupils, shape = ("super", n, s), harper / bear1:
        n 1, s 0     binary diamond                   30.4 / 36.7
        n 0.7, s 0   four-pointed star                30.1
        n 1, s 0.3 / 0.6 / 1.0                        30.8 / 31.1 (36.8) / 31.3
        n 1, s 1.0, size 0.9                          31.2 / 37.6
    One on-axis pupil and NO fold (orders=[("S", 1, size, shape)], same
    optimiser): harper square 0.6 25.7, graded diamond 0.6 / 0.8 / 1.0
    28.3 / 27.1 / 24.8; bear1 square 0.6 31.4, graded 0.8 31.2. Grading and
    folding are complementary: the pupil sharpens how each value arrives,
    the fold makes the arrangements distinct values in the first place
    (58081 vs 3876), and on top of the best single graded pupil the fold
    still adds 3.0 dB (harper) / 6.2 dB (bear1).
    A grey-scale transmission tapering across the pinhole adds ~0.9 dB. The
    binary diamond stays the default because it is free to build; a graded
    pupil needs a coherent amplitude mask (grey-scale photomask, or an
    amplitude LCD between polarisers). A time-dithered DMD does NOT do it:
    it averages the intensities of binary-pupil images instead of weighting
    the field.

    Robustness (square pinholes), harper, old S + i D vs B + i C:
        window 0.5% AR / reflecting gaps (fill 0.9)   -0.65 / -1.15   vs  0.00 / -0.03 dB
        LUT random error 0.01 wave rms                -0.20           vs -0.79 dB
        relative beam phase off 10 / 25 deg           -0.27 / -1.40   vs -0.64 / -2.86 dB
        second beam 20% dimmer                        -0.73           vs -1.46 dB
    so B + i C trades stray-light immunity for twice the sensitivity to its
    own beam balance -- and still clears the old design with a 25 deg error.
    Without the envelope in the model (pixelFillFactor=None, i.e. assuming
    point-like mirrors) the model scored 26.2 / 26.8 / 24.0 dB at pinholes
    0.6 / 0.8 / 1.0 and 11.7 dB for c = 1; intensities added instead
    (orthogonal polarisations) 22.7 dB. With a 50/50 beam splitter the other
    port carries G1 - c G2; the useful port gets ~75% of the light through
    both pinholes, so even after the split it is brighter than pinhole 1 alone.

    Bench options
    -------------
    For the default B + i C: two coherent beams, one tilted by lambda / 2p
    along x and one by lambda / 2p along y (1.68 deg at 632.8 nm), 90 deg
    apart in phase, and ONE on-axis pinhole: the 0.9-order square (1.582 mm
    side at f1 = 60 mm) rotated 45 deg into a diamond, 2.237 mm tip to tip. Each beam's B (resp. C) order then leaves on axis; their
    specular reflections, gap and window light land 1.76 mm off axis and
    are blocked. A grating in the illumination makes both beams through
    shared optics: the two tilts differ by lambda / (sqrt 2 p) along a
    diagonal (grating period sqrt 2 p at the PLM), with the pair steered so
    its midpoint sits at lambda / 4p in x and y. Either sign of each tilt
    works (the B and C bands exist at +-1/2p). The options below describe
    the S + D fold; the same apply per order.
    * Mach-Zehnder fold: pick pinhole 2 off behind the filter plane, steer it
      so it leaves L2 collinear with pinhole 1's beam (i.e. translate it onto
      the axis in the Fourier plane), recombine on a beam splitter, and hold
      the relative phase at its optimum (piezo mirror, dithered on the camera).
      combinePhase's 90 degrees refers to the model's pixel-centred carrier;
      the physical fold differs by a constant (0.20 rad was best in the wave
      check), so set the bench phase by maximising image quality.
    * Input side (common path): light the PLM with a second, coherent beam
      tilted by the D-order angle (lambda / 2p per axis, 1.68 deg at
      632.8 nm), e.g. the +1 order of a grating in the illumination. Its D
      order then leaves ON axis and adds to the first beam's S band in the
      single on-axis pinhole: the same S + c D, the same hologram, the same
      image (wave check 24.83 vs 24.84 dB, same light for a 50/50 split) --
      but no pick-off, periscope or output beam splitter, a relative phase
      that shared optics keep stable (slide the grating to set it), and one
      pinhole. The tilt must match the D order to ~lambda / array width
      (~50 urad); a grating imaged to a period of exactly 2p does that for
      any wavelength.
    * Common path: at an intermediate image of the PLM, a checkerboard phase
      plate (0 / theta, one cell per PLM pixel) multiplies the field by
      a + b (-1)^(x+y), which moves the D order to DC and vice versa; a
      second on-axis pinhole then passes a G1 + b demod(G2), i.e.
      c = -i tan(theta / 2): theta = 90 degrees gives |c| = 1 with no
      interferometric drift, at the price of pixel-registered alignment.
    """

    def __init__(self):
        super().__init__()
        self.combinePhase = np.pi / 2
        self.maxCandidates = 48
        self.secondApertureScale = 0.8
        self.foldOrders = None

    FOLD_CENTERS = {"S": (0.0, 0.0), "B": (0.5, 0.0), "C": (0.0, 0.5), "D": (0.5, 0.5)}
    FOLD_SIGNS = {"S": ((1, 1), (1, 1)), "B": ((1, -1), (1, -1)),
                  "C": ((1, 1), (-1, -1)), "D": ((1, -1), (-1, 1))}
    MODE_INDEX = {"S": 0, "B": 1, "C": 2, "D": 3}

    @property
    def c(self):
        return np.exp(1j * self.combinePhase)

    def pixelWeights(self):
        """The four per-pixel weights (p00, p01, p10, p11) the fold applies:
        sum_M c_M M = sum_k w_k p_k. Distinct weights -> distinct phasors."""
        H = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=np.float64)
        cvec = np.zeros(4, dtype=np.complex128)
        for f in self.foldOrders:
            cvec[self.MODE_INDEX[f["order"]]] += f["weight"]
        return H.T @ cvec / 4.0

    def combinedGamut(self):
        """sum over folded orders of weight * mode, for all 65536 configs."""
        modes = self.configTables()["modes"]
        return sum(f["weight"] * modes[:, self.MODE_INDEX[f["order"]]] for f in self.foldOrders)

    def _buildFold(self):
        hs, ws = self.usable_h // 2, self.usable_w // 2
        self._fold = []
        for f in self.foldOrders:
            W = self._pinholeWeight(self.FOLD_CENTERS[f["order"]], f["aperture"], f["shape"])
            car = np.tile(np.array(self.FOLD_SIGNS[f["order"]], dtype=np.float64), (hs, ws))
            self._fold.append((f["weight"], W, car))

    def combinedField(self, E):
        """Camera field for a padded pixel spectrum E: every folded order's band,
        demodulated to the axis, times its weight."""
        return sum(c * car * self._crop(_ifft2(E * W)) for c, W, car in self._fold)

    def _adjoint(self, R):
        """A^H R for A = sum_M c_M carrier_M A_M."""
        acc = 0.0
        for c, W, car in self._fold:
            acc = acc + np.conj(c) * W * _fft2(self._pad(car * R))
        return self._crop(_ifft2(acc))

    def _curvature4(self):
        """Same-superpixel block of A^H A in the S, B, C, D basis, from the
        response to each mode pattern at a central superpixel."""
        hs, ws = self.usable_h // 2, self.usable_w // 2
        v0, u0 = hs // 2, ws // 2
        V = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=np.float64)
        pos = [(0, 0), (0, 1), (1, 0), (1, 1)]
        Q = np.zeros((4, 4), dtype=np.complex128)
        for n in range(4):
            e = np.zeros((self.usable_h, self.usable_w), dtype=np.complex128)
            for k, (a, b) in enumerate(pos):
                e[2 * v0 + a, 2 * u0 + b] = V[n, k]
            out = self._adjoint(self.combinedField(_fft2(self._pad(e))))
            Q[:, n] = V @ np.array([out[2 * v0 + a, 2 * u0 + b] for (a, b) in pos]) / 4.0
        return Q

    def images(self, cfg):
        """(camera intensity per superpixel, light in the image per unit of
        illumination power when the beams carry powers ~ |weight|^2)."""
        _, E = self._pixelSpectrum(cfg)
        G = self.combinedField(E)
        power = sum(abs(c) ** 2 for c, _, _ in self._fold)
        return self._binIntensity(G), float(np.mean(np.abs(G) ** 2) / power)

    def _sweep(self, cfg, candidates, fixedPhase):
        hs, ws = cfg.shape
        st = np.repeat(np.repeat(np.sqrt(self.imTarget), 2, 0), 2, 1)
        e, E = self._pixelSpectrum(cfg)
        G = self.combinedField(E)
        g = np.sum(np.abs(G) * st) / max(np.sum(st ** 2), 1e-12)
        AhT = self._adjoint(g * st * (1.0 if fixedPhase else np.exp(1j * np.angle(G))))
        Qinv = np.linalg.inv(self._Q + 1e-6 * np.real(np.trace(self._Q)) * np.eye(4))

        flat = cfg.ravel().copy()
        changed = 0
        for k, sites in enumerate(self._groups):
            if k:
                e, E = self._pixelSpectrum(flat.reshape(hs, ws))
            r = self._modes(self._adjoint(self.combinedField(E)) - AhT)
            targets = self._modes(e) - r @ Qinv.T
            new = self._select(sites, candidates, targets, self._metric)
            changed += int(np.count_nonzero(new != flat[sites]))
            flat[sites] = new
        return flat.reshape(hs, ws), changed

    def createCGH(self, DeviceDictionary,
                  filename="", colorChannel="gray",
                  FlipUD=False, FlipLR=False, InvertTarget=False, preserveAspect=True,
                  orders=(("B", 1.0, 0.9, "diamond"), ("C", 1j, 0.9, "diamond")),
                  secondOrder="D", combinePhase=np.pi / 2, combineAmplitude=1.0,
                  amplitudeScale=0.95, apertureScale=0.8, secondApertureScale=0.8,
                  iters=12, fixedPhaseIters=3, maxCandidates=192,
                  updateSpacing=3, pixelFillFactor=1.0):
        """
        orders          list of (order, complex weight[, aperture[, shape]])
                        with order in 'S', 'B', 'C', 'D' and shape 'square',
                        'circle', 'diamond' or 'soft' (equal areas; see
                        pinholeShape); default B + i C (see class doc).
                        Apertures default to secondApertureScale ('S':
                        apertureScale). None -> the older two-order fold
                        S + combineAmplitude e^{i combinePhase} secondOrder
        amplitudeScale  request radius as a fraction of the combined gamut's
        apertureScale, secondApertureScale  pinhole widths (fractions of one order)
        iters           descent passes; fixedPhaseIters of them hold the image
                        phase flat before it is left free
        maxCandidates   configs per superpixel, nearest in the combined gamut
        pixelFillFactor mirror width / pitch for the aperture envelope (keep it:
                        the coherent sum depends on the D order's true
                        amplitude); None assumes point-like mirrors
        """
        secondOrder = str(secondOrder).upper()
        if secondOrder not in self.ORDERS:
            raise ValueError("secondOrder must be 'B', 'C' or 'D'.")
        self.alg = "RECOMBINED"
        self.secondOrder = secondOrder
        self.combinePhase = float(combinePhase)
        self.apertureScale = float(apertureScale)
        self.secondApertureScale = float(secondApertureScale)
        if orders is None:
            orders = [("S", 1.0, apertureScale),
                      (secondOrder, combineAmplitude * np.exp(1j * combinePhase), secondApertureScale)]
        self.foldOrders = []
        for o in orders:
            name = str(o[0]).upper()
            if name not in self.MODE_INDEX:
                raise ValueError("fold orders must be 'S', 'B', 'C' or 'D'.")
            aperture = float(o[2]) if len(o) > 2 else (self.apertureScale if name == "S"
                                                        else self.secondApertureScale)
            shape = o[3] if len(o) > 3 else "square"
            self.foldOrders.append({"order": name, "weight": complex(o[1]), "aperture": aperture,
                                    "shape": shape})
        self.maxCandidates = int(maxCandidates)
        self.pixelFillFactor = pixelFillFactor
        self.propDistance = 0.0
        self._setupDevice(DeviceDictionary)
        self._cfgTables = None
        self._buildPinholes()
        self._buildFold()
        self._groups = self._subLattice(int(updateSpacing))
        self.prepareTarget(filename, colorChannel, FlipUD, FlipLR, InvertTarget, preserveAspect)
        self.imTarget2 = None

        gamut = self.combinedGamut()
        self.amplitudeScale = float(amplitudeScale)
        A = self.amplitudeScale * float(np.abs(gamut).max()) * np.sqrt(np.clip(self.imTarget, 0.0, None))
        A = A / max(float(np.sqrt(self.imTarget.max())), 1e-12)
        pts = np.column_stack([gamut.real, gamut.imag])
        K = min(self.maxCandidates, gamut.size)
        try:
            from scipy.spatial import cKDTree
            _, ids = cKDTree(pts).query(np.column_stack([A.ravel(), np.zeros(A.size)]), k=K, workers=-1)
        except ImportError:
            ids = np.stack([np.argsort(np.abs(gamut - a))[:K] for a in A.ravel()])
        ids = ids.reshape(A.size, K).astype(np.int32)
        candidates = (ids, np.arange(gamut.size, dtype=np.int64)[:, None])
        print("fold %s: %d distinct phasors, radius %.3f, pixel weights %s"
              % (" + ".join("(%.2g%+.2gi)%s" % (f["weight"].real, f["weight"].imag, f["order"])
                            for f in self.foldOrders),
                 len(np.unique(np.round(gamut, 9))), np.abs(gamut).max(),
                 np.round(self.pixelWeights(), 3)))

        self._Q = self._curvature4()
        self._metric = self._selectionMetric(self._Q)
        cfg = ids[:, 0].reshape(A.shape).astype(np.int64)
        self.history = [("direct", self._psnr(self.images(cfg)[0], self.imTarget))]
        print("   KD-tree encoding onto the combined gamut: %.2f dB" % self.history[-1][1])
        for it in range(int(iters)):
            cfg, changed = self._sweep(cfg, candidates, it < fixedPhaseIters)
            self.history.append(("descent", self._psnr(self.images(cfg)[0], self.imTarget)))
            print("   pass %d: %.2f dB (%d changed)" % (it + 1, self.history[-1][1], changed))
            if not changed:
                break

        self.CGH_output_config = cfg
        self._storeResult(DeviceDictionary, self.configsToStates(cfg))

    def recoverImg(self, ShiftFOV=False, propMethod="FOURIER"):
        if self.imTarget is None:
            return
        I, self.imageEfficiency = self.images(self.CGH_output_config)
        self.psnr_disc = self._psnr(I, self.imTarget)
        print("Recombined four-phase (%s): %.2f dB; image holds %.0f%% of the illumination"
              % (" + ".join(f["order"] for f in self.foldOrders), self.psnr_disc,
                 100.0 * self.imageEfficiency))
        gain = np.mean(I * self.imTarget) / (np.mean(I ** 2) + 1e-12)
        self.imRecovered_disc = gain * I
        self.imRecovered_cont = self.imRecovered_disc
