import numpy as np

try:
    import scipy.fft as _fft

    def _fft2c(x):
        return _fft.fftshift(_fft.fft2(_fft.ifftshift(x), workers=-1))
except ImportError:
    def _fft2c(x):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

__all__ = ["CurvedBeamLayout", "CurvedBeamSimulator", "plm067WindowReflections"]

# Converging / diverging illumination for the four-phase PLM 4f.
#
# Measured with CurvedBeamSimulator (632.8 nm, f1 = 60 mm, f2 = 50 mm, PLM at
# L1's front focus, 9 targets: harper, dlp_logo, bear1, bear2, testimag,
# Siemens star, bar chart, gradient + circles, text; paper encoding at 0.6
# order, plus the dual-order and recombined holograms):
#
# * Wave model vs the encoders' FFT model, collimated: within 0.03-0.17 dB
#   for every target (dual harper/logo 23.80/26.38 vs 23.72/26.42).
# * Illumination wavevectors measured on the PLM meet at exactly D; traced
#   through L1 they cross the predicted filter plane within 1e-13 m. The zero
#   and D orders focus exactly there (1% peak left at +-0.3 mm), the D order
#   at (lambda f1 / 2p, lambda f1 / 2p) as predicted.
# * PSNR with the pinholes at the new filter plane, relative to collimated:
#   0.000 .. -0.01 dB (paper, all 9 targets), <= 0.05 dB (dual, recombined)
#   for |D| = 100-200 mm -- the residue is mirror obliquity; 5-7.5 deg
#   incidence at the array corners costs <= 0.03 dB. No PSNR is gained.
# * Pinholes left at the collimated plane (off by f1^2 / D):
#     paper hologram: +0.3 .. -1 dB up to 3.6 mm off (a slightly defocused
#       pinhole acts as a soft-edged one), -6 .. -19 dB at 7.2 mm;
#     dual-order: image 1 / image 2 -0.05 / -0.4 dB at 0.1 mm, -0.4 / -2.5 at
#       0.5 mm, -1.25 / -4.3 at 1 mm, -2.8 / -6.7 at 1.8 mm (|D| = 2 m).
# * Window reflections (ICD: faces 0.70 and ~1.43 mm air-equivalent in front
#   of the mirrors) focus 2t farther than the PLM's source image, so under a
#   curved beam they reach the camera out of focus and interfere as
#   Newton-ring fringes, instead of the uniform offset of collimated light.
#   Harper, image 1, mean over window phases:
#       0.1% AR faces: -0.05 (collimated), -0.15 (|D| 1 m), -1.1 (0.5 m), -2.4 (0.2 m)
#       0.5% AR faces: -0.33, -0.72, -3.6, -6.6 dB;  4% bare: -1.4 vs -13 dB
#   Pinhole 2 (D order) receives none of it.
# * PLM not at L1's front focus: the spectrum zooms (d1 = 30 mm: D = +90 mm
#   -> x0.75, D = -150 mm -> x1.25, camera 70.8 mm behind L2); with pinholes
#   scaled to match PSNR is unchanged (-0.01 dB), fixed pinholes simply see
#   a different effective aperture (0.6 -> 0.8 / 0.48).
# Verdict: no gain; keep the beam collimated (the window fringes and the
# placement sensitivity are the costs), unless the zoom is worth them.


def _prop(d):
    return np.array([[1.0, d], [0.0, 1.0]])


def _lens(f):
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def plm067WindowReflections(reflectance=0.005, index=1.51, phases=(0.0, 0.0)):
    """
    The two window surfaces of the 0.67" PLM package (ICD 2519338, section
    A-A: 1.100 mm window, its inner face 0.703 mm above the active array) as
    reflectors in front of the mirror plane, in air-equivalent distance:
    inner face 0.703 mm, outer face 0.703 + 1.100 / n. `reflectance` is the
    intensity reflectance of each face relative to the mirrors (~0.005 AR
    coated, ~0.04 bare glass); `phases` are the unknown optical-path phases.
    """
    return [
        {"distance": 0.703e-3, "reflectance": reflectance, "phase": phases[0]},
        {"distance": 0.703e-3 + 1.100e-3 / index, "reflectance": reflectance, "phase": phases[1]},
    ]


class CurvedBeamLayout:
    """
    Paraxial layout of PLM -> L1 -> filter plane -> L2 -> camera when the PLM
    is lit by a spherical wave instead of a plane wave.

    Geometry (unfolded about the PLM, so the reflected beam travels +z):
        PLM at z = 0, L1 at d1, L2 at d1 + lensSeparation.
        sourceDistance D: the reflected illumination converges to a point D
        in front of the PLM (D > 0), or diverges from a point |D| behind it
        (D < 0); D = inf is the collimated case.

    The filter plane is the plane where L1 images that point -- the only
    plane where the PLM's spectrum is focused. From the ray matrix
    [[A1, B1], [C1, D1]] of PLM -> filter plane, the field there is
        U_F(xi) = exp(i pi D1 xi^2 / (lambda B1)) * FT[u](xi / (lambda B1))
    because the illumination phase -pi x^2/(lambda D) cancels the kernel's
    pi A1 x^2/(lambda B1) exactly when A1/B1 = 1/D. So
        * spectral scale:  xi = lambda * B1 * (spatial frequency)
        * with the PLM at L1's front focus (d1 = f1): B1 = f1 for ANY D, i.e.
          pinholes keep their transverse positions and sizes and only move
          axially, to s = f1 (1 - f1 / D) behind L1;
        * with d1 != f1: B1 = d1 + s (1 - d1 / f1) depends on D, a zoom.
    The camera plane (PLM image) depends only on the lens positions. Because
    the whole system images (B_total = 0), the second half undoes the
    quadratic phase of the first, and the camera intensity is |u * h|^2 with
    h set by the pinhole in frequency units -- identical to collimated
    illumination. CurvedBeamSimulator checks that with waves.
    """

    def __init__(self, f1, f2, sourceDistance=np.inf, d1=None, lensSeparation=None,
                 wavelength=632.8e-9, pitch=10.8e-6, arrayShape=(800, 1358),
                 lensClearAperture=22.9e-3):
        self.f1, self.f2 = float(f1), float(f2)
        self.D = float(sourceDistance)
        self.d1 = self.f1 if d1 is None else float(d1)
        self.L12 = self.f1 + self.f2 if lensSeparation is None else float(lensSeparation)
        self.lam = float(wavelength)
        self.p = float(pitch)
        self.arrayShape = arrayShape
        self.clearAperture = float(lensClearAperture)

        if np.isinf(self.D):
            self.filterDistance = self.f1
        else:
            s_o = self.d1 - self.D  # source-point distance in front of L1
            if abs(s_o - self.f1) < 1e-12:
                raise ValueError("The source point sits at L1's focus: no spectrum plane "
                                 "(the beam leaves L1 collimated).")
            self.filterDistance = self.f1 * s_o / (s_o - self.f1)
        if self.filterDistance <= 0 or self.filterDistance >= self.L12:
            raise ValueError("The spectrum focuses %.1f mm from L1, outside the L1-L2 gap: "
                             "choose another sourceDistance or d1." % (self.filterDistance * 1e3))
        self.cameraDistance = self._imagePlane()

    # ray matrices ------------------------------------------------------
    def stage1(self, offset=0.0):
        """PLM -> filter plane (+ offset along z)."""
        return _prop(self.filterDistance + offset) @ _lens(self.f1) @ _prop(self.d1)

    def stage2(self, offset=0.0):
        """Filter plane (+ offset) -> camera."""
        return _prop(self.cameraDistance) @ _lens(self.f2) @ _prop(self.L12 - self.filterDistance - offset)

    def _imagePlane(self):
        M = _lens(self.f2) @ _prop(self.L12) @ _lens(self.f1) @ _prop(self.d1)
        # camera at z behind L2 with B_total = M[0,1] + z M[1,1] = 0
        return -M[0, 1] / M[1, 1]

    @property
    def spectralScale(self):
        return self.stage1()[0, 1]

    @property
    def magnification(self):
        return (self.stage2() @ self.stage1())[0, 0]

    def pinholes(self, apertureScale=1.0, secondApertureScale=None, secondOrder="D"):
        """Pinhole centres and full widths (metres) in the filter plane."""
        B = self.spectralScale
        order = {"B": (0.5, 0.0), "C": (0.0, 0.5), "D": (0.5, 0.5)}[secondOrder]
        out = {"pinhole1": {"center": (0.0, 0.0), "width": apertureScale * self.lam * B / (2 * self.p)}}
        if secondApertureScale is not None:
            out["pinhole2"] = {"center": (self.lam * B * order[0] / self.p, self.lam * B * order[1] / self.p),
                               "width": secondApertureScale * self.lam * B / (2 * self.p)}
        return out

    def report(self, apertureScale=1.0, secondApertureScale=None, secondOrder="D", windows=None):
        """Everything that changes with the beam curvature, as a dict (metres)."""
        h, w = self.arrayShape
        hx, hy = w * self.p / 2, h * self.p / 2
        r = np.hypot(hx, hy)
        rep = {
            "filterDistanceAfterL1": self.filterDistance,
            "filterShiftFromCollimated": self.filterDistance - self.f1,
            "spectralScale": self.spectralScale,
            "zoomVsCollimated": self.spectralScale / self.f1,
            "cameraDistanceAfterL2": self.cameraDistance,
            "magnification": self.magnification,
            "pinholes": self.pinholes(apertureScale, secondApertureScale, secondOrder),
        }
        # mirror obliquity: phase depth scales with cos(incidence)
        theta = 0.0 if np.isinf(self.D) else np.arctan(r / abs(self.D))
        rep["maxIncidenceDeg"] = np.degrees(theta)
        rep["maxPhaseDepthError"] = 1.0 - np.cos(theta)
        # beam footprints (half-diagonal, geometric, incl. the D order's spread)
        spread = self.lam / (2 * self.p) * np.sqrt(2)
        slope = 0.0 if np.isinf(self.D) else -1.0 / self.D
        foot = {}
        for name, M in (("L1", _prop(self.d1)),
                        ("L2", _prop(self.L12) @ _lens(self.f1) @ _prop(self.d1))):
            edge = abs(M[0, 0] * r + M[0, 1] * r * slope)
            foot[name] = 2 * (edge + abs(M[0, 1]) * spread)
        rep["beamFootprint"] = foot
        rep["vignetting"] = {k: v > self.clearAperture for k, v in foot.items()}
        # window reflections: their source is 2t farther, so they focus off the
        # filter plane; half-size of their (defocused) spot there
        if windows:
            A, B = self.stage1()[0]
            spots = []
            for wref in windows:
                Dw = np.inf if np.isinf(self.D) else self.D + 2 * wref["distance"]
                slope_w = 0.0 if np.isinf(Dw) else -1.0 / Dw
                spots.append((abs(A * hx + B * hx * slope_w), abs(A * hy + B * hy * slope_w)))
            rep["windowSpotHalfSize"] = spots
        return rep


class CurvedBeamSimulator:
    """
    Wave-optics check of a finished four-phase hologram under curved
    illumination, with nothing assumed about where things focus.

    Field on the PLM (sampled `oversample` x per pixel, so each mirror is a
    flat square and the mirror-aperture envelope is included):
        E = rect(array) * exp(i phi_state * cos(theta)) * exp(-i pi r^2 / (lambda D))
    -- the spherical illumination is carried explicitly (its local wavevector
    k_perp = -k r / D points at the source point; rayCheck() measures it),
    and with obliquity=True each piston's phase depth is scaled by the cosine
    of the local incidence angle. Window reflections (optional) are added as
    unmodulated spherical waves converging 2t farther.

    Propagation: two Collins (ray-matrix Fresnel) integrals, each done as
    chirp * FFT * chirp -- PLM -> any plane near the filter plane, pinholes
    applied in metres there, then -> the camera. This is exact within the
    paraxial model with ideal thin lenses of unlimited aperture (lens
    aberrations and clipping are not modelled; report() flags footprints).
    The camera image is mapped back through the magnification (the 4f
    inverts) and integrated over each superpixel footprint.

    Device realism options (all off by default):
        mirrorFill      square mirrors of width mirrorFill * pitch. On the
                        K-sample grid a mirror is a K-sample block, whose
                        spectrum is sin(pi t) / (K sin(pi t / K)) (t = cycles
                        per pixel; cos(pi t / 2) at K = 2) rather than a real
                        mirror's sinc(a t) -- 0.71 vs 0.64 per axis at the D
                        order. This swaps in the continuous shape exactly
                        (spectral correction, |t| < 0.9; nothing beyond
                        reaches a pinhole). 1.0 = gapless square mirrors.
        gapReflectance, gapPhase  amplitude reflectance / phase of the area
                        between mirrors (relative to a mirror). The gap grid
                        has the pixel period, so it adds only a uniform DC
                        field (1 - a^2) r e^{i psi}: pinhole 1 sees it,
                        pinhole 2 cannot.
        lutScaleError   played phase depth = (1 + s) x the LUT the hologram
                        was designed with (wavelength / bias mismatch)
        lutPhaseErrors  16 per-state errors (waves) added on top
        beamRadius      Gaussian beam, 1/e^2 intensity radius (metres);
                        None = flat top over the array
    """

    def __init__(self, generator, layout, oversample=2, pad=1.25, obliquity=True, windows=None,
                 mirrorFill=None, gapReflectance=0.0, gapPhase=0.0,
                 lutScaleError=0.0, lutPhaseErrors=None, beamRadius=None):
        self.G = generator
        self.L = layout
        self.K = int(oversample)
        self.obliquity = bool(obliquity)
        self.windows = windows or []
        self.mirrorFill = mirrorFill
        self.gapField = (1.0 - (mirrorFill or 1.0) ** 2) * gapReflectance * np.exp(1j * gapPhase)
        self.beamRadius = beamRadius
        H, W = generator.usable_h, generator.usable_w
        self.H, self.W = H, W
        self.dx = generator.pitchW / self.K
        self.N = (self._fastEven(pad * H * self.K), self._fastEven(pad * W * self.K))
        Ny, Nx = self.N
        self.oy, self.ox = (Ny - H * self.K) // 2, (Nx - W * self.K) // 2
        self.y = (np.arange(Ny) - Ny // 2) * self.dx
        self.x = (np.arange(Nx) - Nx // 2) * self.dx
        states = np.asarray(generator.CGH_output_state_disc[:H, :W], dtype=np.int64)
        levels = np.asarray(generator.pLevels, dtype=np.float64) * (1.0 + lutScaleError)
        if lutPhaseErrors is not None:
            levels = levels + np.asarray(lutPhaseErrors, dtype=np.float64)
        phase = 2 * np.pi * levels[states]
        self._phase = np.repeat(np.repeat(phase, self.K, 0), self.K, 1)

    def _mirrorShape(self, U):
        """Replace the K-sample block pixel by a continuous square mirror of
        width mirrorFill * pitch (area-weighted, so gaps lose light)."""
        a, K = float(self.mirrorFill), self.K
        corr = []
        for n in U.shape:
            t = np.fft.fftfreq(n, d=self.dx) * self.G.pitchW
            box = np.ones_like(t)
            nz = np.abs(t) > 1e-12
            box[nz] = np.sin(np.pi * t[nz]) / (K * np.sin(np.pi * t[nz] / K))
            c = np.where(np.abs(t) < 0.9, a * np.sinc(a * t) / np.where(np.abs(t) < 0.9, box, 1.0), 0.0)
            corr.append(c)
        return np.fft.ifft2(np.fft.fft2(U) * corr[0][:, None] * corr[1][None, :])

    @staticmethod
    def _fastEven(n):
        n = int(np.ceil(n))
        while True:
            m, k = n, n
            for f in (2, 3, 5):
                while m % f == 0:
                    m //= f
            if m == 1 and k % 4 == 0:
                return k
            n += 1

    # illumination --------------------------------------------------------
    def illuminationPhase(self, x, y, D=None):
        D = self.L.D if D is None else D
        if np.isinf(D):
            return np.zeros(np.broadcast(x, y).shape)
        return -np.pi * (x ** 2 + y ** 2) / (self.L.lam * D)

    def rayCheck(self, samples=2000, seed=0):
        """
        Measure the illumination's local wavevectors on the PLM (finite
        differences of the field itself) and intersect the rays: returns the
        distance at which they meet (should equal sourceDistance) and, traced
        through L1 with the ray matrix, their spread in the filter plane.
        """
        rng = np.random.default_rng(seed)
        hx, hy = self.W * self.L.p / 2, self.H * self.L.p / 2
        x = rng.uniform(-hx, hx, samples)
        y = rng.uniform(-hy, hy, samples)
        hstep = 1e-8
        E = lambda xx, yy: np.exp(1j * self.illuminationPhase(xx, yy))
        kx = np.angle(E(x + hstep, y) * np.conj(E(x - hstep, y))) / (2 * hstep)
        ky = np.angle(E(x, y + hstep) * np.conj(E(x, y - hstep))) / (2 * hstep)
        k = 2 * np.pi / self.L.lam
        tx, ty = kx / k, ky / k  # ray slopes
        if np.allclose(tx, 0) and np.allclose(ty, 0):
            meet = np.inf
        else:
            # z where x + z tx = 0 and y + z ty = 0 (least squares per ray)
            meet = -(x * tx + y * ty) / (tx ** 2 + ty ** 2)
        A, B = self.L.stage1()[0]
        at_filter = np.hypot(A * x + B * tx, A * y + B * ty)
        return {"convergesAt": meet, "filterPlaneSpread": at_filter,
                "kind": "collimated" if np.isinf(self.L.D) else
                        ("converging" if self.L.D > 0 else "diverging")}

    # propagation ---------------------------------------------------------
    def _collins(self, U, ygrid, xgrid, M, dx_in):
        """Collins integral on a centred grid: returns (field, ygrid, xgrid)."""
        A, B, _, D = M.ravel()
        lam = self.L.lam
        cy = np.exp(1j * np.pi * A * ygrid ** 2 / (lam * B))
        cx = np.exp(1j * np.pi * A * xgrid ** 2 / (lam * B))
        V = _fft2c(U * cy[:, None] * cx[None, :])
        Ny, Nx = U.shape
        yo = (np.arange(Ny) - Ny // 2) * lam * B / (Ny * dx_in[0])
        xo = (np.arange(Nx) - Nx // 2) * lam * B / (Nx * dx_in[1])
        V *= np.exp(1j * np.pi * D * yo ** 2 / (lam * B))[:, None]
        V *= np.exp(1j * np.pi * D * xo ** 2 / (lam * B))[None, :]
        return V, yo, xo

    def plmField(self, withHologram=True, pattern=None):
        """Complex field leaving the PLM (padded grid), illumination included."""
        Ny, Nx = self.N
        Y, X = self.y[self.oy:self.oy + self.H * self.K], self.x[self.ox:self.ox + self.W * self.K]
        illum = self.illuminationPhase(X[None, :], Y[:, None])
        if pattern is not None:
            phi = pattern
        elif withHologram:
            phi = self._phase
        else:
            phi = 0.0
        if self.obliquity and not np.isinf(self.L.D):
            phi = phi / np.sqrt(1.0 + (X[None, :] ** 2 + Y[:, None] ** 2) / self.L.D ** 2)
        U = np.zeros((Ny, Nx), dtype=np.complex128)
        rows = slice(self.oy, self.oy + self.H * self.K)
        cols = slice(self.ox, self.ox + self.W * self.K)
        U[rows, cols] = np.exp(1j * phi)
        if self.mirrorFill is not None:
            U = self._mirrorShape(U)
            U[rows, cols] += self.gapField
        U[rows, cols] *= np.exp(1j * illum)
        for w in self.windows:
            Dw = np.inf if np.isinf(self.L.D) else self.L.D + 2 * w["distance"]
            U[rows, cols] += (np.sqrt(w["reflectance"]) * np.exp(1j * (w["phase"] + self.illuminationPhase(
                X[None, :], Y[:, None], Dw))))
        if self.beamRadius is not None:
            U[rows, cols] *= np.exp(-(X[None, :] ** 2 + Y[:, None] ** 2) / self.beamRadius ** 2)
        return U

    def filterPlane(self, offset=0.0, U=None, **kw):
        """Field in the plane `offset` metres behind the predicted filter plane."""
        U = self.plmField(**kw) if U is None else U
        return self._collins(U, self.y, self.x, self.L.stage1(offset), (self.dx, self.dx))

    def cameraField(self, Vf, yf, xf, offset=0.0):
        """Field in the filter plane -> complex camera field on the PLM's
        sample grid (the 4f inversion undone)."""
        dxf = (yf[1] - yf[0], xf[1] - xf[0])
        Vc, _, _ = self._collins(Vf, yf, xf, self.L.stage2(offset), dxf)
        if self.L.magnification < 0:  # camera sample n <-> object sample N - n
            Vc = np.roll(np.roll(Vc[::-1, ::-1], 1, 0), 1, 1)
        return Vc[self.oy:self.oy + self.H * self.K, self.ox:self.ox + self.W * self.K]

    def binned(self, field):
        """Camera intensity integrated over each superpixel footprint."""
        n = 2 * self.K
        return (np.abs(field) ** 2).reshape(self.H // 2, n, self.W // 2, n).mean(axis=(1, 3))

    def pinholeMask(self, yf, xf, center, width, shape="square"):
        """Pinhole in metres; non-square shapes as the generator defines them
        (same area as the square of this width)."""
        dy = yf[:, None] - center[1]
        dx = xf[None, :] - center[0]
        if shape == "square" or not hasattr(self.G, "pinholeShape"):
            return ((np.abs(dy) <= width / 2) & (np.abs(dx) <= width / 2)).astype(np.float64)
        return self.G.pinholeShape(dx, dy, width / 2, shape)

    def defaultPinholes(self):
        """The generator's apertures at the layout's predicted positions."""
        G = self.G
        second = None
        if getattr(G, "imTarget2", None) is not None or getattr(G, "alg", "") == "RECOMBINED":
            second = G.secondApertureScale
        return self.L.pinholes(G.apertureScale, second, getattr(G, "secondOrder", "D"))

    def images(self, offset=0.0, pinholes=None):
        """Camera images through each pinhole, the pinholes placed `offset`
        behind the predicted filter plane."""
        pinholes = self.defaultPinholes() if pinholes is None else pinholes
        Vf, yf, xf = self.filterPlane(offset)
        return {name: self.binned(self.cameraField(
                    Vf * self.pinholeMask(yf, xf, ph["center"], ph["width"]), yf, xf, offset))
                for name, ph in pinholes.items()}

    def foldedFields(self, offset=0.0):
        """
        RecombinedFourPhaseCGHGenerator on the bench: each folded order's band
        is cut out by a pinhole of its aperture and translated onto the axis
        in the filter plane -- equivalently, one illumination beam per order,
        tilted so that order leaves on axis. Returns (camera fields, weights).
        """
        G = self.G
        Vf, yf, xf = self.filterPlane(offset)
        M = self.L.stage1(offset)
        B1, D1 = M[0, 1], M[1, 1]
        lam, p = self.L.lam, self.L.p
        fields, weights = [], []
        for f in G.foldOrders:
            cx, cy = G.FOLD_CENTERS[f["order"]]
            center = (lam * B1 * cx / p, lam * B1 * cy / p)
            V = Vf * self.pinholeMask(yf, xf, center, f["aperture"] * lam * B1 / (2 * p),
                                      f.get("shape", "square"))
            sy = int(round(center[1] / (yf[1] - yf[0])))
            sx = int(round(center[0] / (xf[1] - xf[0])))
            V = np.roll(np.roll(V, -sy, 0), -sx, 1)
            if abs(D1) > 1e-12 and (sx or sy):  # keep the band's spectral phase
                ey, ex = sy * (yf[1] - yf[0]), sx * (xf[1] - xf[0])
                V *= np.exp(1j * np.pi * D1 * (yf ** 2 - (yf + ey) ** 2) / (lam * B1))[:, None]
                V *= np.exp(1j * np.pi * D1 * (xf ** 2 - (xf + ex) ** 2) / (lam * B1))[None, :]
            fields.append(self.cameraField(V, yf, xf, offset))
            weights.append(f["weight"])
        return fields, weights

    def recombined(self, offset=0.0, phases=32, rounds=2):
        """Folded image with the relative beam phases tuned (coordinate search),
        as the bench's phase adjusters would be. The model's pixel-centred
        carriers differ from the physical tilts by constants, which this
        absorbs. Returns (PSNR, phases, image)."""
        fields, weights = self.foldedFields(offset)

        def score(th):
            I = self.binned(sum(w * np.exp(1j * t) * F for w, t, F in zip(weights, th, fields)))
            return self.G._psnr(I, self.G.imTarget), I

        theta = np.zeros(len(fields))
        best = score(theta)
        for _ in range(rounds):
            for m in range(1, len(fields)):
                for a in np.arange(phases) * 2 * np.pi / phases:
                    trial = theta.copy()
                    trial[m] = a
                    s = score(trial)
                    if s[0] > best[0]:
                        best, theta = s, trial
        return best[0], theta, best[1]

    def psnr(self, offset=0.0, pinholes=None):
        """PSNR per camera image; for a recombined hologram the folded sum."""
        if getattr(self.G, "alg", "") == "RECOMBINED":
            s, a, I = self.recombined(offset)
            return {"recombined": s, "foldPhases": a}, {"recombined": I}
        imgs = self.images(offset, pinholes)
        out = {"pinhole1": self.G._psnr(imgs["pinhole1"], self.G.imTarget)}
        if "pinhole2" in imgs:
            out["pinhole2"] = self.G._psnr(imgs["pinhole2"], self.G.imTarget2)
        return out, imgs

    def focusScan(self, offsets, order="S"):
        """
        Where does each order focus? Illuminates a pure grating for the order
        (flat mirror for 'S', the (-1)^(x+y) checkerboard for 'D') and scans
        the plane: returns (offsets, peak intensity, peak position in metres).
        """
        pattern = None
        if order == "D":
            yy, xx = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
            check = np.pi * ((xx + yy) % 2)
            pattern = np.repeat(np.repeat(check, self.K, 0), self.K, 1)
        U = self.plmField(withHologram=False, pattern=pattern)
        peaks, where = [], []
        for off in offsets:
            V, yf, xf = self.filterPlane(off, U=U)
            I = np.abs(V) ** 2
            if order == "D":
                sel = (yf[:, None] > 0) & (xf[None, :] > 0)
                I = np.where(sel, I, 0.0)
            j = np.unravel_index(np.argmax(I), I.shape)
            peaks.append(I[j])
            where.append((xf[j[1]], yf[j[0]]))
        return np.asarray(offsets), np.asarray(peaks), np.asarray(where)
