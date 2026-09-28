import struct
import zlib
from pathlib import Path

import numpy as np
import cv2

try:
    import scipy.fft as _sfft

    def _fft2(x):
        return _sfft.fft2(x, axes=(-2, -1), workers=-1)

    def _ifft2(x):
        return _sfft.ifft2(x, axes=(-2, -1), workers=-1)
except ImportError:
    def _fft2(x):
        return np.fft.fft2(x, axes=(-2, -1))

    def _ifft2(x):
        return np.fft.ifft2(x, axes=(-2, -1))

from TIPLMSuite import CGHGenerator, DeviceLibrary

__all__ = ["FBXModel", "MultiViewHologramGenerator", "DeviceLibrary"]


class FBXModel:
    """
    Minimal reader for binary FBX 7.x files (Maya, Blender, 3ds Max exports):
    static mesh geometry with the node transforms, per-corner normals and UVs,
    and each material's diffuse texture (embedded in the file, or found next
    to it / in a sibling 'textures' folder). Skinning, animation and blend
    shapes are ignored -- the rest pose is used.

    Everything is converted to a canonical frame (x right, y up, z towards
    the viewer, from GlobalSettings' axis system) and triangulated (fans):
        positions (T, 3, 3), normals (T, 3, 3), uvs (T, 3, 2)
        triTexture (T,)  index into textures (-1: untextured)
        triAlbedo  (T,)  material diffuse luminance (used when untextured)
        textures         list of float32 luminance images in [0, 1]
    """

    def __init__(self, path, textureFile=None):
        self.path = Path(path)
        data = self.path.read_bytes()
        if not data.startswith(b"Kaydara FBX Binary  "):
            raise ValueError("%s is not a binary FBX file (re-export ASCII FBX as binary)" % path)
        self.version = struct.unpack_from("<I", data, 23)[0]
        self._data = data
        self._wide = self.version >= 7500
        self.nodes = self._parseAll()
        del self._data
        self._build(textureFile)

    # binary parsing --------------------------------------------------------
    def _prop(self, o):
        d = self._data
        t = chr(d[o])
        o += 1
        scalar = {"Y": ("<h", 2), "C": ("<?", 1), "I": ("<i", 4), "F": ("<f", 4),
                  "D": ("<d", 8), "L": ("<q", 8)}
        if t in scalar:
            fmt, n = scalar[t]
            return struct.unpack_from(fmt, d, o)[0], o + n
        if t in "SR":
            n = struct.unpack_from("<I", d, o)[0]
            raw = d[o + 4:o + 4 + n]
            return (raw.decode("utf-8", "replace") if t == "S" else raw), o + 4 + n
        if t in "fdlib":
            n, enc, clen = struct.unpack_from("<III", d, o)
            raw = d[o + 12:o + 12 + clen]
            if enc:
                raw = zlib.decompress(raw)
            dt = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "u1"}[t]
            return np.frombuffer(raw, dt, n), o + 12 + clen
        raise ValueError("FBX: unknown property type %r at byte %d" % (t, o - 1))

    def _node(self, o):
        fmt, hl = ("<QQQ", 24) if self._wide else ("<III", 12)
        end, nprop, _ = struct.unpack_from(fmt, self._data, o)
        o += hl
        if end == 0:
            return None, o
        nl = self._data[o]
        name = self._data[o + 1:o + 1 + nl].decode("ascii", "replace")
        o += 1 + nl
        props = []
        for _ in range(nprop):
            v, o = self._prop(o)
            props.append(v)
        kids = []
        while o < end:
            k, o = self._node(o)
            if k is None:
                break
            kids.append(k)
        return (name, props, kids), end

    def _parseAll(self):
        nodes, o = [], 27
        while o < len(self._data) - 32:
            n, o = self._node(o)
            if n is None:
                break
            nodes.append(n)
        return nodes

    @staticmethod
    def _child(node, name):
        for k in node[2]:
            if k[0] == name:
                return k
        return None

    @staticmethod
    def _props70(node):
        out = {}
        p70 = FBXModel._child(node, "Properties70")
        if p70 is not None:
            for p in p70[2]:
                if p[0] == "P" and len(p[1]) >= 4:
                    out[p[1][0]] = p[1][4:]
        return out

    # transforms -------------------------------------------------------------
    @staticmethod
    def _T(v):
        M = np.eye(4)
        M[:3, 3] = v
        return M

    @staticmethod
    def _S(v):
        return np.diag([v[0], v[1], v[2], 1.0])

    @staticmethod
    def _R(deg, order=0):
        """FBX Euler rotation; order enum 0..5 = XYZ XZY YZX YXZ ZXY ZYX (first
        letter applied first)."""
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        mats = {"X": np.array([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]]),
                "Y": np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]]),
                "Z": np.array([[c[2], -s[2], 0], [s[2], c[2], 0], [0, 0, 1]])}
        R = np.eye(3)
        for axis in ("XYZ", "XZY", "YZX", "YXZ", "ZXY", "ZYX")[int(order)]:
            R = mats[axis] @ R
        M = np.eye(4)
        M[:3, :3] = R
        return M

    def _localMatrix(self, model):
        P = self._props70(model)
        g = lambda k, d: np.array(P.get(k, d)[:3], dtype=np.float64)
        z, one = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        order = int(P.get("RotationOrder", (0,))[0])
        Rp, Sp = g("RotationPivot", z), g("ScalingPivot", z)
        return (self._T(g("Lcl Translation", z)) @ self._T(g("RotationOffset", z)) @ self._T(Rp)
                @ self._R(g("PreRotation", z)) @ self._R(g("Lcl Rotation", z), order)
                @ np.linalg.inv(self._R(g("PostRotation", z))) @ self._T(-Rp)
                @ self._T(g("ScalingOffset", z)) @ self._T(Sp) @ self._S(g("Lcl Scaling", one))
                @ self._T(-Sp))

    def _geometricMatrix(self, model):
        P = self._props70(model)
        g = lambda k, d: np.array(P.get(k, d)[:3], dtype=np.float64)
        return (self._T(g("GeometricTranslation", (0.0, 0.0, 0.0)))
                @ self._R(g("GeometricRotation", (0.0, 0.0, 0.0)))
                @ self._S(g("GeometricScaling", (1.0, 1.0, 1.0))))

    def _globalMatrix(self, mid):
        M = self._localMatrix(self.objects[mid])
        for parent, _ in self.parents.get(mid, []):
            if parent in self.objects and self.objects[parent][0] == "Model":
                return self._globalMatrix(parent) @ M
        return M

    # scene assembly -----------------------------------------------------------
    def _layer(self, geom, element, dataName, indexName, width, pvi, cp, poly):
        el = self._child(geom, element)
        if el is None:
            return None
        mapping = self._child(el, "MappingInformationType")[1][0]
        ref = self._child(el, "ReferenceInformationType")[1][0]
        vals = np.asarray(self._child(el, dataName)[1][0], dtype=np.float64).reshape(-1, width)
        base = {"ByPolygonVertex": np.arange(pvi.size), "ByVertice": cp, "ByVertex": cp,
                "ByControlPoint": cp, "ByPolygon": poly, "AllSame": np.zeros(pvi.size, int)}[mapping]
        if ref in ("IndexToDirect", "Index"):
            base = np.asarray(self._child(el, indexName)[1][0])[base]
        return vals[base]

    def _loadTexture(self, material, textureFile):
        if textureFile is not None:
            return self._readImage(Path(textureFile).read_bytes())
        for child, prop in self.children.get(material, []):
            if prop != "DiffuseColor" or self.objects.get(child, ("",))[0] != "Texture":
                continue
            names = []
            for vid, _ in self.children.get(child, []):
                video = self.objects.get(vid)
                if video is None or video[0] != "Video":
                    continue
                content = self._child(video, "Content")
                if content is not None and isinstance(content[1][0], bytes) and len(content[1][0]) > 0:
                    return self._readImage(content[1][0])
                for key in ("RelativeFilename", "Filename"):
                    n = self._child(video, key)
                    if n is not None:
                        names.append(n[1][0])
            for key in ("RelativeFilename", "FileName"):
                n = self._child(self.objects[child], key)
                if n is not None:
                    names.append(n[1][0])
            here = self.path.parent
            for name in names:
                base = Path(name.replace("\\", "/")).name
                for cand in (Path(name), here / name, here / base, here / "textures" / base,
                             here.parent / "textures" / base):
                    try:
                        if cand.is_file():
                            return self._readImage(cand.read_bytes())
                    except OSError:
                        pass
        return None

    @staticmethod
    def _readImage(raw):
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        img = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
        if img.ndim == 3:
            img = img[:, :, :3] @ np.array([0.114, 0.587, 0.299], np.float32)  # BGR luma
        return np.ascontiguousarray(img, dtype=np.float32)

    def _build(self, textureFile):
        top = {n[0]: n for n in self.nodes}
        self.objects = {n[1][0]: n for n in top["Objects"][2]}
        self.parents, self.children = {}, {}
        for c in top["Connections"][2]:
            if c[0] != "C":
                continue
            _, child, parent = c[1][:3]
            prop = c[1][3] if len(c[1]) > 3 else None
            self.parents.setdefault(child, []).append((parent, prop))
            self.children.setdefault(parent, []).append((child, prop))

        # axis system -> x right, y up, z towards the viewer
        gs = self._props70(top["GlobalSettings"]) if "GlobalSettings" in top else {}
        ax = lambda k, d: int(gs.get(k, (d,))[0])
        A = np.zeros((3, 3))
        A[0, ax("CoordAxis", 0)] = ax("CoordAxisSign", 1)
        A[1, ax("UpAxis", 1)] = ax("UpAxisSign", 1)
        A[2, ax("FrontAxis", 2)] = ax("FrontAxisSign", 1)

        pos, nrm, uvs, tex, alb = [], [], [], [], []
        self.textures, self.meshNames = [], []
        texCache = {}
        for gid, geom in self.objects.items():
            if geom[0] != "Geometry" or len(geom[1]) < 3 or geom[1][2] != "Mesh":
                continue
            models = [p for p, _ in self.parents.get(gid, []) if self.objects.get(p, ("",))[0] == "Model"]
            verts = np.asarray(self._child(geom, "Vertices")[1][0], dtype=np.float64).reshape(-1, 3)
            pvi = np.asarray(self._child(geom, "PolygonVertexIndex")[1][0], dtype=np.int64)
            ends = pvi < 0
            cp = np.where(ends, ~pvi, pvi)
            last = np.nonzero(ends)[0]
            first = np.r_[0, last[:-1] + 1]
            poly = np.repeat(np.arange(last.size), last - first + 1)
            ntri = np.maximum(last - first - 1, 0)
            tpoly = np.repeat(np.arange(last.size), ntri)
            k = np.arange(tpoly.size) - np.repeat(np.cumsum(ntri) - ntri, ntri) + 1
            corners = np.stack([first[tpoly], first[tpoly] + k, first[tpoly] + k + 1], axis=1)

            N = self._layer(geom, "LayerElementNormal", "Normals", "NormalsIndex", 3, pvi, cp, poly)
            UV = self._layer(geom, "LayerElementUV", "UV", "UVIndex", 2, pvi, cp, poly)
            matOfCorner = np.zeros(pvi.size, dtype=np.int64)
            el = self._child(geom, "LayerElementMaterial")
            if el is not None and self._child(el, "Materials") is not None:
                mids = np.asarray(self._child(el, "Materials")[1][0], dtype=np.int64)
                mapping = self._child(el, "MappingInformationType")[1][0]
                if mapping == "ByPolygon" and mids.size == last.size:
                    matOfCorner = mids[poly]
                elif mids.size:
                    matOfCorner[:] = mids[0]

            for mid in models or [None]:
                model = self.objects[mid] if mid is not None else None
                if model is not None and float(self._props70(model).get("Visibility", (1.0,))[0]) == 0.0:
                    continue
                M = (self._globalMatrix(mid) @ self._geometricMatrix(model)) if model is not None else np.eye(4)
                P = (verts @ M[:3, :3].T + M[:3, 3]) @ A.T
                Nm = np.linalg.inv(M[:3, :3]).T
                materials = [c for c, _ in self.children.get(mid, [])
                             if self.objects.get(c, ("",))[0] == "Material"] if mid is not None else []
                texIds, albedos = [], []
                for m in materials:
                    if m not in texCache:
                        img = self._loadTexture(m, textureFile)
                        if img is not None:
                            self.textures.append(img)
                            texCache[m] = len(self.textures) - 1
                        else:
                            texCache[m] = -1
                    texIds.append(texCache[m])
                    mp = self._props70(self.objects[m])
                    col = np.array(mp.get("DiffuseColor", mp.get("Diffuse", (0.8, 0.8, 0.8)))[:3], float)
                    albedos.append(float(col @ np.array([0.299, 0.587, 0.114])))
                if not materials:
                    texIds, albedos = [-1], [0.8]
                pos.append(P[cp[corners]])
                if N is not None:
                    n = (N @ Nm.T) @ A.T
                    nrm.append(n[corners])
                else:
                    nrm.append(None)
                uvs.append(UV[corners] if UV is not None else np.zeros(corners.shape + (2,)))
                mi = np.clip(matOfCorner[corners[:, 0]], 0, len(texIds) - 1)
                tex.append(np.asarray(texIds)[mi])
                alb.append(np.asarray(albedos)[mi])
                self.meshNames.append(model[1][1].split("\x00")[0] if model is not None else "mesh")

        if not pos:
            raise ValueError("No visible mesh geometry in %s" % self.path)
        self.positions = np.concatenate(pos)
        faces = [np.repeat(self._faceNormals(p)[:, None, :], 3, axis=1) if n is None else n
                 for p, n in zip(pos, nrm)]
        n = np.concatenate(faces)
        self.normals = n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-12)
        self.uvs = np.concatenate(uvs)
        self.triTexture = np.concatenate(tex).astype(np.int64)
        self.triAlbedo = np.concatenate(alb).astype(np.float64)

    @staticmethod
    def _faceNormals(P):
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def summary(self):
        lo, hi = self.positions.reshape(-1, 3).min(0), self.positions.reshape(-1, 3).max(0)
        return ("%s: %d triangles in %s, %d texture(s), extent %s"
                % (self.path.name, len(self.positions), ", ".join(self.meshNames),
                   len(self.textures), np.round(hi - lo, 3)))


class MultiViewHologramGenerator(CGHGenerator):
    """
    Full-parallax 3D hologram (a holographic stereogram) of a textured FBX
    model on the 0.67 TI PLM, spread over the PLM's ENTIRE diffraction cone.

    How much angle the PLM allows
    -----------------------------
    Mirrors of pitch p send distinct light only into one diffraction zone,
    |sin theta| < lambda / 2p per axis: 3.36 x 3.36 deg full at 632.8 nm.
    Every other zone is a copy of it, so nothing beyond that is new. The
    four-phase encoders pass a quarter of the zone (one superpixel order, half
    the band per axis) to get complex-amplitude control. This generator
    uses the whole zone, phase only: it is cut into views x views sub-pupils
    in the Fourier plane of L1, and the light through sub-pupil (a, b) is
    made to form the object as seen from that sub-pupil's direction. An
    aperture one sub-pupil wide sees one view; moving it across the Fourier
    plane (or a camera moving sideways at a distance behind the image) walks
    around the object. Views sit lambda / (p Nx) apart.

    The space-bandwidth product is fixed: the views share the 1358 x 800
    mirrors, so each view resolves (1358 / Nx) x (800 / Ny) points. 4f
    optics trade image size for angle (etendue): at lateral magnification
    m = f2 / f1 the image is m x (14.7 x 8.6 mm) and the cone 3.36 deg / m.

    3D geometry
    -----------
    Views are rendered as oblique orthographic projections onto the image
    plane -- what a camera focused on that plane, looking through one
    sub-pupil, records. The model is placed with its depth centre on the
    image plane (depthOffset moves it), at true proportions for the chosen
    magnification: a point at depth z in the image appears shifted by
    z * tan(theta_view), with theta_view = asin(lambda f) / m. angleScale > 1
    exaggerates the rotation between views (a multi-view display rather than
    a true-scale 3D image).

    The light budget (why some sub-pupils are blocked)
    --------------------------------------------------
    The camera plane is conjugate to the PLM, and a phase-only PLM is
    uniformly bright: at every point the light summed over ALL directions is
    fixed. A background that is dark in every view is only possible if its
    light leaves in directions nobody looks along. So a few sub-pupils are
    blocked on the bench and serve as the dump -- by default the four
    corners, which the mirror envelope dims most anyway (the four-phase
    paper's filter plays the same role with 3/4 of the zone). With all 25
    sub-pupils used as views the surplus lands in the views as noise
    (13 dB); with the corners blocked 21 views reach ~22 dB and the
    background is dark over the whole frame, so no field stop is needed.
    About 60% of the light goes to the dump.

    Optimisation
    ------------
    The mirror phases are optimised so that the amplitude seen through every
    sub-pupil matches its view, sampled `oversample` x finer than the view's
    resolution (so the camera sees smooth views, not sample-point matches
    with dark gaps between them). The model is exact for the band-limited
    4f: FFT of the mirror field, times the square-mirror envelope
    sinc(fill f p) per axis, cut into sub-pupil blocks, each block inverse-
    FFT'd onto its view grid (a Collins wave simulation with physical
    pinholes agrees to 49-52 dB, 30 dB for views at the zone edge). Stages:
    Adam on continuous phase; Adam with a rising pull onto the device's 16
    LUT phases (plain rounding onto the 632.8 nm LUT, which has a 76 deg gap,
    doubles the error: 21.1 vs 21.6 dB); a discrete projected-gradient polish.

    Measured (macdonald cat, 632.8 nm theoretical LUT, magnification 0.1,
    PSNR per view with camera pixel = view pixel; 2x finer sampling ~2 dB
    lower):
        3 x 3 grid, 5 views, 453 x 267 points      23.0 dB  (80 iterations)
        4 x 4, 12 views, 340 x 200                 21.7 dB  (80 iterations)
        5 x 5, 21 views, 272 x 160 (default)       21.8 dB  (20.1 .. 22.8)
        7 x 7, 45 views, 194 x 115                 20.4 dB  (80 iterations)
        5 x 5 with no blocked sub-pupils           13.1 dB  (80 iterations, even with
                                                            an elliptical field stop)
    Window reflections (0.5% faces) add a pedestal to the centre view only
    (the zero order sits in it); block that sub-pupil too if it matters.
    """

    def __init__(self):
        super().__init__()
        self.alg = "MULTIVIEW3D"

    # device / views ---------------------------------------------------------
    def _setupDevice(self, DeviceDictionary):
        self.H, self.W = int(DeviceDictionary["h"]), int(DeviceDictionary["w"])
        self.usable_h, self.usable_w = self.H, self.W
        self.pitchW = float(DeviceDictionary["pitchW"])
        self.pitchH = float(DeviceDictionary["pitchH"])
        self.lambda_m = float(DeviceDictionary["lambda_m"])
        self.pLevels = np.mod(np.asarray(DeviceDictionary["pLevel"], dtype=np.float64)[:DeviceDictionary["nLevel"]], 1.0)
        self.levelPhases = 2 * np.pi * self.pLevels
        # nearest-level lookup on a fine phase grid (the LUT may be irregular)
        grid = (np.arange(8192) + 0.5) / 8192
        d = np.abs(np.remainder(grid[:, None] - self.pLevels[None, :] + 0.5, 1.0) - 0.5)
        self._qTable = np.argmin(d, axis=1)

    def _quantize(self, phi):
        idx = (np.mod(phi, 2 * np.pi) * (8192 / (2 * np.pi))).astype(np.int64) % 8192
        return self._qTable[idx]

    def _setupViews(self, views, oversample, mirrorFill):
        Nx, Ny = (views, views) if np.isscalar(views) else views
        H, W = self.H, self.W
        ex = np.round(np.linspace(-W / 2, W / 2, Nx + 1)).astype(np.int64)
        ey = np.round(np.linspace(-H / 2, H / 2, Ny + 1)).astype(np.int64)
        self.Nx, self.Ny, self.Nv = int(Nx), int(Ny), int(Nx * Ny)
        self.oversample = int(oversample)
        self.Mx = self.oversample * int(np.max(np.diff(ex)))
        self.My = self.oversample * int(np.max(np.diff(ey)))
        full, local, freq = [], [], []
        for b in range(Ny):
            for a in range(Nx):
                v = b * Nx + a
                ky, kx = np.arange(ey[b], ey[b + 1]), np.arange(ex[a], ex[a + 1])
                KY, KX = np.meshgrid(ky, kx, indexing="ij")
                cy, cx = (ey[b] + ey[b + 1]) // 2, (ex[a] + ex[a + 1]) // 2
                full.append(((KY % H) * W + (KX % W)).ravel())
                local.append((v * self.My * self.Mx + ((KY - cy) % self.My) * self.Mx
                              + (KX - cx) % self.Mx).ravel())
                # block centre in cycles per pixel, and full widths
                freq.append(((ex[a] + ex[a + 1] - 1) / (2 * W), (ey[b] + ey[b + 1] - 1) / (2 * H),
                             (ex[a + 1] - ex[a]) / W, (ey[b + 1] - ey[b]) / H))
        self._full = np.concatenate(full)
        self._local = np.concatenate(local)
        assert np.unique(self._full).size == H * W  # the sub-pupils tile the zone
        self.viewFreq = np.array(freq)
        fy = np.fft.fftfreq(H)[:, None]
        fx = np.fft.fftfreq(W)[None, :]
        a = 1.0 if mirrorFill is None else float(mirrorFill)
        self.mirrorFill = a
        self.env = (np.sinc(a * fy) * np.sinc(a * fx)).astype(np.float32)
        self._c = self.My * self.Mx / (H * W)

    def views(self, u):
        """Complex view fields (Nv, My, Mx) of the mirror field u (H, W)."""
        U = (_fft2(u.astype(np.complex64)) * self.env).ravel()
        X = np.zeros(self.Nv * self.My * self.Mx, dtype=np.complex64)
        X[self._local] = U[self._full]
        return _ifft2(X.reshape(self.Nv, self.My, self.Mx)) * np.float32(self._c)

    def _adjoint(self, r):
        R = _fft2(r.astype(np.complex64)).ravel()
        G = np.empty(self.H * self.W, dtype=np.complex64)
        G[self._full] = R[self._local]
        return _ifft2(G.reshape(self.H, self.W) * self.env)

    def viewDirections(self):
        """Image-space view directions (tan theta_x, tan theta_y) per view."""
        s = self.lambda_m * self.viewFreq[:, :2] / self.pitchW
        return np.tan(np.arcsin(np.clip(s, -1, 1))) * self.angleScale / self.magnification

    # scene ------------------------------------------------------------------
    def loadModel(self, filename, textureFile=None):
        self.model = FBXModel(filename, textureFile)
        print(self.model.summary())
        return self.model

    def _placeScene(self, yaw, pitch, objectHeight, depthOffset, FlipLR, FlipUD, light):
        m = self.model
        P = m.positions.reshape(-1, 3)
        c = 0.5 * (P.min(0) + P.max(0))
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
        R = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        P = (P - c) @ R.T
        Nn = m.normals.reshape(-1, 3) @ R.T
        L = np.asarray(light, dtype=np.float64)
        # model (x right, y up, z to viewer) -> array (column, row, depth), in pixels
        flip = np.array([-1.0 if FlipLR else 1.0, 1.0 if FlipUD else -1.0, 1.0])
        P, Nn, L = P * flip, Nn * flip, L * flip
        ext = P.max(0) - P.min(0)
        s = min(objectHeight * self.H / ext[1], 0.92 * self.W / ext[0])
        P = P * s
        P -= 0.5 * (P.min(0) + P.max(0))
        P += np.array([(self.W - 1) / 2.0, (self.H - 1) / 2.0, depthOffset * objectHeight * self.H])
        self.scene = {"P": P.reshape(-1, 3, 3), "N": Nn.reshape(-1, 3, 3), "light": L / np.linalg.norm(L),
                      "scale": s, "depthRange": (P[:, 2].min(), P[:, 2].max())}

    def renderView(self, tan_dir, ambient=0.35, supersample=3):
        """Oblique orthographic render onto the image plane, sampled on the
        view grid (My x Mx over the full frame). Returns (luminance, coverage)."""
        S = self.scene
        P = S["P"]
        sx, sy = self.W / self.Mx, self.H / self.My           # pixels per view sample
        col = (P[..., 0] - P[..., 2] * tan_dir[0]) / sx
        row = (P[..., 1] - P[..., 2] * tan_dir[1]) / sy
        ss = int(supersample)
        Hs, Ws = self.My * ss, self.Mx * ss
        X = (col + 0.5) * ss - 0.5
        Y = (row + 0.5) * ss - 0.5
        Z = P[..., 2]
        zbuf = np.full((Hs, Ws), -np.inf)
        tbuf = np.full((Hs, Ws), -1, dtype=np.int64)
        bbuf = np.zeros((Hs, Ws, 3))
        for t in range(len(P)):
            x0, x1, x2 = X[t]
            y0, y1, y2 = Y[t]
            area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            if abs(area) < 1e-12:
                continue
            c0, c1 = max(int(np.ceil(min(x0, x1, x2))), 0), min(int(np.floor(max(x0, x1, x2))), Ws - 1)
            r0, r1 = max(int(np.ceil(min(y0, y1, y2))), 0), min(int(np.floor(max(y0, y1, y2))), Hs - 1)
            if c0 > c1 or r0 > r1:
                continue
            xx, yy = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
            w0 = ((x1 - xx) * (y2 - yy) - (x2 - xx) * (y1 - yy)) / area
            w1 = ((x2 - xx) * (y0 - yy) - (x0 - xx) * (y2 - yy)) / area
            w2 = 1.0 - w0 - w1
            zz = w0 * Z[t, 0] + w1 * Z[t, 1] + w2 * Z[t, 2]
            zb = zbuf[r0:r1 + 1, c0:c1 + 1]
            upd = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9) & (zz > zb)
            if upd.any():
                zb[upd] = zz[upd]
                tbuf[r0:r1 + 1, c0:c1 + 1][upd] = t
                bbuf[r0:r1 + 1, c0:c1 + 1][upd] = np.stack([w0[upd], w1[upd], w2[upd]], axis=-1)
        lum = np.zeros((Hs, Ws))
        hit = tbuf >= 0
        t, b = tbuf[hit], bbuf[hit]
        n = np.einsum("ki,kij->kj", b, S["N"][t])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        view = np.array([tan_dir[0], tan_dir[1], 1.0])
        n *= np.where(n @ view < 0, -1.0, 1.0)[:, None]      # two-sided
        shade = ambient + (1.0 - ambient) * np.clip(n @ S["light"], 0.0, None)
        m = self.model
        albedo = m.triAlbedo[t].copy()
        tex = m.triTexture[t]
        for k, img in enumerate(m.textures):
            sel = tex == k
            if not np.any(sel):
                continue
            uv = np.einsum("ki,kij->kj", b[sel], m.uvs[t[sel]])
            th, tw = img.shape
            x = np.mod(uv[:, 0], 1.0) * tw - 0.5          # bilinear, wrapping
            y = (1.0 - np.mod(uv[:, 1], 1.0)) * th - 0.5
            x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
            fx, fy = x - x0, y - y0
            x0, x1 = x0 % tw, (x0 + 1) % tw
            y0, y1 = y0 % th, (y0 + 1) % th
            albedo[sel] = ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy)
                           + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)
        lum[hit] = albedo * shade
        lum = lum.reshape(self.My, ss, self.Mx, ss).mean(axis=(1, 3))
        cov = hit.reshape(self.My, ss, self.Mx, ss).mean(axis=(1, 3))
        return lum, cov

    def _renderTargets(self, ambient, blur):
        dirs = self.viewDirections()
        T, C = [], []
        for v in range(self.Nv):
            lum, cov = self.renderView(dirs[v], ambient)
            T.append(lum)
            C.append(cov)
        T = np.array(T)
        if blur > 0:  # soften to what the view band can carry
            sig = blur * self.oversample
            T = np.array([cv2.GaussianBlur(t, (0, 0), sig) for t in T])
        T /= max(T.max(), 1e-12)
        self.viewTargets = T.astype(np.float32)
        self.viewCoverage = np.array(C, dtype=np.float32)

    def _buildWindow(self, window, margin):
        union = self.viewCoverage.max(axis=0) > 0.02
        rows, cols = np.nonzero(union)
        my, mx = margin * self.My, margin * self.Mx * self.H / self.W
        if window == "silhouette":
            r = max(int(round(margin * self.My)), 1)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            w = cv2.dilate(union.astype(np.uint8), k) > 0
        elif window == "box":
            yy, xx = np.mgrid[:self.My, :self.Mx]
            w = ((yy >= rows.min() - my) & (yy <= rows.max() + my)
                 & (xx >= cols.min() - mx) & (xx <= cols.max() + mx))
        elif window == "ellipse":
            yc, xc = 0.5 * (rows.min() + rows.max()), 0.5 * (cols.min() + cols.max())
            ay, ax = 0.5 * (rows.max() - rows.min()) + 1, 0.5 * (cols.max() - cols.min()) + 1
            k = np.sqrt(np.max(((rows - yc) / ay) ** 2 + ((cols - xc) / ax) ** 2))
            yy, xx = np.mgrid[:self.My, :self.Mx]
            w = ((yy - yc) / (k * ay + my)) ** 2 + ((xx - xc) / (k * ax + mx)) ** 2 <= 1.0
        elif window == "full":
            w = np.ones((self.My, self.Mx), bool)
        else:
            raise ValueError("window must be 'ellipse', 'box', 'silhouette' or 'full'")
        self.windowMask = w.astype(np.float32)

    # optimisation ---------------------------------------------------------------
    def _loss(self, v):
        a = np.abs(v)
        d = a - self._sA
        wd = self._lossWeight * d
        L = float(np.sum(wd * d))
        r = (wd / np.maximum(a, 1e-12)) * v
        return L, r

    def _initialPhase(self, init, rng):
        if init == "random":
            return rng.uniform(0, 2 * np.pi, (self.H, self.W))
        # target amplitudes with one random constant phase per view, back-projected
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, (self.Nv, 1, 1))).astype(np.complex64)
        g = self._adjoint(self._sA * ph)
        return np.angle(g).astype(np.float64)

    def _levelOffset(self, phi):
        """Signed distance (radians) from each phase to its nearest LUT level."""
        return np.angle(np.exp(1j * (phi - self.levelPhases[self._quantize(phi)])))

    def _adam(self, phi, iters, lr, anneal, log, label):
        """Adam on the continuous phase. With anneal = (mu0, mu1) a penalty
        mu * sum(distance to the nearest LUT level)^2, mu rising geometrically
        from mu0 to mu1, walks the phases onto the device's 16 levels while
        the views keep being fitted (the 632.8 nm LUT has a 76 deg gap, so
        plain rounding doubles the error)."""
        m = np.zeros_like(phi)
        s = np.zeros_like(phi)
        b1, b2 = 0.9, 0.999
        pscale = self._L0 / phi.size
        for k in range(1, iters + 1):
            u = np.exp(1j * phi).astype(np.complex64)
            L, r = self._loss(self.views(u))
            g = 2.0 * np.imag(self._adjoint(r) * np.conj(u)).astype(np.float64)
            if anneal is not None:
                mu = anneal[0] * (anneal[1] / anneal[0]) ** (k / iters)
                g += 2.0 * mu * pscale * self._levelOffset(phi)
            m = b1 * m + (1 - b1) * g
            s = b2 * s + (1 - b2) * g * g
            step = lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * k / iters)))  # cosine decay
            sh = np.sqrt(s / (1 - b2 ** k))
            # epsilon relative to the typical gradient: mirrors that barely
            # matter (deep in the noise region) take small steps, not full ones
            eps = self.adamEps * float(np.sqrt(np.mean(sh * sh))) + 1e-30
            phi = phi - step * (m / (1 - b1 ** k)) / (sh + eps)
            if log and (k == 1 or k % log == 0 or k == iters):
                print("  %-10s iter %4d  loss %.4e" % (label, k, L / self._L0), flush=True)
        return phi

    def _polish(self, state, iters, fraction, rng, log):
        """Discrete projected-gradient steps on the LUT levels: each mirror takes
        the level nearest (u - g / kappa), on a random subset per step; a step
        that does not lower the loss is undone and the subset shrinks."""
        u = np.exp(1j * self.levelPhases[state]).astype(np.complex64)
        L, r = self._loss(self.views(u))
        kappa = 2.0 * self._c * float(np.mean(self.env.astype(np.float64) ** 2))
        for k in range(1, iters + 1):
            g = self._adjoint(r)
            cand = self._quantize(np.angle(u - g / kappa))
            pick = (rng.random(state.shape) < fraction) & (cand != state)
            trial = np.where(pick, cand, state)
            ut = np.exp(1j * self.levelPhases[trial]).astype(np.complex64)
            Lt, rt = self._loss(self.views(ut))
            if Lt < L:
                state, u, L, r = trial, ut, Lt, rt
            else:
                fraction *= 0.5
            if log and (k == 1 or k % log == 0 or k == iters):
                print("  %-10s iter %4d  loss %.4e  (subset %.3f)" % ("polish", k, L / self._L0, fraction),
                      flush=True)
            if fraction < 1e-3:
                break
        return state

    def _blockedList(self, blocked):
        if blocked is None or (isinstance(blocked, str) and blocked.lower() == "none"):
            return []
        if isinstance(blocked, str) and blocked.lower() == "corners":
            return sorted({(0, 0), (self.Nx - 1, 0), (0, self.Ny - 1), (self.Nx - 1, self.Ny - 1)})
        return [tuple(int(i) for i in ab) for ab in blocked]

    def createCGH(self, DeviceDictionary, filename, views=(5, 5), blocked="corners", magnification=0.5,
                  angleScale=1.0, objectHeight=0.8, depthOffset=0.0, yaw=0.0, pitch=0.0, FlipLR=True,
                  FlipUD=False, light=(-0.45, 0.55, 0.7), ambient=0.35, textureFile=None,
                  window="full", windowMargin=0.04, fill=2.0, viewFalloff=0.0, viewBalance=0.5,
                  mirrorFill=1.0, oversample=2, targetBlur=0.5, init="views", iters=(200, 60, 20), lr=0.3,
                  anneal=(0.1, 100.0), adamEps=0.0, seed=0, log=25):
        """
        Build the multi-view hologram of an FBX model.

        views           (Nx, Ny) sub-pupils tiling the diffraction zone (or an int)
        blocked         sub-pupils used as the light dump (blocked on the bench):
                        'corners' (default), None, or a list of (a, b) indices
        magnification   4f lateral magnification f2 / f1 of the bench; sets the
                        true-scale rotation between views (cone = 3.36 deg / m)
        angleScale      1 = true 3D at that magnification; > 1 exaggerates the
                        rotation (multi-view display)
        objectHeight    model height as a fraction of the frame height
        depthOffset     depth centre relative to the image plane, in object heights
                        (+ = towards the viewer)
        yaw, pitch      turn the model (degrees) before placing it
        window          region whose darkness/brightness is enforced ('full',
                        'ellipse', 'box', 'silhouette'); outside it anything goes
                        (then block it with a field stop at the image plane)
        fill            the views' share of the local light budget where the object
                        is brightest, before optimisation (the rest goes to the
                        blocked sub-pupils); ~2 is best -- the optimum overshoots
                        because bright pixels borrow light from darker neighbours
        viewFalloff     0 = all views equally bright; 1 = views dim like the
                        mirror envelope towards the zone edges
        viewBalance     weight view k's error by (1 / its envelope^2)^viewBalance
        mirrorFill      square-mirror width / pitch (sets the sinc envelope)
        iters           (continuous Adam, Adam annealed onto the LUT levels,
                        discrete polish) steps
        anneal          (start, end) weight of the pull onto the LUT levels
        """
        self._setupDevice(DeviceDictionary)
        self.magnification = float(abs(magnification))
        self.angleScale = float(angleScale)
        self.flips = (bool(FlipLR), bool(FlipUD))
        self._setupViews(views, oversample, mirrorFill)
        self.viewActive = np.ones(self.Nv, dtype=bool)
        for a, b in self._blockedList(blocked):
            self.viewActive[b * self.Nx + a] = False
        if isinstance(filename, FBXModel):
            self.model = filename
        else:
            self.loadModel(filename, textureFile)
        self._placeScene(yaw, pitch, objectHeight, depthOffset, FlipLR, FlipUD, light)
        self._renderTargets(ambient, targetBlur)
        self.viewTargets *= self.viewActive[:, None, None]
        self.viewCoverage *= self.viewActive[:, None, None]
        self._buildWindow(window, windowMargin)

        # Light budget. The camera plane is conjugate to the PLM, and a
        # phase-only PLM is uniformly bright, so at every point the light
        # summed over ALL sub-pupils is fixed: sum_k I_k / E_k^2 = 1 (E_k the
        # mirror envelope). A dark background in every view is therefore only
        # possible if that light leaves through sub-pupils nobody looks
        # through -- the blocked ones. Without them the surplus lands in the
        # views as noise (13 vs 16-19 dB here). The views are scaled to use
        # `fill` of the budget where the object is brightest.
        e2 = (self.env.astype(np.float64) ** 2).ravel()
        view_of = self._local // (self.My * self.Mx)
        blockE2 = np.array([np.mean(e2[self._full[view_of == v]]) for v in range(self.Nv)])
        self.blockE2 = blockE2
        self.viewWeight = (blockE2 / blockE2.max()) ** float(viewFalloff) * self.viewActive
        wT = self.viewWeight[:, None, None] * self.viewTargets
        budget = np.sum(wT / blockE2[:, None, None], axis=0)
        s2 = float(fill) / float(np.max(budget * (self.windowMask > 0)))
        self._sA = (np.sqrt(s2 * wT) * self.windowMask).astype(np.float32)
        # Error in view k is weighted by (1 / its envelope^2)^viewBalance, so
        # stray light costs the same in every sub-pupil; unweighted, the
        # optimiser parks it where the envelope hides it best (the corners).
        lam = (blockE2.max() / blockE2) ** float(viewBalance) * self.viewActive
        self._lossWeight = (lam[:, None, None] * self.windowMask[None]).astype(np.float32)
        self.adamEps = float(adamEps)
        self._L0 = float(np.sum(self._lossWeight * self._sA ** 2))
        dirs = np.degrees(np.arctan(self.viewDirections()[self.viewActive]))
        print("%d x %d sub-pupils (%d views, %d blocked), %d x %d points per view; view directions "
              "%.2f .. %.2f deg (x), %.2f .. %.2f deg (y) at magnification %.3g%s"
              % (self.Nx, self.Ny, self.viewActive.sum(), self.Nv - self.viewActive.sum(),
                 self.Mx // self.oversample, self.My // self.oversample,
                 dirs[:, 0].min(), dirs[:, 0].max(), dirs[:, 1].min(), dirs[:, 1].max(),
                 self.magnification, "" if self.angleScale == 1 else " (x%.3g exaggerated)" % self.angleScale),
              flush=True)

        rng = np.random.default_rng(seed)
        phi = self._initialPhase(init, rng)
        n1, n2, n3 = iters
        if n1:
            phi = self._adam(phi, n1, lr, None, log, "continuous")
        if n2:
            phi = self._adam(phi, n2, 0.5 * lr, anneal, log, "to levels")
        state = self._quantize(phi)
        if n3:
            state = self._polish(state, n3, 0.25, rng, max(log // 5, 1) if log else 0)
        self._storeResult(DeviceDictionary, state)

    def _storeResult(self, DeviceDictionary, state):
        state = np.asarray(state, dtype=np.int64)
        phase = self.levelPhases[state]
        self.CGH_output_state_disc = state.astype(np.float64)
        self.CGH_output_disc = self.CGH_output_state_disc
        self.CGH_output_phase_disc = phase
        self.CGH_output_cont = phase.copy()
        self.CGH_phase = phase
        self.CGH_mapped = self.deviceLibary.formatPLM(DeviceDictionary, state.astype(np.float64))
        self.recoverImg()

    # evaluation -------------------------------------------------------------------
    def viewIntensities(self, state=None):
        state = self.CGH_output_state_disc if state is None else state
        u = np.exp(1j * self.levelPhases[np.asarray(state, dtype=np.int64)])
        return np.abs(self.views(u)) ** 2

    def binned(self, I):
        o = self.oversample
        return I.reshape(I.shape[:-2] + (self.My // o, o, self.Mx // o, o)).mean(axis=(-3, -1))

    @staticmethod
    def _psnrMasked(I, T, w):
        gain = np.sum(w * I * T) / max(np.sum(w * I * I), 1e-30)
        mse = np.sum(w * (gain * I - T) ** 2) / max(np.sum(w), 1)
        return 10 * np.log10(1.0 / max(mse, 1e-20)), gain

    def recoverImg(self, ShiftFOV=False, propMethod="FOURIER"):
        I = self.viewIntensities()
        wB = self.binned(self.windowMask) > 0.999
        Tb, Ib = self.binned(self.viewTargets), self.binned(I)
        ps, pf, gains = np.full(self.Nv, np.nan), np.full(self.Nv, np.nan), np.full(self.Nv, np.nan)
        for v in np.nonzero(self.viewActive)[0]:
            ps[v], gains[v] = self._psnrMasked(Ib[v], Tb[v], wB)
            pf[v] = self._psnrMasked(I[v], self.viewTargets[v], self.windowMask)[0]
        self.viewPSNR = ps.reshape(self.Ny, self.Nx)          # camera pixel = view pixel
        self.viewPSNRfine = pf.reshape(self.Ny, self.Nx)      # camera sampling 2x finer
        # brightness of each view relative to the brightest (one camera exposure)
        self.viewBrightness = (np.nanmin(gains) / gains).reshape(self.Ny, self.Nx)
        act = self.viewActive
        self.dumpFraction = float(I[~act].sum() / I.sum())
        self.windowEfficiency = float(np.sum(I[act] * self.windowMask) / I[act].sum())
        self.imRecovered_views = I
        self.psnr_disc = float(np.nanmean(self.viewPSNR))
        print("3D hologram: PSNR per view %.1f dB mean (%.1f .. %.1f), %.1f dB at 2x finer camera "
              "sampling; dimmest view %.2f x brightest; %.0f%% of the light leaves through the "
              "blocked sub-pupils"
              % (self.psnr_disc, np.nanmin(self.viewPSNR), np.nanmax(self.viewPSNR),
                 np.nanmean(self.viewPSNRfine), np.nanmin(self.viewBrightness), 100 * self.dumpFraction))

    # bench geometry ------------------------------------------------------------------
    def viewGeometry(self, f1, f2=None):
        """Sub-pupil centres and sizes in the Fourier plane of L1 (focal length
        f1, metres) and the image-space view directions (degrees)."""
        f2 = self.magnification * f1 if f2 is None else f2
        p, lam = self.pitchW, self.lambda_m
        out = []
        for v in range(self.Nv):
            fx, fy, wx, wy = self.viewFreq[v]
            out.append({"view": (v % self.Nx, v // self.Nx), "blocked": not self.viewActive[v],
                        "center": (lam * f1 * fx / p, lam * f1 * fy / p),
                        "size": (lam * f1 * wx / p, lam * f1 * wy / p),
                        "angle_deg": tuple(np.degrees(np.arcsin(lam * np.array([fx, fy]) / p)) * f1 / f2)})
        zone = (lam * f1 / p, lam * f1 / p)
        return {"views": out, "zoneAperture": zone,
                "imageSize": (self.W * p * f2 / f1, self.H * p * f2 / f1),
                "cone_deg": tuple(2 * np.degrees(np.arcsin(lam / (2 * p))) * f1 / f2 * np.ones(2))}

    # pictures ---------------------------------------------------------------------------
    def _crop(self, margin=0.06):
        """The object's region (all views) plus a margin, inside the window."""
        cov = (self.viewCoverage.max(axis=0) > 0.02) & (self.windowMask > 0)
        rows, cols = np.nonzero(cov)
        my, mx = int(margin * self.My), int(margin * self.Mx * self.H / self.W)
        return (slice(max(rows.min() - my, 0), min(rows.max() + my + 1, self.My)),
                slice(max(cols.min() - mx, 0), min(cols.max() + mx + 1, self.Mx)))

    def _display(self, v):
        """Sub-pupil v -> (column, row) in previews: FlipLR / FlipUD mirror the
        scene to suit the EVM, so previews mirror it back (as the 2D previews
        do), which also mirrors the viewpoint order."""
        a, b = v % self.Nx, v // self.Nx
        return (self.Nx - 1 - a if self.flips[0] else a), (self.Ny - 1 - b if self.flips[1] else b)

    def viewFrames(self, I=None, fine=False, height=240):
        """Display frames (uint8) of every sub-pupil, cropped around the object,
        with one common exposure over the views (so view-to-view brightness
        shows); blocked sub-pupils come out as flat grey."""
        I = self.imRecovered_views if I is None else I
        img = I if fine else np.repeat(np.repeat(self.binned(I), self.oversample, -2), self.oversample, -1)
        rs, cs = self._crop()
        img = img[:, rs, cs] * self.windowMask[rs, cs]
        img = img / max(np.percentile(img[self.viewActive], 99.7), 1e-30)
        img[~self.viewActive] = 0.15
        if self.flips[0]:
            img = img[:, :, ::-1]
        if self.flips[1]:
            img = img[:, ::-1, :]
        hh = height
        ww = int(round(hh * (img.shape[2] * self.W / self.Mx) / (img.shape[1] * self.H / self.My)))
        return [(np.clip(cv2.resize(f, (ww, hh), interpolation=cv2.INTER_AREA), 0, 1) ** (1 / 2.2) * 255)
                .astype(np.uint8) for f in img]

    def saveViewSheet(self, path, height=150):
        """Targets (left) and simulated views (right), laid out as the viewpoints
        (grey: blocked sub-pupils)."""
        T = self.viewFrames(self.viewTargets, fine=True, height=height)
        R = self.viewFrames(height=height)
        h, w = T[0].shape
        pad = 4
        sheet = np.full((self.Ny * (h + pad) + pad, 2 * self.Nx * (w + pad) + 3 * pad), 40, np.uint8)
        for v in range(self.Nv):
            a, b = self._display(v)
            y = pad + b * (h + pad)
            sheet[y:y + h, pad + a * (w + pad):pad + a * (w + pad) + w] = T[v]
            x = 2 * pad + self.Nx * (w + pad) + a * (w + pad)
            sheet[y:y + h, x:x + w] = R[v]
        cv2.imwrite(str(path), sheet)
        return sheet

    def saveViewSweep(self, path, height=320, fps=8):
        """Animated GIF: the viewpoint walking a serpentine over the views."""
        from PIL import Image
        frames = self.viewFrames(height=height)
        at = {self._display(v): v for v in range(self.Nv) if self.viewActive[v]}
        order = []
        for b in range(self.Ny):
            row = [at[(a, b)] for a in range(self.Nx) if (a, b) in at]
            order += row if b % 2 == 0 else row[::-1]
        order += order[::-1][1:-1]
        imgs = [Image.fromarray(frames[v]) for v in order]
        imgs[0].save(str(path), save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
