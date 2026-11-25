"""Face swap engine wrapping the InsightFace inswapper_128 ONNX model.

Optionally enhances results with GFPGAN if installed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .coverage import (
    DEFAULT_COVERAGE,
    coverage_alpha,
    face_roi,
    normalize_coverage,
    paste_swapped_face,
    template_128,
)
from .face_analyzer import Face
from .occlusion import _occ_mask
from .pose import face_yaw, repair_landmarks
from .quality import RESTORE_MIN_PX, face_span_px
from .providers import (
    active_providers_from_sessions,
    provider_attempts,
    run_with_cuda_fallback,
    run_with_provider_fallback,
    uses_gpu,
)
from .utils import ensure_inswapper, logger


class FaceSwapper:
    """Swap a target face in a frame with a source face's identity.

    ``execution="auto"`` registers TensorRT, then CUDA, then CPU. A failed
    TensorRT session is retried without it. ``use_gpu=False`` forces CPU.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        enhance: bool = False,
        execution: str = "auto",
    ) -> None:
        self._enhance_requested = bool(enhance)
        self.object_mask = True
        self.precise_edges = False
        self.allow_restore = bool(enhance)
        self.pixel_boost = 128
        self.large_face_px = 0
        self.boost_budget_ms = 40.0
        self.identity_strength = 1.0
        self._precomputed = None
        self._gpen = None
        self.restorer = "gfpgan"
        self.codeformer_fidelity = 0.5
        self._codeformer = None
        self.color_memory: dict[int, np.ndarray] = {}
        self._alpha_memory: dict[int, np.ndarray] = {}
        self._mask_cache: dict = {}
        self._mask_tick = 0
        self._matrix_memory: dict[int, np.ndarray] = {}
        self.last_canvas = None
        self.last_matrix = None
        self.stage_ms: dict[str, float] = {}
        self.enhance_every = 1
        self._enhance_tick = 0
        self.keep_mouth = False
        self.mask_blur = 0
        self.mask_padding = 0
        self._restore_cache = None
        self._color_key: Optional[int] = None
        self._occlusion = None
        self._load(use_gpu=use_gpu, execution=execution, enhance=enhance)

    def _load(self, use_gpu: bool, execution: str, enhance: bool) -> None:
        import insightface  # local import

        model_path = ensure_inswapper()
        mode = "cpu" if not use_gpu else execution
        attempts = provider_attempts(mode)

        def _load(providers):
            model = insightface.model_zoo.get_model(str(model_path), providers=providers)
            session = getattr(model, "session", None)
            active = active_providers_from_sessions([session] if session is not None else [])
            if uses_gpu(providers) and active == ["CPUExecutionProvider"]:
                raise RuntimeError(
                    "InSwapper fell back to CPU. TensorRT or CUDA libraries "
                    "are probably missing from PATH."
                )
            return model, active

        loaded, providers = run_with_provider_fallback(
            attempts, _load, what="InSwapper (inswapper_128)"
        )
        self.swapper, self.active_providers = loaded
        self.providers = providers
        self.execution = mode
        self._on_cpu = not uses_gpu(providers)
        session = getattr(self.swapper, "session", None)
        shapes = []
        live = list(self.active_providers or [])
        if session is not None:
            try:
                live = list(session.get_providers())
                shapes = [(item.name, list(item.shape)) for item in session.get_inputs()]
            except Exception:
                shapes = []
        logger.info(
            "InSwapper session providers=%s inputs=%s (stock batch-1; dynamic batch-4 was slower and is not built)",
            live,
            shapes,
        )
        self._enhancer = None
        if enhance:
            self._enhancer = _try_load_gfpgan(use_gpu=uses_gpu(providers))

    def adopt_cpu(self) -> None:
        """Drop a GPU session that failed while the graph was running."""
        if self._on_cpu:
            return
        self._load(use_gpu=False, execution="cpu", enhance=self._enhance_requested)

    def adopt_execution(self, execution: str) -> None:
        """Reload on DirectML or CPU. InSwapper weights stay on disk."""
        if execution == "cpu":
            self.adopt_cpu()
            return
        self._load(use_gpu=True, execution=execution, enhance=self._enhance_requested)

    def set_enhance(self, enhance: bool) -> Optional[str]:
        """Toggle the selected restorer without reloading InSwapper."""
        return self.prepare_restorer(enhance, getattr(self, "restorer", "gfpgan"))

    def prepare_restorer(self, enhance: bool, restorer: str = "gfpgan") -> Optional[str]:
        """Load GFPGAN or CodeFormer. Returns an install tip when it cannot start."""
        name = (restorer or "gfpgan").strip().lower()
        if name not in {"gfpgan", "codeformer"}:
            name = "gfpgan"
        self.restorer = name
        enhance = bool(enhance)
        if not enhance:
            self._enhance_requested = False
            self._enhancer = None
            self._codeformer = None
            return None
        if name == "gpen":
            from .onnx_restore import gpen_session, gpen_status

            self._gpen = gpen_session(self.execution)
            logger.info(gpen_status(self._gpen))
            if self._gpen is None:
                return self.prepare_restorer(enhance, "gfpgan")
            self._codeformer = None
            self._enhancer = None
            return None
        if name == "codeformer":
            self._enhancer = None
            from .codeformer import codeformer_install_tip, load_codeformer

            tip = codeformer_install_tip()
            if tip is not None:
                self._enhance_requested = False
                self._codeformer = None
                return tip
            if self._codeformer is not None and self._enhance_requested:
                return None
            try:
                self._codeformer = load_codeformer(use_gpu=not self._on_cpu)
            except Exception as exc:
                self._enhance_requested = False
                self._codeformer = None
                logger.warning("CodeFormer did not start: %s", exc)
                return (
                    "CodeFormer did not start. Videoswa downloads codeformer.pth "
                    f"into the models folder on first use. ({exc})"
                )
            self._enhance_requested = True
            return None
        self._codeformer = None
        if enhance and self._enhancer is not None:
            self._enhance_requested = True
            return None
        tip = gfpgan_install_tip()
        if tip is not None:
            self._enhance_requested = False
            self._enhancer = None
            return tip
        self._enhance_requested = True
        self._enhancer = _try_load_gfpgan(use_gpu=not self._on_cpu)
        if self._enhancer is None:
            self._enhance_requested = False
            return (
                "GFPGAN is installed but did not start. "
                "Videoswa downloads GFPGANv1.4.pth into the models folder on first use."
            )
        return None

    def begin_face(self, key: int) -> None:
        """Remember which tracked person the next ``swap`` call belongs to."""
        self._color_key = key

    def reset_color_memory(self) -> None:
        self.color_memory.clear()
        self._alpha_memory.clear()
        self._matrix_memory.clear()
        self._mask_cache = {}
        self._mask_tick = 0
        self._color_key = None
        self.last_canvas = None
        self.last_matrix = None

    def occlusion_models(self):
        precise = bool(self.precise_edges)
        current = self._occlusion
        if (
            current is None
            or getattr(current, "execution", None) != self.execution
            or getattr(current, "precise", False) != precise
        ):
            from .occlusion import OcclusionModels

            self._occlusion = OcclusionModels(self.execution, precise=precise)
        return self._occlusion

    def swap(
        self,
        frame: np.ndarray,
        target_face: Face,
        source_face: Face,
        paste_back: bool = True,
        coverage: str = DEFAULT_COVERAGE,
    ) -> np.ndarray:
        """Replace target_face in frame with source_face identity.

        ``coverage="full"`` (default) extends the paste over the jaw and beard.
        ``coverage="normal"`` keeps the tight face oval.
        """
        # InsightFace expects its own Face objects, but the swapper actually only
        # uses .kps and .normed_embedding. We pass a small shim.
        if not getattr(self, "_accumulate", False):
            self.stage_ms = {}
        prior_inswapper = float(self.stage_ms.get("inswapper", 0.0))
        prior_blend = float(self.stage_ms.get("blend", 0.0))
        src_shim = _FaceShim(source_face)
        tgt_shim = _FaceShim(target_face)
        # Profile landmarks collapse the far eye. Repair them before InSwapper
        # builds the frontal crop, and keep that same set for the paste.
        raw_kps = np.asarray(target_face.kps, dtype=np.float32)
        yaw = float(face_yaw(raw_kps))
        align_kps = repair_landmarks(raw_kps)
        tgt_shim.kps = align_kps
        # paste_back=False returns the 128px swap plus the frame→crop matrix.
        # Videoswa composites that itself so the beard is not cropped off.
        from .pixel_boost import (
            aligned_crop,
            cached_source_latent,
            forward_bgr_tiles,
            join_tiles,
            normalize_pixel_boost,
            split_tiles,
        )

        precomputed = getattr(self, "_precomputed", None)
        if precomputed is not None:
            swapped, matrix, align_kps, yaw = precomputed
            self._add_stage("inswapper", time.perf_counter())
        else:
            boost = self.boost_for_span(face_span_px(target_face.bbox))
            prep_started = time.perf_counter()
            latent = cached_source_latent(
                self.swapper,
                source_face,
                target_face,
                float(getattr(self, "identity_strength", 1.0) or 1.0),
            )
            crop, matrix = aligned_crop(frame, align_kps, boost)
            tiles = [crop] if boost <= 128 else split_tiles(crop)
            self._add_stage("prep", prep_started)
            infer_started = time.perf_counter()
            try:
                outputs = forward_bgr_tiles(self.swapper, tiles, latent)
            except Exception as exc:
                from .providers import is_gpu_memory_error

                if not is_gpu_memory_error(exc) or boost <= 128:
                    if is_gpu_memory_error(exc):
                        self._release_vram(exc)
                        self._mark_wiped(
                            "The GPU ran out of memory. Close any other Videoswa window and run again. "
                            "This frame was left unchanged."
                        )
                        return frame
                    raise
                self._release_vram(exc)
                crop, matrix = aligned_crop(frame, align_kps, 128)
                tiles = [crop]
                boost = 128
                try:
                    outputs = forward_bgr_tiles(self.swapper, tiles, latent)
                except Exception as again:
                    if is_gpu_memory_error(again):
                        self._mark_wiped(
                            "The GPU ran out of memory. Close any other Videoswa window and run again. "
                            "This frame was left unchanged."
                        )
                        return frame
                    raise
            swapped = outputs[0] if boost <= 128 else join_tiles(outputs, boost)
            self._add_stage("inswapper", infer_started)
        swapped = self._restore_before_paste(swapped, target_face)
        if not paste_back:
            return swapped
        self.paste_wiped = False
        self.last_paste_note = ""
        mode = normalize_coverage(coverage)
        alpha = None
        cache_key = self._color_key if self._color_key is not None else 0
        ticks = getattr(self, "_mask_ticks", None)
        if ticks is None:
            ticks = {}
            self._mask_ticks = ticks
        ticks[cache_key] = int(ticks.get(cache_key, 0)) + 1
        cached = self._mask_cache.get(cache_key)
        if self.object_mask and cached is not None and len(cached) >= 3 and (ticks[cache_key] % 3) != 1:
            reuse_started = time.perf_counter()
            alpha = _reuse_mask(
                cached[0], cached[1], cached[2], align_kps, frame.shape[:2], out=self._mask_work(frame.shape[:2])
            )
            self._add_stage("mask_reuse", reuse_started)
        if self.object_mask and alpha is None:
            load_started = time.perf_counter()
            models = self.occlusion_models()
            if not getattr(self, "_mask_warm", False):
                self._add_stage("warmup_mask", load_started)
                self._mask_warm = True
            geom_started = time.perf_counter()
            face_alpha = coverage_alpha(
                frame.shape[:2], align_kps, mode, yaw=yaw, out=self._mask_work(frame.shape[:2])
            )
            self._add_stage("geom", geom_started)
            xseg_started = time.perf_counter()
            ready = getattr(self, "_batched_neurals", None)
            if ready:
                neural = ready.pop(0)
                self._add_stage("xseg", xseg_started)
            else:
                neural = (
                    models.xseg_keepout(frame, matrix, face_alpha, canvas=int(swapped.shape[0]))
                    if models.xseg is not None
                    else None
                )
                if models.xseg is not None:
                    self._add_stage("xseg", xseg_started)
            precise = None
            if self.precise_edges and models.bisenet is not None:
                precise_started = time.perf_counter()
                precise = models.bisenet_keep(frame, matrix, face_alpha, canvas=int(swapped.shape[0]))
                self._add_stage("bisenet", precise_started)
            occ_started = time.perf_counter()
            alpha = _occ_mask(
                frame,
                align_kps,
                mode,
                neural=neural,
                precise=precise,
                yaw=yaw,
                base_alpha=face_alpha,
            )
            self._add_stage("occlusion", occ_started)
            alpha = self._shape_mask(alpha, align_kps)
            smooth_started = time.perf_counter()
            alpha = self._smooth_alpha(alpha, align_kps)
            self._add_stage("smooth", smooth_started)
            if alpha is not None:
                x0, y0, x1, y1 = face_roi(align_kps, alpha.shape[:2])
                self._mask_cache[cache_key] = (
                    np.asarray(align_kps, dtype=np.float32).copy(),
                    np.array(alpha[y0:y1, x0:x1], dtype=np.float32, copy=True),
                    (int(x0), int(y0)),
                )
            if alpha is None or float(np.max(alpha)) < 0.05:
                self._mark_wiped(
                    "The face mask was empty, so the swapped face was not pasted. "
                    "Turn Precise edges off. If the object mask removed the whole face, "
                    "try another frame."
                )
                return frame
        previous = self.color_memory.get(self._color_key) if self._color_key is not None else None
        color_state: dict = {}
        before_box = None
        if getattr(self, "_inplace", False):
            before_box = _roi_copy(frame, target_face.bbox)
        blend_started = time.perf_counter()
        out = paste_swapped_face(
            frame,
            swapped,
            matrix,
            align_kps,
            coverage=mode,
            yaw=yaw,
            alpha=alpha,
            previous_delta=previous,
            color_state=color_state,
            color_match=getattr(self, "color_match", None),
            inplace=bool(getattr(self, "_inplace", False)),
        )
        self._add_stage("blend", blend_started)
        if self._color_key is not None and "delta" in color_state:
            self.color_memory[self._color_key] = color_state["delta"]
        amount = float(getattr(self, "sharpen", 0.0) or 0.0)
        if amount > 0.0:
            sharp_started = time.perf_counter()
            out = _sharpen_roi(out, target_face.bbox, amount)
            self._add_stage("sharpen", sharp_started)
        finish_started = time.perf_counter()
        if _change_against(frame, out, target_face.bbox, before_box) < 1.5:
            plain = coverage_alpha(frame.shape[:2], align_kps, mode, yaw=yaw)
            rescued = paste_swapped_face(
                frame,
                swapped,
                matrix,
                align_kps,
                coverage=mode,
                yaw=yaw,
                alpha=plain,
                color_match=getattr(self, "color_match", None),
                inplace=bool(getattr(self, "_inplace", False)),
            )
            if _change_against(frame, rescued, target_face.bbox, before_box) >= 1.5:
                if not getattr(self, "_mask_fallback_logged", False):
                    logger.info(
                        "The object mask hid a finished swap. Using the face outline so the face is kept."
                    )
                    self._mask_fallback_logged = True
                out = rescued
            else:
                self._mark_wiped(
                    "The paste mask wiped this swap. The face region did not change. "
                    "Turn Precise edges off and preview again. "
                    "A small face that changes by about 8 is a real swap and is kept."
                )
                return frame
        self.last_canvas = np.asarray(swapped).copy()
        self.last_matrix = np.asarray(matrix, dtype=np.float64).reshape(2, 3).copy()
        self._add_stage("finish", finish_started)
        self._recent_ms = (
            float(self.stage_ms.get("inswapper", 0.0)) - prior_inswapper
            + float(self.stage_ms.get("blend", 0.0)) - prior_blend
        )
        return out

    def _release_vram(self, exc: BaseException) -> None:
        """Drop the heavy paths after CUDA runs out of memory."""
        self._vram_low = True
        self.pixel_boost = 128
        self.large_face_px = 0
        model = getattr(self, "swapper", None)
        if model is not None:
            model._videoswa_batch_session = None
            model._videoswa_face_batch = "off"
            model._videoswa_io = False
        if not getattr(self, "_vram_logged", False):
            logger.warning(
                "GPU ran out of memory (%s). Pixel boost is now 128 and the multi-face "
                "batch engine is off. Close any other Videoswa window. The video continues.",
                exc,
            )
            self._vram_logged = True

    def boost_for_span(self, span: float) -> int:
        """128 by default. Quality asks for 256. Balanced may use 256 on a large face."""
        from .pixel_boost import normalize_pixel_boost

        if getattr(self, "_vram_low", False) or _gpu_is_tight():
            return 128
        base = normalize_pixel_boost(getattr(self, "pixel_boost", 128))
        if base > 128:
            return base
        limit = float(getattr(self, "large_face_px", 0) or 0)
        if limit <= 0 or float(span) < limit:
            return 128
        budget = float(getattr(self, "boost_budget_ms", 40.0) or 40.0)
        recent = float(getattr(self, "_recent_ms", 0.0) or 0.0)
        if recent == 0.0 or recent <= budget:
            return 256
        return 128

    def swap_many(self, frame: np.ndarray, pairs: list, coverage: str = DEFAULT_COVERAGE) -> tuple[np.ndarray, list[bool]]:
        """Swap several faces. Batch the 128px network call when that engine is faster."""
        if len(pairs) <= 1:
            target, source = pairs[0][0], pairs[0][1]
            if len(pairs[0]) > 2:
                self.begin_face(pairs[0][2])
            out = self.swap(frame, target, source, coverage=coverage)
            return out, [not self.paste_wiped]
        # Several faces at 256 are four passes each and cannot share one batch.
        saved_boost = self.pixel_boost
        saved_large = self.large_face_px
        self.pixel_boost = 128
        self.large_face_px = 0
        try:
            session = face_batch_session(self.swapper)
            decision = getattr(self.swapper, "_videoswa_face_batch", "")
            if session is None:
                if not getattr(self, "_batch_miss_logged", False):
                    logger.info(
                        "InSwapper batch path not used for %d faces (decision=%s). Per-face 128px calls.",
                        len(pairs),
                        decision or "not ready",
                    )
                    self._batch_miss_logged = True
                return self._swap_many_serial(frame, pairs, coverage)
            if not getattr(self, "_batch_hit_logged", False):
                logger.info(
                    "InSwapper batch path: %d faces in one call (decision=%s).",
                    len(pairs),
                    decision or "on",
                )
                self._batch_hit_logged = True
            return self._swap_many_batched(frame, pairs, coverage, session)
        finally:
            self.pixel_boost = saved_boost
            self.large_face_px = saved_large

    def _swap_many_serial(self, frame, pairs, coverage):
        """One frame copy, one XSeg batch, then every face pasted into that copy.

        Each face used to clear the stage clock and copy the whole frame, so
        the log called the other faces 'gap'.
        """
        from .pixel_boost import aligned_crop, cached_source_latent, forward_bgr_tiles
        from .providers import is_gpu_memory_error

        out = frame.copy()
        self.stage_ms = {}
        self._accumulate = True
        self._inplace = True
        images = []
        matrices = []
        prepared = []
        try:
            for item in pairs:
                target, source = item[0], item[1]
                key = item[2] if len(item) > 2 else id(target)
                raw = np.asarray(target.kps, dtype=np.float32)
                align = repair_landmarks(raw)
                yaw = float(face_yaw(raw))
                prep_started = time.perf_counter()
                latent = cached_source_latent(
                    self.swapper,
                    source,
                    target,
                    float(getattr(self, "identity_strength", 1.0) or 1.0),
                )
                crop, matrix = aligned_crop(frame, align, 128)
                self._add_stage("prep", prep_started)
                infer_started = time.perf_counter()
                try:
                    outputs = forward_bgr_tiles(self.swapper, [crop], latent)
                except Exception as exc:
                    self._add_stage("inswapper", infer_started)
                    if not is_gpu_memory_error(exc):
                        raise
                    self._release_vram(exc)
                    images.append(None)
                    matrices.append(None)
                    prepared.append((target, source, align, yaw, key))
                    continue
                self._add_stage("inswapper", infer_started)
                images.append(self._restore_before_paste(outputs[0], target))
                matrices.append(matrix)
                prepared.append((target, source, align, yaw, key))
            live_matrices = [matrix for matrix in matrices if matrix is not None]
            live_prepared = [item for item, matrix in zip(prepared, matrices) if matrix is not None]
            xseg_started = time.perf_counter()
            self._batched_neurals = _batch_xseg(self, frame, live_matrices, live_prepared)
            self._add_stage("xseg", xseg_started)
            oks = []
            for image, (target, source, align, yaw, key), matrix in zip(images, prepared, matrices):
                self.begin_face(key)
                if image is None or matrix is None:
                    oks.append(False)
                    continue
                self._precomputed = (image, matrix, align, yaw)
                try:
                    out = self.swap(out, target, source, coverage=coverage)
                finally:
                    self._precomputed = None
                oks.append(not self.paste_wiped)
            return out, oks
        finally:
            self._accumulate = False
            self._inplace = False
            self._batched_neurals = None

    def _swap_many_batched(self, frame, pairs, coverage, session):
        from .pixel_boost import aligned_crop, cached_source_latent

        crops = []
        latents = []
        matrices = []
        prepared = []
        for item in pairs:
            target, source = item[0], item[1]
            raw = np.asarray(target.kps, dtype=np.float32)
            align = repair_landmarks(raw)
            latent = cached_source_latent(
                self.swapper, source, target, float(getattr(self, "identity_strength", 1.0) or 1.0)
            )
            crop, matrix = aligned_crop(frame, align, 128)
            crops.append(crop)
            latents.append(latent)
            matrices.append(matrix)
            prepared.append((target, source, align, float(face_yaw(raw)), item[2] if len(item) > 2 else id(target)))
        self.stage_ms = {}
        self._accumulate = True
        self._inplace = True
        try:
            infer_started = time.perf_counter()
            images = _forward_with_session(session, self.swapper, crops, latents)
            self._add_stage("inswapper", infer_started)
            xseg_started = time.perf_counter()
            self._batched_neurals = _batch_xseg(self, frame, matrices, prepared)
            self._add_stage("xseg", xseg_started)
            out = frame.copy()
            oks = []
            for image, (target, source, align, yaw, key), matrix in zip(images, prepared, matrices):
                self.begin_face(key)
                self._precomputed = (image, matrix, align, yaw)
                try:
                    out = self.swap(out, target, source, coverage=coverage)
                finally:
                    self._precomputed = None
                oks.append(not self.paste_wiped)
            return out, oks
        finally:
            self._accumulate = False
            self._inplace = False
            self._batched_neurals = None

    def _restore_before_paste(self, canvas: np.ndarray, target_face: Face) -> np.ndarray:
        """Run GFPGAN or CodeFormer on the aligned swap, then paste that result."""
        if self._enhancer is not None:
            device = _gfpgan_device(self._enhancer)
            if not getattr(self, "_enhance_logged", False):
                logger.info("enhance device=%s cuda_required=yes", device)
                self._enhance_logged = True
            if "cpu" in device.lower():
                logger.warning("GFPGAN is on CPU. Skipping it. A CPU restorer is hundreds of ms per frame.")
                self._enhancer = None
                self.allow_restore = False
                self.stage_ms["enhance"] = self.stage_ms.get("enhance", 0.0)
                return canvas
        if canvas is None or not self.allow_restore:
            if not getattr(self, "_enhance_logged", False):
                logger.info("enhance: off")
                self._enhance_logged = True
            self.stage_ms["enhance"] = self.stage_ms.get("enhance", 0.0)
            return canvas
        if face_span_px(target_face.bbox) < RESTORE_MIN_PX:
            return canvas
        every = max(1, int(getattr(self, "enhance_every", 1) or 1))
        self._enhance_tick += 1
        if every > 1 and (self._enhance_tick % every) != 1:
            cached = getattr(self, "_restore_cache", None)
            if cached is not None and getattr(cached, "shape", None) == canvas.shape:
                return cached
            return canvas
        started = time.perf_counter()
        import cv2

        from .codeformer import blend_restored

        side = int(canvas.shape[0])
        view = canvas
        if side < 256:
            view = cv2.resize(canvas, (256, 256), interpolation=cv2.INTER_CUBIC)
        restored = None
        if getattr(self, "restorer", "gfpgan") == "gpen" and getattr(self, "_gpen", None) is not None:
            from .onnx_restore import restore_gpen

            try:
                restored = restore_gpen(self._gpen, view, float(getattr(self, "restore_strength", 1.0)))
            except Exception as exc:
                logger.debug("GPEN failed before paste: %s", exc)
                restored = None
        elif getattr(self, "restorer", "gfpgan") == "codeformer" and self._codeformer is not None:
            from .codeformer import restore_bgr

            try:
                restored = restore_bgr(
                    self._codeformer,
                    view,
                    float(getattr(self, "codeformer_fidelity", 0.5)),
                )
            except Exception as exc:
                logger.debug("CodeFormer failed before paste: %s", exc)
                restored = None
        elif self._enhancer is not None:
            try:
                _cropped, _faces, restored = self._enhancer.enhance(
                    view, has_aligned=True, only_center_face=False, paste_back=True
                )
            except Exception as exc:
                logger.debug("GFPGAN failed before paste: %s", exc)
                restored = None
        if restored is None:
            self._add_stage("enhance", started)
            return canvas
        if restored.shape[0] != side or restored.shape[1] != side:
            restored = cv2.resize(restored, (side, side), interpolation=cv2.INTER_CUBIC)
        blended = blend_restored(canvas, restored, float(getattr(self, "restore_strength", 1.0)))
        self._restore_cache = blended
        self._add_stage("enhance", started)
        return blended

    def _add_stage(self, name: str, started: float) -> None:
        self.stage_ms[name] = self.stage_ms.get(name, 0.0) + (time.perf_counter() - started) * 1000.0

    def _shape_mask(self, alpha: Optional[np.ndarray], kps: np.ndarray) -> Optional[np.ndarray]:
        """Blur, pad, or cut the mouth out of the paste mask. Defaults leave it unchanged."""
        if alpha is None:
            return None
        import cv2

        started = time.perf_counter()
        out = np.asarray(alpha, dtype=np.float32)
        pad = int(getattr(self, "mask_padding", 0) or 0)
        if pad != 0:
            size = abs(pad) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            out = cv2.dilate(out, kernel) if pad > 0 else cv2.erode(out, kernel)
        blur = int(getattr(self, "mask_blur", 0) or 0)
        if blur > 0:
            ksize = blur * 2 + 1
            out = cv2.GaussianBlur(out, (ksize, ksize), 0)
        if getattr(self, "keep_mouth", False):
            pts = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] >= 5:
                center = pts[3:5].mean(axis=0)
                radius = max(4.0, float(np.linalg.norm(pts[4] - pts[3])) * 0.45)
                cv2.circle(out, (int(round(center[0])), int(round(center[1]))), int(round(radius)), 0.0, -1)
        self._add_stage("mask", started)
        return out

    def _smooth_matrix(self, matrix: np.ndarray) -> np.ndarray:
        from .smooth import smooth_pair

        current = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
        key = self._color_key
        if key is None:
            return current
        previous = self._matrix_memory.get(key)
        if previous is not None and np.linalg.norm(current[:, 2] - previous[:, 2]) > 48.0:
            previous = None
        mixed = smooth_pair(previous, current, keep=0.55)
        self._matrix_memory[key] = mixed
        return mixed

    def _smooth_alpha(self, alpha: Optional[np.ndarray], kps: np.ndarray) -> Optional[np.ndarray]:
        if alpha is None:
            return None
        from .smooth import smooth_pair

        key = self._color_key
        if key is None:
            return alpha
        x0, y0, x1, y1 = face_roi(kps, alpha.shape[:2])
        patch = np.asarray(alpha[y0:y1, x0:x1], dtype=np.float32)
        previous = self._alpha_memory.get(key)
        prev = previous if isinstance(previous, np.ndarray) and previous.shape == patch.shape else None
        mixed = np.clip(smooth_pair(prev, patch, keep=0.65), 0.0, 1.0).astype(np.float32)
        alpha[y0:y1, x0:x1] = mixed
        self._alpha_memory[key] = mixed.copy()
        return alpha

    def _mask_work(self, shape: tuple[int, int]) -> np.ndarray:
        """One reusable full-frame mask. A 1080p float mask is about 8 MB."""
        buf = getattr(self, "_mask_buffer", None)
        if buf is None or buf.shape[:2] != shape:
            buf = np.zeros(shape, dtype=np.float32)
            self._mask_buffer = buf
        return buf

    def _mark_wiped(self, note: str) -> None:
        self.paste_wiped = True
        self.last_paste_note = note
        logger.warning(note)


class _FaceShim:
    """Duck-typed object compatible with insightface's INSwapper.get."""

    def __init__(self, face: Face) -> None:
        self.kps = face.kps
        self.bbox = face.bbox
        self.embedding = face.embedding
        self.normed_embedding = face.normed_embedding
        self.det_score = face.det_score


def _reuse_mask(
    old_kps: np.ndarray,
    patch: np.ndarray,
    origin: tuple[int, int],
    new_kps: np.ndarray,
    shape: tuple[int, int],
    out: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Move the last paste mask with the landmarks. Skips XSeg on in-between frames."""
    import cv2

    if out is None or out.shape[:2] != shape:
        full = np.zeros(shape, dtype=np.float32)
    else:
        full = out
        full.fill(0)
    ox, oy = origin
    ph, pw = patch.shape[:2]
    y1 = min(shape[0], oy + ph)
    x1 = min(shape[1], ox + pw)
    if y1 > oy and x1 > ox:
        full[oy:y1, ox:x1] = patch[: y1 - oy, : x1 - ox]
    old = np.asarray(old_kps, dtype=np.float32).reshape(-1, 2)
    new = np.asarray(new_kps, dtype=np.float32).reshape(-1, 2)
    forward, _inliers = cv2.estimateAffinePartial2D(old.reshape(-1, 1, 2), new.reshape(-1, 1, 2))
    if forward is None:
        delta = new.mean(axis=0) - old.mean(axis=0)
        forward = np.array([[1.0, 0.0, float(delta[0])], [0.0, 1.0, float(delta[1])]], dtype=np.float32)
    x0, y0, x1b, y1b = face_roi(new, shape)
    shifted = np.asarray(forward, dtype=np.float32).copy()
    shifted[0, 2] -= x0
    shifted[1, 2] -= y0
    warped = cv2.warpAffine(
        full,
        shifted,
        (x1b - x0, y1b - y0),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    full.fill(0)
    full[y0:y1b, x0:x1b] = warped
    return full


def _enable_inswapper_batch(model, providers) -> None:
    """Rebuild InSwapper so TensorRT can run 1, 4, or 16 tiles in one call.

    The stock model is batch 1. Four sequential runs were about 50 ms/frame.
    A dynamic batch keeps one cached FP16 engine.
    """
    if getattr(model, "_videoswa_batch_ready", False):
        return
    model_file = getattr(model, "model_file", None)
    if not model_file or not getattr(model, "session", None):
        return
    try:
        import onnx
        import onnxruntime as ort

        from .providers import TENSORRT, cuda_provider_options, tensorrt_provider_options
        from .utils import MODELS_DIR

        onnx_model = onnx.load(str(model_file))
        for value in list(onnx_model.graph.input) + list(onnx_model.graph.output):
            dims = value.type.tensor_type.shape.dim
            if not dims:
                continue
            dims[0].dim_param = "batch"
            dims[0].ClearField("dim_value")
        image_name = model.input_names[0]
        id_name = model.input_names[1]
        id_dim = 512
        id_shape = model.session.get_inputs()[1].shape
        if id_shape and isinstance(id_shape[-1], int):
            id_dim = int(id_shape[-1])
        cache = MODELS_DIR / "trt_cache" / "inswapper_batch"
        cache.mkdir(parents=True, exist_ok=True)
        trt = tensorrt_provider_options(cache)
        shape = (
            f"{image_name}:{{n}}x3x128x128,{id_name}:{{n}}x{id_dim}"
        )
        trt["trt_max_workspace_size"] = 512 * 1024 * 1024
        trt["trt_profile_min_shapes"] = shape.format(n=1)
        trt["trt_profile_opt_shapes"] = shape.format(n=2)
        trt["trt_profile_max_shapes"] = shape.format(n=3)
        # Lower than the default 3 so the first build does not occupy the GPU for minutes.
        trt["trt_builder_optimization_level"] = 1
        active = list(model.session.get_providers())
        chosen = []
        options = []
        for entry in providers or active:
            name = entry[0] if isinstance(entry, tuple) else entry
            if name not in active and name != TENSORRT:
                continue
            chosen.append(name)
            if name == TENSORRT:
                options.append(trt)
            elif "CUDA" in name:
                options.append(cuda_provider_options())
            else:
                options.append({})
        if not chosen:
            chosen = active
            options = [{} for _ in chosen]
        session = ort.InferenceSession(
            onnx_model.SerializeToString(),
            providers=chosen,
            provider_options=options,
        )
        # Do not replace the fast batch-1 session. Measure only after a quiet
        # warmup. A busy GPU made two single calls look like 250 ms and the
        # batch looked like a win.
        _wait_until_gpu_quiet(model)
        single = _median_ms(model.session, model, batch=1)
        batched = _median_ms(session, model, batch=2)
        choice = batch_decision(single, batched)
        decision = cache / "decision_v2.txt"
        model._videoswa_batch_ready = True
        if choice == "skip":
            logger.warning(
                "InSwapper single-call median %.1f ms is too slow to trust "
                "(a quiet pass on this laptop is under 20 ms). Not saving a batch decision.",
                single,
            )
            model._videoswa_face_batch = "off"
            return
        if choice == "off":
            decision.write_text("off")
            model._videoswa_face_batch = "off"
            logger.info(
                "InSwapper batch-2 median %.1f ms vs two singles %.1f ms. Staying on per-face calls.",
                batched,
                single * 2.0,
            )
            return
        decision.write_text("on")
        model._videoswa_batch_session = session
        model._videoswa_face_batch = "on"
        logger.info(
            "InSwapper batch-2 median %.1f ms beats two singles %.1f ms. Multi-face uses batch 1-3.",
            batched,
            single * 2.0,
        )
    except Exception as exc:
        model._videoswa_face_batch = "off"
        model._videoswa_batch_ready = True
        try:
            from .utils import MODELS_DIR

            path = MODELS_DIR / "trt_cache" / "inswapper_batch"
            path.mkdir(parents=True, exist_ok=True)
            (path / "decision_v2.txt").write_text("off")
        except Exception:
            pass
        logger.info("InSwapper face batch stays off (%s). Faces run one at a time.", exc)


def batch_decision(single_ms: float, batch_ms: float) -> str:
    """Return on, off, or skip.

    skip means the single-call time is far above a normal TensorRT pass, so
    the GPU was busy and the comparison is not saved.
    """
    if single_ms <= 0 or single_ms > 40.0:
        return "skip"
    if batch_ms < 0.85 * (single_ms * 2.0):
        return "on"
    return "off"


def _median_ms(session, model, batch: int, warmup: int = 3, runs: int = 5) -> float:
    import time

    image = np.zeros((batch, 3, 128, 128), dtype=np.float32)
    latent = np.zeros((batch, 512), dtype=np.float32)
    feeds = {model.input_names[0]: image, model.input_names[1]: latent}
    for _ in range(warmup):
        session.run(model.output_names, feeds)
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(model.output_names, feeds)
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _wait_until_gpu_quiet(model, timeout_s: float = 20.0) -> None:
    import time

    deadline = time.perf_counter() + timeout_s
    while getattr(model, "_videoswa_busy", False) and time.perf_counter() < deadline:
        time.sleep(0.05)


def _time_runs(session, model, count: int, batch: int = 1) -> float:
    import time

    image = np.zeros((batch, 3, 128, 128), dtype=np.float32)
    latent = np.zeros((batch, 512), dtype=np.float32)
    feeds = {model.input_names[0]: image, model.input_names[1]: latent}
    started = time.perf_counter()
    for _ in range(count):
        session.run(model.output_names, feeds)
    return (time.perf_counter() - started) * 1000.0


def _forward_with_session(session, model, tiles, latents):
    import cv2

    from .pixel_boost import _bgr_from_nchw, forward_bgr_tiles

    blobs = [
        cv2.dnn.blobFromImage(tile, 1.0 / 255.0, (128, 128), (0.0, 0.0, 0.0), swapRB=True)
        for tile in tiles
    ]
    batch = np.concatenate(blobs, axis=0)
    latent = np.concatenate([np.asarray(item, dtype=np.float32).reshape(1, -1) for item in latents], axis=0)
    try:
        pred = session.run(model.output_names, {model.input_names[0]: batch, model.input_names[1]: latent})[0]
        return [_bgr_from_nchw(pred[index:index + 1]) for index in range(pred.shape[0])]
    except Exception as exc:
        logger.info("Face batch run failed (%s). Using per-face calls.", exc)
        model._videoswa_face_batch = "off"
        model._videoswa_batch_session = None
        return [forward_bgr_tiles(model, [tile], one)[0] for tile, one in zip(tiles, latents)]


def _batch_xseg(swapper, frame, matrices, prepared):
    models = swapper.occlusion_models()
    if getattr(models, "xseg", None) is None:
        return None
    from .coverage import coverage_alpha
    from .occlusion import _input_hw, _scale_matrix, _warp_crop, session_masks, warp_into_face

    hw = _input_hw(models.xseg)
    aligned = [_scale_matrix(matrix, 128, hw) for matrix in matrices]
    crops = [_warp_crop(frame, matrix, hw) for matrix in aligned]
    masks = session_masks(models.xseg, crops, occluder=True)
    work = swapper._mask_work(frame.shape[:2])
    neurals = []
    for mask, matrix, item in zip(masks, aligned, prepared):
        if mask is None:
            neurals.append(None)
            continue
        alpha = coverage_alpha(frame.shape[:2], item[2], "full", yaw=item[3], out=work)
        neurals.append(warp_into_face(mask, matrix, frame.shape[:2], alpha, compact=True, kps=item[2]))
    return neurals


def face_batch_session(model):
    """Return a ready batch session. Building happens before the frame loop."""
    if getattr(model, "_videoswa_face_batch", "") != "on":
        return None
    return getattr(model, "_videoswa_batch_session", None)


def prepare_face_batch(model) -> str:
    """Build or load the batch engine before timed frames.

    A build during the frame loop stalled detect and decode for seconds per
    frame. Call this first, then run the video.
    """
    if getattr(model, "_videoswa_face_batch", "") == "on" and getattr(model, "_videoswa_batch_session", None) is not None:
        logger.info("InSwapper batch decision_v2: on (session already loaded)")
        return "on"
    try:
        providers = list(model.session.get_providers())
    except Exception:
        return "off"
    if not any("Tensorrt" in name for name in providers):
        model._videoswa_face_batch = "off"
        logger.info("InSwapper batch decision_v2: off (no TensorRT). Faces run one at a time.")
        return "off"
    from .utils import MODELS_DIR

    decision = MODELS_DIR / "trt_cache" / "inswapper_batch" / "decision_v2.txt"
    if decision.is_file():
        saved = decision.read_text().strip()
        logger.info("InSwapper batch decision_v2: %s", saved or "empty")
        if saved == "off":
            model._videoswa_face_batch = "off"
            return "off"
    else:
        logger.info("InSwapper batch decision_v2: missing. Building before the timed frames.")
    logger.info(
        "Preparing multi-face InSwapper before the timed frames. "
        "Profile is batch 1 / 2 / 3. The frame loop waits until this finishes."
    )
    _enable_inswapper_batch(model, providers)
    result = getattr(model, "_videoswa_face_batch", "off")
    logger.info("InSwapper batch decision_v2 result: %s", result)
    return result


def warm_models(swapper) -> None:
    """One batch-1 forward so TensorRT compiles before the timed frames.

    A dynamic batch-4 engine was slower than the stock 128px pass (64 ms vs
    about 7 ms) and took minutes to build. It is not warmed.
    """
    model = getattr(swapper, "swapper", None)
    if model is not None and hasattr(model, "session"):
        try:
            started = time.perf_counter()
            image = np.zeros((1, 3, 128, 128), dtype=np.float32)
            latent = np.zeros((1, 512), dtype=np.float32)
            model.session.run(
                model.output_names,
                {model.input_names[0]: image, model.input_names[1]: latent},
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            shapes = [(item.name, list(item.shape)) for item in model.session.get_inputs()]
            logger.info(
                "Warmed InSwapper batch-1 in %.1f ms providers=%s inputs=%s",
                elapsed,
                model.session.get_providers(),
                shapes,
            )
        except Exception as exc:
            logger.info("InSwapper warmup skipped: %s", exc)
    try:
        models = swapper.occlusion_models()
    except Exception as exc:
        logger.info("Mask warmup skipped: %s", exc)
        return
    session = getattr(models, "xseg", None)
    if session is None:
        return
    try:
        from .occlusion import _model_input

        blob = _model_input(session, np.zeros((64, 64, 3), dtype=np.uint8), imagenet=False)
        session.run(None, {session.get_inputs()[0].name: blob})
        logger.info("Warmed XSeg")
    except Exception as exc:
        logger.info("XSeg warmup skipped: %s", exc)


def gfpgan_install_tip() -> Optional[str]:
    """How to enable sharpening, or None when GFPGAN can be imported."""
    try:
        import gfpgan  # noqa: F401
    except Exception:
        return (
            "GFPGAN is optional and is not installed in this Python.\n\n"
            "From the Videoswa virtualenv run:\n"
            "pip install -r requirements-enhance.txt\n\n"
            "The first enhanced swap downloads GFPGANv1.4.pth into models/."
        )
    return None


def _try_load_gfpgan(use_gpu: bool):
    try:
        from gfpgan import GFPGANer
    except Exception as exc:
        logger.warning("GFPGAN not available (%s); skipping enhancement", exc)
        return None
    try:
        weight_path = _ensure_gfpgan_weights()
        try:
            import torch

            cuda = bool(use_gpu and torch.cuda.is_available())
        except Exception:
            cuda = False
        device = "cuda" if cuda else "cpu"
        logger.info(
            "GFPGAN device=%s cuda_available=%s (torch must stay on cuda; a CPU restorer costs hundreds of ms per frame)",
            device,
            cuda,
        )
        enhancer = GFPGANer(
            model_path=str(weight_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
        logger.info("GFPGAN parameters are on %s", _gfpgan_device(enhancer))
        return enhancer
    except Exception as exc:
        logger.warning("Failed to initialize GFPGAN: %s", exc)
        return None


def _gfpgan_device(enhancer) -> str:
    try:
        model = getattr(enhancer, "gfpgan", None)
        if model is None:
            return "unknown"
        return str(next(model.parameters()).device)
    except Exception:
        return "unknown"


def _ensure_gfpgan_weights() -> Path:
    from .utils import ensure_gfpgan

    return ensure_gfpgan()


def _gpu_is_tight() -> bool:
    """True when less than 1 GB is free. Pixel boost stays at 128 then."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return False
    return free < 1024 * 1024 * 1024


def _roi_copy(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    """Copy one face box so an in-place paste can still see what changed."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in np.asarray(bbox).reshape(-1)[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array(frame[y1:y2, x1:x2], copy=True)


def _change_against(before: np.ndarray, after: np.ndarray, bbox: np.ndarray, box: Optional[np.ndarray]) -> float:
    """Mean change in the face box. ``box`` is the pre-paste crop when the frame is edited in place."""
    if box is None:
        return _box_mean_abs(before, after, bbox)
    height, width = after.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in np.asarray(bbox).reshape(-1)[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1 or box.shape[:2] != (y2 - y1, x2 - x1):
        return _box_mean_abs(before, after, bbox)
    delta = np.abs(box.astype(np.int16) - after[y1:y2, x1:x2].astype(np.int16))
    return float(np.mean(delta))


def _box_mean_abs(before: np.ndarray, after: np.ndarray, bbox: np.ndarray) -> float:
    """Mean absolute change inside a face box. About 8 is a real small-face swap."""
    if before.shape != after.shape:
        return 0.0
    height, width = before.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in np.asarray(bbox).reshape(-1)[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    delta = np.abs(before[y1:y2, x1:x2].astype(np.int16) - after[y1:y2, x1:x2].astype(np.int16))
    return float(np.mean(delta))


def _sharpen_roi(frame: np.ndarray, bbox: np.ndarray, amount: float) -> np.ndarray:
    """Light unsharp mask on the face box. ``amount`` 0.12 is the Balanced default."""
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in np.asarray(bbox).reshape(-1)[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return frame
    roi = frame[y1:y2, x1:x2]
    blur = cv2.GaussianBlur(roi, (0, 0), 1.2)
    sharp = cv2.addWeighted(roi, 1.0 + amount, blur, -amount, 0)
    out = frame.copy()
    out[y1:y2, x1:x2] = sharp
    return out


def _enhance_bbox(bbox: np.ndarray, coverage: str) -> np.ndarray:
    """Grow the enhancer window so a full-coverage beard is inside it."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if normalize_coverage(coverage) == "full":
        x1 -= 0.30 * width
        x2 += 0.30 * width
        y1 -= 0.20 * height
        y2 += 0.95 * height
    else:
        x1 -= 0.15 * width
        x2 += 0.15 * width
        y1 -= 0.15 * height
        y2 += 0.20 * height
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _enhance_codeformer(frame: np.ndarray, bbox: np.ndarray, model, fidelity: float) -> np.ndarray:
    """Run CodeFormer on the face crop and paste it back at the original size."""
    import cv2

    from .codeformer import restore_bgr

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in np.asarray(bbox).reshape(-1)[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return frame
    crop = frame[y1:y2, x1:x2]
    try:
        restored = restore_bgr(model, crop, fidelity)
    except Exception as exc:
        logger.debug("CodeFormer failed on region: %s", exc)
        return frame
    if restored.shape[:2] != crop.shape[:2]:
        restored = cv2.resize(restored, (crop.shape[1], crop.shape[0]))
    out = frame.copy()
    out[y1:y2, x1:x2] = restored
    return out


def _enhance_face_region(frame: np.ndarray, bbox: np.ndarray, enhancer) -> np.ndarray:
    """Run GFPGAN on the cropped face region with a generous margin."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    pad_x = int((x2 - x1) * 0.4)
    pad_y = int((y2 - y1) * 0.4)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return frame
    crop = frame[y1:y2, x1:x2].copy()
    try:
        _, _, restored = enhancer.enhance(
            crop, has_aligned=False, only_center_face=True, paste_back=True
        )
    except Exception as exc:
        logger.debug("Enhancer failed on region: %s", exc)
        return frame
    if restored is None:
        return frame
    if restored.shape[:2] != crop.shape[:2]:
        import cv2
        restored = cv2.resize(restored, (crop.shape[1], crop.shape[0]))
    out = frame.copy()
    out[y1:y2, x1:x2] = restored
    return out
