#!/usr/bin/env python3
"""
Core REDPAN Implementation
=========================

REDPAN class for continuous seismic phase picking. SeisBench-style
direct array accumulation over sliding windows, with spectrum-matched
noise padding for time alignment at the trace boundaries.

Vendored (minimal subset) from https://github.com/tso1257771/RED-PAN
under its MIT license; relicensed under the RoSE repo's MIT LICENSE.
"""

import gc
import logging
import numpy as np
import tensorflow as tf
from copy import deepcopy
from fractions import Fraction
from typing import List, Tuple
from obspy import Stream
from obspy.signal.trigger import trigger_onset
from obspy.signal.util import smooth
from scipy.signal import resample_poly
from .utils import (
    sac_len_complement, stream_standardize,
    generate_matching_noise, find_reference_signal
)
from .picks import pred_postprocess

logger = logging.getLogger(__name__)


class REDPAN:
    """
    True SeisBench-style RED-PAN implementation with direct array accumulation
    
    Uses spectrum-matched noise padding to ensure proper time alignment
    in sliding window predictions.
    """
    
    def __init__(self,
                 model,
                 pred_npts: int = 6000,
                 dt: float = 0.01,
                 pred_interval_sec: float = 10.0,
                 batch_size: int = 32,
                 use_compiled_infer: bool = True,
                 jit_compile: bool = False,
                 accumulation_mode: str = "loop",
                 seed: int | None = None):
        """
        Initialize the RED-PAN picker

        Args:
            model: TensorFlow model for prediction
            pred_npts: Model input length (samples)
            dt: Sample interval (seconds)
            pred_interval_sec: Sliding window step (seconds)
            batch_size: Batch size for prediction
            seed: Optional integer seed for the spectrum-matched padding
                noise generator. When None, a fresh `np.random.default_rng()`
                is used per call, decoupled from the global `np.random`
                state. Set this for bitwise-reproducible predictions.
        """
        self.model = model
        self.pred_npts = pred_npts
        self.dt = dt
        self.pred_interval_sec = pred_interval_sec
        self.batch_size = batch_size
        self.use_compiled_infer = bool(use_compiled_infer)
        self.jit_compile = bool(jit_compile)
        self.accumulation_mode = str(accumulation_mode).lower()
        if self.accumulation_mode not in {"loop", "addat"}:
            raise ValueError("accumulation_mode must be one of: loop, addat")
        self._noise_rng = (
            np.random.default_rng(seed) if seed is not None else None
        )
        
        # Calculate prediction interval in samples
        self.pred_interval_pt = int(round(pred_interval_sec / dt))
        
        # Use uniform weights for accumulation (matches legacy median filter)
        self.position_weights = np.ones(self.pred_npts, dtype=np.float32)

        self._compiled_infer = None
        if self.use_compiled_infer:
            try:
                self._build_compiled_infer()
            except Exception as e:
                logger.warning(
                    "Failed to build compiled inference graph; falling back to eager mode. "
                    f"reason={e}"
                )
                self.use_compiled_infer = False
                self._compiled_infer = None
        
        gc.collect()
        
        logger.info(f"REDPAN initialized: pred_npts={pred_npts}, "
                   f"pred_interval_sec={pred_interval_sec}, batch_size={batch_size}")

    def _build_compiled_infer(self):
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, self.pred_npts, 3), dtype=tf.float32)
            ],
            reduce_retracing=True,
            jit_compile=self.jit_compile,
        )
        def _infer(x):
            return self.model(x, training=False)

        self._compiled_infer = _infer
    
    def _pad_stream_with_noise(self, wf: Stream, pad_npts: int) -> Stream:
        """
        Pad stream with spectrum-matched noise at both ends.
        
        Args:
            wf: Input ObsPy stream
            pad_npts: Number of samples to pad at each end
            
        Returns:
            Padded stream
        """
        wf_padded = wf.copy()
        
        for trace in wf_padded:
            ref_signal = find_reference_signal(trace.data)
            front_noise = generate_matching_noise(
                ref_signal, pad_npts, rng=self._noise_rng)
            back_noise = generate_matching_noise(
                ref_signal, pad_npts, rng=self._noise_rng)
            trace.data = np.concatenate([front_noise, trace.data, back_noise])
            trace.stats.starttime -= pad_npts * self.dt
        
        return wf_padded
    
    def _prepare_waveform_slices(self, wf: Stream) -> np.ndarray:
        """
        Prepare waveform slices for sliding window prediction.
        
        The waveform should already be padded. This method extracts
        overlapping windows with proper normalization.
        
        Args:
            wf: Padded ObsPy stream
            
        Returns:
            Array of normalized waveform slices (n_windows, pred_npts, 3)
        """
        data_len = len(wf[0].data)
        
        # Calculate number of windows
        n_windows = (data_len - self.pred_npts) // self.pred_interval_pt + 1
        
        logger.debug(f"Preparing {n_windows} windows from {data_len} samples")
        
        # Extract and normalize slices for each channel
        wf_channels = []
        for ch in range(3):
            channel_data = wf[ch].data
            slices = []
            
            for i in range(n_windows):
                start = i * self.pred_interval_pt
                end = start + self.pred_npts
                
                window = channel_data[start:end].copy()
                
                # Normalize: demean and standardize
                window = window - np.mean(window)
                std = np.std(window)
                if std > 1e-10:
                    window = window / std
                
                slices.append(window)
            
            wf_channels.append(np.array(slices))
            
            del slices
        
        # Stack channels: (n_windows, pred_npts, 3)
        wf_slices = np.stack(wf_channels, axis=-1)
        
        del wf_channels
        
        return wf_slices
    
    def _batch_predict(self, wf_slices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run batch prediction on waveform slices.
        
        Args:
            wf_slices: Array of shape (n_windows, pred_npts, 3)
            
        Returns:
            Tuple of (predictions, masks) arrays
        """
        n_slices = len(wf_slices)
        n_batches = (n_slices + self.batch_size - 1) // self.batch_size
        
        all_predictions = []
        all_masks = []
        
        logger.debug(f"Running {n_slices} slices in {n_batches} batches")

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_slices)
            batch_data = wf_slices[start_idx:end_idx]

            predictions, masks = self._infer_batch(batch_data)
            
            all_predictions.append(predictions)
            all_masks.append(masks)
            
            del batch_data

        final_predictions = np.concatenate(all_predictions, axis=0)
        final_masks = np.concatenate(all_masks, axis=0)
        
        del all_predictions, all_masks
        
        logger.debug(f"Batch prediction completed: {final_predictions.shape}")
        return final_predictions, final_masks

    def _infer_batch(self, batch_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast batch inference wrapper.

        Uses direct model call (faster than model.predict for tight loops) and
        returns float32 numpy arrays.
        """
        if self._compiled_infer is not None:
            pred_result = self._compiled_infer(tf.convert_to_tensor(batch_data))
        else:
            pred_result = self.model(batch_data, training=False)

        if isinstance(pred_result, (list, tuple)) and len(pred_result) == 2:
            predictions, masks = pred_result
        else:
            predictions = pred_result
            masks = None

        if tf.is_tensor(predictions):
            predictions = predictions.numpy()
        else:
            predictions = np.asarray(predictions)
        predictions = predictions.astype(np.float32, copy=False)

        if masks is None:
            masks = np.ones_like(predictions[:, :, 0:1], dtype=np.float32)
        else:
            if tf.is_tensor(masks):
                masks = masks.numpy()
            else:
                masks = np.asarray(masks)
            masks = masks.astype(np.float32, copy=False)

        return predictions, masks

    def _build_window_starts(self, npts: int) -> np.ndarray:
        """
        Build sliding-window starts and ensure tail coverage by appending the
        last possible start when the stride does not land exactly at the end.
        """
        if npts <= self.pred_npts:
            return np.array([0], dtype=np.int64)
        starts = np.arange(
            0, npts - self.pred_npts + 1, self.pred_interval_pt, dtype=np.int64
        )
        last = npts - self.pred_npts
        if starts[-1] != last:
            starts = np.append(starts, last)
        return starts

    def _prepare_channel_window_views(self, wf: Stream) -> Tuple[List[np.ndarray], int]:
        """
        Build stride-based window views for each channel without materializing all
        normalized windows at once.
        """
        data_len = len(wf[0].data)
        n_windows = (data_len - self.pred_npts) // self.pred_interval_pt + 1
        if n_windows <= 0:
            return [], 0

        window_views = []
        for ch in range(3):
            channel_data = np.asarray(wf[ch].data, dtype=np.float32)
            view = np.lib.stride_tricks.sliding_window_view(channel_data, self.pred_npts)
            view = view[::self.pred_interval_pt]
            if len(view) > n_windows:
                view = view[:n_windows]
            window_views.append(view)

        return window_views, n_windows

    def _predict_streaming_accumulate(self, wf: Stream) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SeisBench-style inference:
        - window view
        - batch infer
        - immediate accumulation

        This avoids materializing full window tensor and full prediction tensor.
        """
        wf_np = np.stack(
            [np.asarray(wf[ch].data, dtype=np.float32) for ch in range(3)],
            axis=-1,
        )
        original_npts = len(wf_np)
        if original_npts <= 0:
            z = np.zeros(0, dtype=np.float32)
            return z.copy(), z.copy(), z.copy()

        if original_npts < self.pred_npts:
            wf_np = np.pad(
                wf_np,
                ((0, self.pred_npts - original_npts), (0, 0)),
                mode="constant",
            )

        total_samples = len(wf_np)
        starts = self._build_window_starts(total_samples)
        n_windows = len(starts)

        P_acc = np.zeros(total_samples, dtype=np.float32)
        S_acc = np.zeros(total_samples, dtype=np.float32)
        M_acc = np.zeros(total_samples, dtype=np.float32)
        W_acc = np.zeros(total_samples, dtype=np.float32)
        local = np.arange(self.pred_npts, dtype=np.int64)
        w = self.position_weights.astype(np.float32, copy=False)

        for b_start in range(0, n_windows, self.batch_size):
            b_end = min(b_start + self.batch_size, n_windows)
            b_starts = starts[b_start:b_end]
            idx = b_starts[:, None] + local[None, :]
            batch_data = wf_np[idx]

            # Match training-time normalization: demean/std per window/channel
            batch_data -= np.mean(batch_data, axis=1, keepdims=True)
            batch_std = np.std(batch_data, axis=1, keepdims=True)
            batch_std[batch_std < 1e-10] = 1.0
            batch_data /= batch_std

            predictions, masks = self._infer_batch(batch_data)

            if self.accumulation_mode == "addat":
                np.add.at(P_acc, idx, predictions[:, :, 0] * w[None, :])
                np.add.at(S_acc, idx, predictions[:, :, 1] * w[None, :])
                np.add.at(M_acc, idx, masks[:, :, 0] * w[None, :])
                np.add.at(W_acc, idx, w[None, :])
            else:
                for i, st in enumerate(b_starts):
                    ed = st + self.pred_npts
                    P_acc[st:ed] += predictions[i, :, 0] * w
                    S_acc[st:ed] += predictions[i, :, 1] * w
                    M_acc[st:ed] += masks[i, :, 0] * w
                    W_acc[st:ed] += w

        W_acc = np.maximum(W_acc, 1e-8)
        out_p = P_acc / W_acc
        out_s = S_acc / W_acc
        out_m = M_acc / W_acc
        return out_p[:original_npts], out_s[:original_npts], out_m[:original_npts]
    
    def _accumulate_predictions(self, 
                                predictions: np.ndarray, 
                                masks: np.ndarray,
                                total_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Accumulate sliding window predictions using weighted averaging.
        
        This is a clean implementation that directly accumulates predictions
        into an output array of the correct size.
        
        Args:
            predictions: Array of shape (n_windows, pred_npts, 2) for P and S
            masks: Array of shape (n_windows, pred_npts, 1)
            total_samples: Total length of output (padded waveform length)
            
        Returns:
            Tuple of (P_pred, S_pred, M_pred) arrays
        """
        # Pre-allocate accumulation arrays
        P_acc = np.zeros(total_samples, dtype=np.float32)
        S_acc = np.zeros(total_samples, dtype=np.float32)
        M_acc = np.zeros(total_samples, dtype=np.float32)
        W_acc = np.zeros(total_samples, dtype=np.float32)
        
        n_windows = len(predictions)
        logger.debug(f"Accumulating {n_windows} windows into {total_samples} samples")
        
        for i in range(n_windows):
            start_pos = i * self.pred_interval_pt
            end_pos = start_pos + self.pred_npts
            
            # Boundary check
            if end_pos > total_samples:
                end_pos = total_samples
                actual_len = end_pos - start_pos
                if actual_len <= 0:
                    continue
                weights = self.position_weights[:actual_len]
                pp = predictions[i, :actual_len, 0]
                ss = predictions[i, :actual_len, 1]
                mm = masks[i, :actual_len, 0]
            else:
                weights = self.position_weights
                pp = predictions[i, :, 0]
                ss = predictions[i, :, 1]
                mm = masks[i, :, 0]
            
            # Accumulate with weights
            P_acc[start_pos:end_pos] += pp * weights
            S_acc[start_pos:end_pos] += ss * weights
            M_acc[start_pos:end_pos] += mm * weights
            W_acc[start_pos:end_pos] += weights
        
        # Normalize by weights
        W_acc = np.maximum(W_acc, 1e-8)
        P_pred = P_acc / W_acc
        S_pred = S_acc / W_acc
        M_pred = M_acc / W_acc
        
        del P_acc, S_acc, M_acc, W_acc
        
        return P_pred, S_pred, M_pred
    
    def _single_entry_predict_batch(
        self,
        wf: Stream,
        start_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run batched direct model inference on fixed-length windows.
        """
        n_win = len(start_indices)
        if n_win == 0:
            empty = np.zeros((0, self.pred_npts), dtype=np.float32)
            return empty, empty, empty

        tr_data = [tr.data.astype(np.float32, copy=False) for tr in wf[:3]]
        npts = len(tr_data[0])
        batch = np.zeros((n_win, self.pred_npts, 3), dtype=np.float32)

        for i, st_idx in enumerate(start_indices):
            win_st = max(0, int(st_idx))
            win_ed = win_st + self.pred_npts
            for ch in range(3):
                seg = tr_data[ch][win_st:min(win_ed, npts)]
                seg_len = len(seg)
                if seg_len > 0:
                    batch[i, :seg_len, ch] = seg

        # Match training-time standardization: demean/std per window per channel.
        batch -= np.mean(batch, axis=1, keepdims=True)
        batch_std = np.std(batch, axis=1, keepdims=True)
        batch_std[batch_std < 1e-10] = 1.0
        batch /= batch_std

        picks, masks = self._infer_batch(batch)

        p_batch = picks[:, :, 0].astype(np.float32, copy=False)
        s_batch = picks[:, :, 1].astype(np.float32, copy=False)
        m_batch = masks[:, :, 0].astype(np.float32, copy=False)
        return p_batch, s_batch, m_batch

    def _single_entry_predict(self, wf: Stream, start_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run one direct model inference on a fixed-length window starting at start_idx.
        """
        p_batch, s_batch, m_batch = self._single_entry_predict_batch(wf, [int(start_idx)])
        return p_batch[0], s_batch[0], m_batch[0]

    def _detect_events_from_mask(
        self,
        mask: np.ndarray,
        trigger_on: float = 0.30,
        trigger_off: float = 0.30,
        smooth_npts: int = 10,
    ) -> List[Tuple[int, int]]:
        """
        Detect event trigger segments [start, end) from mask probability array.
        """
        m = np.asarray(mask, dtype=np.float32)
        smooth_len = max(1, int(smooth_npts))
        m_s = smooth(m, smooth_len) if smooth_len > 1 else m
        trg = trigger_onset(m_s, trigger_on, trigger_off)
        return [(int(t[0]), int(t[1])) for t in trg]

    def _apply_single_entry_refinement(
        self,
        wf: Stream,
        base_p: np.ndarray,
        base_s: np.ndarray,
        base_m: np.ndarray,
        pre_trigger_sec: float = 5.0,
        trigger_on: float = 0.30,
        trigger_off: float = 0.30,
        smooth_sec: float = 0.10,
        keep_if_refined_mask_lower: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Detect events from mask and replace P/S/M in each trigger with single-entry output.

        Guard rule:
        If keep_if_refined_mask_lower is True, replacement is skipped when
        mean(refined_mask_segment) < mean(original_mask_segment).
        """
        refined_p = base_p.copy()
        refined_s = base_s.copy()
        refined_m = base_m.copy()

        fs = 1.0 / self.dt
        smooth_npts = max(1, int(round(float(smooth_sec) * fs)))
        events = self._detect_events_from_mask(
            base_m,
            trigger_on=trigger_on,
            trigger_off=trigger_off,
            smooth_npts=smooth_npts,
        )

        event_info = []
        pre_npts = max(0, int(round(float(pre_trigger_sec) * fs)))
        npts = len(base_p)
        replace_jobs = []

        for ev_id, (on_idx, off_idx) in enumerate(events):
            if off_idx <= on_idx:
                continue

            win_start = max(0, on_idx - pre_npts)
            rep_st = max(0, on_idx)
            rep_ed = min(npts, off_idx)
            local_st = rep_st - win_start
            local_ed = local_st + (rep_ed - rep_st)
            if local_ed <= local_st or rep_ed <= rep_st:
                continue
            replace_jobs.append(
                {
                    "event_id": ev_id,
                    "trigger_on_idx": int(on_idx),
                    "trigger_off_idx": int(off_idx),
                    "window_start_idx": int(win_start),
                    "replace_start_idx": int(rep_st),
                    "replace_end_idx": int(rep_ed),
                    "local_start_idx": int(local_st),
                    "local_end_idx": int(local_ed),
                }
            )

        if not replace_jobs:
            return refined_p, refined_s, refined_m, event_info

        # Batched single-entry inference (chunked by self.batch_size for memory safety).
        for b_start in range(0, len(replace_jobs), self.batch_size):
            b_end = min(b_start + self.batch_size, len(replace_jobs))
            jobs = replace_jobs[b_start:b_end]
            start_indices = [j["window_start_idx"] for j in jobs]
            p_batch, s_batch, m_batch = self._single_entry_predict_batch(wf, start_indices)

            for i, j in enumerate(jobs):
                rep_st = j["replace_start_idx"]
                rep_ed = j["replace_end_idx"]
                local_st = j["local_start_idx"]
                local_ed = j["local_end_idx"]
                old_seg_m = base_m[rep_st:rep_ed]
                new_seg_m = m_batch[i, local_st:local_ed]
                old_mean_m = float(np.mean(old_seg_m)) if len(old_seg_m) > 0 else 0.0
                new_mean_m = float(np.mean(new_seg_m)) if len(new_seg_m) > 0 else 0.0

                applied = True
                if keep_if_refined_mask_lower and new_mean_m < old_mean_m:
                    applied = False
                else:
                    refined_p[rep_st:rep_ed] = p_batch[i, local_st:local_ed]
                    refined_s[rep_st:rep_ed] = s_batch[i, local_st:local_ed]
                    refined_m[rep_st:rep_ed] = new_seg_m

                event_info.append(
                    {
                        "event_id": j["event_id"],
                        "trigger_on_idx": j["trigger_on_idx"],
                        "trigger_off_idx": j["trigger_off_idx"],
                        "window_start_idx": j["window_start_idx"],
                        "replace_start_idx": rep_st,
                        "replace_end_idx": rep_ed,
                        "applied": bool(applied),
                        "orig_mean_mask": old_mean_m,
                        "refined_mean_mask": new_mean_m,
                    }
                )

        return refined_p, refined_s, refined_m, event_info

    def predict(
        self,
        wf: Stream,
        postprocess: bool = False,
        use_single_entry_refined: bool = False,
        pre_trigger_sec: float = 5.0,
        trigger_on: float = 0.30,
        trigger_off: float = 0.30,
        smooth_sec: float = 0.10,
        keep_if_refined_mask_lower: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main prediction function with proper time alignment.
        
        Strategy:
        - If waveform <= model receptive field: Direct single-window prediction
        - If waveform > model receptive field: 
          1. Pad waveform with spectrum-matched noise (pred_npts at each end)
          2. Run sliding window accumulation
          3. Truncate padding from predictions
        
        Args:
            wf: Input waveform stream (3-component)
            postprocess: Whether to apply postprocessing
            use_single_entry_refined: If True, detect triggers from mask and
                replace triggered segments by single-entry inference results.
            pre_trigger_sec: Single-entry window starts this many seconds before trigger-on.
            trigger_on: Trigger-on threshold for mask detection.
            trigger_off: Trigger-off threshold for mask detection.
            smooth_sec: Smoothing duration (seconds) before triggering.
            keep_if_refined_mask_lower: If True, keep original segment when
                refined segment has lower mean mask probability.
            
        Returns:
            Tuple of (P_predictions, S_predictions, Mask_predictions)
        """
        original_npts = len(wf[0].data)
        
        # Case 1: Short waveform - direct prediction (no sliding window)
        # Use direct prediction if waveform fits within model receptive field (+ 1 second tolerance)
        if original_npts <= self.pred_npts + 100:
            logger.debug(f"Short waveform ({original_npts} samples): using direct prediction")
            
            # Simply pad with zeros at the end to reach model input size
            _wf = wf.copy()
            for tr in _wf:
                data = tr.data.astype(np.float32)
                if len(data) < self.pred_npts:
                    # Pad with zeros at the end
                    data = np.pad(data, (0, self.pred_npts - len(data)), mode='constant', constant_values=0)
                else:
                    # Truncate if slightly longer
                    data = data[:self.pred_npts]
                tr.data = data
            
            _wf = stream_standardize(_wf, data_length=self.pred_npts)
            
            # Single prediction
            batch_data = np.stack([W.data for W in _wf], axis=-1)[np.newaxis, ...]
            picks, masks = self._infer_batch(batch_data)
            
            # Extract predictions - truncate or pad to match original length
            if original_npts <= self.pred_npts:
                array_P = picks[0, :original_npts, 0]
                array_S = picks[0, :original_npts, 1]
                array_M = masks[0, :original_npts, 0]
            else:
                # Pad with zeros for samples beyond pred_npts
                extra_samples = original_npts - self.pred_npts
                array_P = np.concatenate([picks[0, :, 0], np.zeros(extra_samples, dtype=np.float32)])
                array_S = np.concatenate([picks[0, :, 1], np.zeros(extra_samples, dtype=np.float32)])
                array_M = np.concatenate([masks[0, :, 0], np.zeros(extra_samples, dtype=np.float32)])
            
            del batch_data, picks, masks, _wf
        
        # Case 2: Long waveform - sliding window with noise padding
        else:
            logger.debug(f"Long waveform ({original_npts} samples): using sliding window")
            
            # Pad with spectrum-matched noise at both ends
            pad_npts = self.pred_npts
            wf_padded = self._pad_stream_with_noise(wf, pad_npts)
            padded_len = len(wf_padded[0].data)
            
            logger.debug(f"Padded: {original_npts} -> {padded_len} samples (pad={pad_npts})")
            
            # Streaming inference and accumulation (lower memory, less Python overhead).
            P_padded, S_padded, M_padded = self._predict_streaming_accumulate(wf_padded)

            del wf_padded
            
            # Truncate padding: keep only [pad_npts : pad_npts + original_npts]
            array_P = P_padded[pad_npts:pad_npts + original_npts]
            array_S = S_padded[pad_npts:pad_npts + original_npts]
            array_M = M_padded[pad_npts:pad_npts + original_npts]
            
            del P_padded, S_padded, M_padded
        
        # Zero any sample where P, S, or M is NaN/Inf so find_peaks /
        # trigger_onset downstream never see invalid floats.
        invalid_mask = (
            np.isnan(array_P) | np.isinf(array_P)
            | np.isnan(array_S) | np.isinf(array_S)
            | np.isnan(array_M) | np.isinf(array_M)
        )
        array_P[invalid_mask] = 0.0
        array_S[invalid_mask] = 0.0
        array_M[invalid_mask] = 0.0

        if use_single_entry_refined:
            array_P, array_S, array_M, _ = self._apply_single_entry_refinement(
                wf=wf,
                base_p=array_P,
                base_s=array_S,
                base_m=array_M,
                pre_trigger_sec=pre_trigger_sec,
                trigger_on=trigger_on,
                trigger_off=trigger_off,
                smooth_sec=smooth_sec,
                keep_if_refined_mask_lower=keep_if_refined_mask_lower,
            )
        
        # Apply postprocessing if requested
        if postprocess and hasattr(self, 'postprocess_config') and self.postprocess_config:
            array_P, array_S, array_M = pred_postprocess(
                array_P, array_S, array_M,
                dt=self.dt,
                **self.postprocess_config,
            )
        
        return array_P, array_S, array_M
    
    def annotate_stream(
        self,
        wf: Stream,
        postprocess: bool = False,
        use_single_entry_refined: bool = False,
        pre_trigger_sec: float = 5.0,
        trigger_on: float = 0.30,
        trigger_off: float = 0.30,
        smooth_sec: float = 0.10,
        keep_if_refined_mask_lower: bool = True,
    ) -> Tuple[Stream, Stream, Stream]:
        """
        Annotate stream with REDPAN predictions, creating ObsPy streams.
        
        Args:
            wf: Input ObsPy stream (3-component seismic data)
            postprocess: Whether to apply postprocessing
            use_single_entry_refined: If True, apply single-entry refinement.
            
        Returns:
            Tuple of (P_stream, S_stream, M_stream) as ObsPy Stream objects
        """
        # Get predictions
        array_P, array_S, array_M = self.predict(
            wf,
            postprocess=postprocess,
            use_single_entry_refined=use_single_entry_refined,
            pre_trigger_sec=pre_trigger_sec,
            trigger_on=trigger_on,
            trigger_off=trigger_off,
            smooth_sec=smooth_sec,
            keep_if_refined_mask_lower=keep_if_refined_mask_lower,
        )
        
        # Create output streams
        P_stream, S_stream, M_stream = Stream(), Stream(), Stream()
        
        W_data = [array_P, array_S, array_M]
        W_chn = ["redpan_P", "redpan_S", "redpan_mask"]
        W_sac = [P_stream, S_stream, M_stream]
        
        for k in range(3):
            W = deepcopy(wf[0])
            W.data = W_data[k]
            W.stats.channel = W_chn[k]
            W_sac[k].append(W)
        
        # Slice to match original time window (safety check)
        P_stream = P_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        S_stream = S_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        M_stream = M_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        
        del wf, array_P, array_S, array_M, W_data, W_sac
        
        logger.debug(f"Created annotated streams: P={len(P_stream[0].data)}, "
                    f"S={len(S_stream[0].data)}, M={len(M_stream[0].data)} samples")
        
        return P_stream, S_stream, M_stream


class REDPANDualPass(REDPAN):
    """
    Dual-pass RED-PAN continuous predictor.

    Pass A (fast): original sampling rate (typically 100 Hz) for precise picks.
    Pass B (slow): downsampled sampling rate (typically 50 Hz) with same pred_npts
    to enlarge temporal context; predictions are upsampled back to fast rate.

    Merge policy:
    - Keep Pass A predictions where Pass A mask already detects event.
    - Fill only missing portions of long Pass B detections.
    """

    def __init__(
        self,
        model,
        pred_npts: int = 9000,
        fast_dt: float = 0.01,
        slow_dt: float = 0.02,
        pred_interval_sec: float = 10.0,
        batch_size: int = 32,
        fast_mask_threshold: float = 0.30,
        slow_mask_threshold: float = 0.25,
        long_event_min_sec: float = 8.0,
        fast_coverage_threshold: float = 0.70,
    ):
        super().__init__(
            model=model,
            pred_npts=pred_npts,
            dt=fast_dt,
            pred_interval_sec=pred_interval_sec,
            batch_size=batch_size,
        )
        self.fast_dt = float(fast_dt)
        self.slow_dt = float(slow_dt)
        self.fast_fs = 1.0 / self.fast_dt
        self.slow_fs = 1.0 / self.slow_dt
        self.fast_mask_threshold = float(fast_mask_threshold)
        self.slow_mask_threshold = float(slow_mask_threshold)
        self.long_event_min_sec = float(long_event_min_sec)
        self.fast_coverage_threshold = float(fast_coverage_threshold)

        self._slow_predictor = REDPAN(
            model=model,
            pred_npts=pred_npts,
            dt=slow_dt,
            pred_interval_sec=pred_interval_sec,
            batch_size=batch_size,
        )

    @staticmethod
    def _find_true_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
        """Return contiguous [start, end) segments where mask is True."""
        m = np.asarray(mask, dtype=bool)
        padded = np.pad(m.astype(np.int8), (1, 1), mode="constant", constant_values=0)
        dm = np.diff(padded)
        starts = np.where(dm == 1)[0]
        ends = np.where(dm == -1)[0]
        return list(zip(starts.tolist(), ends.tolist()))

    @staticmethod
    def _resample_ratio(source_rate: float, target_rate: float) -> Tuple[int, int]:
        """Compute stable integer (up, down) ratio for resample_poly."""
        frac = Fraction(target_rate / source_rate).limit_denominator(1000)
        return frac.numerator, frac.denominator

    def _resample_stream(self, wf: Stream, target_dt: float) -> Stream:
        """
        Resample stream to target_dt with polyphase filtering.
        """
        wf_out = wf.copy()
        source_rate = float(wf[0].stats.sampling_rate)
        target_rate = 1.0 / target_dt
        up, down = self._resample_ratio(source_rate, target_rate)
        expected_npts = int(round(len(wf[0].data) * up / down))

        for tr in wf_out:
            data = tr.data.astype(np.float32, copy=False)
            rs = resample_poly(data, up=up, down=down).astype(np.float32, copy=False)
            if len(rs) > expected_npts:
                rs = rs[:expected_npts]
            elif len(rs) < expected_npts:
                pad_n = expected_npts - len(rs)
                rs = np.pad(rs, (0, pad_n), mode="edge")
            tr.data = rs
            tr.stats.sampling_rate = target_rate
            tr.stats.delta = target_dt
        return wf_out

    @staticmethod
    def _upsample_to_length(arr: np.ndarray, target_npts: int) -> np.ndarray:
        """
        Linearly upsample/downsample prediction array to exact target length.
        """
        arr = np.asarray(arr, dtype=np.float32)
        if len(arr) == target_npts:
            return arr
        src_x = np.linspace(0.0, target_npts - 1.0, num=len(arr), dtype=np.float64)
        dst_x = np.arange(target_npts, dtype=np.float64)
        return np.interp(dst_x, src_x, arr).astype(np.float32)

    def _build_fill_mask(self, fast_m: np.ndarray, slow_m_up: np.ndarray) -> np.ndarray:
        """
        Build mask of samples to fill from slow pass.
        """
        fast_event = fast_m >= self.fast_mask_threshold
        slow_event = slow_m_up >= self.slow_mask_threshold

        fill_mask = np.zeros_like(fast_event, dtype=bool)
        min_len = max(1, int(round(self.long_event_min_sec * self.fast_fs)))

        for seg_start, seg_end in self._find_true_segments(slow_event):
            seg_len = seg_end - seg_start
            if seg_len < min_len:
                continue
            fast_cov = float(np.mean(fast_event[seg_start:seg_end]))
            if fast_cov < self.fast_coverage_threshold:
                missing = ~fast_event[seg_start:seg_end]
                fill_mask[seg_start:seg_end] = missing

        return fill_mask

    def _run_dual_pass(self, wf: Stream):
        """
        Run fast + slow passes and merge predictions.
        """
        fast_p, fast_s, fast_m = REDPAN.predict(self, wf, postprocess=False)

        wf_slow = self._resample_stream(wf, target_dt=self.slow_dt)
        slow_p, slow_s, slow_m = self._slow_predictor.predict(wf_slow, postprocess=False)

        target_npts = len(fast_p)
        slow_p_up = self._upsample_to_length(slow_p, target_npts)
        slow_s_up = self._upsample_to_length(slow_s, target_npts)
        slow_m_up = self._upsample_to_length(slow_m, target_npts)

        fill_mask = self._build_fill_mask(fast_m, slow_m_up)

        merged_p = fast_p.copy()
        merged_s = fast_s.copy()
        merged_m = fast_m.copy()

        merged_p[fill_mask] = np.maximum(merged_p[fill_mask], slow_p_up[fill_mask])
        merged_s[fill_mask] = np.maximum(merged_s[fill_mask], slow_s_up[fill_mask])
        merged_m[fill_mask] = np.maximum(merged_m[fill_mask], slow_m_up[fill_mask])

        return (
            merged_p,
            merged_s,
            merged_m,
            fast_p,
            fast_s,
            fast_m,
            slow_p_up,
            slow_s_up,
            slow_m_up,
            fill_mask,
        )

    def predict(
        self,
        wf: Stream,
        postprocess: bool = False,
        use_single_entry_refined: bool = False,
        pre_trigger_sec: float = 5.0,
        trigger_on: float = 0.30,
        trigger_off: float = 0.30,
        smooth_sec: float = 0.10,
        keep_if_refined_mask_lower: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return merged dual-pass predictions.

        The signature matches REDPAN.predict (Liskov substitutable). The
        legacy kwarg name `use_trial_refined` is no longer accepted —
        update callers if you were using it (no external callers existed
        at the time of the rename).
        """
        merged_p, merged_s, merged_m, *_ = self._run_dual_pass(wf)

        if use_single_entry_refined:
            merged_p, merged_s, merged_m, _ = self._apply_trial_single_entry_refinement(
                wf=wf,
                base_p=merged_p,
                base_s=merged_s,
                base_m=merged_m,
                pre_trigger_sec=pre_trigger_sec,
                trigger_on=trigger_on,
                trigger_off=trigger_off,
                smooth_sec=smooth_sec,
                keep_if_refined_mask_lower=keep_if_refined_mask_lower,
            )

        if postprocess and hasattr(self, "postprocess_config") and self.postprocess_config:
            merged_p, merged_s, merged_m = pred_postprocess(
                merged_p,
                merged_s,
                merged_m,
                dt=self.fast_dt,
                **self.postprocess_config,
            )

        return merged_p, merged_s, merged_m

    def predict_with_details(self, wf: Stream, postprocess: bool = False) -> dict:
        """
        Return merged predictions plus both pass outputs and merge mask.
        """
        (
            merged_p,
            merged_s,
            merged_m,
            fast_p,
            fast_s,
            fast_m,
            slow_p_up,
            slow_s_up,
            slow_m_up,
            fill_mask,
        ) = self._run_dual_pass(wf)

        if postprocess and hasattr(self, "postprocess_config") and self.postprocess_config:
            merged_p, merged_s, merged_m = pred_postprocess(
                merged_p,
                merged_s,
                merged_m,
                dt=self.fast_dt,
                **self.postprocess_config,
            )

        return {
            "merged": {"P": merged_p, "S": merged_s, "M": merged_m},
            "pass_a_100hz": {"P": fast_p, "S": fast_s, "M": fast_m},
            "pass_b_50hz_upsampled": {"P": slow_p_up, "S": slow_s_up, "M": slow_m_up},
            "fill_mask": fill_mask.astype(np.uint8),
        }

    def _single_entry_predict(self, wf: Stream, start_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run one direct model inference on a fixed-length window starting at start_idx.
        """
        return REDPAN._single_entry_predict(self, wf, start_idx)

    def _detect_events_from_mask(
        self,
        mask: np.ndarray,
        trigger_on: float = 0.30,
        trigger_off: float = 0.30,
        smooth_npts: int = 10,
    ) -> List[Tuple[int, int]]:
        """
        Detect event trigger segments [start, end) from mask probability array.
        """
        return REDPAN._detect_events_from_mask(
            self,
            mask=mask,
            trigger_on=trigger_on,
            trigger_off=trigger_off,
            smooth_npts=smooth_npts,
        )

    def _apply_trial_single_entry_refinement(
        self,
        wf: Stream,
        base_p: np.ndarray,
        base_s: np.ndarray,
        base_m: np.ndarray,
        pre_trigger_sec: float = 5.0,
        trigger_on: float = 0.30,
        trigger_off: float = 0.30,
        smooth_sec: float = 0.10,
        keep_if_refined_mask_lower: bool = True,
    ):
        """
        Trial refinement:
        1) detect events on base mask,
        2) run one single-entry inference per trigger (start 5s before on),
        3) replace P/S/M only over trigger segment with single-entry results.
        """
        return REDPAN._apply_single_entry_refinement(
            self,
            wf=wf,
            base_p=base_p,
            base_s=base_s,
            base_m=base_m,
            pre_trigger_sec=pre_trigger_sec,
            trigger_on=trigger_on,
            trigger_off=trigger_off,
            smooth_sec=smooth_sec,
            keep_if_refined_mask_lower=keep_if_refined_mask_lower,
        )

    def predict_with_trial_details(
        self,
        wf: Stream,
        postprocess: bool = False,
        pre_trigger_sec: float = 5.0,
        trigger_on: float = 0.30,
        trigger_off: float = 0.30,
        smooth_sec: float = 0.10,
    ) -> dict:
        """
        Return dual-pass outputs plus trial single-entry refinement output.
        """
        details = self.predict_with_details(wf, postprocess=postprocess)
        merged = details["merged"]

        ref_p, ref_s, ref_m, events = self._apply_trial_single_entry_refinement(
            wf=wf,
            base_p=merged["P"],
            base_s=merged["S"],
            base_m=merged["M"],
            pre_trigger_sec=pre_trigger_sec,
            trigger_on=trigger_on,
            trigger_off=trigger_off,
            smooth_sec=smooth_sec,
        )

        details["trial_single_entry_refined"] = {"P": ref_p, "S": ref_s, "M": ref_m}
        details["trial_events"] = events
        return details

    def annotate_stream(
        self,
        wf: Stream,
        postprocess: bool = False,
        use_single_entry_refined: bool = False,
        pre_trigger_sec: float = 5.0,
        trigger_on: float = 0.30,
        trigger_off: float = 0.30,
        smooth_sec: float = 0.10,
        keep_if_refined_mask_lower: bool = True,
    ) -> Tuple[Stream, Stream, Stream]:
        """
        Annotate stream with dual-pass predictions, matching REDPAN output style.

        Signature matches REDPAN.annotate_stream (Liskov substitutable).

        Returns:
            Tuple of (P_stream, S_stream, M_stream), each as ObsPy Stream.
        """
        array_P, array_S, array_M = self.predict(
            wf=wf,
            postprocess=postprocess,
            use_single_entry_refined=use_single_entry_refined,
            pre_trigger_sec=pre_trigger_sec,
            trigger_on=trigger_on,
            trigger_off=trigger_off,
            smooth_sec=smooth_sec,
            keep_if_refined_mask_lower=keep_if_refined_mask_lower,
        )

        P_stream, S_stream, M_stream = Stream(), Stream(), Stream()
        W_data = [array_P, array_S, array_M]
        W_chn = ["redpan_P", "redpan_S", "redpan_mask"]
        W_sac = [P_stream, S_stream, M_stream]

        for k in range(3):
            W = deepcopy(wf[0])
            W.data = W_data[k]
            W.stats.channel = W_chn[k]
            W_sac[k].append(W)

        P_stream = P_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        S_stream = S_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)
        M_stream = M_stream.slice(wf[0].stats.starttime, wf[0].stats.endtime)

        return P_stream, S_stream, M_stream


# Legacy compatibility classes and functions below
# =================================================

class PhasePicker:
    """
    Legacy PhasePicker class for backward compatibility.
    """
    
    def __init__(
        self,
        model=None,
        dt=0.01,
        pred_npts=3000,
        pred_interval_sec=10,
        STMF_max_sec=1200,
        postprocess_config=None,
    ):
        self.model = model
        self.dt = dt
        self.pred_npts = pred_npts
        self.pred_interval_sec = pred_interval_sec
        self.STMF_max_sec = STMF_max_sec
        if postprocess_config is None:
            postprocess_config = {
                "mask_trigger": [0.1, 0.1],
                "mask_len_thre": 0.5,
                "mask_err_win": 0.5,
                "detection_threshold": 0.3,
                "P_threshold": 0.1,
                "S_threshold": 0.1,
            }
        self.postprocess_config = postprocess_config

        if model is None:
            raise AssertionError("The Phase picker model should be defined!")
        
        # Use new REDPAN implementation internally
        self._picker = REDPAN(
            model=model,
            pred_npts=pred_npts,
            dt=dt,
            pred_interval_sec=pred_interval_sec,
            batch_size=32
        )
        self._picker.postprocess_config = postprocess_config

    def predict(self, wf=None, postprocess=False):
        if wf is None:
            raise AssertionError("Obspy.stream should be assigned as `wf=?`!")
        return self._picker.predict(wf, postprocess=postprocess)
    
    def annotate_stream(self, wf, STMF_max_sec=None, postprocess=False):
        return self._picker.annotate_stream(wf, postprocess=postprocess)


def conti_standard_wf_fast(wf, pred_npts, pred_interval_sec, dt, pad_zeros=True):
    """
    Legacy waveform preparation function - kept for compatibility.
    """
    from copy import deepcopy
    
    raw_n = len(wf[0].data)
    pred_rate = int(pred_interval_sec / dt)
    full_len = int(pred_npts + pred_rate * np.ceil(raw_n - pred_npts) / pred_rate)
    n_marching_win = int((full_len - pred_npts) / pred_rate) + 1

    wf = sac_len_complement(wf.copy(), max_length=full_len)
    pad_bef = pred_npts - pred_rate
    pad_aft = pred_npts
    for W in wf:
        W.data = np.insert(W.data, 0, np.zeros(pad_bef))
        W.data = np.insert(W.data, len(W.data), np.zeros(pad_aft))

    wf_n = []
    for w in range(3):
        wf_ = np.array([
            deepcopy(wf[w].data[pred_rate * i : pred_rate * i + pred_npts])
            for i in range(n_marching_win)
        ])
        wf_dm = np.array([i - np.mean(i) for i in wf_])
        wf_std = np.array([np.std(i) for i in wf_dm])
        wf_std[wf_std == 0] = 1
        wf_norm = np.array([wf_dm[i] / wf_std[i] for i in range(len(wf_dm))])
        wf_n.append(wf_norm)

    wf_slices = np.stack([wf_n[0], wf_n[1], wf_n[2]], -1)
    return np.array(wf_slices), pad_bef, pad_aft


def pred_MedianFilter(preds, masks, wf_npts, dt, pred_npts, pred_interval_sec, pad_bef, pad_aft):
    """
    Legacy median filter function - kept for compatibility.
    """
    wf_n = wf_npts + (pad_bef + pad_aft)
    pred_array_P = [[] for _ in range(wf_n)]
    pred_array_S = [[] for _ in range(wf_n)]
    pred_array_mask = [[] for _ in range(wf_n)]
    pred_interval_pt = int(round(pred_interval_sec / dt))

    init_pt = 0
    for i in range(len(preds)):
        pp = np.array_split(preds[i].T[0], pred_npts)
        ss = np.array_split(preds[i].T[1], pred_npts)
        mm = np.array_split(masks[i].T[0], pred_npts)
        j = 0
        for p, s, m in zip(pp, ss, mm):
            pred_array_P[init_pt + j].append(p)
            pred_array_S[init_pt + j].append(s)
            pred_array_mask[init_pt + j].append(m)
            j += 1
        init_pt += pred_interval_pt

    pred_array_P = np.array(pred_array_P, dtype="object")
    pred_array_S = np.array(pred_array_S, dtype="object")
    pred_array_mask = np.array(pred_array_mask, dtype="object")
    
    lenP = np.array([len(p) for p in pred_array_P])
    nums = np.unique(lenP)
    array_P_med = np.zeros(wf_n)
    array_S_med = np.zeros(wf_n)
    array_M_med = np.zeros(wf_n)
    
    for k in nums:
        num_idx = np.where(lenP == k)[0]
        array_P_med[num_idx] = np.median(np.hstack(np.take(pred_array_P, num_idx)), axis=0)
        array_S_med[num_idx] = np.median(np.hstack(np.take(pred_array_S, num_idx)), axis=0)
        array_M_med[num_idx] = np.median(np.hstack(np.take(pred_array_mask, num_idx)), axis=0)
    
    del pred_array_P, pred_array_S, pred_array_mask

    array_P_med = array_P_med[pad_bef:-pad_aft]
    array_S_med = array_S_med[pad_bef:-pad_aft]
    array_M_med = array_M_med[pad_bef:-pad_aft]
    assert len(array_P_med) == wf_npts

    return array_P_med, array_S_med, array_M_med
