#!/usr/bin/env python3

# Standard library
import os
import json
from pathlib import Path
from argparse import Namespace

# Data processing
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from load_config import load_inference_config

# Audio processing (for Step 6)
import librosa
from scipy import signal

# SyncTalk modules
from nerf_triplane.network import NeRFNetwork, AudioEncoder
from nerf_triplane.utils import Trainer, get_audio_features, get_rays, get_bg_coords
from nerf_triplane.provider import nerf_matrix_to_ngp


class StreamingInference:
    def __init__(
        self,
        data_path,
        workspace,
        checkpoint="latest",
        portrait=False,
        torso=False,
        W=512,
        H=512,
        fps=25,
        config_path="config/inference_config.yaml",
        debug=False,
    ):
        """
        Initialize streaming inference for SyncTalk.

        Args:
            data_path: Path to training data directory (e.g., 'data/May')
            workspace: Path to trained model workspace (e.g., 'model/trial_may')
            checkpoint: Checkpoint name or 'latest'
            portrait: Enable portrait mode
            torso: Enable torso mode
            W, H: Output dimensions
            fps: Frames per second
        """

        self.config_path = config_path
        self.data_path = data_path
        self.workspace = workspace
        self.checkpoint = checkpoint
        self.portrait = portrait
        self.torso = torso
        self.W = W
        self.H = H
        self.fps = fps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # To be initialized in subsequent steps
        self.opt = None
        self.model = None
        self.trainer = None
        self.audio_encoder = None

        # Scene data (to be loaded)
        self.intrinsics = None
        self.poses = None
        self.bg_img = None
        self.eye_areas = None

        # Audio features (will be set when processing audio)
        self.current_audio_features = None

        # DEBUG mode
        self.debug = debug

        self._load_model()
        self._load_scene_data()
        self._load_audio_encoder()

    def _load_model(self):
        """Load NeRF model and checkpoint via Trainer."""
        print("Loading model...")

        # Create opt configuration using load_config module
        self.opt = load_inference_config(
            data_path=self.data_path,
            workspace=self.workspace,
            checkpoint=self.checkpoint,
            portrait=self.portrait,
            torso=self.torso,
            W=self.W,
            H=self.H,
            fps=self.fps,
            config_path=getattr(self, "config_path", "config/inference_config.yaml"),
        )

        # Create model
        self.model = NeRFNetwork(self.opt)

        # Create trainer - automatically loads checkpoint
        criterion = torch.nn.L1Loss(reduction="none")
        self.trainer = Trainer(
            "ngp",
            self.opt,
            self.model,
            device=self.device,
            workspace=self.workspace,
            criterion=criterion,
            fp16=self.opt.fp16,
            metrics=[],
            use_checkpoint=self.opt.ckpt,
        )

        print(f"✓ Model loaded from {self.workspace}")

        # DEBUG: Check if density grids are loaded
        print(f"  Density grid loaded: {hasattr(self.model, 'mean_density')}")

        if self.debug:
            if hasattr(self.model, "mean_density"):
                print(f"  Mean density shape: {self.model.mean_density}")

        self.model.eval()
        print("✓ Model set to eval mode")

    def _load_scene_data(self):
        """
        Load scene data from the training data directory.
        This includes camera intrinsics, poses, background image, and blendshapes.

        References:
            provider.py lines 125-128 (transforms loading)
            provider.py lines 256-258 (blendshape loading)
            provider.py lines 387-403 (background loading)
            provider.py lines 492-509 (intrinsics loading)
        """
        print("Loading scene data...")

        # 1. Load transforms.json
        # TODO: Implement this
        transform_path = os.path.join(self.data_path, "transforms_train.json")
        if not os.path.exists(transform_path):
            raise FileNotFoundError(
                f"transforms_val.json not found at {transform_path}"
            )

        with open(transform_path, "r") as f:
            transform = json.load(f)

        # 2. Extract camera intrinsics
        # TODO: Implement this
        if "h" in transform and "w" in transform:
            self.H = int(transform["h"])
            self.W = int(transform["w"])
        else:
            self.H = int(transform["cy"]) * 2
            self.W = int(transform["cx"]) * 2

        if "focal_len" in transform:
            fl_x = fl_y = transform["focal_len"]
        else:
            fl_x = transform.get("fl_x", transform.get("fl_y"))
            fl_y = transform.get("fl_y", transform.get("fl_x"))

        # Get principal point
        cx = transform.get("cx", self.W / 2)
        cy = transform.get("cy", self.H / 2)

        # Store intrinsics [fl_x, fl_y, cx, cy]
        self.intrinsics = np.array([fl_x, fl_y, cx, cy], dtype=np.float32)

        # 3. Extract and process camera poses
        # TODO: Implement this
        poses = []
        img_ids = []

        for frame in transform["frames"]:
            pose = np.array(frame["transform_matrix"], dtype=np.float32)  # [4, 4]
            # Convert to NGP format
            pose = nerf_matrix_to_ngp(
                pose, scale=self.opt.scale, offset=self.opt.offset
            )
            poses.append(pose)

            img_ids.append(frame["img_id"])

        # Stack into array [N, 4, 4]
        self.poses = np.stack(poses, axis=0)
        self.num_poses = len(poses)
        self.img_ids = img_ids

        print(f"  Loaded {self.num_poses} camera poses")

        # 4. Load background image
        # TODO: Implement this
        bg_path = os.path.join(self.data_path, "bc.jpg")

        if not os.path.exists(bg_path):
            print(f"  [WARN] Background image not found, using white background")
            bg_img = np.ones((self.H, self.W, 3), dtype=np.float32)
        else:
            bg_img = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
            bg_img = cv2.resize(bg_img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = bg_img.astype(np.float32) / 255.0

        # Convert to tensor and move to device
        self.bg_img = torch.from_numpy(bg_img).to(self.device)

        # 5. Load blendshape data (eye movements)
        # TODO: Implement this

        # Load blendshapes for eye/face movements
        bs_path = os.path.join(self.data_path, "bs.npy")

        if not os.path.exists(bs_path):
            print(f"  [WARN] Blendshape file not found, using zeros")
            self.eye_areas = torch.zeros(self.num_poses, 7, dtype=torch.float32).to(
                self.device
            )
        else:
            # Load FULL blendshape array
            bs_full = np.load(bs_path)

            # Extract blendshapes based on bs_area config
            if self.opt.bs_area == "upper":
                bs_full = np.hstack((bs_full[:, 0:5], bs_full[:, 8:10]))
            elif self.opt.bs_area == "eye":
                bs_full = bs_full[:, 8:10]

            # Extract ONLY the frames we need (using img_id)
            # Reference: provider.py line 361
            bs_frames = []
            for img_id in img_ids:
                bs_frames.append(bs_full[img_id])  # ← Use img_id to index!

            bs_frames = np.stack(bs_frames, axis=0)  # [N, 7]

            # Convert to tensor
            self.eye_areas = torch.from_numpy(bs_frames.astype(np.float32)).to(
                self.device
            )

            print(f"  Loaded blendshapes: {self.eye_areas.shape}")
            print(
                f"  Poses: {self.poses.shape[0]}, Eye areas: {self.eye_areas.shape[0]}"
            )
            assert (
                self.poses.shape[0] == self.eye_areas.shape[0]
            ), "Poses and eye_areas must match!"

        # 6. If portrait mode, set up portrait data paths
        # TODO: Implement this

        # Set up portrait mode paths (we'll load images on-demand later)
        if self.portrait:
            self.ori_imgs_dir = os.path.join(self.data_path, "ori_imgs")
            self.parsing_dir = os.path.join(self.data_path, "parsing")

            if not os.path.exists(self.ori_imgs_dir):
                print(f"  [WARN] Portrait mode enabled but ori_imgs not found")
                self.portrait = False
            elif not os.path.exists(self.parsing_dir):
                print(f"  [WARN] Portrait mode enabled but parsing not found")
                self.portrait = False
            else:
                print(f"  Portrait mode enabled")

                print(f"✓ Scene data loaded")

        # Pre-compute background coordinates (static, doesn't change per frame)
        self.bg_coords = get_bg_coords(self.H, self.W, self.device)  # [1, H*W, 2]

        print(f"✓ Scene data loaded")

    def _load_audio_encoder(self):
        """
        Initialize the AVE (Audio-Visual Encoder) for processing audio.

        The AVE encoder converts mel-spectrogram windows to 512-dim feature vectors.

        References:
            provider.py lines 172-175 (AVE loading)

        Note: Only needed if asr_model == 'ave'. For other models (deepspeech, hubert),
                audio features must be pre-computed and provided as .npy files.
        """
        # Only load AVE if using ave model
        if self.opt.asr_model != "ave":
            print(
                f"  ASR model: {self.opt.asr_model} (requires pre-computed .npy features)"
            )
            self.audio_encoder = None
            return

        print("Loading AVE audio encoder...")

        # 1. Create AudioEncoder model
        # TODO: Implement this
        self.audio_encoder = AudioEncoder().to(self.device)

        # 2. Load checkpoint
        # TODO: Implement this
        ave_checkpoint_path = "./nerf_triplane/checkpoints/audio_visual_encoder.pth"

        if not os.path.exists(ave_checkpoint_path):
            raise FileNotFoundError(
                f"AVE checkpoint not found at {ave_checkpoint_path}\n"
                "Please download the audio_visual_encoder.pth checkpoint."
            )
        # Load checkpoint
        ckpt = torch.load(ave_checkpoint_path, map_location=self.device)

        # The checkpoint keys need 'audio_encoder.' prefix
        self.audio_encoder.load_state_dict(
            {f"audio_encoder.{k}": v for k, v in ckpt.items()}
        )

        # 3. Set to eval mode and freeze
        # TODO: Implement this
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # 4. Warmup (optional but recommended for consistent performance)
        # TODO: Implement this
        # Warmup: Run a few dummy forward passes for consistent timing
        # Input shape: [batch, 1, 80, 16] - mel spectrogram
        print("  Warming up encoder...")
        dummy_input = torch.randn(1, 1, 80, 16).to(self.device)

        with torch.no_grad():
            for _ in range(3):
                _ = self.audio_encoder(dummy_input)

        print(f"  AVE encoder ready (output dim: 512)")

        print("✓ AVE audio encoder loaded")

    def mirror_index(self, index):
        """
        Mirror index for pose cycling.

        When generating frames beyond the training video length, we cycle through
        poses in a ping-pong pattern to avoid abrupt transitions:
        - 1st pass: 0 -> N-1 (forward)
        - 2nd pass: N-1 -> 0 (backward)
        - 3rd pass: 0 -> N-1 (forward)
        - etc.

        Args:
            index: Frame index (can be >= number of training poses)

        Returns:
            Mapped pose index in range [0, N-1]

        Example:
            If we have 10 poses (0-9):
            - mirror_index(0) = 0
            - mirror_index(5) = 5
            - mirror_index(9) = 9
            - mirror_index(10) = 9  (start going backward)
            - mirror_index(15) = 4
            - mirror_index(19) = 0  (reached start)
            - mirror_index(20) = 0  (start going forward again)

        Reference:
            provider.py lines 515-522
        """
        size = self.num_poses  # or self.poses.shape[0]
        turn = index // size  # Which "pass" are we on?
        res = index % size  # Position within the current pass

        if turn % 2 == 0:
            # Even passes: forward (0 -> N-1)
            return res
        else:
            # Odd passes: backward (N-1 -> 0)
            return size - res - 1

    def _preemphasis(self, wav, k=0.97):
        """
        Apply pre-emphasis filter to amplify high frequencies.

        Reference: utils.py line 1584-1585
        """
        return signal.lfilter([1, -k], [1], wav)

    def _build_mel_basis(self):
        """
        Build mel filterbank (cached for efficiency).

        Reference: utils.py line 1605-1606
        """
        if not hasattr(self, "_mel_basis"):
            self._mel_basis = librosa.filters.mel(
                sr=16000,  # Sample rate
                n_fft=800,  # FFT size
                n_mels=80,  # Number of mel bands
                fmin=55,  # Min frequency
                fmax=7600,  # Max frequency
            )
        return self._mel_basis

    def _stft(self, wav):
        """
        Compute Short-Time Fourier Transform.

        Reference: utils.py line 1595-1596
        """
        return librosa.stft(y=wav, n_fft=800, hop_length=200, win_length=800)

    def _amp_to_db(self, x):
        """
        Convert amplitude to decibels.

        Reference: utils.py line 1609-1611
        """
        min_level = np.exp(-5 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _normalize(self, S):
        """
        Normalize spectrogram to [-4, 4] range.

        Reference: utils.py line 1614-1615
        """
        return np.clip((2 * 4.0) * ((S - -100) / (--100)) - 4.0, -4.0, 4.0)

    def wav_to_melspectrogram(self, wav):
        """
        Convert raw audio waveform to mel-spectrogram.

        Args:
            wav: Audio waveform (numpy array, 16kHz)

        Returns:
            Mel-spectrogram [T, 80] where T is number of time steps

        Reference: utils.py line 1588-1592
        """
        # 1. Apply pre-emphasis
        wav_preemph = self._preemphasis(wav, k=0.97)

        # 2. Compute STFT
        D = self._stft(wav_preemph)

        # 3. Convert to mel scale
        mel_basis = self._build_mel_basis()
        S = np.dot(mel_basis, np.abs(D))

        # 4. Convert to dB and normalize
        S = self._amp_to_db(S) - 20
        S = self._normalize(S)

        # 5. Transpose to [T, 80]
        return S.T

    def extract_ave_features(self, wav):
        """
        Extract AVE audio features from raw audio waveform.

        This processes the entire audio file and returns features for all frames.

        Args:
            wav: Audio waveform (numpy array, 16kHz)

        Returns:
            Audio features [N, 512] where N is number of frames

        Reference:
            utils.py lines 1618-1653 (AudDataset)
            provider.py lines 176-187 (feature extraction)
        """
        if self.audio_encoder is None:
            raise RuntimeError(
                "AVE encoder not loaded. Only works with asr_model='ave'"
            )

        # 1. Convert to mel-spectrogram
        mel = self.wav_to_melspectrogram(wav)  # [T, 80]

        # 2. Calculate number of output frames
        # Formula from AudDataset line 1623
        num_frames = int((mel.shape[0] - 16) / 80.0 * float(self.fps)) + 2

        # 3. Extract features for each frame
        features = []
        for idx in range(num_frames):
            # Calculate the 16-frame window for this frame
            # Formula from AudDataset lines 1633-1639
            start_idx = int(80.0 * (idx / float(self.fps)))
            end_idx = start_idx + 16

            # Handle edge case at the end
            if end_idx > mel.shape[0]:
                end_idx = mel.shape[0]
                start_idx = end_idx - 16

            # Crop mel window [16, 80]
            mel_window = mel[start_idx:end_idx, :]

            # Convert to tensor [1, 1, 80, 16]
            mel_tensor = torch.FloatTensor(mel_window.T).unsqueeze(0).unsqueeze(0)
            mel_tensor = mel_tensor.to(self.device)

            # Extract features with AVE encoder
            with torch.no_grad():
                feat = self.audio_encoder(mel_tensor)  # [1, 512]

            features.append(feat)

        # 4. Concatenate all features
        features = torch.cat(features, dim=0)  # [N, 512]

        # 5. Add padding (first 2 and last 2 frames repeated)
        first_frame = features[:1]
        last_frame = features[-1:]
        features = torch.cat(
            [first_frame.repeat(2, 1), features, last_frame.repeat(2, 1)], dim=0
        )  # [N+4, 512]

        # 6. Match provider.py format (lines 219-223)
        # Add batch dimension and permute to [N, 1, 512]
        features = features.unsqueeze(0)  # [1, N, 512]
        features = features.permute(1, 0, 2)  # [N, 1, 512]

        return features

    def load_audio_file(self, audio_path):
        """
        Load audio file from disk.

        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)

        Returns:
            Audio waveform (numpy array, 16kHz)

        Reference: utils.py line 1580-1581
        """
        wav, sr = librosa.load(audio_path, sr=16000)
        print(f"Loaded audio: {len(wav)/sr:.2f}s @ {sr}Hz")
        return wav

    @torch.inference_mode()
    def generate_frame(self, audio_features, frame_index, debug=False):
        """Generate a single frame from audio features."""

        # CRITICAL: Set model attributes for update_extra_states
        # Reference: main.py lines 196-197
        # Set BOTH singular and plural (model uses both internally!)
        self.trainer.model.aud_features = audio_features.to(self.device)
        self.trainer.model.eye_areas = self.eye_areas  # Plural (external reference)
        self.trainer.model.eye_area = (
            self.eye_areas
        )  # Singular (used in renderer.py:439)

        # Verify attributes are set (only for first frame)
        if debug and frame_index == 0:
            print("\n=== MODEL ATTRIBUTES (should match main.py:196-197) ===")
            print(f"  aud_features: {self.trainer.model.aud_features.shape}")
            print(f"  eye_areas: {self.trainer.model.eye_areas.shape}")
            print(f"  eye_area: {self.trainer.model.eye_area.shape}")

        # 1. Get mirrored index for pose/eye cycling (provider.py:538)
        pose_index = self.mirror_index(frame_index)

        img_id = self.img_ids[pose_index]  # Get img_id

        # 2. Get audio features with attention window (provider.py:534)
        # Note: audio uses ORIGINAL index, not mirrored (important!)
        auds = get_audio_features(
            audio_features, att_mode=self.opt.att, index=frame_index
        )
        auds = auds.to(self.device)  # [8, 1, 512] for AVE model

        # 3. Prepare pose (provider.py:540)
        pose = self.poses[pose_index : pose_index + 1]  # [1, 4, 4]
        pose = torch.from_numpy(pose).to(self.device)

        # 4. Generate camera rays (provider.py:547)
        # For inference: num_rays=-1 (all rays), patch_size from opt
        rays = get_rays(pose, self.intrinsics, self.H, self.W, -1, self.opt.patch_size)

        # 5. Get eye area (provider.py:576)
        if self.opt.exp_eye:
            eye = self.eye_areas[pose_index : pose_index + 1]  # [1, 7]
        else:
            eye = None

        # 6. Prepare background color (provider.py:596-604)
        # For inference (not torso mode): use background image
        bg_color = self.bg_img.view(1, -1, 3)  # [1, H*W, 3]

        # 6. Prepare background color (provider.py:587-604)
        # Load torso image and composite over background
        # torso_path = os.path.join(self.data_path, "torso_imgs", f"{img_id}.png")
        # torso_img = cv2.imread(torso_path, cv2.IMREAD_UNCHANGED)  # [H, W, 4]
        # torso_img = cv2.cvtColor(torso_img, cv2.COLOR_BGRA2RGBA)
        # torso_img = torso_img.astype(np.float32) / 255.0  # [H, W, 4]
        # torso_img = torch.from_numpy(torso_img).unsqueeze(0).to(self.device)  # [1, H, W, 4]

        # # Composite: torso RGB * alpha + bg * (1 - alpha)
        # bg_torso_img = torso_img[..., :3] * torso_img[..., 3:] + self.bg_img.unsqueeze(0) * (1 - torso_img[..., 3:])
        # bg_torso_img = bg_torso_img.view(1, -1, 3)  # [1, H*W, 3]

        # # For non-torso mode, use composited torso as bg
        # if not self.opt.torso:
        #     bg_color = bg_torso_img
        # else:
        #     bg_color = self.bg_img.view(1, -1, 3)  # [1, H*W, 3]

        # 7. Build data dictionary (provider.py:530-604)
        # # 7. Load ground truth image (provider.py:629-640)
        # gt_path = os.path.join(self.ori_imgs_dir, f"{img_id}.jpg")
        # images = cv2.imread(gt_path)  # [H, W, 3]
        # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        # images = images.astype(np.float32) / 255.0  # [H, W, 3]
        # images = torch.from_numpy(images).unsqueeze(0).to(self.device)  # [1, H, W, 3]

        # 8. Build data dictionary (provider.py:530-653)
        data = {
            "rays_o": rays["rays_o"],  # [1, H*W, 3]
            "rays_d": rays["rays_d"],  # [1, H*W, 3]
            "auds": auds,  # [8, 1, 512] for AVE model
            "bg_coords": self.bg_coords,  # [1, H*W, 2]
            "poses": pose,  # [1, 4, 4]
            "eye": eye,  # [1, 7] or None
            "bg_color": bg_color,  # [1, H*W, 3]
            # "images": images,  # [1, H, W, 3] - Ground truth image
            "index": [pose_index],  # For individual code
            "H": self.H,
            "W": self.W,
        }

        # ADD PORTRAIT MODE SUPPORT
        if self.portrait:
            # Load GT image
            gt_path = os.path.join(self.ori_imgs_dir, f"{img_id}.jpg")
            gt_img = cv2.imread(gt_path)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_img = gt_img.astype(np.float32) / 255.0
            gt_img = torch.from_numpy(gt_img).unsqueeze(0).to(self.device)

            # Load face mask
            mask_path = os.path.join(self.parsing_dir, f"{img_id}_face.png")
            face_mask = cv2.imread(mask_path)
            face_mask = (255 - face_mask[:, :, 1]) / 255.0
            face_mask = face_mask.astype(
                np.float32
            )  # ← FIX: Convert to float32 BEFORE tensor
            face_mask = torch.from_numpy(face_mask).unsqueeze(0).to(self.device)

            data["bg_gt_images"] = gt_img
            data["bg_face_mask"] = face_mask

            # Debug portrait mode
            if debug and frame_index == 0:
                print("\n=== PORTRAIT MODE DEBUG ===")
                print(
                    f"  GT image: {gt_img.shape}, range [{gt_img.min():.3f}, {gt_img.max():.3f}]"
                )
                print(
                    f"  Face mask: {face_mask.shape}, range [{face_mask.min():.3f}, {face_mask.max():.3f}]"
                )
                print(
                    f"  Face mask mean: {face_mask.mean():.3f} (should be ~0.2-0.5 for typical face)"
                )
                # Save mask for visual inspection
                mask_vis = (face_mask[0].cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(f"debug_face_mask_frame{frame_index}.png", mask_vis)
                print(f"  Saved mask to debug_face_mask_frame{frame_index}.png")

        if debug:
            print("\n=== YOUR generate_frame OUTPUT ===")
            for key, val in data.items():
                if isinstance(val, torch.Tensor):
                    print(
                        f"{key:20s}: shape={val.shape}, dtype={val.dtype}, device={val.device}"
                    )
                elif isinstance(val, list):
                    print(f"{key:20s}: {val}")
                else:
                    print(f"{key:20s}: {val}")
            print(f"auds min/max: {data['auds'].min():.3f} / {data['auds'].max():.3f}")
            if eye is not None:
                print(f"eye min/max: {data['eye'].min():.3f} / {data['eye'].max():.3f}")

        # 8. Call trainer.test_step() to render (utils.py:958-987)
        with torch.cuda.amp.autocast(enabled=self.opt.fp16):
            pred_rgb, pred_depth = self.trainer.test_step(data)

        # 9. Apply color space conversion if needed (utils.py:1074-1075)
        from nerf_triplane.utils import linear_to_srgb

        if self.opt.color_space == "linear":
            pred_rgb = linear_to_srgb(pred_rgb)

        if self.portrait and "bg_gt_images" in data and "bg_face_mask" in data:
            from nerf_triplane.utils import blend_with_mask_cuda

            pred = blend_with_mask_cuda(
                pred_rgb[0],
                data["bg_gt_images"].squeeze(0),
                data["bg_face_mask"].squeeze(0),
            )
            pred = (pred * 255).astype(np.uint8)
        else:
            # 10. Extract first (and only) image from batch
            pred = pred_rgb[0].detach().cpu().numpy()  # [H, W, 3]

            # Convert to uint8
            pred = (pred * 255).astype(np.uint8)

        return pred


if __name__ == "__main__":
    inference = StreamingInference(
        data_path="data/May", workspace="model/trial_may", portrait=True, torso=False
    )

    # Test if mirror index function is working properly
    print("\n=== Testing mirror_index ===")
    print(f"\nnum_poses: {inference.num_poses}")

    # Test with indices that actually cross boundaries
    test_indices = [
        0,  # First pose
        10,  # Early in first pass
        552,  # Last pose of first pass
        553,  # First index of second pass (should mirror)
        560,  # Should go backward
        inference.num_poses - 1,  # Last valid pose
        inference.num_poses,  # Start of second pass
        inference.num_poses + 10,  # Second pass
        inference.num_poses * 2 - 1,  # End of second pass
        inference.num_poses * 2,  # Start of third pass
    ]

    print("\nTesting mirror_index:")
    for i in test_indices:
        result = inference.mirror_index(i)
        turn = i // inference.num_poses
        direction = "forward" if turn % 2 == 0 else "backward"
        print(f"  index {i:4d} -> pose {result:3d} (pass {turn}, {direction})")

    # Test audio processing
    print("\n=== Testing Audio Processing ===")

    # Load a test audio file
    test_audio = "data/May/aud.wav"  # Use your training audio
    wav = inference.load_audio_file(test_audio)

    # Convert to mel
    mel = inference.wav_to_melspectrogram(wav)
    print(f"Mel-spectrogram shape: {mel.shape}")

    # Extract features
    features = inference.extract_ave_features(wav)
    print(f"Audio features shape: {features.shape}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")

    # Test get_audio_features with attention window
    print("\n=== Testing Attention Window ===")

    # Test different frames
    for frame_idx in [0, 10, 100, features.shape[0] - 1]:
        # Get features with bi-directional attention (mode 2)
        frame_features = get_audio_features(features, att_mode=2, index=frame_idx)
        print(f"Frame {frame_idx:4d}: input shape {frame_features.shape}")

    # Test edge cases
    print("\nEdge case - first frame (should pad with zeros):")
    frame_features = get_audio_features(features, att_mode=2, index=0)
    print(f"  Shape: {frame_features.shape} (still returns 8 frames with zero padding)")

    print("\nEdge case - last frame (should pad with zeros):")
    last_idx = features.shape[0] - 1
    frame_features = get_audio_features(features, att_mode=2, index=last_idx)
    print(f"  Shape: {frame_features.shape} (still returns 8 frames with zero padding)")

    print("\n=== Testing Frame Generation ===")

    # Load and process audio
    test_audio = "data/May/aud.wav"
    wav = inference.load_audio_file(test_audio)
    audio_features = inference.extract_ave_features(wav)

    # Generate a few test frames
    test_frames = [0, 10, 100, 200]

    for frame_idx in test_frames:
        print(f"\nGenerating frame {frame_idx}...")
        frame = inference.generate_frame(audio_features, frame_idx, debug=True)
        print(f"  Frame shape: {frame.shape}")
        print(f"  Frame dtype: {frame.dtype}")
        print(f"  Frame range: [{frame.min()}, {frame.max()}]")

        # Save frame for visual inspection
        import cv2

        output_path = f"test_frame_{frame_idx:04d}.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"  Saved to: {output_path}")
