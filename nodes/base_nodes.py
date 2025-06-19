import torch
import math
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import node_helpers


class WanWeightedControlToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "width": ("INT", {"default": 832, "min": 16, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 2048, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 1024, "step": 4}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "control_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "control_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "control_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "falloff_percentage": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.49, "step": 0.01}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                "start_image": ("IMAGE", ),
                "control_video": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, 
               control_weight, control_start, control_end, falloff_percentage,
               clip_vision_output=None, start_image=None, control_video=None):
        
        device = comfy.model_management.intermediate_device()
        
        # Initialize base latents (empty latent for generation)
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                           device=device, dtype=torch.float32)
        
        # Initialize concat latent for control/image conditioning
        # This follows the native WanFunControlToVideo implementation
        concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                                  device=device, dtype=torch.float32)
        
        # Apply Wan2.1 format processing
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
        concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)  # Double channels: [:16] for control, [16:] for image
        
        total_latent_frames = concat_latent.shape[2]
        
        # Process control video (channels 0-15)
        if control_video is not None:
            with torch.no_grad():
                # Prepare control video frames
                control_video_frames = control_video[:length].movedim(-1, 1)
                control_video_frames = comfy.utils.common_upscale(
                    control_video_frames, width, height, "bilinear", "center"
                ).movedim(1, -1)
                
                # Encode control video
                control_latent = vae.encode(control_video_frames[:, :, :, :3])
                control_latent = control_latent.to(device)
                
                control_frames = min(control_latent.shape[2], total_latent_frames)
                
                # Calculate control influence range
                start_frame = int(control_frames * control_start)
                end_frame = int(control_frames * control_end)
                
                # Create temporal mask with smooth falloffs
                control_mask = self._create_temporal_mask(
                    total_frames=control_frames,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    falloff_percentage=falloff_percentage,
                    device=device
                )
                
                # Apply control weight and expand dimensions
                control_mask = control_mask * control_weight
                control_mask = control_mask.unsqueeze(-1).unsqueeze(-1)
                
                # Apply weighted control to first 16 channels
                weighted_control = control_latent[:, :, :control_frames] * control_mask
                concat_latent[:, :16, :control_frames] = weighted_control
        
        # Process start image (channels 16-31)
        if start_image is not None:
            with torch.no_grad():
                # Prepare start image frames
                start_image_frames = start_image[:length].movedim(-1, 1)
                start_image_frames = comfy.utils.common_upscale(
                    start_image_frames, width, height, "bilinear", "center"
                ).movedim(1, -1)
                
                # Encode start image
                start_latent = vae.encode(start_image_frames[:, :, :, :3])
                start_latent = start_latent.to(device)
                
                # Apply start image to all frames (or up to available frames)
                start_frames = min(start_latent.shape[2], total_latent_frames)
                concat_latent[:, 16:, :start_frames] = start_latent[:, :, :start_frames]
        
        # Prepare conditioning data
        conditioning_data = {"concat_latent_image": concat_latent}
        
        # Add CLIP vision output if provided (for reference/style guidance)
        if clip_vision_output is not None:
            conditioning_data["clip_vision_output"] = clip_vision_output
        
        # Apply conditioning to positive and negative
        positive_out = node_helpers.conditioning_set_values(positive, conditioning_data)
        negative_out = node_helpers.conditioning_set_values(negative, conditioning_data)
        
        # Return outputs
        out_latent = {"samples": latent}
        return (positive_out, negative_out, out_latent)
    
    def _create_temporal_mask(self, total_frames, start_frame, end_frame, falloff_percentage, device):
        """Create a temporal mask with smooth cosine-based falloffs"""
        mask = torch.zeros([1, 1, total_frames], device=device, dtype=torch.float32)
        
        if start_frame >= end_frame:
            return mask
        
        # Set full influence in the specified range
        mask[:, :, start_frame:end_frame] = 1.0
        
        # Calculate falloff length
        falloff_frames = max(1, int(total_frames * falloff_percentage))
        
        # Apply smooth falloff at the start
        if start_frame > 0:
            falloff_start = max(0, start_frame - falloff_frames)
            for i in range(falloff_start, start_frame):
                t = (i - falloff_start) / (start_frame - falloff_start)
                # Cosine interpolation for smooth transition
                mask[:, :, i] = 0.5 - 0.5 * math.cos(t * math.pi)
        
        # Apply smooth falloff at the end
        if end_frame < total_frames:
            falloff_end = min(total_frames, end_frame + falloff_frames)
            for i in range(end_frame, falloff_end):
                t = (i - end_frame) / (falloff_end - end_frame)
                # Cosine interpolation for smooth transition
                mask[:, :, i] = 0.5 + 0.5 * math.cos(t * math.pi)
        
        return mask


NODE_CLASS_MAPPINGS = {
    "WanWeightedControlToVideo": WanWeightedControlToVideo,
}