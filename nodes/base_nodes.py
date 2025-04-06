import nodes
import node_helpers
import torch
import math
import comfy.model_management
import comfy.utils
import comfy.latent_formats


class WanWeightedControlToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "control_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                             "control_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "control_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "falloff_percentage": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.49, "step": 0.01}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                             "control_video": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, control_weight, control_start, control_end, falloff_percentage, start_image=None, end_image=None, clip_vision_output=None, control_video=None):
        try:
            # Explicitly set device and ensure consistency
            device = comfy.model_management.intermediate_device()
            
            # Setup base latents with explicit device placement
            latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                                device=device, dtype=torch.float32).contiguous()
            concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                                      device=device, dtype=torch.float32).contiguous()
            concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
            concat_latent = concat_latent.repeat(1, 2, 1, 1, 1).contiguous()
            
            # Calculate total frames in latent space
            total_latent_frames = concat_latent.shape[2]

            # Create a unified influence mask for the timeline
            timeline_mask = torch.zeros([1, 1, total_latent_frames], device=device, dtype=torch.float32).contiguous()
            
            # Process start image if provided
            start_latent = None
            if start_image is not None:
                with torch.no_grad():  # Reduce memory usage
                    start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), 
                                                          width, height, "bilinear", "center").movedim(1, -1)
                    start_latent = vae.encode(start_image[:, :, :, :3])
                    start_latent = start_latent.to(device).contiguous()  # Ensure on correct device
                    
                    # Calculate how much of the timeline the start image influences
                    start_frames = min(start_latent.shape[2], total_latent_frames)
                    start_influence = torch.ones([1, 1, start_frames], device=device, dtype=torch.float32).contiguous()
                    
                    # Apply falloff to the end of start influence
                    falloff_length = max(1, int(total_latent_frames * falloff_percentage))
                    if start_frames > falloff_length:
                        # Create a smooth falloff at the end of the start image influence
                        t = torch.linspace(0, 1, falloff_length, device=device)
                        falloff = 0.5 + 0.5 * torch.cos(t * math.pi)  # 1->0 falloff
                        start_influence[0, 0, -falloff_length:] = falloff
                    
                    # Place start latent in the concatenated latent
                    if start_frames > 0:
                        concat_latent[:,16:,:start_frames] = start_latent[:,:,:start_frames]
                    
                    # Update the timeline mask for the start image's influence
                    timeline_mask[:, :, :start_frames] = start_influence
            
            # Process end image if provided
            end_latent = None
            if end_image is not None:
                with torch.no_grad():  # Reduce memory usage
                    end_image = comfy.utils.common_upscale(end_image[:length].movedim(-1, 1), 
                                                        width, height, "bilinear", "center").movedim(1, -1)
                    end_latent = vae.encode(end_image[:, :, :, :3])
                    end_latent = end_latent.to(device).contiguous()  # Ensure on correct device
                    
                    # Calculate how much of the timeline the end image influences
                    end_frames = min(end_latent.shape[2], total_latent_frames)
                    end_start_idx = max(0, total_latent_frames - end_frames)
                    
                    # Create a smoother influence transition for the end image
                    end_influence = torch.zeros([1, 1, total_latent_frames], 
                                              device=device, dtype=torch.float32).contiguous()
                    
                    # Calculate when to start end image influence
                    influence_start = 0
                    if control_video is not None and control_end < 1.0:
                        influence_start = max(0, int(total_latent_frames * control_end) - falloff_length)
                    else:
                        influence_start = max(0, int(total_latent_frames * 0.6))
                    
                    # Improved end image transition: use a smoother S-curve
                    # with longer lead-in for better motion guidance
                    transition_length = total_latent_frames - influence_start
                    if transition_length > 0:
                        # Create a smoother S-curve for influence
                        curve_positions = torch.linspace(0, 1, transition_length, device=device)
                        
                        # Apply a smoother cubic ease-in-out curve
                        # This grows more slowly at first, then accelerates smoothly
                        for i, pos in enumerate(curve_positions):
                            idx = influence_start + i
                            if idx < total_latent_frames:
                                # Cubic ease-in-out curve: even smoother than quadratic
                                if pos < 0.5:
                                    # First half: slower growth (cubic ease-in)
                                    influence = 4 * pos * pos * pos
                                else:
                                    # Second half: accelerating (cubic ease-out)
                                    p = pos - 1
                                    influence = 1 + 4 * p * p * p
                                
                                # Ensure full influence in final frames
                                if idx >= total_latent_frames - 3:
                                    influence = 1.0
                                
                                end_influence[0, 0, idx] = influence
                    
                    # Improved blending for end image transition
                    # Calculate start of blending period
                    blend_start = influence_start
                    blend_length = total_latent_frames - blend_start
                    
                    if blend_length > 0 and end_frames > 0:
                        # Create a reference to the end latent section for better blending
                        # This uses the last frames of the end latent, repeated if necessary
                        reference_end_latent = torch.zeros_like(concat_latent[:, 16:, blend_start:])
                        
                        for i in range(blend_length):
                            # Calculate which frame from end_latent to use
                            # Map from blend timeline to end_latent timeline
                            src_idx = min(i * end_frames // blend_length, end_frames - 1)
                            reference_end_latent[:, :, i] = end_latent[:, :, src_idx]
                        
                        # Now blend between existing content and end reference using influence curve
                        for i in range(blend_length):
                            idx = blend_start + i
                            if idx < total_latent_frames:
                                weight = end_influence[0, 0, idx].item()
                                if weight > 0:
                                    # Improved blending with temporal coherence
                                    concat_latent[:, 16:, idx] = (
                                        (1.0 - weight) * concat_latent[:, 16:, idx] +
                                        weight * reference_end_latent[:, :, i]
                                    )
                    
                    # Ensure the final frames are exactly the end image
                    last_frames = min(3, end_frames)
                    if last_frames > 0:
                        end_offset = total_latent_frames - last_frames
                        if end_offset >= 0:
                            concat_latent[:, 16:, end_offset:] = end_latent[:, :, :last_frames]
                    
                    # Update the timeline mask
                    timeline_mask = torch.max(timeline_mask, end_influence)
            
            # Process control video if provided
            control_latent = None
            if control_video is not None:
                with torch.no_grad():  # Reduce memory usage
                    control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), 
                                                            width, height, "bilinear", "center").movedim(1, -1)
                    control_latent = vae.encode(control_video[:, :, :, :3])
                    control_latent = control_latent.to(device).contiguous()  # Ensure on correct device
                    
                    # Calculate control influence range in frames
                    total_frames = control_latent.shape[2]
                    start_frame = max(0, min(total_frames-1, int(total_frames * control_start)))
                    end_frame = max(start_frame+1, min(total_frames, int(total_frames * control_end)))
                    
                    # Create control weight mask
                    control_mask = torch.zeros([1, 1, total_frames], 
                                            device=device, dtype=torch.float32).contiguous()
                    
                    # Set full control weight in the main section
                    if start_frame < end_frame:
                        control_mask[:, :, start_frame:end_frame] = 1.0
                    
                    # Create falloff before start_frame
                    falloff_length = max(2, int(total_frames * falloff_percentage))
                    if start_frame > 0:
                        falloff_start = max(0, start_frame - falloff_length)
                        t = torch.linspace(0, 1, start_frame - falloff_start, device=device)
                        smooth_t = 0.5 - 0.5 * torch.cos(t * math.pi)  # Smooth 0->1 transition
                        control_mask[:, :, falloff_start:start_frame] = smooth_t.reshape(1, 1, -1)
                    
                    # Create falloff after end_frame - with special handling for end image transition
                    if end_frame < total_frames:
                        falloff_end = min(total_frames, end_frame + falloff_length)
                        
                        # Check if we have end image influence in this region
                        if end_image is not None:
                            # Create a special falloff that complements the end image influence
                            for i in range(end_frame, falloff_end):
                                if i < total_frames:
                                    # Calculate original falloff
                                    fade_pos = (i - end_frame) / falloff_length
                                    original_falloff = 0.5 + 0.5 * math.cos(fade_pos * math.pi)
                                    
                                    # Get end image influence at this point
                                    end_influence_here = timeline_mask[0, 0, i].item() if i < timeline_mask.shape[2] else 0
                                    
                                    # Adjust control falloff to complement end influence
                                    # More gradual handoff to reduce artifacts
                                    adjusted_falloff = original_falloff * (1.0 - (end_influence_here * 0.8))
                                    
                                    # Apply adjusted falloff
                                    control_mask[0, 0, i] = adjusted_falloff
                        else:
                            # Standard falloff if no end image
                            t = torch.linspace(0, 1, falloff_end - end_frame, device=device)
                            smooth_t = 0.5 + 0.5 * torch.cos(t * math.pi)  # Smooth 1->0 transition
                            control_mask[:, :, end_frame:falloff_end] = smooth_t.reshape(1, 1, -1)
                    
                    # Apply control weight
                    control_mask = control_mask * control_weight
                    
                    # Expand dimensions to match latent dimensions
                    control_mask = control_mask.unsqueeze(-1).unsqueeze(-1)
                    
                    # Apply weighted control to the control section of the concat_latent
                    weighted_control = control_latent * control_mask
                    concat_latent[:,:16,:weighted_control.shape[2]] = weighted_control[:,:,:concat_latent.shape[2]]
            
            # Make sure final tensor is contiguous
            concat_latent = concat_latent.contiguous()
            
            # Set conditioning values
            positive_out = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent})
            negative_out = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent})

            if clip_vision_output is not None:
                positive_out = node_helpers.conditioning_set_values(positive_out, {"clip_vision_output": clip_vision_output})
                negative_out = node_helpers.conditioning_set_values(negative_out, {"clip_vision_output": clip_vision_output})

            # Clean up references to large tensors to help with memory management
            del start_latent, end_latent, control_latent, weighted_control
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return outputs
            out_latent = {}
            out_latent["samples"] = latent
            return (positive_out, negative_out, out_latent)
            
        except Exception as e:
            print(f"Error in WanWeightedControlToVideo: {str(e)}")
            # Return original conditioning on error to avoid breaking the graph
            out_latent = {}
            out_latent["samples"] = torch.zeros([batch_size, 4, ((length - 1) // 8) + 1, height // 8, width // 8], 
                                           device=device)
            return (positive, negative, out_latent)


NODE_CLASS_MAPPINGS = {
    "WanWeightedControlToVideo": WanWeightedControlToVideo,
}
