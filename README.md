# WAN Control Nodes for ComfyUI

This pack provides enhanced control nodes for working with Wan video models in ComfyUI. It is under active development and may change regularly, or may not. Depends entirely on my free time and waning interest. Please don't come to rely on it for anything, but you are welcome to improve on it.

## Features

- **WanWeightedControlToVideo**: Control video generation with weighted influence and smooth transitions
  - Full timeline control with start/end positions and falloff
  - Keyframe capabilities with start and end images
  - Flexible control strength adjustment

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:

cd ComfyUI/custom_nodes

git clone https://github.com/wordbrew/comfyui-wan-control-nodes.git

2. Restart ComfyUI

## Usage

### WanWeightedControlToVideo

This node allows for precise control over how and when control video influences your generation:

- **control_weight**: Adjusts the strength of the control video (0.0-5.0)
- **control_start/end**: Set when the control begins and ends (0.0-1.0 timeline position)
- **falloff_percentage**: Controls how gradually the influence fades in/out (0.0-0.49)

#### Example Workflows:

1. **Basic Control**: Use only a control video to guide the entire generation
2. **Keyframe Animation**: Use start and end images to define keyframes with control in between
3. **Partial Control**: Apply control to only a portion of the timeline

## License

Apache License 2.0
