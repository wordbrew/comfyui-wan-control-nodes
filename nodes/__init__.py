from .base_nodes import NODE_CLASS_MAPPINGS as BASE_NODE_CLASS_MAPPINGS
# Will import advanced nodes in the future
# from .advanced_nodes import NODE_CLASS_MAPPINGS as ADVANCED_NODE_CLASS_MAPPINGS

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **BASE_NODE_CLASS_MAPPINGS,
    # **ADVANCED_NODE_CLASS_MAPPINGS,  # Uncomment when adding advanced nodes
}

# Export the combined mapping
__all__ = ['NODE_CLASS_MAPPINGS']