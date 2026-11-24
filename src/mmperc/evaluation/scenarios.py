"""Named degradation scenarios shared by every evaluation script (identical across models)."""
SCENARIOS = {
    "clean": {},
    "occlusion L2": {"vision_occlusion": 2}, "occlusion L4": {"vision_occlusion": 4},
    "vis noise L3": {"vision_noise": 3}, "illumination L3": {"vision_illumination": 3},
    "proprio noise L3": {"proprio_noise": 3}, "proprio bias L3": {"proprio_bias": 3},
    "proprio dropout L3": {"proprio_dropout": 3},
    "contact noise L2": {"contact_noise": 2}, "contact noise L4": {"contact_noise": 4},
    "random occl L4": {"vision_occlusion_random": 4}, "blur L3": {"vision_blur": 3},
    "vision dropout L3": {"vision_dropout": 3},
    "vision dropout L4 + proprio dropout L3": {"vision_dropout": 4, "proprio_dropout": 3},
    "occl L3 + proprio noise L2": {"vision_occlusion": 3, "proprio_noise": 2},
    "occl L4 + proprio bias L3": {"vision_occlusion": 4, "proprio_bias": 3},
}
