import depthai as dai, sys

out = open("probe_out.txt", "w")

out.write(f"depthai version: {dai.__version__}\n")

stereo_cls = dai.node.StereoDepth
out.write(f"StereoDepth attrs: {[a for a in dir(stereo_cls) if 'preset' in a.lower() or 'Preset' in a]}\n")

# try to enumerate PresetMode
pm = stereo_cls.PresetMode
out.write(f"PresetMode type: {type(pm)}\n")
for name in ["HIGH_DENSITY", "HIGH_ACCURACY", "FAST_DENSITY", "HIGH_DETAIL",
             "DEFAULT", "ROBOTICS", "FACE", "HAND"]:
    try:
        val = getattr(pm, name)
        out.write(f"  {name} = {val}\n")
    except AttributeError:
        out.write(f"  {name} -> NOT FOUND\n")

# list all non-dunder attrs of PresetMode
out.write(f"PresetMode dir: {[a for a in dir(pm) if not a.startswith('_')]}\n")

out.close()
print("done")

