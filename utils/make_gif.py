import imageio.v2 as imageio
import glob
import re

frame_files = glob.glob("displacement_t_*.png")

def extract_time(fname):
    return float(re.search(r"t_(\d+\.\d+)", fname).group(1))

frame_files = sorted(frame_files, key=extract_time)

output_gif = "hood_displacement.gif"

with imageio.get_writer(
    output_gif,
    mode="I",
    duration=0.3,
    loop=0      # <-- THIS is the key
) as writer:
    for fname in frame_files:
        image = imageio.imread(fname)
        writer.append_data(image)

print("Looping GIF saved.")
