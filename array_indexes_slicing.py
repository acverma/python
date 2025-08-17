import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tempfile
from matplotlib import animation

# ------------------------
# Streamlit Page Setup
# ------------------------
st.set_page_config(page_title="3D Array Slicing Visualizer", layout="centered")
st.title("📊 3D Array Slicing Visualizer")
st.markdown("Visualize how 3D arrays are structured and sliced. Explore slices, highlight elements, and generate animations.")

# ------------------------
# Array Input
# ------------------------
st.sidebar.header("🧮 Define 3D Array")

depth = st.sidebar.number_input("Depth (Z)", min_value=1, value=2)
rows = st.sidebar.number_input("Rows (Y)", min_value=1, value=2)
cols = st.sidebar.number_input("Cols (X)", min_value=1, value=2)

default_text = "\n".join([
    "1 2",
    "3 4",
    "---",
    "5 6",
    "7 8"
])

user_input = st.sidebar.text_area("Enter 3D array (use '---' to separate layers)", default_text, height=150)

def parse_array(text, depth, rows, cols):
    layers = text.strip().split('---')
    if len(layers) != depth:
        raise ValueError("Number of layers does not match depth.")
    array = np.zeros((depth, rows, cols), dtype=int)
    for z, layer in enumerate(layers):
        lines = layer.strip().splitlines()
        if len(lines) != rows:
            raise ValueError(f"Layer {z} has {len(lines)} rows, expected {rows}.")
        for y, line in enumerate(lines):
            vals = list(map(int, line.strip().split()))
            if len(vals) != cols:
                raise ValueError(f"Row {y} in layer {z} has {len(vals)} values, expected {cols}.")
            array[z, y, :] = vals
    return array

# ------------------------
# Slice & Highlight Settings
# ------------------------
axis = st.sidebar.selectbox("📐 Slice Axis", ["Z", "Y", "X"])
show_labels = st.sidebar.checkbox("🔢 Show Number Labels", value=True)

# ------------------------
# Drawing the 3D Slice
# ------------------------
def draw_slice(array, axis, frame, show_labels=True):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    depth, rows, cols = array.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_zlim(0, depth)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    if axis == 'Z':
        slice_data = array[frame]
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        z = np.full_like(x, frame)
    elif axis == 'Y':
        slice_data = array[:, frame, :]
        x, z = np.meshgrid(np.arange(cols), np.arange(depth))
        y = np.full_like(x, frame)
    else:
        slice_data = array[:, :, frame]
        y, z = np.meshgrid(np.arange(rows), np.arange(depth))
        x = np.full_like(y, frame)

    norm = slice_data / slice_data.max() if slice_data.max() else slice_data

    ax.plot_surface(
        x, y, z,
        facecolors=plt.cm.viridis(norm),
        edgecolor='k', rstride=1, cstride=1,
        alpha=0.9, shade=False
    )

    if show_labels:
        for i in range(slice_data.shape[0]):
            for j in range(slice_data.shape[1]):
                val = slice_data[i, j]
                xi, yi, zi = x[i, j], y[i, j], z[i, j]
                ax.text(xi, yi, zi + 0.1, str(val), color='black', ha='center', va='center', fontsize=9)

    ax.set_title(f"{axis}-Axis Slice @ Index {frame}")
    return fig, ax

# ------------------------
# Animation Generator
# ------------------------
def generate_animation(array, axis, file_type="gif"):
    frames = {"Z": array.shape[0], "Y": array.shape[1], "X": array.shape[2]}[axis]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        depth, rows, cols = array.shape
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_zlim(0, depth)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if axis == 'Z':
            slice_data = array[frame]
            x, y = np.meshgrid(np.arange(cols), np.arange(rows))
            z = np.full_like(x, frame)
        elif axis == 'Y':
            slice_data = array[:, frame, :]
            x, z = np.meshgrid(np.arange(cols), np.arange(depth))
            y = np.full_like(x, frame)
        else:
            slice_data = array[:, :, frame]
            y, z = np.meshgrid(np.arange(rows), np.arange(depth))
            x = np.full_like(y, frame)

        norm = slice_data / slice_data.max() if slice_data.max() else slice_data

        ax.plot_surface(
            x, y, z,
            facecolors=plt.cm.viridis(norm),
            edgecolor='k', rstride=1, cstride=1,
            alpha=0.9, shade=False
        )

        if show_labels:
            for i in range(slice_data.shape[0]):
                for j in range(slice_data.shape[1]):
                    val = slice_data[i, j]
                    xi, yi, zi = x[i, j], y[i, j], z[i, j]
                    ax.text(xi, yi, zi + 0.1, str(val), color='black', ha='center', va='center', fontsize=9)

        ax.set_title(f"{axis}-Axis Slice @ Index {frame}")

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000, repeat=False)

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        if file_type == "gif":
            ani.save(tmp.name, writer="pillow", fps=1)
        else:
            ani.save(tmp.name, writer="ffmpeg", fps=1)
        return tmp.name

# ------------------------
# Run the App
# ------------------------
try:
    array = parse_array(user_input, depth, rows, cols)
    st.success("✅ Array parsed successfully.")

    # Initialize session state for frame index
    if "frame" not in st.session_state:
        st.session_state.frame = 0

    # Display the current slice
    fig, _ = draw_slice(array, axis, st.session_state.frame, show_labels)
    st.pyplot(fig)

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    if col1.button("⬅️ Previous"):
        st.session_state.frame = max(0, st.session_state.frame - 1)
    if col3.button("➡️ Next"):
        max_frames = {"Z": array.shape[0], "Y": array.shape[1], "X": array.shape[2]}[axis]
        st.session_state.frame = min(max_frames - 1, st.session_state.frame + 1)

    # Animation export
    st.subheader("🎞️ Export Animation")
    col1, col2 = st.columns([1, 2])
    file_type = col1.selectbox("Format", ["gif", "mp4"])
    if col2.button("Generate & Download"):
        st.info("Generating animation...")
        filepath = generate_animation(array, axis, file_type)
        with open(filepath, "rb") as f:
            data = f.read()
        if file_type == "gif":
            st.image(data)
        else:
            st.video(data)
        st.download_button("📥 Download", data, f"array_slicing.{file_type}", mime="video/mp4" if file_type == "mp4" else "image/gif")

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
