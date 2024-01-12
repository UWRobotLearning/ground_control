import os

def generate_box_urdf(dimensions, friction):
    # Validate friction value
    friction = max(0, min(5, friction))

    # Calculate color based on friction value (linear interpolation from blue to red)
    color_r = min(1.0, friction / 5.0)
    color_b = max(0.0, 1.0 - friction / 5.0)
    color_g = 0.0

    # URDF template for a box
    urdf_template = """<?xml version="1.0" ?>
    <robot name="box">
        <link name="base_link">
            <visual>
                <geometry>
                    <box size="{length} {width} {height}"/>
                </geometry>
                <material name="box_color">
                    <color rgba="{r} {g} {b} 1"/>
                </material>
            </visual>
            <collision>
                <geometry>
                    <box size="{length} {width} {height}"/>
                </geometry>
            </collision>
        </link>
    </robot>
    """

    # Format the template with dimensions and color
    urdf_content = urdf_template.format(
        length=dimensions[0],
        width=dimensions[1],
        height=dimensions[2],
        r=color_r,
        g=color_g,
        b=color_b
    )

    # Save the URDF content to a file or return it as needed
    return urdf_content

# Example usage:
# dimensions = (100, 100, 0.05)
# friction = 2.5  # Example friction value
# urdf_content = generate_box_urdf(dimensions, friction)

friction_values = [0.1*i for i in range(0, 51)]

save_dir_small = os.path.join(os.getcwd(), "planes_small")
if not os.path.exists(save_dir_small):
    os.makedirs(save_dir_small)

save_dir_large = os.path.join(os.getcwd(), "planes_large")
if not os.path.exists(save_dir_large):
    os.makedirs(save_dir_large)


large_dimensions = (100, 100, 0.05)
small_dimensions = (1, 1, 0.05)
# Generate planes:
for i, friction in enumerate(friction_values):
    friction_str = f"{friction:0.1f}".replace(".", "-")
    file_name = f"plane_{friction_str}.urdf"

    urdf_small = generate_box_urdf(small_dimensions, friction)
    urdf_large = generate_box_urdf(large_dimensions, friction)

    with open(os.path.join(save_dir_small, file_name), "w") as f:
        f.write(urdf_small)

    with open(os.path.join(save_dir_large, file_name), "w") as f:
        f.write(urdf_large)