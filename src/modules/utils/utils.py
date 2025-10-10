import math

def follow_articulation_with_camera(sim, articulation, angle_rad=0.0, radius=3.0, height=1.0):
    # Get the root position of the first environment's robot
    root_pos = articulation.data.root_pos_w[0].cpu().numpy()  # shape: (3,)
    target = root_pos.tolist()

    # Compute camera position in orbit
    cam_x = root_pos[0] + radius * math.cos(angle_rad)
    cam_y = root_pos[1] + radius * math.sin(angle_rad)
    cam_z = root_pos[2] + height

    sim.set_camera_view(eye=[cam_x, cam_y, cam_z], target=[target[0], target[1], 0.5])