from typing import List, Tuple

import numpy as np

import torch


def process_data(odom, cmd_vel):
    # odom columns are: Time,time,xPos,yPos,yaw,xVel,yVel,zAngVel
    # cmd_vel columns are: Time,linear.x,linear.y,linear.z,angular.x,angular.y,angular.z

    sampling_interval = 0.1

    # Filter odom at 10 Hz based on the "time" column. 
    # Note that "time" is the measurement time, and 
    # "Time" is the ROS bag recording time. 
    filtered_rows = [odom[0]]
    last_time = odom[0][0]
    for row in odom[1:]:
        if row[0] - last_time >= sampling_interval:
            filtered_rows.append(row)
            last_time = row[0]

    odom = np.array(filtered_rows)

    # Filter cmd_vel to match the time of odom
    indices = np.abs(cmd_vel[:, 0, None] - odom[:, 0]).argmin(axis=0)
    cmd_vel = cmd_vel[indices]

    # Unwrap the yaw measurements
    odom[:, 4] = np.unwrap(odom[:, 4])

    # Drop Time column from odom and cmd_vel
    odom = odom[:, 1:]
    cmd_vel = cmd_vel[:, 1:]

    # Drop the empty cmd_vel columns
    cmd_vel = cmd_vel[:, [0, 5]]

    # -------------------------

    odom = torch.tensor(odom, dtype=torch.float32)
    cmd_vel = torch.tensor(cmd_vel, dtype=torch.float32)

    # Split into inputs and targets
    inputs = odom[:-1].clone()
    targets = odom[1:].clone()

    cmd_vel = cmd_vel[:-1].clone()

    # Body frame transformation
    targets = pos_inertial_to_body(refs=inputs, data=targets)
    targets = vel_body_to_inertial(refs=targets, data=targets)
    targets = vel_inertial_to_body(refs=inputs, data=targets)

    # Take the change in heading
    targets[:, 3] = targets[:, 3] - inputs[:, 3]

    # Zero out the position and heading.
    inputs[:, 1:4] = 0

    # Concatenate inputs and cmd_vel
    inputs = torch.cat((inputs, cmd_vel), dim=1)

    return inputs, targets


def pos_inertial_to_body(refs, data):

    yaws = refs[:, 3]
    c = torch.cos(yaws)
    s = torch.sin(yaws)

    R = torch.stack(
        [
            torch.stack([c, s], dim=1),
            torch.stack([-s, c], dim=1),
        ],
        dim=1,
    )

    # Apply the rotation matrix to the odom data
    data[:, 1:3] = torch.bmm(R, (data[:, 1:3] - refs[:, 1:3]).unsqueeze(-1)).squeeze(-1)

    return data


def vel_body_to_inertial(refs, data):

    yaws = refs[:, 3]
    c = torch.cos(yaws)
    s = torch.sin(yaws)

    R = torch.stack(
        [
            torch.stack([c, -s], dim=1),
            torch.stack([s, c], dim=1),
        ],
        dim=1,
    )

    # Apply the rotation matrix to the odom velocity data
    data[:, 4:6] = torch.bmm(R, data[:, 4:6].unsqueeze(-1)).squeeze(-1)

    return data


def vel_inertial_to_body(refs, data):

    yaws = refs[:, 3]
    c = torch.cos(yaws)
    s = torch.sin(yaws)

    R = torch.stack(
        [
            torch.stack([c, s], dim=1),
            torch.stack([-s, c], dim=1),
        ],
        dim=1,
    )

    # Apply the rotation matrix to the odom velocity data
    data[:, 4:6] = torch.bmm(R, data[:, 4:6].unsqueeze(-1)).squeeze(-1)

    return data
