import torch

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tensorboardX import SummaryWriter

"""
NOTE:
density is in g/cm3 (from kg/m3 divide by 1000)
viscosity is in g/(cm*s) (from kg/(m*s)=Pa*s multiply by 10)

TODO: 
Do:

Maybe:
- instead of setting input velocity field as a boundary condition, use it as a external force field
- multiply output by X so the output range is bigger
"""

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
np.random.seed(seed)

# Config variables.
EXPERIMENT_TITLE = "default"
FIELD_SIZE = 48
BATCH_SIZE = 10
TRAINING_SET_SIZE = 1000
EPOCH_COUNT = 1000
ACTIVE_FIELD_SIZE = 8
VAL_DENSITIES = [0.5, 0.75, 1.0, 1.25, 1.5]
VAL_VISCOSITIES = [1e-3, 1e-2, 0.5, 1.0, 10.0]
TRAIN_DENSITY_MIN = 0.5
TRAIN_DENSITY_MAX = 1.0
TRAIN_VISCOSITY_MIN = 1e-3
TRAIN_VISCOSITY_MAX = 10.0

# Helper variables.
ACTIVE_FIELD_START = FIELD_SIZE //  2 - ACTIVE_FIELD_SIZE // 2
ACTIVE_FIELD_END = FIELD_SIZE //  2 + ACTIVE_FIELD_SIZE // 2
VAL_FLOW_START = np.array([ACTIVE_FIELD_START, ACTIVE_FIELD_START])
VAL_FLOW_END = np.array([ACTIVE_FIELD_END, ACTIVE_FIELD_END])
VAL_DENSITIES_COUNT = len(VAL_DENSITIES)
VAL_VISCOSITIES_COUNT = len(VAL_VISCOSITIES)
VAL_SAMPLES_COUNT = VAL_DENSITIES_COUNT * VAL_VISCOSITIES_COUNT

def draw_velocity_field(field, flow_start, flow_end):
    velocity = flow_end - flow_start
    norm = np.sqrt(np.dot(velocity, velocity))
    velocity = velocity / norm
    cv2.line(field, tuple(flow_start), tuple(flow_end), tuple(velocity), 1)
    return field

# Create validation velocity data.
VAL_VELOCITY = draw_velocity_field(np.zeros((FIELD_SIZE, FIELD_SIZE, 2)), VAL_FLOW_START, VAL_FLOW_END)

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        # Create array of random velocity fields.
        velocity_fields = np.zeros((size, FIELD_SIZE, FIELD_SIZE, 2))
        for i in range(size):
            start = np.random.randint(ACTIVE_FIELD_START, ACTIVE_FIELD_END, size=2)
            end = start + np.random.randint(ACTIVE_FIELD_SIZE // 2, ACTIVE_FIELD_SIZE, size=2) * np.random.choice([-1, 1], size=2)
            velocity_fields[i, :] = draw_velocity_field(velocity_fields[i, :], start, end)

        # Create Torch TensorDataset out of the velocity field array.
        velocity_fields = torch.tensor(velocity_fields.transpose((0, 3, 1, 2))).float().cuda()
        self.velocity_dataset = torch.utils.data.TensorDataset(velocity_fields)

    def __getitem__(self, index):
        target_velocity = self.velocity_dataset[index][0]
        density = torch.rand([]).cuda() * (TRAIN_DENSITY_MAX - TRAIN_DENSITY_MIN) + TRAIN_DENSITY_MIN
        viscosity = torch.rand([]).cuda() * (TRAIN_VISCOSITY_MAX - TRAIN_VISCOSITY_MIN) + TRAIN_VISCOSITY_MIN
        viscosity /= density
        return target_velocity, density, viscosity

    def __len__(self):
        return len(self.velocity_dataset)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv7 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv8 = nn.ConvTranspose2d(512, 256, 3, padding=1, stride=2, output_padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv11 = nn.Conv2d(256, 3, 3, padding=1, bias=False)

    def forward(self, input_velocity, density, viscosity):
        field_height, field_width = input_velocity.size()[2:4]

        # Convert density and viscosity scalars to 4D tensors.
        density = density.view([density.size()[0], 1, 1, 1]).repeat([1, 1, field_height, field_width])
        viscosity = viscosity.view([viscosity.size()[0], 1, 1, 1]).repeat([1, 1, field_height, field_width])

        # DNN input is target velocity, density and viscosity.
        x = torch.cat((input_velocity, density, viscosity), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.tanh(self.conv11(x))

        # Outputs are velocity field and pressure field.
        velocity = x[:, :2, :, :]
        pressure = x[:, 2:3, :, :]
        return velocity, pressure

gradient_x_kernel = torch.tensor([[-1, 0, 1]], requires_grad=False).unsqueeze(0).unsqueeze(0).float().cuda()
gradient_y_kernel = torch.tensor([[-1], [0], [1]], requires_grad=False).unsqueeze(0).unsqueeze(0).float().cuda()

def gradient_x(field):
    return F.conv2d(field, gradient_x_kernel, padding=[0, 1])

def gradient_y(field):
    return F.conv2d(field, gradient_y_kernel, padding=[1, 0])

def get_loss(velocity, pressure, velocity_target, density, viscosity):
    # Convert to 4D tensors from scalars.
    density = density.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    viscosity = viscosity.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    # Set velocity boundary conditions - replace values in output velocity field
    # by non-zero values from target velocity field.
    velocity_mask = 1 - torch.where(torch.abs(velocity_target) > 0, torch.ones(velocity.shape).cuda(), torch.zeros(velocity.shape).cuda())
    velocity = velocity * velocity_mask + velocity_target
    vx = velocity[:, :1, :, :]
    vy = velocity[:, 1:2, :, :]

    # Velocity derivatives.
    vx_x = gradient_x(vx)
    vx_y = gradient_y(vx)
    vy_x = gradient_x(vy)
    vy_y = gradient_y(vy)
    vx_xx = gradient_x(vx_x)
    vx_yy = gradient_y(vx_y)
    vy_xx = gradient_x(vy_x)
    vy_yy = gradient_y(vy_y)
    
    # Pressure derivatives.
    p_x = gradient_x(pressure)
    p_y = gradient_y(pressure)
    p_xx = gradient_x(p_x)
    p_yy = gradient_y(p_y)

    # Compute navier stokes momentum equation (assuming zero external field and dv/dt = 0 - steady flow) set to zero and optimize for.
    convection_x = vx_x * vx + vx_y * vy
    convection_y = vy_x * vx + vy_y * vy
    diffusion_x = (vx_xx + vx_yy) * viscosity
    diffusion_y = (vy_xx + vy_yy) * viscosity
    internal_source_x = p_x / density
    internal_source_y = p_y / density
    navier_stokes_momentum_loss_x = (convection_x - diffusion_x + internal_source_x) ** 2
    navier_stokes_momentum_loss_y = (convection_y - diffusion_y + internal_source_y) ** 2
    navier_stokes_momentum_loss = navier_stokes_momentum_loss_x.mean() + navier_stokes_momentum_loss_y.mean()

    # Divergence needs to be zero.
    divergence = vx_x + vy_y
    divergence_loss = (divergence ** 2).mean()

    # Pressure field loss.
    # See:
    # - https://physics.stackexchange.com/questions/187889/incompressible-navier-stokes-pressure-solve-in-simulations
    # - https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/13_Step_10.ipynb
    pressure_eq_lhs = p_xx + p_yy
    pressure_eq_rhs = (vx_x * vx_x + 2 * vx_y * vy_x + vy_y * vy_y) * density
    pressure_loss = (pressure_eq_lhs + pressure_eq_rhs) ** 2
    pressure_loss = pressure_loss.mean()
    
    return divergence_loss, navier_stokes_momentum_loss, pressure_loss

def compose_velocity_field_matrix(velocities, row_labels, col_labels):
    rows, cols = len(row_labels), len(col_labels)
    assert len(velocities) == rows * cols
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=100)
    field_height, field_width = velocities.shape[2:4]
    for y in range(rows):
        for x in range(cols):
            index = y * cols + x
            ax = axs[y][x]
            ax.cla()
            ax.set(ylabel=str(row_labels[y]), xlabel=str(col_labels[x]))
            ax.label_outer()
            ax.quiver(np.arange(0, field_width * 10, 10), np.arange(0, field_height * 10, 10)[::-1], velocities[index, 0, :, :], -velocities[index, 1, :, :], scale=20, headwidth=5)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = np.reshape(img, fig.canvas.get_width_height()[::-1] + (3,))
    img = np.transpose(img, (2, 0, 1))
    return img

def compose_img_matrix(array, rows, cols):
    assert len(array) == rows * cols
    num_channels, field_height, field_width = array.shape[1:4]
    array = np.transpose(array, (0, 3, 2, 1))
    array = np.reshape(array, (rows, cols * field_width, field_height, num_channels))
    array = np.transpose(array, (0, 2, 1, 3))
    array = np.reshape(array, (rows * field_height, cols * field_width, num_channels))
    array = np.transpose(array, (2, 0, 1))
    return array

def train():
    # Set up training data.
    training_dataset = TrainingDataset(TRAINING_SET_SIZE)
    data_loader = torch.utils.data.DataLoader(training_dataset, BATCH_SIZE, True)

    # Set up validation data.
    val_velocities = torch.tensor(VAL_VELOCITY.transpose((2, 0, 1))).float().cuda().unsqueeze(0).repeat((VAL_SAMPLES_COUNT, 1, 1, 1))
    val_densities = torch.tensor(VAL_DENSITIES).cuda().view([VAL_DENSITIES_COUNT, 1]).repeat([1, VAL_VISCOSITIES_COUNT]).view([VAL_SAMPLES_COUNT])
    val_viscosities = torch.tensor(VAL_VISCOSITIES).cuda().repeat([VAL_DENSITIES_COUNT]) / val_densities

    # Set up model.
    model = Model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Set up outputs/logging.
    writer = SummaryWriter(comment=EXPERIMENT_TITLE)

    step = 0
    for epoch in range(EPOCH_COUNT):
        for input_velocity, density, viscosity in data_loader:
            # Training step.
            optimizer.zero_grad()
            velocity, pressure = model.forward(input_velocity, density, viscosity)
            divergence_loss, momentum_loss, pressure_loss = get_loss(velocity, pressure, input_velocity, density, viscosity)
            loss = divergence_loss + momentum_loss + pressure_loss
            loss.backward()
            optimizer.step()

            # Loss logging.
            if step % 10 == 0:
                print(step, loss, divergence_loss, momentum_loss, pressure_loss)
                writer.add_scalar("loss", loss, step)

            # "Validation step" - producing pressure and velocity field images.
            if step % 1000 == 0:
                velocity, pressure = model.forward(val_velocities, val_densities, val_viscosities)
                velocity = velocity.detach().cpu().numpy()
                pressure = pressure.detach().cpu().numpy()

                writer.add_image("velocity", compose_velocity_field_matrix(velocity, VAL_DENSITIES, VAL_VISCOSITIES), step)
                writer.add_image("pressure", compose_img_matrix(pressure, VAL_DENSITIES_COUNT, VAL_VISCOSITIES_COUNT), step)

            step += 1

    torch.save(model, "model_{}.pt".format(EXPERIMENT_TITLE))

if __name__ == '__main__':
    train()
