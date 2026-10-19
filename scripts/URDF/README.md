# Onshape to URDF Export Setup

This directory contains the configuration and scripts needed to export your Onshape CAD model to URDF format.

## Important: Single-Arm URDF for Bilateral Systems

**For bilateral (two-arm) robots: Export only ONE arm (6 or 7 DOF) from Onshape.**

The gravity compensation system automatically mirrors the single-arm URDF to work with bilateral systems:
- **Single-arm URDF (6 or 7 DOF)** → Automatically mirrored to **12 or 14 DOF** for bilateral control
- The URDF should represent the **right arm** (the system will mirror it for the left arm)
- This avoids needing to model both arms separately in Onshape

## Quick Start

### 1. Install Dependencies

```bash
pip install onshape-to-robot python-dotenv
```

### 2. Set Up API Credentials

1. Get your Onshape API keys from: https://dev.onshape.com/keys
2. Copy `.env.example` to `.env`:
   ```bash
   cp scripts/URDF/.env.example scripts/URDF/.env
   ```
3. Edit `scripts/URDF/.env` and fill in your credentials:
   ```
   ONSHAPE_ACCESS_KEY=your_actual_access_key
   ONSHAPE_SECRET_KEY=your_actual_secret_key
   ```

### 3. Configure Export Settings

Set `url` in `scripts/URDF/config.json` to your Onshape document URL before
exporting. The supplied config leaves it empty. The checked-in URDF and STL
files can be used directly without Onshape access.

Other export settings:
- `package_name`: ROS package name (optional, for ROS workflows)
- `joint_properties`: Override joint limits, effort, velocity, friction
- `set_zero_mass_to_fixed`: Set to `true` to exclude inertial data for fixed bodies
- `additional_xml`: Path to additional XML file to include in URDF

### 4. Export URDF

```bash
# Basic export (outputs to ./urdf_output)
python scripts/URDF/onshape_to_urdf.py

# Custom output directory
python scripts/URDF/onshape_to_urdf.py --output ./my_robot_urdf

# Use custom config file
python scripts/URDF/onshape_to_urdf.py --config ./custom_config.json
```

## Configuration Options

### `config.json` Structure

```json
{
    "url": "https://cad.onshape.com/documents/...",
    "output_format": "urdf",
    "package_name": "my_robot",
    "additional_xml": "",
    "set_zero_mass_to_fixed": false,
    "joint_properties": {
        "*": {
            "max_effort": 10.0,
            "max_velocity": 6.0,
            "friction": 0.5
        },
        "joint_name": {
            "max_effort": 20.0,
            "max_velocity": 10.0,
            "friction": 0.1,
            "limits": [0.5, 1.2],
            "type": "revolute"
        }
    }
}
```

### Joint Properties

- `max_effort`: Maximum effort (torque/force) for the joint
- `max_velocity`: Maximum velocity for the joint
- `friction`: Joint friction coefficient
- `limits`: `[min, max]` joint limits (overrides automatic detection)
- `type`: Joint type (`revolute`, `prismatic`, `continuous`, `fixed`, etc.)

Use `"*"` as a key to set default properties for all joints.

## Output Structure

After export, you'll get:

```
urdf_output/
├── robot.urdf          # Main URDF file (single arm, 6 or 7 DOF)
├── meshes/             # STL mesh files for each link
│   ├── link1.stl
│   ├── link2.stl
│   └── ...
└── config.json         # Copy of export config
```

**Note:** For bilateral systems, the URDF should contain only one arm. The system will automatically mirror it for the second arm during gravity compensation.

## Troubleshooting

### API Key Errors

- Verify your API keys are correct in `.env`
- Check that `.env` file is in the `scripts/` directory
- Ensure no extra spaces or quotes around values in `.env`

### Export Fails

- Verify the Onshape URL in `config.json` is correct
- Check that you have access to the Onshape document
- Ensure your Onshape model has proper mate connectors for joints

### Joint Issues

- Review joint properties in `config.json`
- Check mate connectors in Onshape are properly configured
- Manually adjust joint limits if needed

## References

- [Onshape-to-Robot Documentation](https://onshape-to-robot.readthedocs.io/en/latest/exporter_urdf.html)
- [Onshape API Keys](https://dev.onshape.com/keys)
- [URDF Specification](http://wiki.ros.org/urdf/XML)

