# BluPe deployment entry

The current implementation is documented in:

- [First hardware trial preparation](BLUPE_HARDWARE_TRIAL.md)
- [Platform interfaces and software deployment](PLATFORM_DEPLOYMENT.md)
- [Local model services](CMU_EVALUATION.md)

For the initial trial, use `scripts/serve_blupe_hardware.py` with its default
read-only mode. Complete the measured site configuration and operator setup
before enabling hardware Run. Opening the UI does not join the robot queue;
a later explicitly enabled Run does. The official Runner retains queue and Stop.

The framework generates native joint trajectories through our IK adapter.
BluPe does not supply an assumed grasp skill, and simulation force priors are
not calibrated hardware force commands. No public hardware performance is
claimed by the repository's local simulation tests.
