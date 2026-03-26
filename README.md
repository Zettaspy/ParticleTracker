# ParticleTracker

Tracks particles in an ocean current using video feed. Provides data about the current and particles in that given system.

## How does it work?
- Utilizes dense optical flow with the Farnback algorithm to calcuate particle information.
- Draws arrows to indicate stats to the particles in the video.
- Prints out the average magnitude and speed of all particles.

## Known Issues:

- Particles can still be lost if too faint.
- Data is avergaed with all the motion of the water (should be segmented due to terbulant flow)
- Weights for masks need to be tweaked.

## Upcoming Stuff:

- Fix known issues
- Add user implemented weights if default are not satisfactory
- Turn it into a UI interface
