# ParticleTracker

Tracks particles in an ocean current using video feed. Provides data about the current and particles in that given system.

## How does it work?
- Utilizes dense optical flow with the Optical Motion Tracking algorithm to calcuate particle information.
- Draws circles to indicate id's to the particles in the video.
- Prints out the average magnitude and speed of all particles.
- Create a compiled video with the circle identifiers

## Known Issues:

- Particles can still be lost if too faint.
- Other ghosting problems with particles

## Upcoming Stuff:

- Fix known issues
- Add user implemented weights if default are not satisfactory
- Turn it into a UI interface (maybe)
