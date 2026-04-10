# AUSM-WebGPU
A high-performance 2D inviscid compressible fluid solver accelerated by WebGPU

This project implements AUSM-family finite volume schemes on curvilinear body-fitted grids to simulate transonic and supersonic flows directly in the browser.

### Features
* O-grid generation using elliptic PDE
* SLAU2, SLAU, AUSM<sup>+</sup>-up implementations
  * Supports all-speed flows, from Mach 0.1 to 10+
* MUSCL reconstructed interface states for sharper shocks
* TVD RK3 explicit time integration
* Automatic CFL-based dt calculation
* Automatic steps/frame adjustment based on frame rate
* NACA 4-digit airfoil generator
* Airfoil .dat loader
* Render numerical schlieren, density, temp, pressure, mach, velocity, and more
* Contour rendering in frag shader

### Real-time interactive performance
* 60 fps @ ~65 steps/frame (~4000 steps/sec) on RTX 4070 mobile with 512\*384 grid size
  * Simulation timestep is constrained by CFL condition to maintain numerical stability
* Compute and rendering are done entirely on GPU with minimal CPU-GPU data transfer
* If you are on a laptop you may need to assign your browser to use discrete GPU for good performance

### Planned features
* Automatically determine when to stop grid PDE solver from residuals
* C grid generation
* Hyperbolic PDE-based grid generation
* Boundary pressure based lift and drag calculation
* Explicitly define bind group layouts to allow running on devices that don't support `f32-filterable`

### Potential future features
* Flow field visualization
* Viscosity
* Real units

### Sources

Kitamura, K., & Shima, E. (2013). Towards shock-stable and accurate hypersonic heating computations: A new pressure flux for AUSM-family schemes. Journal of Computational Physics, 245, 62–83. https://doi.org/10.1016/j.jcp.2013.02.046

Liou, M. (n.d.). A sequel to AUSM, Part II: AUSM+-up for all speeds. Journal of Computational Physics, 214(1), 137–170. https://doi.org/10.1016/j.jcp.2005.09.020

Shima, E., & Kitamura, K. (2011). Parameter-Free Simple Low-Dissipation AUSM-Family Scheme for all speeds. AIAA Journal, 49(8), 1693–1709. https://doi.org/10.2514/1.j050905

Thompson, J. F., Thames, F. C., & Mastin, C. (1974). Automatic numerical generation of body-fitted curvilinear coordinate system for field containing any number of arbitrary two-dimensional bodies. Journal of Computational Physics, 15(3), 299–319. https://doi.org/10.1016/0021-9991(74)90114-4
