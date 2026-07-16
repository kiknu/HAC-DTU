import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import mitsuba as mi
    import numpy as np
    import drjit as dr

    import matplotlib.pyplot as plt
    import imageio_ffmpeg

    from hac.data import get_model_stl_path
    from hac.constants import OBSERVERS

    import marimo as mo

    return OBSERVERS, dr, get_model_stl_path, imageio_ffmpeg, mi, mo, np, plt


@app.cell
def _(OBSERVERS, imageio_ffmpeg, plt):
    # Setup
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    MODEL_NUM = 3
    EXPERIMENT = "0_hor1"
    OBSERVER = OBSERVERS[EXPERIMENT]
    return EXPERIMENT, MODEL_NUM, OBSERVER


@app.cell
def _(MODEL_NUM, get_model_stl_path):
    # Load
    model_stl_path = get_model_stl_path(model_num=MODEL_NUM)
    return (model_stl_path,)


@app.cell
def _(OBSERVER, mi, model_stl_path, np):
    from hac.mitsuba import make_scene
    from hac.mitsuba import MitsubaMesh, RenderingSettings

    m = MitsubaMesh.from_stl_path(model_stl_path)

    width, height = 128, 128
    ssp = 32  # Number of samples per pixel

    rendering_settings = RenderingSettings(width=width, height=height, spp=32)

    phi = 0.0
    mi.set_variant("scalar_mono")  # Need to recompile mitsuba to get scalar_mono
    scene = make_scene(
        mesh=m,
        theta=OBSERVER.theta,
        alpha=OBSERVER.alpha,
        phi=phi,
        observer_distance=OBSERVER.r_obs,
        asteroid_scale=0.2,  # TODO set dynamically
        rendering_settings=rendering_settings,
        albedo=1.0,
    )

    params = mi.traverse(scene)

    base_vertex_positions = (
        np.asarray(params["asteroid.vertex_positions"]).reshape(-1, 3).copy()
    )
    base_vertex_normals = (
        np.asarray(params["asteroid.vertex_normals"]).reshape(-1, 3).copy()
    )


    def rotate_about_minus_z(values, angle_deg):
        angle = np.radians(-angle_deg)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotated = values.copy()
        x = values[:, 0]
        y = values[:, 1]
        rotated[:, 0] = cos_angle * x - sin_angle * y
        rotated[:, 1] = sin_angle * x + cos_angle * y
        return rotated.reshape(-1)


    def set_asteroid_phi(phi):
        params["asteroid.vertex_positions"] = rotate_about_minus_z(
            base_vertex_positions, phi
        )
        params["asteroid.vertex_normals"] = rotate_about_minus_z(base_vertex_normals, phi)
        params.update()


    params
    return scene, set_asteroid_phi


@app.cell
def _(mi, plt, scene, set_asteroid_phi):
    def render_scene(
        scene: mi.Scene,
    ):
        """
        Render one frame and return the bitmap before lightcurve integration.
        """
        return mi.render(scene)


    def display_rendering(img, ax=None):
        """
        Display the rendered image using matplotlib.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        ax.axis("off")
        ax.imshow(mi.util.convert_to_bitmap(img))

        return ax


    set_asteroid_phi(0.0)
    img = render_scene(scene)
    ax = display_rendering(img)

    ax
    return (render_scene,)


@app.cell
def _(np):
    # Render a little movie
    N_PHI = 360
    phis_anim = np.linspace(0, 360, N_PHI)
    return (phis_anim,)


@app.cell
def _(phis_anim, render_scene, scene, set_asteroid_phi):
    # Render frames by mutating the single loaded Mitsuba scene.
    renders = []
    for phi_ in phis_anim:
        set_asteroid_phi(phi_)
        renders.append(render_scene(scene))
    return (renders,)


@app.cell
def _(mi, renders):
    bitmaps = [mi.util.convert_to_bitmap(img) for img in renders]
    return (bitmaps,)


@app.cell
def _(bitmaps, mo, np, plt):
    # Animation for rotating asteroid (notebook-friendly)
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    # Convert Mitsuba renders to RGB bitmaps (uint8 HxWx3)

    fig, ax_anim = plt.subplots(figsize=(6, 6))
    ax_anim.axis("off")
    im = ax_anim.imshow(np.asarray(bitmaps[0]))


    def update(frame):
        im.set_data(np.asarray(bitmaps[frame]))
        return (im,)


    anim = FuncAnimation(
        fig,
        update,
        frames=len(bitmaps),
        blit=True,
        repeat=True,
    )
    anim.save("asteroid_rotation.mp4", writer="ffmpeg", fps=len(bitmaps) // 10)


    vid = HTML(anim.to_html5_video())
    # mo.Html(vid)
    mo.video("asteroid_rotation.mp4")
    return


@app.cell
def _(mi, np, renders):
    from hac.lightcurve_integration import intensity

    lightcurve = [
        intensity(np.asarray(mi.util.convert_to_bitmap(render))) for render in renders
    ]
    return intensity, lightcurve


@app.cell
def _(dr, renders):
    renders_dr = dr.auto.Float(
        [dr.sum(render) for render in renders]
    )  # Convert to DrJit array for DrJit operations
    return


@app.cell
def _(dr, np, renders):
    lightcurve_dr = [dr.sum(render).numpy() for render in renders]
    lightcurve_dr = dr(lightcurve_dr)
    # Threshold
    threshold_mask = lightcurve_dr > 25.5
    lightcurve_dr = dr.where
    lightcurve_dr /= np.mean(lightcurve_dr)
    return (lightcurve_dr,)


@app.cell
def _(intensity, mi, np, renders):
    # Try DrJit
    def intensity_from_render(render):
        """
        Compute the intensity of a rendered image using DrJit.
        """
        # Convert the Mitsuba render to a DrJit array
        bitmap = mi.util.convert_to_bitmap(render)

        return intensity(np.asarray(bitmap))


    intensity_from_render(renders[0])
    return


@app.cell
def _(lightcurve, np):
    lightcurve_normalised = lightcurve / np.mean(lightcurve)
    return (lightcurve_normalised,)


@app.cell
def _(EXPERIMENT, MODEL_NUM, lightcurve_dr, lightcurve_normalised, np, plt):
    from hac.data import load_data

    ds = load_data(model_num=MODEL_NUM, origin="simulated", curve_type="intensity")
    ds = load_data(model_num=MODEL_NUM, origin="simulated", curve_type="intensity")

    plt.plot(lightcurve_normalised, label="Mitsuba simulation")
    plt.plot(ds.data[EXPERIMENT], label="HAC simulation")
    plt.plot(np.asarray(lightcurve_dr), label="Mitsuba DrJit simulation")
    plt.legend()
    return


@app.cell
def _():
    # Source - https://stackoverflow.com/a/71177358
    # Posted by It_is_Chris
    # Retrieved 2026-07-15, License - CC BY-SA 4.0

    import matplotlib

    print(matplotlib.get_cachedir())  # should return something like ~/.matplotlib

    return


if __name__ == "__main__":
    app.run()
