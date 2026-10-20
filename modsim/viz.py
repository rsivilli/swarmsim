from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from .geo import km_to_latlon
from .models import ScenarioConfig, SimState
from .sim import run_iter, build_sensors

# Try to import cartopy for map visualization
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def add_map_background(ax, map_style: str = "terrain"):
    """
    Add background imagery to a cartopy GeoAxes.

    Args:
        ax: Cartopy GeoAxes
        map_style: One of "terrain", "satellite", "street", "toner", or "basic"
    """
    if not HAS_CARTOPY:
        return

    # Check if scipy or pykdtree is available for image tiles
    try:
        import scipy  # noqa: F401  (availability check)

        HAS_IMAGE_TILES = True
    except ImportError:
        try:
            import pykdtree  # noqa: F401  (availability check)

            HAS_IMAGE_TILES = True
        except ImportError:
            HAS_IMAGE_TILES = False

    def _fallback_to_basic():
        """Helper to render basic map features"""
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.6, zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.5, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=":", zorder=1)
        ax.add_feature(cfeature.STATES, linewidth=0.5, linestyle=":", alpha=0.7, zorder=1)

    if map_style == "basic":
        # Basic cartopy features (original style, but with higher opacity)
        _fallback_to_basic()
    elif not HAS_IMAGE_TILES:
        # Fallback to basic features if image tiles not available
        print(
            f"Warning: Map style '{map_style}' requires scipy or pykdtree. Falling back to 'basic' style."
        )
        print("Install with: pip install scipy")
        _fallback_to_basic()
    else:
        # Try to fetch tiles, fall back to basic if it fails (no internet, service down, etc.)
        try:
            if map_style == "terrain":
                # Stadia Maps terrain tiles (formerly Stamen)
                tiles = cimgt.Stamen("terrain-background")
                ax.add_image(tiles, 8)
            elif map_style == "satellite":
                # Google satellite imagery (requires internet)
                tiles = cimgt.GoogleTiles(style="satellite")
                ax.add_image(tiles, 8)
            elif map_style == "street":
                # OpenStreetMap tiles
                tiles = cimgt.OSM()
                ax.add_image(tiles, 8)
            elif map_style == "toner":
                # Stadia Maps toner tiles (formerly Stamen)
                tiles = cimgt.Stamen("toner-lite")
                ax.add_image(tiles, 8)
            else:
                # Stock image fallback
                ax.stock_img()
        except (OSError, ConnectionError, RuntimeError, ValueError) as e:
            # If tile fetching fails (no internet, service down, etc.), fall back to basic
            print(f"Warning: Could not fetch '{map_style}' map tiles: {str(e)}")
            print("Falling back to 'basic' style (offline mode)")
            _fallback_to_basic()


def plot_final(state: SimState, cfg: ScenarioConfig, map_style: str = "terrain") -> None:
    # Check if we should use geographic projection
    use_geo = cfg.reference_origin is not None and HAS_CARTOPY

    # Create figure with different subplot configuration based on map availability
    if use_geo:
        # Use cartopy projection for first panel
        fig = plt.figure(figsize=(20, 7))
        ax_xy = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
        ax_xz = fig.add_subplot(1, 3, 2)
        ax2 = fig.add_subplot(1, 3, 3)
        axes = [ax_xy, ax_xz, ax2]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        ax_xy = axes[0]
        ax_xz = axes[1]
        ax2 = axes[2]

    # --- Panel 0: Top-down view (X-Y) or Map projection ---
    if use_geo:
        # Convert bounds to lat/lon for map view
        origin_lat, origin_lon = cfg.reference_origin
        min_lat, min_lon = km_to_latlon(0, 0, cfg.reference_origin)
        max_lat, max_lon = km_to_latlon(cfg.area.width_km, cfg.area.height_km, cfg.reference_origin)

        # Set extent with some padding
        lat_padding = (max_lat - min_lat) * 0.1
        lon_padding = (max_lon - min_lon) * 0.1
        ax_xy.set_extent(
            [
                min_lon - lon_padding,
                max_lon + lon_padding,
                min_lat - lat_padding,
                max_lat + lat_padding,
            ],
            crs=ccrs.PlateCarree(),
        )

        # Add map background (terrain, satellite, street, toner, or basic)
        add_map_background(ax_xy, map_style)

        ax_xy.gridlines(
            draw_labels=True,
            dms=True,
            x_inline=False,
            y_inline=False,
            alpha=0.5,
            linestyle="--",
            zorder=10,
        )
        ax_xy.set_title(f"{cfg.name} — Geographic Map View")
    elif cfg.reference_origin is not None:
        # Fallback to simple lat/lon plot if cartopy not available
        min_lat, min_lon = km_to_latlon(0, 0, cfg.reference_origin)
        max_lat, max_lon = km_to_latlon(cfg.area.width_km, cfg.area.height_km, cfg.reference_origin)

        ax_xy.set_xlim(min_lon, max_lon)
        ax_xy.set_ylim(min_lat, max_lat)
        ax_xy.set_aspect("equal")
        ax_xy.set_xlabel("Longitude (°)")
        ax_xy.set_ylabel("Latitude (°)")
        ax_xy.set_title(f"{cfg.name} — Geographic View (install cartopy for map)")
        ax_xy.grid(True, alpha=0.3, linestyle="--")
    else:
        ax_xy.set_xlim(0, cfg.area.width_km)
        ax_xy.set_ylim(0, cfg.area.height_km)
        ax_xy.set_aspect("equal")
        ax_xy.set_xlabel("x (km)")
        ax_xy.set_ylabel("y (km)")
        ax_xy.set_title(f"{cfg.name} — Top-Down View")

    # Sensor range circles + markers
    for i, sensor in enumerate(state.sensors):
        if use_geo:
            # Convert sensor position to lat/lon
            sensor_lat, sensor_lon = km_to_latlon(sensor.x, sensor.y, cfg.reference_origin)
            # For geographic view, approximate range circle in degrees
            # This is a rough approximation - proper implementation would use geodesic circles
            range_deg = state.sensor_range[i] / 111.0  # km to degrees (approximate)
            circle = plt.Circle(
                (sensor_lon, sensor_lat),
                range_deg,
                fill=True,
                color="blue",
                alpha=0.08,
                linestyle="--",
                linewidth=0.5,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            ax_xy.add_patch(circle)
            ax_xy.plot(
                sensor_lon,
                sensor_lat,
                "b^",
                markersize=6,
                alpha=0.8,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
        elif cfg.reference_origin is not None:
            # Fallback without cartopy
            sensor_lat, sensor_lon = km_to_latlon(sensor.x, sensor.y, cfg.reference_origin)
            range_deg = state.sensor_range[i] / 111.0
            circle = plt.Circle(
                (sensor_lon, sensor_lat),
                range_deg,
                fill=True,
                color="blue",
                alpha=0.08,
                linestyle="--",
                linewidth=0.5,
            )
            ax_xy.add_patch(circle)
            ax_xy.plot(sensor_lon, sensor_lat, "b^", markersize=5, alpha=0.7)
        else:
            circle = plt.Circle(
                (sensor.x, sensor.y),
                state.sensor_range[i],
                fill=True,
                color="blue",
                alpha=0.08,
                linestyle="--",
                linewidth=0.5,
            )
            ax_xy.add_patch(circle)
            ax_xy.plot(sensor.x, sensor.y, "b^", markersize=5, alpha=0.7)

    # Detection events as scatter
    if state.detection_log:
        if use_geo:
            # Convert detection positions to lat/lon
            det_coords = [
                km_to_latlon(d.sensor_x, d.sensor_y, cfg.reference_origin)
                for d in state.detection_log
            ]
            det_lats = [coord[0] for coord in det_coords]
            det_lons = [coord[1] for coord in det_coords]
            ax_xy.scatter(
                det_lons,
                det_lats,
                c="red",
                s=3,
                alpha=0.08,
                label="detections",
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
        elif cfg.reference_origin is not None:
            # Fallback without cartopy
            det_coords = [
                km_to_latlon(d.sensor_x, d.sensor_y, cfg.reference_origin)
                for d in state.detection_log
            ]
            det_lats = [coord[0] for coord in det_coords]
            det_lons = [coord[1] for coord in det_coords]
            ax_xy.scatter(det_lons, det_lats, c="red", s=2, alpha=0.05, label="detections")
        else:
            det_x = [d.sensor_x for d in state.detection_log]
            det_y = [d.sensor_y for d in state.detection_log]
            ax_xy.scatter(det_x, det_y, c="red", s=2, alpha=0.05, label="detections")

    ax_xy.legend(loc="upper right", fontsize=8)

    # --- Panel 1: Side profile view (X-Z) ---
    ax_xz.set_xlim(0, cfg.area.width_km)

    # Calculate max altitude from sensors and objects
    max_z = 1.0  # minimum default
    if state.sensors:
        max_z = max(max_z, max(s.z for s in state.sensors))
    if state.object_snapshots:
        # Get max altitude from all snapshots
        for snapshot in state.object_snapshots:
            if snapshot.objects:
                max_z = max(max_z, max(o.z for o in snapshot.objects))

    ax_xz.set_ylim(0, max_z * 1.1)
    ax_xz.set_xlabel("x (km)")
    ax_xz.set_ylabel("altitude (km)")
    ax_xz.set_title("Side Profile (X-Z)")

    # Plot sensors and ranges in side view
    for i, sensor in enumerate(state.sensors):
        ax_xz.plot(sensor.x, sensor.z, "b^", markersize=5, alpha=0.7)
        circle = plt.Circle(
            (sensor.x, sensor.z),
            state.sensor_range[i],
            fill=False,
            color="blue",
            alpha=0.3,
            linestyle="--",
            linewidth=0.5,
        )
        ax_xz.add_patch(circle)

    # --- Panel 2: Coverage over time ---
    if state.timestep_stats:
        times = [s.time_sec for s in state.timestep_stats]
        coverages = [s.coverage_pct for s in state.timestep_stats]
        obj_counts = [s.total_objects for s in state.timestep_stats]

        ax2.plot(times, coverages, "g-", linewidth=1.5, label="Coverage %")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Coverage %", color="green")
        ax2.set_title("Coverage & Object Count Over Time")
        ax2.set_ylim(0, 105)

        ax2b = ax2.twinx()
        ax2b.plot(times, obj_counts, "b--", linewidth=1, alpha=0.6, label="Objects in area")
        ax2b.set_ylabel("Objects in area", color="blue")

        ax2.legend(loc="upper left", fontsize=8)
        ax2b.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


def animate_scenario(
    cfg: ScenarioConfig,
    interval_ms: int = 50,
    save_path: str | None = None,
    map_style: str = "terrain",
    frame_step: int = 1,
) -> None:
    if frame_step < 1:
        raise ValueError(f"frame_step must be >= 1, got {frame_step}")

    sensor_positions, sensor_ranges = build_sensors(cfg)

    # Collect frames first (for small/medium sims this is fine), keeping every
    # frame_step-th timestep so long runs don't render one frame per dt
    frames: list[tuple] = []
    for i, frame_data in enumerate(run_iter(cfg)):
        if i % frame_step == 0:
            frames.append(frame_data)

    if not frames:
        print("No frames to animate.")
        return

    # Calculate max altitude for side profile
    max_z = 1.0
    if sensor_positions:
        max_z = max(max_z, max(s.z for s in sensor_positions))
    for frame in frames:
        _, _, objects, _, _, _ = frame
        if objects:
            max_z = max(max_z, max(o.z for o in objects))

    # Check if we should use geographic projection
    use_geo = cfg.reference_origin is not None and HAS_CARTOPY

    # Create figure with different configuration based on map availability
    if use_geo:
        # Use cartopy projection for first panel
        fig = plt.figure(figsize=(16, 8))
        ax_xy = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax_xz = fig.add_subplot(1, 2, 2)
    else:
        fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(16, 8))

    # Setup top-down view (X-Y) or Map view
    if use_geo:
        # Convert bounds to lat/lon for map view
        min_lat, min_lon = km_to_latlon(0, 0, cfg.reference_origin)
        max_lat, max_lon = km_to_latlon(cfg.area.width_km, cfg.area.height_km, cfg.reference_origin)

        # Set extent with some padding
        lat_padding = (max_lat - min_lat) * 0.1
        lon_padding = (max_lon - min_lon) * 0.1
        ax_xy.set_extent(
            [
                min_lon - lon_padding,
                max_lon + lon_padding,
                min_lat - lat_padding,
                max_lat + lat_padding,
            ],
            crs=ccrs.PlateCarree(),
        )

        # Add map background (terrain, satellite, street, toner, or basic)
        add_map_background(ax_xy, map_style)

        ax_xy.gridlines(
            draw_labels=True,
            dms=True,
            x_inline=False,
            y_inline=False,
            alpha=0.5,
            linestyle="--",
            zorder=10,
        )
        ax_xy.set_title("Geographic Map View")
    elif cfg.reference_origin is not None:
        # Fallback to simple lat/lon plot if cartopy not available
        min_lat, min_lon = km_to_latlon(0, 0, cfg.reference_origin)
        max_lat, max_lon = km_to_latlon(cfg.area.width_km, cfg.area.height_km, cfg.reference_origin)
        ax_xy.set_xlim(min_lon, max_lon)
        ax_xy.set_ylim(min_lat, max_lat)
        ax_xy.set_aspect("equal")
        ax_xy.set_xlabel("Longitude (°)")
        ax_xy.set_ylabel("Latitude (°)")
        ax_xy.set_title("Geographic View (install cartopy for map)")
        ax_xy.grid(True, alpha=0.3, linestyle="--")
    else:
        ax_xy.set_xlim(0, cfg.area.width_km)
        ax_xy.set_ylim(0, cfg.area.height_km)
        ax_xy.set_aspect("equal")
        ax_xy.set_xlabel("x (km)")
        ax_xy.set_ylabel("y (km)")
        ax_xy.set_title("Top-Down View")

    # Setup side profile view (X-Z)
    ax_xz.set_xlim(0, cfg.area.width_km)
    ax_xz.set_ylim(0, max_z * 1.1)
    ax_xz.set_xlabel("x (km)")
    ax_xz.set_ylabel("altitude (km)")
    ax_xz.set_title("Side Profile")

    # Static: sensor circles on both views
    for i, sensor in enumerate(sensor_positions):
        # Top-down view / Map view
        if use_geo:
            sensor_lat, sensor_lon = km_to_latlon(sensor.x, sensor.y, cfg.reference_origin)
            range_deg = sensor_ranges[i] / 111.0
            circle_xy = plt.Circle(
                (sensor_lon, sensor_lat),
                range_deg,
                fill=True,
                color="blue",
                alpha=0.06,
                linestyle="--",
                linewidth=0.5,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            ax_xy.add_patch(circle_xy)
            ax_xy.plot(
                sensor_lon,
                sensor_lat,
                "b^",
                markersize=4,
                alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
        elif cfg.reference_origin is not None:
            sensor_lat, sensor_lon = km_to_latlon(sensor.x, sensor.y, cfg.reference_origin)
            range_deg = sensor_ranges[i] / 111.0
            circle_xy = plt.Circle(
                (sensor_lon, sensor_lat),
                range_deg,
                fill=True,
                color="blue",
                alpha=0.06,
                linestyle="--",
                linewidth=0.5,
            )
            ax_xy.add_patch(circle_xy)
            ax_xy.plot(sensor_lon, sensor_lat, "b^", markersize=4, alpha=0.5)
        else:
            circle_xy = plt.Circle(
                (sensor.x, sensor.y),
                sensor_ranges[i],
                fill=True,
                color="blue",
                alpha=0.06,
                linestyle="--",
                linewidth=0.5,
            )
            ax_xy.add_patch(circle_xy)
            ax_xy.plot(sensor.x, sensor.y, "b^", markersize=4, alpha=0.5)

        # Side profile view
        circle_xz = plt.Circle(
            (sensor.x, sensor.z),
            sensor_ranges[i],
            fill=False,
            color="blue",
            alpha=0.2,
            linestyle="--",
            linewidth=0.5,
        )
        ax_xz.add_patch(circle_xz)
        ax_xz.plot(sensor.x, sensor.z, "b^", markersize=4, alpha=0.5)

    # Dynamic elements for both views
    if use_geo:
        obj_scatter_xy = ax_xy.scatter(
            [], [], c="black", s=10, zorder=5, transform=ccrs.PlateCarree()
        )
        det_scatter_xy = ax_xy.scatter(
            [], [], c="red", s=30, marker="x", zorder=6, transform=ccrs.PlateCarree()
        )
    else:
        obj_scatter_xy = ax_xy.scatter([], [], c="black", s=10, zorder=5)
        det_scatter_xy = ax_xy.scatter([], [], c="red", s=30, marker="x", zorder=6)

    obj_scatter_xz = ax_xz.scatter([], [], c="black", s=10, zorder=5)
    det_scatter_xz = ax_xz.scatter([], [], c="red", s=30, marker="x", zorder=6)

    title = fig.suptitle("")
    stats_text = ax_xy.text(
        0.02,
        0.98,
        "",
        transform=ax_xy.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    def update(frame_idx: int):
        step, time_sec, objects, detections, track_states, stats = frames[frame_idx]

        # Update top-down view (X-Y) or Map view
        if objects:
            if use_geo or cfg.reference_origin is not None:
                # Convert to lat/lon for geographic view
                obj_coords = [km_to_latlon(o.x, o.y, cfg.reference_origin) for o in objects]
                obj_xy = np.array(
                    [[lon, lat] for lat, lon in obj_coords]
                )  # lon, lat order for plotting
            else:
                obj_xy = np.array([[o.x, o.y] for o in objects])
            obj_scatter_xy.set_offsets(obj_xy)
        else:
            obj_scatter_xy.set_offsets(np.empty((0, 2)))

        # Update side profile view (X-Z)
        if objects:
            obj_xz = np.array([[o.x, o.z] for o in objects])
            obj_scatter_xz.set_offsets(obj_xz)
        else:
            obj_scatter_xz.set_offsets(np.empty((0, 2)))

        # Detection flashes (both views)
        if detections:
            det_obj_ids = {d.obj_id for d in detections}
            det_objs = [o for o in objects if o.obj_id in det_obj_ids]
            if det_objs:
                if use_geo or cfg.reference_origin is not None:
                    # Convert to lat/lon for geographic view
                    det_coords = [km_to_latlon(o.x, o.y, cfg.reference_origin) for o in det_objs]
                    det_xy = np.array(
                        [[lon, lat] for lat, lon in det_coords]
                    )  # lon, lat order for plotting
                else:
                    det_xy = np.array([[o.x, o.y] for o in det_objs])
                det_xz = np.array([[o.x, o.z] for o in det_objs])
                det_scatter_xy.set_offsets(det_xy)
                det_scatter_xz.set_offsets(det_xz)
            else:
                det_scatter_xy.set_offsets(np.empty((0, 2)))
                det_scatter_xz.set_offsets(np.empty((0, 2)))
        else:
            det_scatter_xy.set_offsets(np.empty((0, 2)))
            det_scatter_xz.set_offsets(np.empty((0, 2)))

        title.set_text(f"{cfg.name} — t={time_sec:.0f}s (step {step})")
        msgs = sum(stats.detections_per_sensor)
        active_sensors = sum(1 for c in stats.detections_per_sensor if c > 0)
        stats_text.set_text(
            f"Objects: {stats.total_objects}\n"
            f"Detected: {stats.objects_detected}\n"
            f"Coverage: {stats.coverage_pct:.1f}%\n"
            f"Msgs: {msgs}  ({active_sensors} sensors)"
        )

        return obj_scatter_xy, det_scatter_xy, obj_scatter_xz, det_scatter_xz, title, stats_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if save_path:
        fps = max(1, 1000 // interval_ms)
        ext = Path(save_path).suffix.lower()
        if ext == ".gif":
            ani.save(save_path, writer="pillow", fps=fps)
        elif ext in (".mp4", ".mov", ".avi", ".mkv"):
            ani.save(save_path, writer="ffmpeg", fps=fps)
        elif ext == ".html":
            ani.save(save_path, writer="html", fps=fps)
        else:
            # Default to pillow for unknown extensions
            ani.save(save_path, writer="pillow", fps=fps)
        plt.close(fig)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
