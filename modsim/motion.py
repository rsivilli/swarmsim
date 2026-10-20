from __future__ import annotations

import math
import random as stdlib_random

from .models import Area, GroupMode, ObjectGroupConfig, SimObject, SpacingProfile


def spawn_group(group: ObjectGroupConfig, area: Area, rng: stdlib_random.Random) -> list[SimObject]:
    mag = math.hypot(group.heading.x, group.heading.y)
    if mag == 0:
        raise ValueError(f"Group {group.group_id} has zero heading vector")
    hx, hy = group.heading.x / mag, group.heading.y / mag

    objects: list[SimObject] = []
    for i in range(group.count):
        # Position offset from spawn center
        if group.spacing_profile == SpacingProfile.UNIFORM:
            angle = 2 * math.pi * i * (1 + math.sqrt(5)) / 2
            r = group.spawn_radius * math.sqrt(i / max(group.count - 1, 1))
        elif group.spacing_profile == SpacingProfile.GAUSSIAN:
            angle = rng.uniform(0, 2 * math.pi)
            r = abs(rng.gauss(0, group.spawn_radius / 3))
        else:  # RANDOM
            angle = rng.uniform(0, 2 * math.pi)
            r = group.spawn_radius * math.sqrt(rng.random())

        ox = group.spawn_center.x + r * math.cos(angle)
        oy = group.spawn_center.y + r * math.sin(angle)
        ox = max(0.0, min(area.width_km, ox))
        oy = max(0.0, min(area.height_km, oy))

        # Altitude spawning
        if group.spawn_altitude_range > 0:
            oz = group.spawn_altitude + rng.uniform(
                -group.spawn_altitude_range, group.spawn_altitude_range
            )
        else:
            oz = group.spawn_altitude

        # Speed
        if group.speed is not None:
            spd = group.speed
        else:
            spd = rng.uniform(group.speed_min, group.speed_max)

        # Velocity
        if group.mode == GroupMode.SWARM:
            perturb = rng.gauss(0, 0.1)
            cos_p, sin_p = math.cos(perturb), math.sin(perturb)
            obj_hx = hx * cos_p - hy * sin_p
            obj_hy = hx * sin_p + hy * cos_p
            vx = obj_hx * spd
            vy = obj_hy * spd
        else:  # FORMATION — shared velocity
            vx = hx * spd
            vy = hy * spd

        # Vertical velocity
        vz = group.vertical_speed if group.vertical_speed is not None else 0.0

        objects.append(
            SimObject(
                obj_id=f"{group.group_id}-{i:04d}",
                group_id=group.group_id,
                x=ox,
                y=oy,
                z=oz,
                vx=vx,
                vy=vy,
                vz=vz,
            )
        )

    return objects


def step_objects(objects: list[SimObject], dt: float, area: Area) -> list[SimObject]:
    alive: list[SimObject] = []
    for obj in objects:
        obj.x += obj.vx * dt
        obj.y += obj.vy * dt
        obj.z += obj.vz * dt
        # Keep objects within XY bounds and with non-negative altitude
        if 0 <= obj.x <= area.width_km and 0 <= obj.y <= area.height_km and obj.z >= 0:
            alive.append(obj)
    return alive


def step_formation(objects: list[SimObject], dt: float, area: Area) -> list[SimObject]:
    if not objects:
        return []
    vx, vy, vz = objects[0].vx, objects[0].vy, objects[0].vz
    alive: list[SimObject] = []
    for obj in objects:
        obj.x += vx * dt
        obj.y += vy * dt
        obj.z += vz * dt
        # Keep objects within XY bounds and with non-negative altitude
        if 0 <= obj.x <= area.width_km and 0 <= obj.y <= area.height_km and obj.z >= 0:
            alive.append(obj)
    return alive
