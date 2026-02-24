"""Map payload generation and migration utilities."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime

from analyzer import gradient_color_from_t


MAP_PAYLOAD_VERSION = 5

logger = logging.getLogger(__name__)


class MapPayloadBuilder:
    """Build, normalize, and backfill map payloads for activity records."""

    def __init__(self, db):
        self.db = db

    def _calculate_distance_from_speed(self, speed_stream, timestamps):
        """
        Calculate cumulative distance from speed and timestamps.

        Args:
            speed_stream: List[float] - speed in m/s
            timestamps: List[datetime] - timestamp for each record

        Returns:
            List[float] - cumulative distance in meters
        """
        distance_stream = []
        cumulative_distance = 0.0

        for i in range(len(speed_stream)):
            if i == 0:
                distance_stream.append(0.0)
                continue

            if timestamps[i] and timestamps[i - 1]:
                time_delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
            else:
                time_delta = 1.0

            speed = speed_stream[i] if speed_stream[i] is not None else (
                speed_stream[i - 1] if i > 0 and speed_stream[i - 1] is not None else 0.0
            )

            distance_increment = speed * time_delta
            cumulative_distance += distance_increment
            distance_stream.append(cumulative_distance)

        return distance_stream

    def _normalize_bounds(self, bounds_value):
        """Validate and normalize bounds into [[min_lat, min_lon], [max_lat, max_lon]]."""
        try:
            if not isinstance(bounds_value, (list, tuple)) or len(bounds_value) != 2:
                return None
            if not isinstance(bounds_value[0], (list, tuple)) or not isinstance(bounds_value[1], (list, tuple)):
                return None
            if len(bounds_value[0]) != 2 or len(bounds_value[1]) != 2:
                return None

            lat1 = float(bounds_value[0][0])
            lon1 = float(bounds_value[0][1])
            lat2 = float(bounds_value[1][0])
            lon2 = float(bounds_value[1][1])

            min_lat, max_lat = sorted([lat1, lat2])
            min_lon, max_lon = sorted([lon1, lon2])

            if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
                return None
            if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
                return None
            if abs(max_lat - min_lat) < 1e-6:
                min_lat -= 0.0005
                max_lat += 0.0005
            if abs(max_lon - min_lon) < 1e-6:
                min_lon -= 0.0005
                max_lon += 0.0005

            return [[min_lat, min_lon], [max_lat, max_lon]]
        except Exception:
            return None

    def _hex_to_rgb(self, color):
        """Convert #RRGGBB to (r, g, b), or None for invalid input."""
        if not isinstance(color, str):
            return None
        color = color.strip()
        if not color.startswith('#') or len(color) != 7:
            return None
        try:
            return (
                int(color[1:3], 16),
                int(color[3:5], 16),
                int(color[5:7], 16),
            )
        except Exception:
            return None

    def _rgb_to_hex(self, r, g, b):
        """Convert RGB ints to #RRGGBB."""
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))
        return f'#{r:02x}{g:02x}{b:02x}'

    def _projection_on_dual_tone(self, rgb):
        """Project RGB onto dual-tone Zinc->Emerald ramp and return (t, distance)."""
        start = (113.0, 113.0, 122.0)
        end = (52.0, 211.0, 153.0)
        vx, vy, vz = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
        wx, wy, wz = (rgb[0] - start[0], rgb[1] - start[1], rgb[2] - start[2])
        vv = vx * vx + vy * vy + vz * vz
        if vv <= 0:
            return 0.5, float('inf')
        t = (wx * vx + wy * vy + wz * vz) / vv
        t = max(0.0, min(1.0, t))
        px = start[0] + t * vx
        py = start[1] + t * vy
        pz = start[2] + t * vz
        dist = math.sqrt((rgb[0] - px) ** 2 + (rgb[1] - py) ** 2 + (rgb[2] - pz) ** 2)
        return t, dist

    def _looks_like_dual_tone_segments(self, segments):
        """Heuristic: detect old Zinc->Emerald palette so we can migrate to 5-color scale."""
        if not isinstance(segments, list) or not segments:
            return False

        sample = segments[: min(120, len(segments))]
        valid = 0
        near_line = 0
        for seg in sample:
            if not isinstance(seg, (list, tuple)) or len(seg) < 5:
                continue
            rgb = self._hex_to_rgb(seg[4])
            if rgb is None:
                continue
            valid += 1
            _, dist = self._projection_on_dual_tone(rgb)
            if dist <= 10.0:
                near_line += 1

        if valid < 10:
            return False
        return (near_line / valid) >= 0.85

    def _convert_dual_tone_segments_to_multicolor(self, segments):
        """Convert old dual-tone segment colors to the Garmin 5-color palette."""
        converted = []
        for seg in segments:
            if not isinstance(seg, (list, tuple)) or len(seg) < 5:
                converted.append(seg)
                continue
            rgb = self._hex_to_rgb(seg[4])
            if rgb is None:
                converted.append(list(seg))
                continue
            t, _ = self._projection_on_dual_tone(rgb)
            converted.append([seg[0], seg[1], seg[2], seg[3], gradient_color_from_t(t)])
        return converted

    def _build_map_payload_from_segments(self, route_segments, max_segments=500):
        """Convert legacy route_segments into versioned map payload with robust bounds."""
        if not isinstance(route_segments, list) or not route_segments:
            return None

        cleaned = []
        for seg in route_segments:
            try:
                if not isinstance(seg, (list, tuple)) or len(seg) < 4:
                    continue
                lat1, lon1, lat2, lon2 = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
                if not (-90 <= lat1 <= 90 and -180 <= lon1 <= 180 and -90 <= lat2 <= 90 and -180 <= lon2 <= 180):
                    continue
                if (abs(lat1) < 1e-6 and abs(lon1) < 1e-6) or (abs(lat2) < 1e-6 and abs(lon2) < 1e-6):
                    continue
                pace_color = seg[4] if len(seg) > 4 and isinstance(seg[4], str) and seg[4].startswith('#') else '#10b981'
                hr_color = seg[5] if len(seg) > 5 and isinstance(seg[5], str) and seg[5].startswith('#') else None
                normalized = [lat1, lon1, lat2, lon2, pace_color]
                if hr_color:
                    normalized.append(hr_color)
                cleaned.append(normalized)
            except Exception:
                continue

        if not cleaned:
            return None

        if self._looks_like_dual_tone_segments(cleaned):
            cleaned = self._convert_dual_tone_segments_to_multicolor(cleaned)

        if len(cleaned) > max_segments:
            step = max(1, len(cleaned) // max_segments)
            sampled = cleaned[::step]
            if sampled[-1] != cleaned[-1]:
                sampled.append(cleaned[-1])
            cleaned = sampled

        lats = [s[0] for s in cleaned] + [s[2] for s in cleaned]
        lons = [s[1] for s in cleaned] + [s[3] for s in cleaned]
        if not lats or not lons:
            return None

        sorted_lats = sorted(lats)
        sorted_lons = sorted(lons)

        def percentile(values, p):
            if not values:
                return 0.0
            idx = (len(values) - 1) * p
            lo = int(idx)
            hi = min(lo + 1, len(values) - 1)
            if lo == hi:
                return float(values[lo])
            frac = idx - lo
            return float(values[lo] + (values[hi] - values[lo]) * frac)

        if len(sorted_lats) > 20:
            min_lat = percentile(sorted_lats, 0.005)
            max_lat = percentile(sorted_lats, 0.995)
            min_lon = percentile(sorted_lons, 0.005)
            max_lon = percentile(sorted_lons, 0.995)
        else:
            min_lat, max_lat = float(sorted_lats[0]), float(sorted_lats[-1])
            min_lon, max_lon = float(sorted_lons[0]), float(sorted_lons[-1])

        bounds = self._normalize_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        if not bounds:
            return None

        center = [
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2,
        ]

        return {
            'v': MAP_PAYLOAD_VERSION,
            'segments': cleaned,
            'bounds': bounds,
            'center': center,
            'segment_count': len(cleaned),
        }

    def _activity_needs_map_payload_backfill(self, activity):
        """Decide whether an activity needs map payload migration."""
        if not isinstance(activity, dict):
            return False
        map_payload = activity.get('map_payload')
        if not isinstance(map_payload, dict):
            return True
        version = int(activity.get('map_payload_version') or map_payload.get('v') or 1)
        segments = map_payload.get('segments')
        bounds = self._normalize_bounds(map_payload.get('bounds'))
        if not isinstance(segments, list) or not segments:
            return True
        if not bounds:
            return True
        if version < MAP_PAYLOAD_VERSION:
            return version < 4
        return False

    def _get_or_backfill_map_payload(self, activity):
        """Return normalized map payload, creating/persisting current-version payload when needed."""
        if not isinstance(activity, dict):
            return None

        map_payload = activity.get('map_payload')
        if isinstance(map_payload, dict):
            bounds = self._normalize_bounds(map_payload.get('bounds'))
            segments = map_payload.get('segments')
            version = int(activity.get('map_payload_version') or map_payload.get('v') or 1)
            if bounds and isinstance(segments, list) and segments:
                map_payload['bounds'] = bounds
                activity['map_payload'] = map_payload
                activity['map_payload_version'] = version
                if version >= MAP_PAYLOAD_VERSION:
                    return map_payload
                if version >= 4:
                    return map_payload

        legacy_segments = activity.get('route_segments')
        if (not legacy_segments) and isinstance(map_payload, dict):
            legacy_segments = map_payload.get('segments')

        new_payload = self._build_map_payload_from_segments(legacy_segments, max_segments=500)
        if not new_payload:
            return None

        activity['map_payload'] = new_payload
        activity['map_payload_version'] = new_payload.get('v', MAP_PAYLOAD_VERSION)
        activity['route_segments'] = new_payload.get('segments', [])
        activity['bounds'] = new_payload.get('bounds', [[0, 0], [0, 0]])

        activity_hash = activity.get('hash') or activity.get('db_hash')
        if activity_hash:
            try:
                self.db.update_activity_map_payload(
                    activity_hash,
                    new_payload,
                    route_segments=new_payload.get('segments'),
                    bounds=new_payload.get('bounds'),
                    map_payload_version=new_payload.get('v', MAP_PAYLOAD_VERSION),
                )
            except Exception as ex:
                context = json.dumps({
                    'activity_hash': activity_hash,
                    'version': new_payload.get('v', MAP_PAYLOAD_VERSION),
                })
                logger.warning("Map payload backfill warning %s: %s", context, ex)

        return new_payload

    async def _backfill_map_payloads_for_loaded_activities(self, activities_data):
        """Background migration for loaded activities to versioned map payloads."""
        if not activities_data:
            return

        started_at = datetime.utcnow().isoformat()
        for idx, activity in enumerate(activities_data):
            try:
                if self._activity_needs_map_payload_backfill(activity):
                    self._get_or_backfill_map_payload(activity)
            except Exception as ex:
                logger.warning("Background map payload backfill warning: %s", ex)

            if idx % 20 == 0:
                await asyncio.sleep(0)

        logger.debug(
            "Map payload backfill completed (%s)",
            json.dumps({'count': len(activities_data), 'started_at': started_at}),
        )

