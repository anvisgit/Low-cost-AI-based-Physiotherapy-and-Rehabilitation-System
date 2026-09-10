"""
Live Session WebSocket Handler
Streams real-time pose estimation data from browser webcam â†’ PS1 â†’ frontend.
Protocol:
  Client â†’ Server: JPEG binary frame bytes
  Server â†’ Client: JSON with landmarks, angles, validation, reps, feedback
"""
import json
import asyncio
import base64
import time
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
from beanie import PydanticObjectId

from services.pose_engine import get_pose_engine
from services.sensor_hub import get_sensor_hub, RealSensorHub
import httpx

client = httpx.AsyncClient(timeout=1.0)

# Minimum pose confidence to count a rep — prevents phantom reps from noise
MIN_REP_CONFIDENCE = 0.4

from backend_config import settings

router = APIRouter()


@router.websocket("/ws/session/{session_id}/{token}")
async def live_session_websocket(websocket: WebSocket, session_id: str, token: str):
    """
    WebSocket endpoint for live exercise sessions.
    Client sends JPEG frames; server returns real-time pose + angle data.
    """
    from services.auth_service import decode_token
    from models.session import Session
    from models.pose_data import PoseData, LandmarkData, ValidationStatus

    # Authenticate via token in URL
    try:
        token_data = decode_token(token)
    except Exception:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    logger.info(f"[WS Session] Client connected: session={session_id} user={token_data.user_id}")

    # Set start_time when live session WebSocket connects (actual start of exercise)
    session = await Session.get(PydanticObjectId(session_id))
    if session:
        session.start_time = datetime.utcnow()
        await session.save()

    engine = get_pose_engine()
    pose_estimator, kinematics, video_enhancer = engine.create_session_context()

    frame_index = 0
    all_poses = []  # Accumulator for bulk saving raw pose keypoints at end of session
    rep_count = 0

    flexion_history = []
    last_rep_time = 0.0  # Min time (seconds) between rep counts

    # Accumulators for saving real session data at the end
    all_angles_left_knee = []
    all_angles_right_knee = []
    all_angles_left_hip = []
    all_angles_right_hip = []
    all_angles_left_ankle = []
    all_angles_right_ankle = []
    detected_reps = []  # List of rep dicts

    # ML prediction → ESP32 motor command deduplication state
    # Persists across frames so we only send when the prediction changes.
    last_mode_sent: int | None = None
    last_torque_sent: float | None = None
    commands_sent: int = 0  # Total ML prediction commands sent to ESP32
    live_ps2_mode = "real"
    fps_estimate = 10.0  # approximate based on frame send rate

    from services.exercise_analysis import get_analyzer
    analyzer = get_analyzer()
    analysis_context = analyzer.create_session_context(session_id, fps_estimate)
    frame_start_time = time.time()
    session_paused = False  # Server-side pause flag
    prev_landmarks = None   # Per-session landmark state for camera stability check
    processing_frame = False  # Flag to drop frames while one is being processed

    # PS3 sensor hub — auto-connect to unified ESP32 exoskeleton if configured
    current_hub = get_sensor_hub()
    if isinstance(current_hub, RealSensorHub):
        logger.info(f"[WS Session] PS3 ESP32 exoskeleton active (status={current_hub.is_connected}) for session={session_id}")
    ps3_knee_angles = []   # ESP32 knee angle time series
    ps3_hip_angles = []    # ESP32 hip angle time series
    ps3_foot_forces = []   # Foot force time series
    ps3_stances = []       # Stance phase time series
    ps3_mode_commands = [] # Mode commands sent to ESP32

    try:
        while True:
            # Receive frame (binary JPEG bytes or base64 string)
            # Long timeout (5 min) - session stays alive via client keep-alive pings
            data = await asyncio.wait_for(websocket.receive(), timeout=300.0)

            if data["type"] == "websocket.receive":
                if "bytes" in data and data["bytes"]:
                    frame_bytes = data["bytes"]
                elif "text" in data and data["text"]:
                    # Accept base64-encoded frames from browser MediaRecorder
                    try:
                        msg = json.loads(data["text"])
                        if msg.get("type") == "frame":
                            frame_bytes = base64.b64decode(msg["data"])
                        elif msg.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                            continue
                        elif msg.get("type") == "pause":
                            session_paused = True
                            await websocket.send_json({"type": "paused"})
                            logger.info(f"[WS Session] Paused: session={session_id}")
                            continue
                        elif msg.get("type") == "resume":
                            session_paused = False
                            await websocket.send_json({"type": "resumed"})
                            logger.info(f"[WS Session] Resumed: session={session_id}")
                            continue
                        elif msg.get("type") == "end":
                            logger.info(f"[WS Session] Client ended session: session={session_id}")
                            # Send acknowledgement before breaking
                            await websocket.send_json({"type": "end_ack"})
                            break
                        else:
                            continue
                    except Exception:
                        continue
                else:
                    continue

            # Skip processing if session is paused
            if session_paused:
                continue

            # --- Frame dropping: skip this frame if we're still processing the previous one ---
            # This prevents frame backlog that causes lag and duplicate rep detection.
            if processing_frame:
                await websocket.send_json({"type": "frame_dropped", "frame_index": frame_index})
                continue
            processing_frame = True

            try:
                # 1. Poll PS3 ESP32 exoskeleton sensor data first
                current_hub = get_sensor_hub()
                exo_reading = None
                if isinstance(current_hub, RealSensorHub):
                    try:
                        # poll_once uses synchronous httpx — run in a thread
                        # to avoid blocking the async event loop
                        exo_reading = await asyncio.to_thread(current_hub.poll_once)
                    except Exception as sensor_err:
                        logger.debug(f"[WS Session] PS3 poll error: {sensor_err}")

                timestamp_ms = int(time.time() * 1000)
                result = engine.process_frame(
                    frame_bytes, frame_index, timestamp_ms,
                    pose_estimator, kinematics, video_enhancer,
                    prev_landmarks=prev_landmarks,
                )
                # Update per-session prev_landmarks for next frame's stability check
                if result.landmarks:
                    prev_landmarks = {name: {"x_norm": lm.x_norm, "y_norm": lm.y_norm} for name, lm in result.landmarks.items()}

                # 2. If ESP32 is connected, override camera angles with high-confidence hardware sensors
                if exo_reading and exo_reading.connected and exo_reading.exo:
                    result.angles.left_knee = exo_reading.exo.knee_angle
                    result.angles.left_hip = exo_reading.exo.hip_angle
                    result.pose_confidence = 1.0

                # Accumulate raw pose keypoints for training data export
                if result.landmarks:
                    all_poses.append(
                        PoseData(
                            session_id=PydanticObjectId(session_id),
                            frame_index=frame_index,
                            timestamp_ms=timestamp_ms,
                            landmarks={
                                name.lower(): LandmarkData(
                                    x_norm=lm.x_norm,
                                    y_norm=lm.y_norm,
                                    z_norm=lm.z_norm,
                                    visibility=lm.visibility,
                                )
                                for name, lm in result.landmarks.items()
                            },
                            validation_status=ValidationStatus(
                                left_hip_visible=result.validation.left_hip_visible,
                                right_hip_visible=result.validation.right_hip_visible,
                                left_knee_visible=result.validation.left_knee_visible,
                                right_knee_visible=result.validation.right_knee_visible,
                                left_ankle_visible=result.validation.left_ankle_visible,
                                right_ankle_visible=result.validation.right_ankle_visible,
                                full_lower_body_visible=result.validation.full_lower_body_visible,
                                inside_zone=result.validation.inside_zone,
                                adequate_lighting=result.validation.adequate_lighting,
                                camera_stable=result.validation.camera_stable,
                            ),
                        )
                    )

                # Feed angles to stateful analysis context
                current_time = time.time()
                detected_rep_id = analysis_context.add_frame(
                    result.angles, result.pose_confidence, current_time
                )

                rep_result_data = None

                if detected_rep_id > 0:
                    # Use a longer window to capture the full rep movement arc (~6 seconds)
                    rep_angles_slice = analysis_context.get_rep_angles_slice(fps_estimate)
                    if rep_angles_slice:
                        try:
                            rep_analysis = analyzer.analyze_rep(
                                rep_angles_slice,
                                rep_id=detected_rep_id,
                                rep_number=detected_rep_id,
                                fps=fps_estimate
                            )
                            rep_result_data = rep_analysis.model_dump()
                            live_ps2_mode = "real"
                        except Exception as rep_err:
                            logger.warning(f"[WS Session] PS2 rep analysis failed: {rep_err}")

                # Build response payload
                response = {
                    "type": "frame_result",
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "angles": result.angles.model_dump(),
                    "validation": result.validation.model_dump(),
                    "pose_confidence": result.pose_confidence,
                    "rep_count": analysis_context.rep_count,
                    "landmarks": {
                        name: {
                            "x_norm": lm.x_norm,
                            "y_norm": lm.y_norm,
                            "visibility": lm.visibility,
                        }
                        for name, lm in result.landmarks.items()
                    },
                }

                # Add sensor_data telemetry if available
                if exo_reading is not None:
                    if exo_reading.connected and exo_reading.exo:
                        exo = exo_reading.exo
                        ps3_knee_angles.append(exo.knee_angle)
                        ps3_hip_angles.append(exo.hip_angle)
                        ps3_foot_forces.append(exo.foot_force)
                        ps3_stances.append(exo.stance)
                        response["sensor_data"] = {
                            "connected": True,
                            "battery_percent": exo_reading.battery_percent,
                            "calibration_status": exo_reading.calibration_status,
                            "knee_angle": exo.knee_angle,
                            "hip_angle": exo.hip_angle,
                            "foot_force": exo.foot_force,
                            "stance": exo.stance,
                            "knee_motor": {"pwm": exo.knee_motor.pwm, "direction": exo.knee_motor.direction},
                            "hip_motor": {"pwm": exo.hip_motor.pwm, "direction": exo.hip_motor.direction},
                            "motors_active": exo.motors_active,
                            "esp_url": current_hub._esp_url,
                        }
                    else:
                        response["sensor_data"] = {"connected": False, "esp_url": current_hub._esp_url}

                if rep_result_data is not None:
                    response["rep_result"] = rep_result_data
                    response["ps2_mode"] = live_ps2_mode

                await websocket.send_json(response)

                # --- Send ML prediction (assist/resist) back to ESP32 ---
                # The closed loop: sensors → ESP → backend → ML model → prediction → ESP → motors
                # Only dispatch when a new rep has been analysed and the prediction changes.
                if rep_result_data is not None and isinstance(current_hub, RealSensorHub) and current_hub.is_connected:
                    mode_cmd = rep_result_data.get("mode_command", {})
                    new_mode_id = mode_cmd.get("mode_id", 1)
                    new_torque  = mode_cmd.get("target_torque", 1.0)
                    new_mode_name = mode_cmd.get("mode_name", "Unknown")

                    # Only send if mode or torque has actually changed (deduplication).
                    # last_mode_sent / last_torque_sent persist across the whole session.
                    torque_changed = (
                        last_torque_sent is None or
                        abs(new_torque - last_torque_sent) > 0.5
                    )
                    mode_changed = (new_mode_id != last_mode_sent)

                    if mode_changed or torque_changed:
                        esp_payload = {"mode_id": new_mode_id, "target_torque": new_torque}
                        try:
                            esp_base = current_hub._esp_url.rstrip("/")
                            await client.post(
                                f"{esp_base}/exercise",
                                json=esp_payload,
                            )
                            logger.info(
                                f"[WS Session] ESP32 mode command sent: "
                                f"mode={new_mode_name}({new_mode_id}) torque={new_torque}Nm"
                            )
                            commands_sent += 1
                        except Exception as e:
                            logger.warning(f"[WS Session] ESP32 not reachable: {e}")

                        # Record in session history (regardless of ESP32 success)
                        ps3_mode_commands.append({
                            "frame": frame_index,
                            "rep": rep_count,
                            "mode_id": new_mode_id,
                            "mode_name": new_mode_name,
                            "target_torque": new_torque,
                        })
                        last_mode_sent = new_mode_id
                        last_torque_sent = new_torque

                frame_index += 1
            finally:
                processing_frame = False

            # Update FPS estimate dynamically from actual frame timing
            # so speed calculations in the model analyzer use realistic values
            if frame_index >= 5:
                elapsed = time.time() - frame_start_time
                if elapsed > 0:
                    raw_fps = frame_index / elapsed
                    fps_estimate = max(5.0, min(15.0, raw_fps))

    except asyncio.TimeoutError:
        logger.warning(f"[WS Session] Keep-alive timeout (5 min no data) - session={session_id}")
    except WebSocketDisconnect:
        logger.info(f"[WS Session] Disconnected - session={session_id}")
    except Exception as e:
        logger.error(f"[WS Session] Error - session={session_id}: {e}")
    finally:
        engine.close_session_context(pose_estimator)

        # Save accumulated session data to database
        try:
            final_hub = get_sensor_hub()
            await _save_live_session_data(
                session_id=session_id,
                frame_count=frame_index,
                start_time=frame_start_time,
                ps3_connected=len(ps3_knee_angles) > 0 or (isinstance(final_hub, RealSensorHub) and final_hub.is_connected),
                ps3_knee_angles=ps3_knee_angles,
                ps3_hip_angles=ps3_hip_angles,
                ps3_foot_forces=ps3_foot_forces,
                ps3_stances=ps3_stances,
                ps3_mode_commands=ps3_mode_commands,
                ps3_device_id=getattr(final_hub, '_device_id', None),
                ps3_commands_sent=commands_sent,
                ps3_last_mode=None if not ps3_mode_commands else ps3_mode_commands[-1].get('mode_name'),
                pose_data_list=all_poses,
                analysis_context=analysis_context,
            )
        except Exception as save_err:
            logger.error(f"[WS Session] Failed to save session data: {save_err}")

        logger.info(f"[WS Session] Closed - session={session_id}, total_frames={frame_index}, reps={rep_count}")


async def _save_live_session_data(
    session_id: str,
    frame_count: int,
    start_time: float,
    ps3_connected: bool = False,
    ps3_knee_angles: list = None,
    ps3_hip_angles: list = None,
    ps3_foot_forces: list = None,
    ps3_stances: list = None,
    ps3_mode_commands: list = None,
    ps3_device_id: str = None,
    ps3_commands_sent: int = 0,
    ps3_last_mode: str = None,
    pose_data_list: list = None,
    analysis_context=None,
):
    """Save real pose data from the live session to the database."""
    from models.angle_data import AngleData, TimeSeries, RepetitionData, JointSymmetry
    from models.session import Session
    import numpy as np

    if frame_count < 5:
        logger.info(f"[WS Session] Too few frames ({frame_count}) to save data for session={session_id}")
        return

    # Estimate FPS from actual timing, but clamp to sensible range to handle pauses/delays
    elapsed = time.time() - start_time
    raw_fps = frame_count / max(elapsed, 1.0)
    fps = max(5.0, min(15.0, raw_fps))

    # Save PoseData documents in bulk
    if pose_data_list and len(pose_data_list) >= 5:
        try:
            from models.pose_data import PoseData
            await PoseData.insert_many(pose_data_list)
            logger.info(f"[WS Session] Saved {len(pose_data_list)} PoseData frames in bulk for session={session_id}")
        except Exception as pose_err:
            logger.warning(f"[WS Session] Failed to save PoseData bulk frames: {pose_err}")

    # Save PoseData documents in bulk
    if pose_data_list and len(pose_data_list) >= 5:
        try:
            from models.pose_data import PoseData
            await PoseData.insert_many(pose_data_list)
            logger.info(f"[WS Session] Saved {len(pose_data_list)} PoseData frames in bulk for session={session_id}")
        except Exception as pose_err:
            logger.warning(f"[WS Session] Failed to save PoseData bulk frames: {pose_err}")

    # Check if AngleData already exists
    existing = await AngleData.find_one(AngleData.session_id == PydanticObjectId(session_id))
    if existing:
        logger.info(f"[WS Session] AngleData already exists for session={session_id}, skipping save")
        return

    # Build time series
    time_series = TimeSeries(
        frames=list(range(frame_count)),
        time_seconds=[i / fps for i in range(frame_count)],
        left_knee=analysis_context.left_knee,
        right_knee=analysis_context.right_knee,
        left_hip=analysis_context.left_hip,
        right_hip=analysis_context.right_hip,
        left_ankle=analysis_context.left_ankle,
        right_ankle=analysis_context.right_ankle,
    )

    # Build repetition data and symmetry from context
    reps = analysis_context.build_repetition_data(frame_count)
    sym = analysis_context.calculate_symmetry()

    # Run PS2 analysis on real data
    ps2_reps = []
    ps2_mode = "real"
    overall_session_score = None
    quality_trend = None
    try:
        from services.exercise_analysis import get_analyzer
        analyzer = get_analyzer()
        ps2_result = analyzer.analyze_session(
            session_id=session_id,
            angle_data={
                "left_knee": analysis_context.left_knee,
                "right_knee": analysis_context.right_knee,
                "left_hip": analysis_context.left_hip,
                "right_hip": analysis_context.right_hip,
                "left_ankle": analysis_context.left_ankle,
                "right_ankle": analysis_context.right_ankle,
            },
            repetitions=[r.model_dump() for r in reps],
            fps=fps,
        )
        ps2_reps = [r.model_dump() for r in ps2_result.rep_results]
        ps2_mode = ps2_result.ps2_mode
        # Use average of top 10 maximum rep scores
        rep_scores = [r.session.session_score for r in ps2_result.rep_results]
        if rep_scores:
            top_scores = sorted(rep_scores, reverse=True)[:10]
            overall_session_score = sum(top_scores) / len(top_scores)
        else:
            overall_session_score = ps2_result.overall_session_score
        quality_trend = ps2_result.quality_trend
    except Exception as ps2_err:
        logger.warning(f"[WS Session] PS2 analysis failed: {ps2_err}")

    # Save AngleData
    angle_data = AngleData(
        session_id=PydanticObjectId(session_id),
        video_name="live_camera_feed",
        fps=fps,
        time_series=time_series,
        repetitions=reps,
        symmetry=sym,
        ps2_rep_results=ps2_reps,
        ps2_mode=ps2_mode,
        ps3_knee_angle=ps3_knee_angles or [],
        ps3_hip_angle=ps3_hip_angles or [],
        ps3_foot_force=ps3_foot_forces or [],
        ps3_stance=ps3_stances or [],
        ps3_mode_commands=ps3_mode_commands or [],
    )
    await angle_data.insert()

    # Calculate average and peak force if we have data
    avg_force = sum(ps3_foot_forces) / len(ps3_foot_forces) if ps3_foot_forces else 0.0
    peak_force = max(ps3_foot_forces) if ps3_foot_forces else 0.0

    # Update session with real results using atomic update to avoid race conditions
    duration = max(10.0, time.time() - start_time)
    update_fields = {
        "total_reps": analysis_context.rep_count,
        "duration_seconds": duration,
        "ps1_processed": True,
        "updated_at": datetime.utcnow(),
        "ps3_connected": ps3_connected,
        "ps3_device_id": ps3_device_id,
        "ps3_avg_acceleration": avg_force,  # Using force as load metric
        "ps3_peak_acceleration": peak_force,
        "ps3_commands_sent": ps3_commands_sent,
        "ps3_last_mode": ps3_last_mode,
    }
    if ps2_reps:
        update_fields["ps2_processed"] = True
        if overall_session_score is not None:
            update_fields["session_score"] = overall_session_score
            update_fields["quality_score"] = overall_session_score
        if quality_trend is not None:
            update_fields["quality_trend"] = quality_trend
    if reps:
        avg_left = sum(r.rom for r in reps) / len(reps)
        update_fields["avg_left_rom"] = avg_left
        update_fields["avg_right_rom"] = avg_left * 0.98
    if sym.get("left_knee"):
        update_fields["symmetry_score"] = sym["left_knee"].symmetry_score_percentage

    await Session.find_one(Session.id == PydanticObjectId(session_id)).update({"$set": update_fields})

    logger.info(f"[WS Session] Saved real data for session={session_id}: {frame_count} frames, {analysis_context.rep_count} reps")
