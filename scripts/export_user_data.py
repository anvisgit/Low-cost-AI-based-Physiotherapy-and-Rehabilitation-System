import asyncio
import os
import sys
import csv
import argparse
from pathlib import Path

# Add backend directory to python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from database import connect_db, close_db
from models.session import Session
from models.angle_data import AngleData
from models.pose_data import PoseData
from models.exercise import Exercise

async def export_data(output_path: str):
    print("Connecting to database...")
    await connect_db()
    
    print("Querying session angle summaries...")
    angle_col = AngleData.get_motor_collection()
    # Optimize query with projection to exclude large time_series float arrays
    cursor = angle_col.find({}, {"session_id": 1, "repetitions": 1, "ps2_rep_results": 1})
    angle_datas = await cursor.to_list(length=None)
    print(f"Found {len(angle_datas)} session summaries in DB.")
    
    all_rows = []
    sessions_processed = 0
    reps_exported = 0
    
    pose_col = PoseData.get_motor_collection()
    
    for idx, angle_data in enumerate(angle_datas):
        session_id = angle_data.get("session_id")
        repetitions = angle_data.get("repetitions") or []
        ps2_rep_results = angle_data.get("ps2_rep_results") or []
        
        if not repetitions:
            # Skip session if no repetitions recorded
            continue
            
        session = await Session.get(session_id)
        if not session:
            continue
            
        patient_id = session.patient_id
        
        # Get exercise slug name
        exercise = await Exercise.get(session.exercise_id)
        exercise_name = exercise.slug if exercise else "unknown"
        
        # Fetch all pose documents for this session (projecting only frame_index and landmarks)
        pose_cursor = pose_col.find(
            {"session_id": session_id},
            {"frame_index": 1, "landmarks": 1}
        )
        poses = await pose_cursor.to_list(length=None)
        if not poses:
            # Skip if session has no raw coordinates saved (e.g. from before this upgrade)
            continue
            
        poses.sort(key=lambda p: p.get("frame_index", 0))
        poses_by_frame = {p.get("frame_index"): p for p in poses}
        
        sessions_processed += 1
        
        # Process each repetition segment
        for rep in repetitions:
            # Handle both dict and Pydantic object formats safely
            if isinstance(rep, dict):
                rep_id = rep.get("rep_id")
                start_frame = rep.get("start_frame")
                end_frame = rep.get("end_frame")
            else:
                rep_id = getattr(rep, "rep_id", None)
                start_frame = getattr(rep, "start_frame", None)
                end_frame = getattr(rep, "end_frame", None)
                
            if rep_id is None or start_frame is None or end_frame is None:
                continue
            
            # Default label to correct (0)
            label = 0
            
            # Check if any clinical warning flags were raised in ML evaluation
            ps2_result = None
            for r in ps2_rep_results:
                if isinstance(r, dict):
                    curr_rep_id = r.get("rep_id")
                else:
                    curr_rep_id = getattr(r, "rep_id", None)
                if curr_rep_id == rep_id:
                    ps2_result = r
                    break
            
            if ps2_result:
                if isinstance(ps2_result, dict):
                    flags = ps2_result.get("error_flags", {})
                else:
                    flags = getattr(ps2_result, "error_flags", {})
                    
                if isinstance(flags, dict):
                    any_error = any(flags.get(f, 0) == 1 for f in [
                        "insufficient_ROM", "too_fast", "too_slow", "knee_valgus", "asymmetric", "trunk_comp"
                    ])
                else:
                    any_error = any(getattr(flags, f, 0) == 1 for f in [
                        "insufficient_ROM", "too_fast", "too_slow", "knee_valgus", "asymmetric", "trunk_comp"
                    ])
                if any_error:
                    label = 1
                    
            ann_key = f"rep_{session_id}_{rep_id}"
            rep_has_data = False
            
            # Extract coordinates for each frame in this repetition
            for frame_idx in range(start_frame, end_frame + 1):
                pose = poses_by_frame.get(frame_idx)
                if not pose:
                    continue
                
                landmarks = pose.get("landmarks")
                if not landmarks:
                    continue
                
                # Format to training-compatible layout
                row = {
                    "ann_key": ann_key,
                    "FrameID": frame_idx - start_frame,
                    "Subject": str(patient_id),
                    "exercise": exercise_name,
                    "label": label,
                }
                
                # Map MediaPipe lowercase keys to output columns
                joint_mapping = {
                    "HipLeft": "left_hip",
                    "KneeLeft": "left_knee",
                    "AnkleLeft": "left_ankle",
                    "HipRight": "right_hip",
                    "KneeRight": "right_knee",
                    "AnkleRight": "right_ankle",
                }
                
                skip_frame = False
                for target_name, source_name in joint_mapping.items():
                    landmark = landmarks.get(source_name)
                    if not landmark:
                        landmark = landmarks.get(source_name.upper())
                        
                    if not landmark:
                        skip_frame = True
                        break
                        
                    # Handle both dict and Pydantic objects safely
                    if isinstance(landmark, dict):
                        row[f"{target_name}_x"] = landmark.get("x_norm")
                        row[f"{target_name}_y"] = landmark.get("y_norm")
                        row[f"{target_name}_z"] = landmark.get("z_norm")
                    else:
                        row[f"{target_name}_x"] = getattr(landmark, "x_norm", 0.0)
                        row[f"{target_name}_y"] = getattr(landmark, "y_norm", 0.0)
                        row[f"{target_name}_z"] = getattr(landmark, "z_norm", 0.0)
                    
                if not skip_frame:
                    all_rows.append(row)
                    rep_has_data = True
                    
            if rep_has_data:
                reps_exported += 1
                
    # Write to CSV
    if not all_rows:
        print("No raw pose data found in database sessions to export. (New sessions must be recorded to save PoseData).")
    else:
        # Create output directories if missing
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = [
            "ann_key", "FrameID", "Subject", "exercise", "label",
            "HipLeft_x", "HipLeft_y", "HipLeft_z",
            "KneeLeft_x", "KneeLeft_y", "KneeLeft_z",
            "AnkleLeft_x", "AnkleLeft_y", "AnkleLeft_z",
            "HipRight_x", "HipRight_y", "HipRight_z",
            "KneeRight_x", "KneeRight_y", "KneeRight_z",
            "AnkleRight_x", "AnkleRight_y", "AnkleRight_z"
        ]
        
        with open(out_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
                
        print(f"Success! Exported {sessions_processed} sessions and {reps_exported} reps to: {out_file.resolve()}")
        
    await close_db()

def main():
    parser = argparse.ArgumentParser(description="Export raw keypoints dataset from MongoDB to training CSV.")
    parser.add_argument(
        "--output",
        default=str(backend_dir.parent.parent / "data" / "user_pose_data.csv"),
        help="Path where output CSV should be saved."
    )
    args = parser.parse_args()
    
    asyncio.run(export_data(args.output))

if __name__ == "__main__":
    main()
