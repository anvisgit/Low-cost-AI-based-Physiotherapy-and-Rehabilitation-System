import cv2

from rgda_dashboard_clean import load_rgda_model
from rgda_live_dashboard_runtime import LiveExerciseSession

SELECTED_EXERCISE = "squat"
PATIENT_ID = "P001"

# Model is optional.
# Use None if you only want biomechanical flag-based output.
MODEL_PATH = None
# MODEL_PATH = "models/rgda_rehabnet_best.pth"

model = load_rgda_model(MODEL_PATH) if MODEL_PATH else None

session = LiveExerciseSession(
    exercise=SELECTED_EXERCISE,
    patient_id=PATIENT_ID,
    model=model,
    hardware_mode=0,
)

cap = cv2.VideoCapture(0)

print("Camera started.")
print("Press q to stop and get final rep summary.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    overlay_frame, live = session.process_frame(frame)

    feedback = live.get("live_feedback", "")
    angles = live.get("live_angles", {})

    cv2.putText(
        overlay_frame,
        feedback,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Live Rehab Tracking", overlay_frame)

    print(
        "feedback:",
        feedback,
        "| angles:",
        angles,
        "| flags:",
        live.get("live_flags", []),
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

summary = session.finalize_session()

cap.release()
cv2.destroyAllWindows()
session.close()

print("\nFINAL SESSION SUMMARY")
print(summary)