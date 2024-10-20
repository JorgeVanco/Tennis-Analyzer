import pickle
import sys
import cv2
from ultralytics import YOLO

sys.path.append("../")
from utils import get_center_of_bbox, measure_distance


class PlayerTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(
            court_keypoints, player_detections_first_frame
        )
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {
                track_id: bbox
                for track_id, bbox in player_dict.items()
                if track_id in chosen_players
            }
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections

    def choose_players(self, court_keypoints, player_detections):
        distances = []
        for track_id, bbox in player_detections.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float("inf")
            for i in range(len(court_keypoints), 2):
                distance = measure_distance(player_center, court_keypoints[i : i + 2])
                if distance < min_distance:
                    distance = min_distance

            distances.append((track_id, min_distance))

        # sort the distances in ascending order
        distances.sort(key=lambda x: x[1])
        # Choose the first two tracks
        chosen_players = (distances[0][0], distances[1][0])
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as fh:
                player_detections = pickle.load(fh)
            return player_detections

        player_detections = [self.detect_frame(frame) for frame in frames]
        if stub_path is not None:
            with open(stub_path, "wb") as fh:
                pickle.dump(player_detections, fh)
        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(
                    frame,
                    f"Player ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            output_frames.append(frame)
        return output_frames
