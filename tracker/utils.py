# Kalman Filter on 8 dim state space
# u, v, gamma, h (bounding box center, aspect ratio, height)
# x., y., gamma., h. (their velocity)
import numpy as np
from typing import Dict

ESTIMATION_COVARIANCE = np.diag([0.01, 0.01, 0.1, 0.01])
REMOVE_ON_AGE = 1.5 # Remove track after this seconds of no detection association
TENTATIVE_ON_AGE = 0.5 # Add to track after this seconds of consecutive detection association

def xyxy2uvgh(xyxy: np.ndarray):
    # xyxy should have shape [N, 4]
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    gamma = h / w
    ret = np.zeros_like(xyxy)
    ret[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    ret[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    ret[:, 2] = gamma
    ret[:, 3] = h
    ret = np.round(ret, decimals=4)
    return ret

def uvgh2xyxy(uvgh):
    # uvgh should have shape [N, 4]
    # Transform u,v,gamma,h format to xyxy format
    ret = np.zeros_like(uvgh)
    halfw = uvgh[:, 3] / (2 * uvgh[:, 2])
    ret[:, 0] = uvgh[:, 0] - halfw
    ret[:, 2] = uvgh[:, 0] + halfw
    halfh = uvgh[:, 3] / 2
    ret[:, 1] = uvgh[:, 1] - halfh
    ret[:, 3] = uvgh[:, 1] + halfh    
    
    ret = np.round(ret, decimals=4)
    return ret

def iou(track, det):
    # Works on uvgh format
    track_xyxy = uvgh2xyxy(np.expand_dims(track, axis=0))[0]
    det_xyxy = uvgh2xyxy(np.expand_dims(det, axis=0))[0]
    # Intersect rectangle
    iw = min(track_xyxy[2], det_xyxy[2]) \
        - max(track_xyxy[0], det_xyxy[0])
    if iw <= 0:
        return 0
    ih = min(track_xyxy[3], det_xyxy[3]) \
        - max(track_xyxy[1], det_xyxy[1])
    if ih <= 0:
        return 0
    intersect = iw * ih
    area1 = track[3]**2 / track[2]
    area2 = det[3]**2 / det[2]
    iou_value = intersect / (area1 + area2 - intersect)
    return iou_value

def measure_and_assign(tracks_uvgh: np.ndarray, tracks_cov: np.ndarray, 
                    dets_uvgh: np.ndarray, iou_threshold: float = 0.4, 
                    use_ma: bool = True):
    # tracks_uvgh: [N, 5], tracks_cov: [N, 4, 4]
    # dets_uvgh: [M, 4], iou_threshold: float
    # use_ma if True, use Mahalanobis dist instead of IOU.
    # Measure IOU or Mahalanobis distance before assignment.

    # We look for maximization of association, hence everything
    # initialized as -np.inf at first.
    assoc_dict = {}
    if use_ma:
        # Use Mahalanobis Distance, should have better behavior.
        for track_ind, track in enumerate(tracks_uvgh):
            track_id = track[-1] # Scalar
            assoc_dict[track_id] = []
            # Below runs only if there is at least one detection.
            if dets_uvgh.size > 0:
                track_cov = tracks_cov[track_ind] # [4, 4]
                dif = dets_uvgh - track[:4] # [M, 4]
                cov_inv = np.linalg.inv(track_cov) # [4,4]
                ma_dist = np.matmul(cov_inv, dif[:, :, np.newaxis])
                ma_dist = np.matmul(dif[:, np.newaxis, :], ma_dist)
                ma_dist = -ma_dist[:, 0, 0]
                # Appply Mahalanobis threshold for 95% matching.
                ma_dist[ma_dist<-9.4877] = -np.inf 
                assoc_dict[track_id] = ma_dist.tolist()
    else:
        # Use IOU.
        for track in tracks_uvgh:
            track_id = track[-1]
            assoc_dict[track_id] = []
            for det in dets_uvgh:
                iou_value = iou(track[:4], det)
                # Apply iou threshold
                if iou_value < iou_threshold:
                    iou_value = -np.inf
                assoc_dict[track_id].append(iou_value)
    if not assoc_dict:
        # No tracks, can have or has no detections
        return {
            "unmatched_tracks": [],
            "unmatched_detections": list(range(dets_uvgh.shape[0])),
            "pairs": {},
        }
    if not list(assoc_dict.values())[0]:
        # Have tracks but no detections
        return {
            "unmatched_tracks": list(tracks_uvgh[:, -1]),
            "unmatched_detections": [],
            "pairs": {},
        }
    # Have both tracks and detections
    track_ids = []
    mat = []
    for track_id, ious in assoc_dict.items():
        track_ids.append(int(track_id))
        mat.append(ious)
    mat = np.array(mat)
    det_ids = list(range(mat.shape[1]))
    pairing = {}
    # We assume iou matrix already got thresholded.
    matmax = mat.max()
    while matmax > -np.inf:
        max_ind = np.where(mat == matmax)
        curr_track_id = track_ids[max_ind[0][0]]
        curr_det_id = det_ids[max_ind[1][0]]
        mat = np.delete(mat, max_ind[0][0], axis=0)
        mat = np.delete(mat, max_ind[1][0], axis=1)
        track_ids.pop(max_ind[0][0])
        det_ids.pop(max_ind[1][0])
        pairing[curr_track_id] = curr_det_id
        if mat.size == 0: # Assigned all tracks or dets
            break
        matmax = mat.max()
    # print("Unmatched tracks", track_ids)
    # print("Unmatched detections", det_ids )
    # print("Pairing", pairing)
    return {
        "unmatched_tracks": track_ids,
        "unmatched_detections": det_ids,
        "pairs": pairing,
    }


def get_state_transition(delta_t):
    F = np.eye(8)
    F[:4, 4:] = delta_t * np.eye(4)
    return F

def solve_relation(split_config: dict) -> np.ndarray:
    # Line equation will be ax+by+c = 0
    # Return numpy array of shape [3]: value [a,b,c].
    line_coefs: np.ndarray = np.array([0.,0.,0.]) # [a,b,c]
    x1, y1 = split_config["points"][0]
    x2, y2 = split_config["points"][1]
    # X diff version
    if x1 == x2 and y1 == y2:
        raise ValueError("The two endpoints cannot be the same.")
    if x1 != x2:
        line_coefs[1] = 1 # Can let b = 1 since x1 - x2 != 0
        line_coefs[[0,2]] = (1/(x1-x2)) * np.array([y2-y1, x2*y1 - x1*y2])
    else:
        line_coefs[0] = 1 # Can let a = 1 since y1 - y2 != 0
        line_coefs[1:] = (1/(y1-y2)) * np.array([x2-x1, x1*y2 - x2*y1])
    # Make sure inside is positive side
    inside_point = np.array(split_config["in"])
    dot_value = line_coefs[:2].dot(inside_point) + line_coefs[2]
    if dot_value == 0:
        raise ValueError(f"We couldn't determine side information. Pick another point for 'in'. ")
    elif dot_value < 0:
        # Reverse coefficient sign to ensure inside is positive
        line_coefs = -line_coefs
    return line_coefs

class Tracker():
    def __init__(self, x_8: np.ndarray, ID: int):
        # x_8 value must be normalized (independent of image size)
        assert x_8.shape == (8,)
        self.x : np.ndarray = x_8
        self.p : np.ndarray = 0.5 * np.eye(8) # process noise
        self.visible : bool = False 
        self.ID : int = ID
        # Kalman Filter matrices
        self.H = np.eye(4, 8)
        # Aging
        self.unmatch_age = 0 # This is in term of seconds, not number of frames
        self._tentative_age = 0 # Consecutive association with any detection

    def make_visible(self):
        self.visible = True 

    @property
    def tentative_age(self):
        return self._tentative_age

    @tentative_age.setter
    def tentative_age(self, new_value):
        self._tentative_age = new_value
        if self._tentative_age > TENTATIVE_ON_AGE:
            # make track visible when tentative age surpass a threshold
            self.make_visible()


    def step(self, z: np.ndarray, r: np.ndarray, F: np.ndarray):
        # Kalman Filter workflow

        # State Extrapolation
        self.x = F.dot(self.x)
        # Covariance Extrapolation
        self.p = F.dot(self.p.dot(F.transpose()))
        # Calculate Kalman Gain
        kalman_inv = self.H.dot(self.p.dot(self.H.transpose()))
        kalman_inv = np.linalg.inv(kalman_inv)
        K = self.p.dot(self.H.transpose().dot(kalman_inv))
        # State Update
        innovation = z - self.H.dot(self.x)
        self.x = self.x + K.dot(innovation)
        # Covariance Update
        factor: np.ndarray = np.eye(8) - K.dot(self.H)
        self.p = factor.dot(self.p.dot(factor.transpose())) + K.dot(r.dot(K.transpose()))

    def step_without_obs(self, F: np.ndarray):
        # State Extrapolation
        self.x = F.dot(self.x)
        # Covariance Extrapolation
        self.p = F.dot(self.p.dot(F.transpose()))

    def get_side(self, coef: np.ndarray) -> int:
        value = self.x[:2].dot(coef[:2]) + coef[2]
        return int(2 * (value >= 0) - 1) # 1 inside or on line, -1 outside


class Tracks():
    def __init__(self, split_config : dict = None):
        self.tracks : list[Tracker] = []
        self.latest_ID : int = 0
        self.in_traffic: int = 0 # How many people going inside
        self.out_traffic: int = 0 # How many people going outside 
        
        self.prev_track_ids: list = None
        self.curr_track_ids: list = None
        
        self.split_config: dict = None
        self.split_coef: np.ndarray = None

        if split_config is not None:
            self.split_config = split_config
            self.split_coef = solve_relation(split_config) # [3]
        self.r = ESTIMATION_COVARIANCE

    def add_track(self, x_8: np.ndarray):
        track = Tracker(x_8, ID = self.latest_ID + 1)
        self.tracks.append(track)
        self.latest_ID = track.ID
        return track
    
    def remove_track(self, track_ID):
        found = False
        for i, track in enumerate(self.tracks):
            if track.ID == track_ID:
                found = True
                del self.tracks[i]
                break
        if not found:
            print(f"Track ID {track_ID} not found, nothing changed.")

    def retrieve(self, use_visible = False):
        # Return tracks first four state in numpy ndarray [N, 5] format, 
        # [u, v, gamma, h, track_ID].
        # Return tracks covariance matrices [N, 4, 4] format.
        mat = []
        cov_mat = []
        for track in self.tracks:
            # Guard Clause: Use visible on invisible track
            if use_visible and not track.visible:
                continue
            mat.append(list(track.x[:4]) + [track.ID])
            cov_mat.append(track.p[:4, :4].tolist())
        return np.array(mat), np.array(cov_mat)
    
    def retrieve_ids(self, use_visible = False):
        id_and_sides = []
        for track in self.tracks:
            # Guard Clause: Use visible on invisible track
            if use_visible and not track.visible:
                continue
            if self.split_coef is None:
                id_and_sides.append([track.ID, None])
            else:
                id_and_sides.append([track.ID, track.get_side(self.split_coef)])
        return id_and_sides
    
    def __getitem__(self, track_ID: int):
        for track in self.tracks:
            if track.ID == track_ID:
                return track
        raise IndexError(f"There is no track with track ID {track_ID}")        

    def __len__(self):
        return len(self.tracks)
    
    def update_traffic(self):
        # Guard Clause: No previous track record or not using split
        if self.prev_track_ids is None or self.split_coef is None:
            return
        # We are sure prev_track_ids and curr_track_ids are both valid lists.
        # State description: [1] inside, [0] outside, [None] disappear
        state_record = {
            track_id: [prev_indicator] for track_id, prev_indicator in self.prev_track_ids
        }
        for track_id, curr_indicator in self.curr_track_ids:
            if track_id in state_record:
                state_record[track_id].append(curr_indicator)
                if state_record[track_id][0] > state_record[track_id][1] \
                    and self.out_traffic < self.in_traffic:
                    # Being [1, -1], outflow traffic
                    # Only increment when outtraffic < intraffic
                    self.out_traffic += 1
                elif state_record[track_id][0] < state_record[track_id][1]:
                    # Being [-1, 1], inflow traffic
                    self.in_traffic += 1
            else:   
                # Appearance
                state_record[track_id] = [None, curr_indicator]
                # Since it could be a person already in store reappear in store,
                # or pedestrain reappear on the road, we do not count traffic here.
        for track_id in list(state_record.keys()):
            if len(state_record[track_id]) == 1:
                # Disappear
                state_record[track_id].append(None)

        return state_record

    def step(self, dets_uvgh: np.ndarray, pairing_dict: dict, delta_t: float):
        """Perform a step in Kalman Filter.
        Update track ids.

        Args:
            dets_uvgh (np.ndarray): Shape [N, 4], uvgh format.
            pairing_dict (dict): Format {
                            "unmatched_tracks": track_ids,
                            "unmatched_detections": det_ids,
                            "pairs": Dict[track_ID, det_index],
                        },
                        contain assignment information.
            delta_t (float): Time passed from previous step, in seconds.
        """
        F = get_state_transition(delta_t) 
        # Update previous track IDs
        if self.curr_track_ids is not None:
            self.prev_track_ids = self.curr_track_ids.copy()
        # Get visibility
        # print([(track.unmatch_age, track.tentative_age) for track in self.tracks])
        # Remove unmatched tracks
        for track_ID in pairing_dict["unmatched_tracks"]:
            track = self[track_ID]
            track.unmatch_age += delta_t 
            if (track.tentative_age <= TENTATIVE_ON_AGE and \
                not track.visible) or \
                track.unmatch_age > REMOVE_ON_AGE:
                # Remove upon disappearance during tentative period
                # Or unmatch after predefined seconds of disappearance
                self.remove_track(track_ID = track_ID)
            else:
                track.tentative_age = 0 # reset on unmatch 
                # Forward a step on this track without detection
                track.step_without_obs(F)
        # Kalman Filter on paired track-detections
        for track_ID, det_index in pairing_dict["pairs"].items():
            track = self[int(track_ID)]
            det_uvgh_z = dets_uvgh[det_index]
            track.unmatch_age = 0 # Refresh unmatch_age
            track.tentative_age = track.tentative_age + delta_t
            track.step(det_uvgh_z, self.r, F)
            
        # Add unmatched detection into new tracks
        for det_index in pairing_dict["unmatched_detections"]:
            # Assume velocities are zero
            x_8 = np.zeros(8)
            x_8[:4] = dets_uvgh[det_index]
            this_track = self.add_track(np.copy(x_8))
            this_track.tentative_age = this_track.tentative_age + delta_t
        # print("No of tracks:", len(self), ", Max ID:", self.latest_ID)
        # Update current track ids
        self.curr_track_ids = self.retrieve_ids(use_visible = True)
        # try:
        #     print(len(self.prev_track_ids), len(self.curr_track_ids))
        # except TypeError:
        #     print("No info yet...")

        # Update traffic
        self.update_traffic()

        

if __name__ == "__main__":
    T = Tracks()
    T.add_track(np.array([0.1,0.2,0.3,0.4,0,0,0,0]))
    T.add_track(np.array([0.2,0.2,0.3,0.4,0,0,0,0]))
    T.add_track(np.array([0.3,0.2,0.3,0.4,0,0,0,0]))
    T.add_track(np.array([0.4,0.2,0.3,0.4,0,0,0,0]))
    print(len(T.tracks))
    T.remove_track(2)
    print(len(T.tracks))
    # tracks_xyxy = np.array([
    #     [0, 0, .2, .3, 7],
    #     [.3, .4, .4, .7, 9],
    #     [.35, .4, .7, .6, 11],
    #     [.6, .1, .9, .4, 16],
    # ])
    # dets_xyxy = np.array([
    #     [.05, 0, .2, .35],
    #     [.65, .2, .9, .4],
    #     [.32, .4, .42, .71],
    #     [.6, .5, .9, .9],
    # ])
    # tracks_uvgh = np.copy(tracks_xyxy)
    # tracks_uvgh[:, :4] = xyxy2uvgh(tracks_uvgh[:, :4])
    # dets_uvgh = xyxy2uvgh(dets_xyxy)
    # pairing_dict = iou_and_assign(tracks_uvgh, dets_uvgh)
    # print(pairing_dict)
