
# from __future__ import print_function
import cv2
import argparse
import time #for time
import imageio
import numpy as np
from scipy.spatial import distance
from filterpy.kalman import KalmanFilter
import glob
from skimage import io
import os
np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
            [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
            the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

#kalman Filter
class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
            print("kfffffffff")
            print(self.kf.x[6])
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

#Associating detections to tracker
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.25):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
            print("aaaaaaaaa")
            print(a)            
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
            #print("unmatched")
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
	def __init__(self, max_age=10, min_hits=4, iou_threshold=0.25):
		"""
		Sets key parameters for SORT
		"""
		self.max_age = max_age
		self.min_hits = min_hits
		self.iou_threshold = iou_threshold
		self.trackers = []
		self.frame_count = 0

	def update(self, dets: np.array = np.empty((0, 5))):
		"""
		Params:
			dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
		Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""
		self.frame_count += 1
		# get predicted locations from existing trackers.
		trks = np.zeros((len(self.trackers), 5))
		to_del = []
		ret = []
     
		for t, trk in enumerate(trks):
          
			pos = self.trackers[t].predict()[0]
            
			trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
        
			if np.any(np.isnan(pos)):
				to_del.append(t)
                
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
		for t in reversed(to_del):
			self.trackers.pop(t)
          
		matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
			dets, trks, self.iou_threshold)
		for i,m in enumerate(matched):
			
			self.trackers[m[1]].update(dets[m[0]])
			
		# create and initialise new trackers for unmatched detections
		for i in unmatched_dets:
			trk = KalmanBoxTracker(dets[i])
			self.trackers.append(trk)
			
		i = len(self.trackers)
		for trk in reversed(self.trackers):
			d = trk.get_state()[0]
			if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
				# +1 as MOT benchmark requires positive
				ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
			i -= 1
			# remove dead tracklet
			if (trk.time_since_update > self.max_age):
				self.trackers.pop(i)
		if (len(ret) > 0):
			return np.concatenate(ret)
		return np.empty((0, 5))

#Finding angle of turn
def find_angle_distance(points):
    d = calculate_covered_distance(points[-20:])
    print("pountss  s s s")
    print(points)
    if(d > 10):
        points = points[-40:]
        size = len(points)//4
        points = points[::size]
        p1,p2,p3,p4 = points[-4:]
        
        if(calculate_covered_distance([p2,p4]) > 5):
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p4) - np.array(p3)
            unit_v1 = v1 / np.linalg.norm(v1)
            unit_v2 = v2 / np.linalg.norm(v2)
            angle = np.degrees(np.arccos(np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)))
            # print(angle)
            if(0<=angle<=10):
                return "straight"
            elif (180<=angle):
                return "Reverse"
            elif(angle>10 and angle<90):
                # diff = v2[0] - v1[0]
                A,B,C = p1,p2,p3
                """
                Assuming the points are (Ax,Ay) (Bx,By) and (Cx,Cy), you need to compute:
                (Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)
                This will equal zero if the point C is on the line formed by points A and B, and will have a different sign depending on the side. Which side this is depends on the orientation of your (x,y) coordinates, but you can plug test values for A,B and C into this formula to determine whether negative values are to the left or to the right                        
                """
                diff = (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])

                if(diff >=1 ):
                    return "right"
                elif(diff<=-1):
                    return "left"
                else:
                    return "straight"
            else:
                return "Stright"    
        else:
            return "Stright"

    
    else:
        return "stopped"

#Calculating distance travelled    
def calculate_covered_distance(points):
    d = 0        
    for i in range(len(points)-1):
        d+=distance.euclidean(points[i],points[i+1])
    return d


#Object Tracking
class Tracker:
    CLASS_NAMES= ["bicycle","car","motorbike","aeroplane","bus","train","truck","boat","person"]

    def __init__(self):
        super(Tracker,self).__init__()
        
        self.car_tracker = Sort()
        self.detections = None
        self.__saved_track_ids = []
        self.frame = None
        self.trackers_centers = {}

    
    def track(self,output):
        dets = []
        track_results = []
        turn = "None"
        if(len(output)>0):
            for c in output:
                if(not c[0] in Tracker.CLASS_NAMES):
                    continue
                else:
                    x,y,w,h = c[1:5]
                    dets.append(np.array([x,y,(x+w),(y+h),c[5]]))
                    
            detts = np.array(dets)
            
            track_bbs_ids = self.car_tracker.update(detts)

            for d in track_bbs_ids :
                d = d.astype(np.int32)
               
                tid , x,y = str(d[4]) , d[0],d[1]
                w = d[2] - d[0]
                h = d[3] - d[1]
                if(not tid in self.trackers_centers):
                    self.trackers_centers[tid] = []
                self.trackers_centers[tid].append([x+w//2 , y+h//2])
                if(len(self.trackers_centers[tid]) >= 20):
                    turn = find_angle_distance(self.trackers_centers[tid])
                
                track_results.append({"track_id":tid,"points":[x,y,w,h],"class":"vehicle","track-dict":self.trackers_centers,"direction":turn})
                self.__saved_track_ids.append(tid)
                
        return track_results

#Object Detection 
class YoloDetection():
    def __init__(self, model_path: str, config: str, classes: str, width: int, height: int,
                 scale=0.00392, thr=0.5, nms=0.4, backend=0,
                 framework=3,
                 target=0, mean=[0, 0, 0]):
        
        super(YoloDetection,self).__init__()
        choices = ['caffe', 'tensorflow', 'torch', 'darknet']
        backends = (
            cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_BACKEND_HALIDE, cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
            cv2.dnn.DNN_BACKEND_OPENCV)
        targets = (
            cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL, cv2.dnn.DNN_TARGET_OPENCL_FP16, cv2.dnn.DNN_TARGET_MYRIAD)

        self.__confThreshold = thr
        self.__nmsThreshold = nms
        self.__mean = mean
        self.__scale = scale
        self.__width = width
        self.__height = height

        # Load a network
        self.__net = cv2.dnn.readNet(model_path, config, choices[framework])
        self.__net.setPreferableBackend(backends[backend])
        self.__net.setPreferableTarget(targets[target])
        self.__classes = None

        if classes:
            with open(classes, 'rt') as f:
                self.__classes = f.read().rstrip('\n').split('\n')


    def get_output_layers_name(self, net):
        all_layers_names = net.getLayerNames()
        return [all_layers_names[i-1] for i in net.getUnconnectedOutLayers()]

    def post_process_output(self, frame, outs):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        class_ids = []
        confidences = []
        boxes = []

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.__confThreshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = center_x - width / 2
                    top = center_y - height / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.__confThreshold, self.__nmsThreshold)
        return (indices, boxes, confidences, class_ids)

    def process_frame(self, frame: np.ndarray):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        blob = cv2.dnn.blobFromImage(frame, self.__scale, (self.__width, self.__height), self.__mean, True, crop=False)

        # Run a model
        self.__net.setInput(blob)
        outs = self.__net.forward(self.get_output_layers_name(self.__net))
        (indices, boxes, confidences, class_ids) = self.post_process_output(frame, outs)
        detected_objects = []

        for i in indices:
            
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            x = int(left)
            y = int(top)
            nw = int(width)
            nh = int(height)
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + nw > frame_width:
                nw = frame_width - x
            if y + nh > frame_height:
                nh = frame_height - y
            detected_objects.append([self.__classes[class_ids[i]], x, y, nw, nh, confidences[i]])
        return detected_objects
CONFIG_FILE = None
model = None
def load_config(config_path):
    global CONFIG_FILE
    CONFIG_FILE = eval(open(config_path).read())


def load_model():
    global model
    model = YoloDetection(CONFIG_FILE["model-parameters"]["model-weights"],
                    CONFIG_FILE["model-parameters"]["model-config"],
                    CONFIG_FILE["model-parameters"]["model-names"],
                    CONFIG_FILE["shape"][0],
                    CONFIG_FILE["shape"][1])


def start_detection(media_path):
    vehicle_tracker = Tracker()    
    cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(media_path)
    writer = imageio.get_writer("demo.mp4")
    ret = True
    while ret:
        ret , frame = cap.read()
        if(ret):
            st = time.time()
            detections = model.process_frame(frame)
            tracked_results = vehicle_tracker.track(detections)
            print(tracked_results)
            for r in tracked_results:
               
                x,y,w,h = r["points"]
                track_id = r["track_id"]
               
                cv2.rectangle(frame,(x,y),(x+w,y+h),thickness=2,color=(255,255,0))
                cv2.putText(frame,track_id+"-"+str(r["direction"]),(x,y-3),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            et = time.time()
            cv2.putText(frame,f"FPS : {round(1/(et-st) , 2)}",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            cv2.imshow("Video",frame)
            writer.append_data(frame[:,:,::-1])
            key = cv2.waitKey(30)
            if(key==27):
                break
            if(key==32):
                cv2.waitKey(-1)
    
    writer.close()

    cv2.destroyAllWindows()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Give config file and media file path")
    parser.add_argument("--config","-c")
    parser.add_argument("--debug","-d")
    parser.add_argument("--video","-v")
    args = parser.parse_args()
    config_path = args.config
    load_config(config_path)
    load_model()
    start_detection(args.video)

