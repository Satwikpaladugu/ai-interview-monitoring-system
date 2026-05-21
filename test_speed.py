import time
from deepface import DeepFace

# warmup
start = time.time()
DeepFace.represent('uploads/reference_candidate_1779386650468_dae762vy0.jpg', detector_backend='mtcnn', enforce_detection=False)
print('Warmup Time:', time.time()-start)

# test
start = time.time()
DeepFace.represent('uploads/reference_candidate_1779386650468_dae762vy0.jpg', detector_backend='mtcnn', enforce_detection=False)
print('Execution Time:', time.time()-start)
