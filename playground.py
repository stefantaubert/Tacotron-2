import tensorflow as tf
#tf.__version__sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from tensorflow.python.platform import build_info as tf_build_info
tf.config.list_physical_devices('GPU') 
#print(tf_build_info.cuda_version_number)
# 9.0 in v1.10.0
#print(tf_build_info.cudnn_version_number)
