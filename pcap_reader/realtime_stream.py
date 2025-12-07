import time
import numpy as np
from scapy.all import rdpcap
from utils.preprocessor_unsw import UNSWPreprocessor
import tensorflow as tf
import joblib

class RealTimeAnalyzer:

    def __init__(self, model_path, scaler_path, feature_config, threshold):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = feature_config
        self.threshold = threshold
        self.preprocessor = UNSWPreprocessor()

    def predict_packet(self, parsed_pkt):
        df = self.preprocessor.packet_to_dataframe(parsed_pkt)
        x = self.scaler.transform(df[self.features])
        y = self.model.predict(x)[0][0]
        return y, y > self.threshold

    def stream_pcap(self, pcap_path):
        packets = rdpcap(pcap_path)

        for pkt in packets:
            yield pkt
            time.sleep(0.02)  # simulate real-time
