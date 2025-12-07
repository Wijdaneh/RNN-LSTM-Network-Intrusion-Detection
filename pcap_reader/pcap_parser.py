from scapy.layers.inet import IP, TCP, UDP

def parse_packet(pkt):
    if IP not in pkt:
        return None

    src = pkt[IP].src
    dst = pkt[IP].dst
    length = len(pkt)
    
    if TCP in pkt:
        protocol = "TCP"
    elif UDP in pkt:
        protocol = "UDP"
    else:
        protocol = "OTHER"

    return {
        "src_ip": src,
        "dst_ip": dst,
        "protocol": protocol,
        "length": length
    }
