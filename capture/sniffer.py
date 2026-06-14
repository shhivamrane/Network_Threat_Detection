"""Live packet capture with Scapy. Requires admin privileges and Npcap (Windows)."""
import time
from collections import defaultdict
from capture.features import Flow, flow_to_features

try:
    from scapy.all import sniff, IP, TCP, UDP
except Exception:  # noqa: BLE001
    sniff = None


class FlowTracker:
    def __init__(self, timeout: float = 15.0):
        self.flows: dict = {}
        self.timeout = timeout

    def _key(self, ip, l4):
        return (ip.src, ip.dst, getattr(l4, "sport", 0),
                getattr(l4, "dport", 0), ip.proto)

    def process(self, pkt):
        if IP not in pkt:
            return None
        ip = pkt[IP]
        l4 = pkt[TCP] if TCP in pkt else (pkt[UDP] if UDP in pkt else None)
        if l4 is None:
            return None

        key = self._key(ip, l4)
        rev = (ip.dst, ip.src, getattr(l4, "dport", 0),
               getattr(l4, "sport", 0), ip.proto)

        if key in self.flows:
            flow, from_src = self.flows[key], True
        elif rev in self.flows:
            flow, from_src = self.flows[rev], False
        else:
            flow = Flow(ip.src, ip.dst, getattr(l4, "sport", 0),
                        getattr(l4, "dport", 0), ip.proto)
            self.flows[key] = flow
            from_src = True

        flow.add(len(pkt), time.time(), from_src)
        return None

    def expired(self):
        """Yield features for flows that have timed out."""
        now = time.time()
        done = [k for k, f in self.flows.items() if now - f.last > self.timeout]
        for k in done:
            yield flow_to_features(self.flows.pop(k))


def capture(callback, iface=None, count=0):
    """Sniff packets and call `callback(features_dict)` for each completed flow."""
    if sniff is None:
        raise RuntimeError("Scapy unavailable. Install scapy and Npcap.")
    tracker = FlowTracker()

    def handle(pkt):
        tracker.process(pkt)
        for feats in tracker.expired():
            callback(feats)

    sniff(prn=handle, iface=iface, count=count, store=False)
