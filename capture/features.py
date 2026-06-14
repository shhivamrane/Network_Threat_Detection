"""Convert a network flow (list of packets) into the model's feature vector."""
from dataclasses import dataclass, field
from ml.data.nslkdd import COLUMNS

# NSL-KDD feature columns (excluding label/difficulty).
FEATURE_COLUMNS = [c for c in COLUMNS if c not in ("label", "difficulty")]

PROTO_MAP = {6: "tcp", 17: "udp", 1: "icmp"}


@dataclass
class Flow:
    """Accumulates packets sharing the same 5-tuple."""
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    packets: list = field(default_factory=list)
    src_bytes: int = 0
    dst_bytes: int = 0
    start: float = 0.0
    last: float = 0.0

    def add(self, pkt_len: int, ts: float, from_src: bool):
        if not self.packets:
            self.start = ts
        self.last = ts
        self.packets.append(ts)
        if from_src:
            self.src_bytes += pkt_len
        else:
            self.dst_bytes += pkt_len


def _service_from_port(port: int) -> str:
    well_known = {
        80: "http", 443: "http", 21: "ftp", 22: "ssh",
        25: "smtp", 53: "dns", 23: "telnet",
    }
    return well_known.get(port, "other")


def flow_to_features(flow: Flow) -> dict:
    """Map a Flow to the NSL-KDD feature schema with safe defaults."""
    duration = max(0.0, flow.last - flow.start)
    feats = {col: 0 for col in FEATURE_COLUMNS}

    feats["duration"] = duration
    feats["protocol_type"] = PROTO_MAP.get(flow.proto, "tcp")
    feats["service"] = _service_from_port(flow.dport)
    feats["flag"] = "SF"
    feats["src_bytes"] = flow.src_bytes
    feats["dst_bytes"] = flow.dst_bytes
    feats["count"] = len(flow.packets)
    feats["srv_count"] = len(flow.packets)
    # Remaining statistical fields default to 0; refine with CICIDS2017 schema.
    return feats
