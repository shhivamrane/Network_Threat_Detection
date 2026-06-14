"""Replay a PCAP file through the flow tracker (no live capture needed)."""
from capture.sniffer import FlowTracker

try:
    from scapy.all import rdpcap
except Exception:  # noqa: BLE001
    rdpcap = None


def replay(pcap_path: str):
    """Return a list of feature dicts extracted from a PCAP file."""
    if rdpcap is None:
        raise RuntimeError("Scapy unavailable. Install scapy.")
    tracker = FlowTracker(timeout=0)  # finalize everything at the end
    for pkt in rdpcap(pcap_path):
        tracker.process(pkt)
    # Flush all remaining flows.
    return [
        __import__("capture.features", fromlist=["flow_to_features"])
        .flow_to_features(f)
        for f in tracker.flows.values()
    ]
