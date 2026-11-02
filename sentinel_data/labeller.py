from datetime import datetime
from time import mktime

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
)


def cicids2017(day, flow, label_reverse=False, signal_reverse=False):
    if day == MONDAY:
        label = monday(flow)
    elif day == TUESDAY:
        label = tuesday(flow)
    elif day == WEDNESDAY:
        label = wednesday(flow)
    elif day == THURSDAY:
        label = thursday(flow)
    elif day == FRIDAY:
        label = friday(flow)
    else:
        raise Exception(f"Given day doesn't exists: {day}")

    if label_reverse and label == "BENIGN":
        label = rcicids2017(day, flow, signal_reverse)
    elif signal_reverse:
        flow.udps.reversed = False

    if (
        flow.udps.src2dst_payload == 0
        and flow.protocol == 6
        and label not in ["PortScan"]
    ):  # ["Web Attack - Brute Force", "Web Attack - XSS", "Bot"]):
        label = "BENIGN"

    return label


def rcicids2017(day, flow, signal_reversed=False):
    if day == MONDAY:
        label = monday(flow)
    elif day == TUESDAY:
        label = rtuesday(flow)
    elif day == WEDNESDAY:
        label = rwednesday(flow)
    elif day == THURSDAY:
        label = rthursday(flow)
    elif day == FRIDAY:
        label = rfriday(flow)
    else:
        raise Exception(f"Given day doesn't exists: {day}")

    if signal_reversed and label != "BENIGN":
        flow.udps.reversed = True

    if (
        flow.udps.dst2src_payload == 0
        and flow.protocol == 6
        and label not in ["PortScan"]
    ):
        label = "BENIGN"

    return label


def utime(YYYY, MM, DD, hh, mm):
    time_shift = 5  # Shifting hours because of different timezones.
    return mktime(datetime(YYYY, MM, DD, hh + time_shift, mm).timetuple()) * 1000


def monday(flow):
    return "BENIGN"


def tuesday(flow):
    if (
        flow.src_ip == "172.16.0.1"
        and flow.dst_ip == "192.168.10.50"
        and flow.dst_port == 21
        and flow.protocol == 6
    ):
        return "FTP-Patator"
    elif (
        flow.src_ip == "172.16.0.1"
        and flow.dst_ip == "192.168.10.50"
        and flow.dst_port == 22
        and flow.protocol == 6
        # and flow.bidirectional_first_seen_ms <= utime(2017,7,4,15,16)
    ):
        return "SSH-Patator"
    return "BENIGN"


def rtuesday(flow):
    if (
        flow.dst_ip == "172.16.0.1"
        and flow.src_ip == "192.168.10.50"
        and flow.src_port == 21
        and flow.protocol == 6
    ):
        return "FTP-Patator"
    elif (
        flow.dst_ip == "172.16.0.1"
        and flow.src_ip == "192.168.10.50"
        and flow.src_port == 22
        and flow.protocol == 6
    ):
        return "SSH-Patator"
    return "BENIGN"


def wednesday(flow):
    if (
        flow.src_ip == "172.16.0.1"
        and flow.dst_ip == "192.168.10.50"
        and flow.dst_port == 80
        and flow.protocol == 6
    ):
        if (
            utime(2017, 7, 5, 9, 43)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 5, 10, 12)
        ):
            return "DoS slowloris"
        elif (
            utime(2017, 7, 5, 10, 13)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 5, 10, 40)
        ):
            return "DoS Slowhttptest"
        elif (
            utime(2017, 7, 5, 10, 41)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 5, 11, 8)
        ):
            return "DoS Hulk"
        elif (
            utime(2017, 7, 5, 11, 9)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 5, 11, 24)
        ):
            return "DoS GoldenEye"
    elif (
        flow.src_ip == "172.16.0.1"
        and flow.dst_ip == "192.168.10.51"
        and flow.dst_port == 444
        and flow.src_port == 45022
        and flow.protocol == 6
    ):
        return "Heartbleed"
    return "BENIGN"


def rwednesday(flow):
    if (
        flow.dst_ip == "172.16.0.1"
        and flow.src_ip == "192.168.10.50"
        and flow.src_port == 80
        and flow.protocol == 6
    ):
        if (
            utime(2017, 7, 5, 9, 43)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 5, 10, 12)
        ):
            return "DoS slowloris"
        elif (
            utime(2017, 7, 5, 10, 13)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 5, 10, 40)
        ):
            return "DoS Slowhttptest"
        elif (
            utime(2017, 7, 5, 10, 41)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 5, 11, 8)
        ):
            return "DoS Hulk"
        elif (
            utime(2017, 7, 5, 11, 9)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 5, 11, 24)
        ):
            return "DoS GoldenEye"
    elif (
        flow.src_ip == "192.168.10.51"
        and flow.dst_ip == "172.16.0.1"
        and flow.dst_port == 45022
        and flow.src_port == 444
        and flow.protocol == 6
    ):
        return "Heartbleed"
    return "BENIGN"


def thursday(flow):
    if (
        flow.src_ip == "172.16.0.1"
        and flow.dst_ip == "192.168.10.50"
        and flow.dst_port == 80
        and flow.protocol == 6
    ):
        # 9:15-10:00
        if (
            utime(2017, 7, 6, 9, 10)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 6, 10, 5)
        ):
            return "Web Attack - Brute Force"
        # 10:15-10:35
        elif (
            utime(2017, 7, 6, 10, 10)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 6, 10, 38)
        ):
            return "Web Attack - XSS"
        # 10:40-10:42
        elif (
            utime(2017, 7, 6, 10, 38)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 6, 10, 47)
        ):
            return "Web Attack - Sql Injection"
    # (+)
    elif (
        flow.src_ip == "192.168.10.8"
        and flow.dst_ip == "205.174.165.73"
        and flow.dst_port == 444
        and flow.protocol == 6
    ):
        return "Infiltration"
    elif (
        flow.src2dst_packets == 1
        and 0 <= flow.dst2src_packets <= 1
        and (flow.src_ip == "192.168.10.8" or flow.dst_ip == "192.168.10.8")
        and flow.protocol == 6
        and flow.udps.src2dst_payload == 0
        and flow.bidirectional_duration_ms <= 1
    ):
        return "PortScan"
    return "BENIGN"


def rthursday(flow):
    if (
        flow.dst_ip == "172.16.0.1"
        and flow.src_ip == "192.168.10.50"
        and flow.src_port == 80
        and flow.protocol == 6
    ):
        # 9:15-10:00
        if (
            utime(2017, 7, 6, 9, 10)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 6, 10, 5)
        ):
            return "Web Attack - Brute Force"
        # 10:15-10:35
        elif (
            utime(2017, 7, 6, 10, 10)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 6, 10, 38)
        ):
            return "Web Attack - XSS"
        # 10:40-10:42
        elif (
            utime(2017, 7, 6, 10, 38)
            <= flow.bidirectional_first_seen_ms
            <= utime(2017, 7, 6, 10, 47)
        ):
            return "Web Attack - Sql Injection"
    # (+)
    elif (
        flow.dst_ip == "192.168.10.8"
        and flow.src_ip == "205.174.165.73"
        and flow.src_port == 444
        and flow.protocol == 6
    ):
        return "Infiltration"
    elif (
        flow.dst2src_packets == 1
        and 0 <= flow.src2dst_packets <= 1
        and (flow.src_ip == "192.168.10.8" or flow.dst_ip == "192.168.10.8")
        and flow.protocol == 6
        and flow.udps.dst2src_payload == 0
        and flow.bidirectional_duration_ms <= 1
    ):
        return "PortScan"
    return "BENIGN"


def friday(flow):
    if (
        flow.src_ip == "172.16.0.1"
        and flow.dst_ip == "192.168.10.50"
        and (flow.dst_port == 80)
        and flow.protocol == 6
        # 15:56-16:16
        and utime(2017, 7, 7, 15, 51)
        <= flow.bidirectional_first_seen_ms
        <= utime(2017, 7, 7, 16, 21)
    ):
        return "DDoS"
    elif (
        flow.src_ip == "172.16.0.1"
        and flow.dst_ip == "192.168.10.50"
        and flow.protocol == 6
        # 13:05-15:23
        and utime(2017, 7, 7, 13, 0)
        <= flow.bidirectional_first_seen_ms
        <= utime(2017, 7, 7, 15, 28)
    ):
        return "PortScan"
    elif (
        (
            flow.src_ip == "192.168.10.5"
            or flow.src_ip == "192.168.10.8"
            or flow.src_ip == "192.168.10.9"
            or flow.src_ip == "192.168.10.14"
            or flow.src_ip == "192.168.10.15"
            # or flow.src_ip == '205.174.165.73'
            or flow.src_ip == "192.168.10.12"
            or flow.src_ip == "192.168.10.17"
        )
        and (  # flow.dst_ip == '192.168.10.5'
            # or flow.dst_ip == '192.168.10.8'
            # or flow.dst_ip == '192.168.10.9'
            # or flow.dst_ip == '192.168.10.14'
            # or flow.dst_ip == '192.168.10.15'
            flow.dst_ip == "205.174.165.73"
            or flow.dst_ip == "52.6.13.28"
            or flow.dst_ip == "52.7.235.158"
        )
        and flow.protocol == 6
        # 9:34-12:59
        and utime(2017, 7, 7, 9, 0)
        <= flow.bidirectional_first_seen_ms
        <= utime(2017, 7, 7, 13, 0)
    ):
        return "Bot"

    return "BENIGN"


def rfriday(flow):
    if (
        flow.dst_ip == "172.16.0.1"
        and flow.src_ip == "192.168.10.50"
        and flow.src_port == 80
        and flow.protocol == 6
        # 15:56-16:16
        and utime(2017, 7, 7, 15, 51)
        <= flow.bidirectional_first_seen_ms
        <= utime(2017, 7, 7, 16, 21)
    ):
        return "DDoS"
    elif (
        flow.dst_ip == "172.16.0.1"
        and flow.src_ip == "192.168.10.50"
        and flow.protocol == 6
        # 13:05-15:23
        and utime(2017, 7, 7, 13, 0)
        <= flow.bidirectional_first_seen_ms
        <= utime(2017, 7, 7, 15, 28)
    ):
        return "PortScan"
    elif (
        (
            flow.dst_ip == "192.168.10.5"
            or flow.dst_ip == "192.168.10.8"
            or flow.dst_ip == "192.168.10.9"
            or flow.dst_ip == "192.168.10.14"
            or flow.dst_ip == "192.168.10.15"
            or flow.dst_ip == "192.168.10.12"
            or flow.dst_ip == "192.168.10.17"
        )
        and (  # flow.dst_ip == '192.168.10.5'
            # or flow.dst_ip == '192.168.10.8'
            # or flow.dst_ip == '192.168.10.9'
            # or flow.dst_ip == '192.168.10.14'
            # or flow.dst_ip == '192.168.10.15'
            flow.src_ip == "205.174.165.73"
            or flow.src_ip == "52.6.13.28"
            or flow.src_ip == "52.7.235.158"
        )
        and flow.protocol == 6
        # 9:34-12:59
        and utime(2017, 7, 7, 9, 0)
        <= flow.bidirectional_first_seen_ms
        <= utime(2017, 7, 7, 13, 0)
    ):
        return "Bot"

    return "BENIGN"
