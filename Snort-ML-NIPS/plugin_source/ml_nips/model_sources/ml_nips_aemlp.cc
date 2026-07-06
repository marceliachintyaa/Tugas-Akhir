//--------------------------------------------------------------------------
// Copyright (C) 2014-2026 Cisco and/or its affiliates. All rights reserved.
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License Version 2 as published
// by the Free Software Foundation.  You may not use, modify or distribute
// this program under any other version of the GNU General Public License.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
//--------------------------------------------------------------------------
// MlNips.cc author Russ Combs <rcombs@sourcefire.com>

#include "detection/detection_engine.h"
#include "framework/inspector.h"
#include "framework/module.h"
#include "log/messages.h"
#include "profiler/profiler.h"
#include "protocols/packet.h"
#include "trace/trace_api.h"
#include <algorithm>
#include <cstdint>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include "protocols/tcp.h"
#include "scaler_params_quantile.h"

using namespace snort;

#define ML_NIPS_GID 256
#define ML_NIPS_SID 1

static const char* s_name = "ml_nips";
static const char* s_help = "ML-based NIPS inspector";

static constexpr float ML_NIPS_MATURE_THRESHOLD = 0.683f;    // model threshold from thresholds.json
static constexpr float ML_NIPS_EARLY_THRESHOLD = 0.683f;     // same ML threshold for early/runtime scoring
static constexpr uint64_t ML_MIN_PACKETS = 5;               // minimum packet count before any ML scoring
static constexpr uint64_t ML_RECHECK_EVERY_PACKETS = 5;     // early re-score every 5 packets
static constexpr uint64_t ML_MIN_DURATION_US = 20000000ULL; // mature by time: 20 seconds for low-rate traffic patterns
static constexpr uint64_t ML_MATURE_MIN_PACKETS = 20;       // mature by packet count for fast but sufficiently large flows
static constexpr uint64_t ML_MATURE_RECHECK_US = 2000000ULL;// mature re-score every 2 seconds
static constexpr uint64_t ML_SHORT_FLOW_GUARD_US = 1000000ULL; // guard short HTTP flows: 1 second
static constexpr uint64_t ML_AUTH_MIN_PACKETS = 5;         // SSH authentication-related can be short; allow mature decision on port 22 after 5 packets
static constexpr uint64_t ML_ACTIVE_TIMEOUT_US = 5000000ULL;    // CICFlowMeter-like active/idle split: gap > 5s = idle
static constexpr uint16_t ML_HTTP_PORT = 80;                  // HTTP service port for short-flow confirmation guard
static constexpr uint16_t ML_SSH_PORT = 22;                   // allow SSH authentication-related flows without changing model features
static constexpr size_t ML_MAX_FLOWS = 100000;
static constexpr uint32_t ML_BLOCK_TIMEOUT_SEC = 300;       // ipset block duration after ML detects attack
static const char* ML_IPSET_NAME = "ml_nips_blocklist";

static THREAD_LOCAL ProfileStats MlNipsPerfStats;

static THREAD_LOCAL SimpleStats ml_nips_stats;

static std::unique_ptr<tflite::FlatBufferModel> encoder_model;
static std::unique_ptr<tflite::FlatBufferModel> ae_model;
static std::unique_ptr<tflite::FlatBufferModel> mlp_model;
static std::unique_ptr<tflite::Interpreter> encoder_interpreter;
static std::unique_ptr<tflite::Interpreter> ae_interpreter;
static std::unique_ptr<tflite::Interpreter> mlp_interpreter;
// Dual CSV outputs:
// 1) IDS log: all evaluated ML detections, used for detection analysis/report.
// 2) IPS log: only rows where ML decides to block, used for prevention/blocking report.
static std::ofstream ids_log("/home/nadyahra/ml_ids_log_aemlp.csv", std::ios::app);
static std::ofstream ips_log("/home/nadyahra/ml_ips_log_aemlp.csv", std::ios::app);

struct FlowState
{
    uint64_t first_seen = 0;
    uint64_t last_seen = 0;
    uint64_t active_start_time = 0;   // start of current active period, used for CICFlowMeter-like Active/Idle

    uint64_t fwd_pkts = 0;
    uint64_t bwd_pkts = 0;

    uint64_t fwd_bytes = 0;
    uint64_t bwd_bytes = 0;

    uint32_t pkt_len_min = 0xffffffff;
    uint32_t pkt_len_max = 0;
    uint64_t pkt_len_sum = 0;
    uint64_t pkt_count = 0;
    uint64_t fwd_header_len = 0;
    uint64_t bwd_header_len = 0;

    uint64_t ack_cnt = 0;
    uint64_t syn_cnt = 0;  // used only for feature-readiness; ML decision still uses score threshold
    uint64_t fin_cnt = 0;
    uint64_t rst_cnt = 0;
    uint64_t psh_cnt = 0;
    uint64_t urg_cnt = 0;
    uint64_t cwe_cnt = 0;

    uint64_t fwd_psh_flags = 0;

    // CICFlowMeter-compatible Fwd Seg Size Min.
    // Use minimum forward TCP payload/application-data size, not TCP header size
    // and not full packet length. Pure SYN/ACK packets commonly have 0 payload.
    uint32_t fwd_seg_size_min = UINT32_MAX;

    uint32_t init_fwd_win_bytes = 0;
    uint32_t init_bwd_win_bytes = 0;

    uint64_t last_pkt_time = 0;
    uint64_t last_fwd_time = 0;
    uint64_t last_bwd_time = 0;

    uint64_t flow_iat_max = 0;
    uint64_t flow_iat_sum = 0;
    uint64_t flow_iat_count = 0;
    double flow_iat_sq_sum = 0.0;

    uint64_t fwd_iat_sum = 0;
    uint64_t fwd_iat_count = 0;
    double fwd_iat_sq_sum = 0.0;

    uint64_t bwd_iat_max = 0;
    uint64_t bwd_iat_min = UINT64_MAX;
    uint64_t bwd_iat_sum = 0;
    uint64_t bwd_iat_count = 0;
    double bwd_iat_sq_sum = 0.0;
    uint64_t fwd_pkt_len_sum = 0;
    uint64_t fwd_pkt_len_sq_sum = 0;
    uint32_t fwd_pkt_len_min = UINT32_MAX;
    uint32_t fwd_pkt_len_max = 0;
    uint64_t fwd_pkt_len_count = 0;

    uint64_t bwd_pkt_len_sum = 0;
    uint64_t bwd_pkt_len_sq_sum = 0;
    uint32_t bwd_pkt_len_min = UINT32_MAX;
    uint32_t bwd_pkt_len_max = 0;
    uint64_t bwd_pkt_len_count = 0;

    uint64_t pkt_len_sq_sum = 0;

    uint64_t active_max = 0;
    uint64_t active_min = UINT64_MAX;
    double active_sq_sum = 0.0;
    uint64_t active_sum = 0;
    uint64_t active_count = 0;

    uint64_t idle_max = 0;
    double idle_sq_sum = 0.0;
    uint64_t idle_sum = 0;
    uint64_t idle_count = 0;

    uint64_t last_ml_pkt_count = 0;              // packet-count based early scoring
    uint64_t last_mature_ml_duration = 0;        // time-based mature scoring
    float max_score_seen = 0.0f;                 // logging only; final decision uses current ML score
    bool early_hit = false;                      // true when current ML score crosses threshold in early stage
    bool mature_hit = false;                     // true when current ML score crosses threshold in mature stage
    bool flow_detected = false;                  // sticky label/alert state for this flow
    bool alert_queued = false;                   // alert only once per flow
    bool blocked = false;                        // ML block decision, set once per flow
    bool ipset_applied = false;                  // true only if ipset command returns success
};

static std::unordered_map<std::string, FlowState> flow_table;


// Pure-ML feature readiness guard.
// This does not classify a flow as attack or normal. It only prevents ML inference
// before the bidirectional/statistical flow features are representative enough.
// Final attack decision remains: score >= active_threshold.
//------------------------------------------------------------------------------
// Feature Readiness Check
//------------------------------------------------------------------------------

static inline bool ml_feature_ready(bool is_tcp, bool is_udp,
    uint64_t pkt_count, uint64_t bwd_pkt_count, uint64_t bwd_iat_sum,
    uint64_t syn_count)
{
    // UDP has no TCP handshake/backward requirement. Wait for a minimal packet count
    // so rate/IAT/statistical features are not built from a single packet.
    if (is_udp)
        return pkt_count >= ML_MIN_PACKETS;

    if (is_tcp)
    {
        // Normal TCP readiness gate. One-packet SYN handling is decided in eval()
        // with source-IP context, so normal private-lab curl SYN packets are not
        // evaluated too early.
        if (pkt_count < ML_MIN_PACKETS)
            return false;

        // SYN-heavy flows such as hping/SYN flood may be forward-only. Allow inference
        // once enough SYN evidence exists, but ML still decides from the score.
        if (syn_count >= 3)
            return true;

        // Normal bidirectional TCP/HTTP/SSH flows:
        // Do NOT infer only because one backward packet exists. In the false-positive
        // curl logs, bwd_pkts was already non-zero but Bwd_IAT_Tot was still 0, so
        // the model saw an incomplete/unstable backward timing feature.
        if (bwd_pkt_count > 0 && bwd_iat_sum > 0)
            return true;

        // Fallback for short bidirectional TCP flows: wait until the flow has more
        // packets, so FIN/ACK or additional response packets can complete the stats.
        // This is not a benign/attack rule; it only prevents inference on half-built
        // flow features.
        if (bwd_pkt_count > 0 && pkt_count >= 10)
            return true;

        return false;
    }

    return false;
}

//------------------------------------------------------------------------------
// IP Validation and Filtering
//------------------------------------------------------------------------------

static bool is_valid_ipv4_for_shell(const std::string& ip)
{
    if (ip.empty() || ip.size() > 15)
        return false;

    int dots = 0;
    int digits = 0;
    int value = 0;

    for (char c : ip)
    {
        if (c == '.')
        {
            if (digits == 0 || value > 255)
                return false;
            dots++;
            digits = 0;
            value = 0;
            continue;
        }

        if (c < '0' || c > '9')
            return false;

        value = (value * 10) + (c - '0');
        digits++;

        if (digits > 3 || value > 255)
            return false;
    }

    return dots == 3 && digits > 0 && value <= 255;
}


static bool parse_ipv4_octets(const std::string& ip, int octets[4])
{
    if (!is_valid_ipv4_for_shell(ip))
        return false;

    int idx = 0;
    int value = 0;

    for (char c : ip)
    {
        if (c == '.')
        {
            if (idx >= 4)
                return false;

            octets[idx++] = value;
            value = 0;
            continue;
        }

        value = (value * 10) + (c - '0');
    }

    if (idx != 3)
        return false;

    octets[idx] = value;
    return true;
}

static bool is_non_unicast_ipv4(const std::string& ip)
{
    int o[4] = {0, 0, 0, 0};

    if (!parse_ipv4_octets(ip, o))
        return true;

    // 0.0.0.0/8, multicast 224.0.0.0/4, experimental 240.0.0.0/4,
    // and limited/directed broadcast patterns commonly seen in VM labs.
    if (o[0] == 0 || o[0] >= 224)
        return true;

    if (o[3] == 255)
        return true;

    return false;
}

static bool should_skip_flow_address(const std::string& src, const std::string& dst)
{
    return is_non_unicast_ipv4(src) || is_non_unicast_ipv4(dst);
}


static bool is_private_ipv4(const std::string& ip)
{
    int o[4] = {0, 0, 0, 0};

    if (!parse_ipv4_octets(ip, o))
        return false;

    // RFC1918 private ranges commonly used by lab VMs.
    if (o[0] == 10)
        return true;

    if (o[0] == 172 && o[1] >= 16 && o[1] <= 31)
        return true;

    if (o[0] == 192 && o[1] == 168)
        return true;

    return false;
}

//------------------------------------------------------------------------------
// Runtime Blocking Using IPSet
//------------------------------------------------------------------------------

static bool block_client_with_ipset(const std::string& client_ip)
{
    if (!is_valid_ipv4_for_shell(client_ip))
    {
        LogMessage("ml_nips: skip blocking invalid/non-IPv4 client_ip=%s\n", client_ip.c_str());
        return false;
    }

    char cmd[512];
    std::snprintf(cmd, sizeof(cmd),
        "ipset create %s hash:ip timeout %u -exist >/dev/null 2>&1 && "
        "ipset add %s %s timeout %u -exist >/dev/null 2>&1",
        ML_IPSET_NAME, ML_BLOCK_TIMEOUT_SEC,
        ML_IPSET_NAME, client_ip.c_str(), ML_BLOCK_TIMEOUT_SEC);

    int rc = std::system(cmd);

    if (rc == 0)
    {
        LogMessage("ml_nips: BLOCKED client_ip=%s\n",
            client_ip.c_str());
        return true;
    }

    LogMessage("ml_nips: IPSET_NOT_APPLIED client_ip=%s rc=%d. Check ipset exists, iptables policy, and sudo permission.\n",
        client_ip.c_str(), rc);
    return false;
}

THREAD_LOCAL const Trace* ml_nips_trace = nullptr;

//-------------------------------------------------------------------------
// class stuff
//-------------------------------------------------------------------------

class MlNips : public Inspector
{
public:
    MlNips(uint16_t port, uint16_t max);

    void show(const SnortConfig*) const override;
    void eval(Packet*) override;

private:
    uint16_t port;
    uint16_t max;
};

//------------------------------------------------------------------------------
// Model Initialization
//------------------------------------------------------------------------------

MlNips::MlNips(uint16_t p, uint16_t m)
{
    port = p;
    max = m;

    encoder_model = tflite::FlatBufferModel::BuildFromFile("/opt/snort-ml/ae_encoder_final.tflite");
    ae_model = tflite::FlatBufferModel::BuildFromFile("/opt/snort-ml/ae_model_final.tflite");
    mlp_model = tflite::FlatBufferModel::BuildFromFile("/opt/snort-ml/ae_mlp_final.tflite");

    if (!encoder_model || !ae_model || !mlp_model)
    {
        LogMessage("ml_nips: FAILED load one or more TFLite models (encoder/ae/mlp)\n");
        return;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*encoder_model, resolver)(&encoder_interpreter);
    tflite::InterpreterBuilder(*ae_model, resolver)(&ae_interpreter);
    tflite::InterpreterBuilder(*mlp_model, resolver)(&mlp_interpreter);

    if (!encoder_interpreter || !ae_interpreter || !mlp_interpreter)
    {
        LogMessage("ml_nips: FAILED create one or more interpreters (encoder/ae/mlp)\n");
        return;
    }

    if (encoder_interpreter->AllocateTensors() != kTfLiteOk ||
        ae_interpreter->AllocateTensors() != kTfLiteOk ||
        mlp_interpreter->AllocateTensors() != kTfLiteOk)
    {
        LogMessage("ml_nips: FAILED allocate tensors for encoder/ae/mlp\n");
        return;
    }

    LogMessage("ml_nips: AE+MLP models loaded SUCCESS (encoder + autoencoder + mlp)\n");

    auto write_ids_csv_header = [](std::ofstream& out)
    {
        if (out.is_open() && out.tellp() == 0)
        {
            out <<
    "timestamp_us,"
    "ACK_Flag_Cnt,"
    "Active_Max,"
    "Active_Min,"
    "Active_Std,"
    "Bwd_Header_Len,"
    "Bwd_IAT_Max,"
    "Bwd_IAT_Mean,"
    "Bwd_IAT_Min,"
    "Bwd_IAT_Std,"
    "Bwd_IAT_Tot,"
    "Bwd_Pkt_Len_Min,"
    "CWE_Flag_Count,"
    "Down_Up_Ratio,"
    "FIN_Flag_Cnt,"
    "Flow_Byts_s,"
    "Flow_Duration,"
    "Flow_IAT_Max,"
    "Flow_IAT_Std,"
    "Flow_Pkts_s,"
    "Fwd_Header_Len,"
    "Fwd_IAT_Mean,"
    "Fwd_IAT_Std,"
    "Fwd_PSH_Flags,"
    "Fwd_Pkt_Len_Max,"
    "Fwd_Pkt_Len_Mean,"
    "Fwd_Pkt_Len_Min,"
    "Fwd_Seg_Size_Min,"
    "Idle_Max,"
    "Idle_Std,"
    "Init_Bwd_Win_Byts,"
    "Init_Fwd_Win_Byts,"
    "PSH_Flag_Cnt,"
    "Pkt_Len_Max,"
    "Pkt_Len_Mean,"
    "Pkt_Len_Min,"
    "Pkt_Len_Var,"
    "RST_Flag_Cnt,"
    "URG_Flag_Cnt,"
    "client_ip,server_ip,client_port,server_port,"
    "score,max_score,threshold,stage,avg_fwd_l4_len,label\n";
        }
    };

    auto write_ips_csv_header = [](std::ofstream& out)
    {
        if (out.is_open() && out.tellp() == 0)
        {
            out <<
    "timestamp_us,client_ip,server_ip,client_port,server_port,"
    "score,threshold,stage,block_decision,ipset_applied,action\n";
        }
    };

    write_ids_csv_header(ids_log);
    write_ips_csv_header(ips_log);
}

void MlNips::show(const SnortConfig*) const
{
    ConfigLogger::log_value("port", port);
    ConfigLogger::log_value("max", max);
}

//------------------------------------------------------------------------------
// Packet Processing Pipeline
//------------------------------------------------------------------------------

void MlNips::eval(Packet* p)
{
    if (!p->has_ip())
        return;

    SfIpString src_ip;
    SfIpString dst_ip;

    p->ptrs.ip_api.get_src()->ntop(src_ip);
    p->ptrs.ip_api.get_dst()->ntop(dst_ip);

    const bool is_tcp_packet = p->is_tcp();
    const bool is_udp_packet = p->is_udp();

    // NIPS scope: inspect all TCP/UDP transport flows on all ports.
    // ICMP/ARP/other non-TCP-UDP traffic is skipped because the trained flow features
    // are TCP/UDP-oriented. This is not an attack/benign rule.
    if (!is_tcp_packet && !is_udp_packet)
        return;

    uint16_t src_port = p->ptrs.sp;
    uint16_t dst_port = p->ptrs.dp;

    std::string src = src_ip;
    std::string dst = dst_ip;

    // Ignore non-unicast/broadcast traffic before creating a flow entry.
    // This removes VM broadcast noise such as 192.168.x.255 from CSV logs.
    if (should_skip_flow_address(src, dst))
        return;

    // CICFlowMeter-like direction:
    // Prefer Snort's native flow metadata: client -> server = forward.
    // If p->flow is not available, use first-seen 5-tuple direction and match
    // the reverse key on later packets. No port/service heuristic is used.
    std::string client_ip = src;
    std::string server_ip = dst;
    uint16_t client_port = src_port;
    uint16_t server_port = dst_port;
    bool is_forward = true;

    if (p->flow)
    {
        SfIpString flow_client_ip;
        SfIpString flow_server_ip;

        p->flow->client_ip.ntop(flow_client_ip);
        p->flow->server_ip.ntop(flow_server_ip);

        client_ip = flow_client_ip;
        server_ip = flow_server_ip;
        client_port = p->flow->client_port;
        server_port = p->flow->server_port;
        is_forward = p->is_from_client();
    }
    else
    {
        std::string bwd_key = dst + ":" + std::to_string(dst_port) + "->" +
                              src + ":" + std::to_string(src_port);

        auto bwd_it = flow_table.find(bwd_key);
        if (bwd_it != flow_table.end())
        {
            client_ip = dst;
            server_ip = src;
            client_port = dst_port;
            server_port = src_port;
            is_forward = false;
        }
    }

    std::string key = client_ip + ":" + std::to_string(client_port) + "->" +
                      server_ip + ":" + std::to_string(server_port);

    if (flow_table.size() > ML_MAX_FLOWS)
        flow_table.clear();

    FlowState& fs = flow_table[key];

    uint64_t now = (uint64_t)p->pkth->ts.tv_sec * 1000000ULL + p->pkth->ts.tv_usec;

    // Balanced CICFlowMeter-style packet length approximation.
    // Do NOT use p->pkth->pktlen directly: it includes Ethernet/IP headers and made
    // benign curl look too large/attack-like.
    // Do NOT use p->dsize only: many TCP/UDP attack packets have zero application
    // payload, so payload-only removed too much signal and made attacks look normal.
    //
    // Use Layer-4 length instead:
    //   TCP: TCP header length + decoded payload size
    //   UDP: UDP header length + decoded payload size
    // This keeps transport-flow signal for SYN/UDP/HTTP floods while removing L2/L3
    // headers. It does not create any manual attack/benign rule; ML still decides.
    uint32_t transport_header_len_for_len = 0;
    if (is_tcp_packet)
    {
        transport_header_len_for_len = 20;
        const tcp::TCPHdr* len_tcph = p->ptrs.tcph;
        if (len_tcph)
        {
            transport_header_len_for_len = len_tcph->off();
            if (transport_header_len_for_len < 20 || transport_header_len_for_len > 60)
                transport_header_len_for_len = 20;
        }
    }
    else if (is_udp_packet)
    {
        transport_header_len_for_len = 8;
    }

    uint32_t pkt_len = transport_header_len_for_len + (uint32_t)p->dsize;

    if (fs.first_seen == 0)
    {
        fs.first_seen = now;
        fs.last_seen = now;
        fs.active_start_time = now;
    }

    fs.last_seen = now;

    uint64_t duration = 0;
    if (fs.first_seen > 0 && fs.last_seen >= fs.first_seen)
        duration = fs.last_seen - fs.first_seen; // microseconds, same scale used by CICFlowMeter

    // Flow IAT + CICFlowMeter-like Active/Idle
    // IAT is always the gap between consecutive packets in the same bidirectional flow.
    // Active/Idle is NOT the same as IAT: if a gap is > 5 seconds, CICFlowMeter treats
    // the previous continuous burst as an active period and the gap itself as idle time.
    if (fs.last_pkt_time != 0 && now >= fs.last_pkt_time)
    {
        uint64_t iat = now - fs.last_pkt_time;

        fs.flow_iat_sum += iat;
        fs.flow_iat_sq_sum += (double)iat * (double)iat;
        fs.flow_iat_count++;

        if (iat > fs.flow_iat_max)
            fs.flow_iat_max = iat;

        if (iat > ML_ACTIVE_TIMEOUT_US)
        {
            uint64_t active_duration = 0;

            if (fs.active_start_time != 0 && fs.last_pkt_time >= fs.active_start_time)
                active_duration = fs.last_pkt_time - fs.active_start_time;

            fs.active_sum += active_duration;
            fs.active_sq_sum += (double)active_duration * (double)active_duration;
            fs.active_count++;

            if (active_duration > fs.active_max)
                fs.active_max = active_duration;

            if (active_duration < fs.active_min)
                fs.active_min = active_duration;

            fs.idle_sum += iat;
            fs.idle_sq_sum += (double)iat * (double)iat;
            fs.idle_count++;

            if (iat > fs.idle_max)
                fs.idle_max = iat;

            fs.active_start_time = now;
        }
    }

    fs.last_pkt_time = now;

    // Directional IAT
    if (is_forward)
    {
        if (fs.last_fwd_time != 0 && now >= fs.last_fwd_time)
        {
            uint64_t iat = now - fs.last_fwd_time;
            fs.fwd_iat_sum += iat;
            fs.fwd_iat_sq_sum += (double)iat * (double)iat;
            fs.fwd_iat_count++;
        }

        fs.last_fwd_time = now;
    }
    else
    {
        if (fs.last_bwd_time != 0 && now >= fs.last_bwd_time)
        {
            uint64_t iat = now - fs.last_bwd_time;
            fs.bwd_iat_sum += iat;
            fs.bwd_iat_sq_sum += (double)iat * (double)iat;
            fs.bwd_iat_count++;

            if (iat > fs.bwd_iat_max)
                fs.bwd_iat_max = iat;

            if (iat < fs.bwd_iat_min)
                fs.bwd_iat_min = iat;
        }

        fs.last_bwd_time = now;
    }

    if (is_forward)
    {
        fs.fwd_pkts++;
        fs.fwd_bytes += pkt_len;
    }
    else
    {
        fs.bwd_pkts++;
        fs.bwd_bytes += pkt_len;
    }

    if (p->is_tcp())
    {
        uint32_t tcp_header_len = 20;
        const tcp::TCPHdr* tcph = p->ptrs.tcph;

        if (tcph)
        {
            // TCPHdr::off() in this Snort API already gives header length in bytes.
            tcp_header_len = tcph->off();

            if (tcp_header_len < 20 || tcp_header_len > 60)
                tcp_header_len = 20;
        }

        if (is_forward)
            fs.fwd_header_len += tcp_header_len;
        else
            fs.bwd_header_len += tcp_header_len;

        // CICFlowMeter-compatible Fwd_Seg_Size_Min value: forward TCP payload size.
        // Feature name, position, and count are unchanged.
        uint32_t tcp_payload_len = (uint32_t)p->dsize;
        if (is_forward)
            fs.fwd_seg_size_min = std::min(fs.fwd_seg_size_min, tcp_payload_len);

        if (tcph)
        {
            if (tcph->is_ack()) fs.ack_cnt++;
            if (tcph->is_fin()) fs.fin_cnt++;
            if (tcph->is_rst()) fs.rst_cnt++;

            if (tcph->is_psh())
            {
                fs.psh_cnt++;
                if (is_forward)
                    fs.fwd_psh_flags++;
            }

            const uint8_t tcp_flags = tcph->th_flags;
            if (tcp_flags & 0x02)  // SYN bit; readiness only, not a manual attack rule
                fs.syn_cnt++;

            if (tcp_flags & 0x20)  // URG bit
                fs.urg_cnt++;

            if (is_forward && fs.init_fwd_win_bytes == 0)
                fs.init_fwd_win_bytes = tcph->win();

            if (!is_forward && fs.init_bwd_win_bytes == 0)
                fs.init_bwd_win_bytes = tcph->win();
        }
    }

    fs.pkt_count++;
    fs.pkt_len_sum += pkt_len;
    fs.pkt_len_min = std::min(fs.pkt_len_min, pkt_len);
    fs.pkt_len_max = std::max(fs.pkt_len_max, pkt_len);
    fs.pkt_len_sq_sum += (uint64_t)pkt_len * (uint64_t)pkt_len;

    if (is_forward)
    {
        fs.fwd_pkt_len_sum += pkt_len;
        fs.fwd_pkt_len_sq_sum += (uint64_t)pkt_len * (uint64_t)pkt_len;
        fs.fwd_pkt_len_count++;
        fs.fwd_pkt_len_min = std::min(fs.fwd_pkt_len_min, pkt_len);
        fs.fwd_pkt_len_max = std::max(fs.fwd_pkt_len_max, pkt_len);
    }
    else
    {
        fs.bwd_pkt_len_sum += pkt_len;
        fs.bwd_pkt_len_sq_sum += (uint64_t)pkt_len * (uint64_t)pkt_len;
        fs.bwd_pkt_len_count++;
        fs.bwd_pkt_len_min = std::min(fs.bwd_pkt_len_min, pkt_len);
        fs.bwd_pkt_len_max = std::max(fs.bwd_pkt_len_max, pkt_len);
    }

    double pkt_len_mean_for_log = fs.pkt_count ? (double)fs.pkt_len_sum / fs.pkt_count : 0.0;
    double pkt_len_var_for_log = 0.0;

    if (fs.pkt_count > 0)
    {
        double mean_sq = (double)fs.pkt_len_sq_sum / fs.pkt_count;
        pkt_len_var_for_log = mean_sq - (pkt_len_mean_for_log * pkt_len_mean_for_log);

        if (pkt_len_var_for_log < 0.0)
            pkt_len_var_for_log = 0.0;
    }

    double flow_bytes_per_sec = 0.0;
    if (duration > 0)
        flow_bytes_per_sec = ((double)(fs.fwd_bytes + fs.bwd_bytes) * 1000000.0) / duration;

    /* CLEAN_LOG: noisy terminal debug disabled
LogMessage(
        "ml_nips flow=%s fwd_pkts=%lu bwd_pkts=%lu fwd_bytes=%lu bwd_bytes=%lu duration_us=%lu l4_len_min=%u l4_len_max=%u l4_len_mean=%.2f\n",
        key.c_str(), fs.fwd_pkts, fs.bwd_pkts, fs.fwd_bytes, fs.bwd_bytes,
        duration, fs.pkt_len_min, fs.pkt_len_max, pkt_len_mean_for_log
    );
*/

    // Pure ML scoring mode:
    // The final attack decision is based only on the ML model score and the validation threshold.
    // Protocol/flow metadata below is used only to decide when to run inference, not to add rules.
    const bool is_ssh_flow = (server_port == ML_SSH_PORT);

    const bool feature_ready_base = ml_feature_ready(
        is_tcp_packet, is_udp_packet, fs.pkt_count, fs.bwd_pkts,
        fs.bwd_iat_sum, fs.syn_cnt);

    // Pure-ML spoofed SYN readiness:
    // This only decides whether inference is allowed. It does not decide attack.
    // Normal lab traffic from RFC1918/private IPs, such as curl from 192.168.x.x,
    // must not be evaluated at the first SYN packet because that caused false blocks.
    const bool spoofed_syn_ready =
        is_tcp_packet &&
        fs.syn_cnt >= 1 &&
        fs.bwd_pkts == 0 &&
        fs.pkt_count >= 1 &&
        !is_private_ipv4(client_ip);

    const bool feature_ready = feature_ready_base || spoofed_syn_ready;

    if (!feature_ready)
    {
        /* CLEAN_LOG: noisy terminal debug disabled
LogMessage("ml_nips: WAIT FEATURE_READY flow=%s proto=%s pkt_count=%lu bwd_pkts=%lu bwd_iat_sum=%lu syn_cnt=%lu duration_us=%lu private_src=%d\n",
            key.c_str(), is_tcp_packet ? "TCP" : "UDP",
            fs.pkt_count, fs.bwd_pkts, fs.bwd_iat_sum, fs.syn_cnt, duration,
            is_private_ipv4(client_ip) ? 1 : 0);
*/
        return;
    }

    const bool mature_by_time = (duration >= ML_MIN_DURATION_US);
    const bool mature_by_packets = (fs.pkt_count >= ML_MATURE_MIN_PACKETS);
    const bool mature_by_auth_pattern = is_ssh_flow && fs.pkt_count >= ML_AUTH_MIN_PACKETS;

    const bool mature_flow =
        fs.pkt_count >= ML_MIN_PACKETS &&
        (mature_by_time || mature_by_packets || mature_by_auth_pattern);

    const bool syn_only_flow_ready =
        is_tcp_packet &&
        fs.syn_cnt >= 1 &&
        fs.bwd_pkts == 0 &&
        fs.pkt_count >= 1 &&
        !is_private_ipv4(client_ip);

    const bool early_score_ready =
        !mature_flow &&
        (syn_only_flow_ready || fs.pkt_count >= ML_MIN_PACKETS) &&
        (fs.last_ml_pkt_count == 0 || fs.pkt_count - fs.last_ml_pkt_count >= ML_RECHECK_EVERY_PACKETS);

    const bool mature_score_ready =
        mature_flow &&
        (fs.last_mature_ml_duration == 0 ||
         duration - fs.last_mature_ml_duration >= ML_MATURE_RECHECK_US ||
         mature_by_auth_pattern);

    if (early_score_ready || mature_score_ready)
    {
        const bool is_mature_score = mature_score_ready;
        const float active_threshold = is_mature_score ? ML_NIPS_MATURE_THRESHOLD : ML_NIPS_EARLY_THRESHOLD;

        if (is_mature_score)
            fs.last_mature_ml_duration = duration;
        else
            fs.last_ml_pkt_count = fs.pkt_count;

        LogMessage("ml_nips: READY FOR ML flow=%s stage=%s pkt_count=%lu duration_us=%lu threshold=%.2f\n",
            key.c_str(), is_mature_score ? "mature" : "early",
            fs.pkt_count, duration, active_threshold);

        float duration_us = (float)duration;
        float duration_sec = duration / 1000000.0f;
        float pkt_len_mean = fs.pkt_len_sum / (float)fs.pkt_count;
        float pkt_len_var = (float)pkt_len_var_for_log;

        float flow_iat_mean = fs.flow_iat_count ? fs.flow_iat_sum / (float)fs.flow_iat_count : 0.0f;
        float flow_iat_var = fs.flow_iat_count
            ? (float)(fs.flow_iat_sq_sum / fs.flow_iat_count - flow_iat_mean * flow_iat_mean)
            : 0.0f;
        float flow_iat_std = flow_iat_var > 0 ? sqrtf(flow_iat_var) : 0.0f;

        float fwd_iat_mean = fs.fwd_iat_count ? fs.fwd_iat_sum / (float)fs.fwd_iat_count : 0.0f;
        float fwd_iat_var = fs.fwd_iat_count
            ? (float)(fs.fwd_iat_sq_sum / fs.fwd_iat_count - fwd_iat_mean * fwd_iat_mean)
            : 0.0f;
        float fwd_iat_std = fwd_iat_var > 0 ? sqrtf(fwd_iat_var) : 0.0f;

        float bwd_iat_mean = fs.bwd_iat_count ? fs.bwd_iat_sum / (float)fs.bwd_iat_count : 0.0f;
        float bwd_iat_var = fs.bwd_iat_count
            ? (float)(fs.bwd_iat_sq_sum / fs.bwd_iat_count - bwd_iat_mean * bwd_iat_mean)
            : 0.0f;
        float bwd_iat_std = bwd_iat_var > 0 ? sqrtf(bwd_iat_var) : 0.0f;
        float bwd_iat_min = fs.bwd_iat_min == UINT64_MAX ? 0.0f : (float)fs.bwd_iat_min;

        float fwd_pkt_len_mean = fs.fwd_pkt_len_count
            ? fs.fwd_pkt_len_sum / (float)fs.fwd_pkt_len_count
            : 0.0f;

        float fwd_pkt_len_min = fs.fwd_pkt_len_min == UINT32_MAX
            ? 0.0f
            : (float)fs.fwd_pkt_len_min;

        float bwd_pkt_len_mean = fs.bwd_pkt_len_count
            ? fs.bwd_pkt_len_sum / (float)fs.bwd_pkt_len_count
            : 0.0f;

        float bwd_pkt_len_max = (float)fs.bwd_pkt_len_max;

        float bwd_pkt_len_min = fs.bwd_pkt_len_min == UINT32_MAX
            ? 0.0f
            : (float)fs.bwd_pkt_len_min;

        // Include the currently-open active segment when scoring live traffic.
        // This keeps real-time detection usable without waiting for FIN/RST or flow timeout.
        uint64_t active_sum_eff = fs.active_sum;
        double active_sq_sum_eff = fs.active_sq_sum;
        uint64_t active_count_eff = fs.active_count;
        uint64_t active_max_eff = fs.active_max;
        uint64_t active_min_eff = fs.active_min;

        if (fs.active_start_time != 0 && now >= fs.active_start_time)
        {
            uint64_t current_active = now - fs.active_start_time;
            active_sum_eff += current_active;
            active_sq_sum_eff += (double)current_active * (double)current_active;
            active_count_eff++;

            if (current_active > active_max_eff)
                active_max_eff = current_active;

            if (current_active < active_min_eff)
                active_min_eff = current_active;
        }

        float active_std = 0.0f;
        float active_min = active_min_eff == UINT64_MAX ? 0.0f : (float)active_min_eff;

        if (active_count_eff > 0)
        {
            float active_mean = active_sum_eff / (float)active_count_eff;
            float active_var = (float)(active_sq_sum_eff / active_count_eff - active_mean * active_mean);

            if (active_var < 0.0f)
                active_var = 0.0f;

            active_std = sqrtf(active_var);
        }

        float idle_std = 0.0f;

        if (fs.idle_count > 0)
        {
            float idle_mean = fs.idle_sum / (float)fs.idle_count;
            float idle_var = (float)(fs.idle_sq_sum / fs.idle_count - idle_mean * idle_mean);

            if (idle_var < 0.0f)
                idle_var = 0.0f;

            idle_std = sqrtf(idle_var);
        }

        // Feature Extraction

        std::vector<float> features(38, 0.0f);

        // Feature order is unchanged. Do not reorder these unless you retrain the model.
        features[0]  = (float)fs.ack_cnt;             // ACK Flag Cnt
        features[1]  = (float)active_max_eff;          // Active Max
        features[2]  = active_min;                    // Active Min
        features[3]  = active_std;                    // Active Std
        features[4]  = (float)fs.bwd_header_len;      // Bwd Header Len
        features[5]  = (float)fs.bwd_iat_max;         // Bwd IAT Max
        features[6]  = bwd_iat_mean;                  // Bwd IAT Mean
        features[7]  = bwd_iat_min;                   // Bwd IAT Min
        features[8]  = bwd_iat_std;                   // Bwd IAT Std
        features[9]  = (float)fs.bwd_iat_sum;         // Bwd IAT Tot
        features[10] = fs.bwd_pkt_len_min == UINT32_MAX ? 0.0f : (float)fs.bwd_pkt_len_min; // Bwd Pkt Len Min
        features[11] = (float)fs.cwe_cnt;             // CWE Flag Count
        features[12] = fs.fwd_pkts ? (float)fs.bwd_pkts / fs.fwd_pkts : 0.0f; // Down/Up Ratio
        features[13] = (float)fs.fin_cnt;             // FIN Flag Cnt
        features[14] = (float)flow_bytes_per_sec;     // Flow Byts/s
        features[15] = duration_us;                   // Flow Duration
        features[16] = (float)fs.flow_iat_max;        // Flow IAT Max
        features[17] = flow_iat_std;                  // Flow IAT Std
        features[18] = duration_sec > 0 ? (float)fs.pkt_count / duration_sec : 0.0f; // Flow Pkts/s
        features[19] = (float)fs.fwd_header_len;      // Fwd Header Len
        features[20] = fwd_iat_mean;                  // Fwd IAT Mean
        features[21] = fwd_iat_std;                   // Fwd IAT Std
        features[22] = (float)fs.fwd_psh_flags;       // Fwd PSH Flags
        features[23] = (float)fs.fwd_pkt_len_max;     // Fwd Pkt Len Max
        features[24] = fwd_pkt_len_mean;              // Fwd Pkt Len Mean
        features[25] = fwd_pkt_len_min;               // Fwd Pkt Len Min
        features[26] = fs.fwd_seg_size_min == UINT32_MAX ? 0.0f : (float)fs.fwd_seg_size_min; // Fwd Seg Size Min
        features[27] = (float)fs.idle_max;            // Idle Max
        features[28] = idle_std;                      // Idle Std
        features[29] = (float)fs.init_bwd_win_bytes;  // Init Bwd Win Byts
        features[30] = (float)fs.init_fwd_win_bytes;  // Init Fwd Win Byts
        features[31] = (float)fs.psh_cnt;             // PSH Flag Cnt
        features[32] = (float)fs.pkt_len_max;         // Pkt Len Max
        features[33] = pkt_len_mean;                  // Pkt Len Mean
        features[34] = (float)fs.pkt_len_min;         // Pkt Len Min
        features[35] = pkt_len_var;                   // Pkt Len Var
        features[36] = (float)fs.rst_cnt;             // RST Flag Cnt
        features[37] = (float)fs.urg_cnt;             // URG Flag Cnt

        /* CLEAN_LOG: noisy terminal debug disabled
LogMessage("ml_nips features38: ack=%.0f fin=%.0f rst=%.0f psh=%.0f fwd_hdr=%.0f bwd_hdr=%.0f win_fwd=%.0f win_bwd=%.0f pkt_mean=%.2f | flow_iat_max=%.2f flow_iat_std=%.2f fwd_iat_mean=%.2f bwd_iat_mean=%.2f\n",
            features[0], features[13], features[36], features[31],
            features[19], features[4], features[30], features[29], features[33],
            features[16], features[17], features[20], features[6]);
*/

        /* CLEAN_LOG: noisy terminal debug disabled
LogMessage("ml_nips payload-len direction check: fwd_min=%.2f fwd_mean=%.2f fwd_max=%.2f | bwd_min=%.2f bwd_mean=%.2f bwd_max=%.2f | fwd_count=%lu bwd_count=%lu\n",
            fwd_pkt_len_min, fwd_pkt_len_mean, (float)fs.fwd_pkt_len_max,
            bwd_pkt_len_min, bwd_pkt_len_mean, bwd_pkt_len_max,
            fs.fwd_pkt_len_count, fs.bwd_pkt_len_count);
*/

        /* CLEAN_LOG: noisy terminal debug disabled
LogMessage("ml_nips fwd seg size min check: %.2f\n", features[26]);
*/

        if (encoder_interpreter && ae_interpreter && mlp_interpreter)
        {
            // Feature Scaling

            float scaled_features[38];

            for (size_t i = 0; i < features.size(); i++)
                scaled_features[i] = quantile_scale(i, features[i]);

            /* CLEAN_LOG: noisy terminal debug disabled
LogMessage("ml_nips scaled debug: raw_win_fwd=%.2f scaled_win_fwd=%.4f raw_pkt_mean=%.2f scaled_pkt_mean=%.4f\n",
                features[30], scaled_features[30],
                features[33], scaled_features[33]);
*/

            // 1) Encoder: 38 scaled features -> 8 latent features.
            // Encoder Inference

            float* encoder_input = encoder_interpreter->typed_input_tensor<float>(0);
            for (int i = 0; i < 38; i++)
                encoder_input[i] = scaled_features[i];

            if (encoder_interpreter->Invoke() != kTfLiteOk)
            {
                LogMessage("ml_nips: encoder Invoke failed\n");
                return;
            }

            float* latent = encoder_interpreter->typed_output_tensor<float>(0);

            // 2) Full autoencoder: 38 scaled features -> 38 reconstructed features.
            // Reconstruction error becomes the 9th input to the MLP.
            // Autoencoder Reconstruction

            float* ae_input = ae_interpreter->typed_input_tensor<float>(0);
            for (int i = 0; i < 38; i++)
                ae_input[i] = scaled_features[i];

            if (ae_interpreter->Invoke() != kTfLiteOk)
            {
                LogMessage("ml_nips: autoencoder Invoke failed\n");
                return;
            }

            float* reconstructed = ae_interpreter->typed_output_tensor<float>(0);
            float recon_mse = 0.0f;
            for (int i = 0; i < 38; i++)
            {
                float diff = scaled_features[i] - reconstructed[i];
                recon_mse += diff * diff;
            }
            recon_mse /= 38.0f;

            // 3) MLP: [latent_dim=8 + reconstruction_error=1] -> attack probability/score.
            // MLP Classification

            float* mlp_input = mlp_interpreter->typed_input_tensor<float>(0);
            for (int i = 0; i < 8; i++)
                mlp_input[i] = latent[i];
            mlp_input[8] = recon_mse;

            if (mlp_interpreter->Invoke() != kTfLiteOk)
            {
                LogMessage("ml_nips: mlp Invoke failed\n");
                return;
            }

            float* mlp_output = mlp_interpreter->typed_output_tensor<float>(0);
            float score = mlp_output[0];

            LogMessage("ml_nips prediction: score=%.4f threshold=%.2f stage=%s\n",
                score, active_threshold, is_mature_score ? "mature" : "early");

            // Pure ML decision logic.
            // The final decision does not use handcrafted helpers, open-flow counts,
            // payload-size heuristics, protocol-specific bypasses, or max-score voting.
            // A flow is detected only when the current ML score crosses the model threshold.
            if (score > fs.max_score_seen)
                fs.max_score_seen = score;

            const bool ml_score_hit = score >= active_threshold;

            if (ml_score_hit)
            {
                LogMessage(
                    "ml_nips: ATTACK score=%.4f threshold=%.2f\n",
                    score,
                    active_threshold
                );
            }
            else
            {
                LogMessage(
                    "ml_nips: BENIGN score=%.4f threshold=%.2f\n",
                    score,
                    active_threshold
                );
            }

            if (ml_score_hit && !is_mature_score)
                fs.early_hit = true;

            if (ml_score_hit && is_mature_score)
                fs.mature_hit = true;

            const double avg_fwd_l4_len =
                fs.fwd_pkts > 0 ? (double)fs.fwd_bytes / (double)fs.fwd_pkts : 0.0;

            const char* stage_name = is_mature_score ? "mature" : "early";

            // Pure ML decision: metadata/readiness only controls when inference is run.
            // Attack label/blocking is decided only by the current ML score.
            // ML-based Detection and Prevention

            if (ml_score_hit)
                fs.flow_detected = true;

            bool detected = fs.flow_detected;

            /* CLEAN_LOG: noisy terminal debug disabled
LogMessage("ml_nips decision: stage=%s score=%.4f threshold=%.2f ml_score_hit=%d detected=%d duration_us=%lu pkt_count=%lu syn_cnt=%lu rst_cnt=%lu\n",
                stage_name, score, active_threshold,
                ml_score_hit ? 1 : 0, detected ? 1 : 0,
                duration, fs.pkt_count, fs.syn_cnt, fs.rst_cnt);
*/

            if (ml_score_hit)
            {
                LogMessage("ml_nips: ATTACK DETECTED score=%.4f threshold=%.2f stage=%s\n",
                    score, active_threshold, stage_name);

                if (!fs.alert_queued)
                {
                    DetectionEngine::queue_event(ML_NIPS_GID, ML_NIPS_SID);
                    fs.alert_queued = true;
                }

                // Runtime block is triggered only on current ML score hit.
                if (!fs.blocked)
                {
                    fs.blocked = true;
                    fs.ipset_applied = block_client_with_ipset(client_ip);
                }
            }

            // IDS and IPS Logging

            auto write_ids_csv_row = [&](std::ofstream& out)
            {
                if (!out.is_open())
                    return;

                out << now << ",";

                for (int i = 0; i < 38; i++)
                    out << features[i] << ",";

                out << client_ip << ","
                    << server_ip << ","
                    << client_port << ","
                    << server_port << ","
                    << score << ","
                    << fs.max_score_seen << ","
                    << active_threshold << ","
                    << stage_name << ","
                    << avg_fwd_l4_len << ","
                    << (ml_score_hit ? 1 : 0) << "\n";
                out.flush();
            };

            auto write_ips_csv_row = [&](std::ofstream& out)
            {
                if (!out.is_open())
                    return;

                const char* action = fs.ipset_applied ? "BLOCKED" : "BLOCK_DECISION_ONLY";

                out << now << ","
                    << client_ip << ","
                    << server_ip << ","
                    << client_port << ","
                    << server_port << ","
                    << score << ","
                    << active_threshold << ","
                    << stage_name << ","
                    << (ml_score_hit ? 1 : 0) << ","
                    << (fs.ipset_applied ? 1 : 0) << ","
                    << action << "\n";
                out.flush();
            };

            // IDS CSV: write every evaluated ML row.
            write_ids_csv_row(ids_log);

            // IPS CSV: write only for the current evaluation when ML score hits threshold.
            // This avoids writing low-score rows as blocked only because the flow was detected earlier.
            if (ml_score_hit)
                write_ips_csv_row(ips_log);
        }
        else
        {
            LogMessage("ml_nips: one or more interpreters are NULL\n");
        }
    }

    ++ml_nips_stats.total_packets;
}

//-------------------------------------------------------------------------
// module stuff
//-------------------------------------------------------------------------

static const Parameter
ml_nips_params[] =
{
    { "port", Parameter::PT_PORT, nullptr, nullptr,
      "port to check" },

    { "max", Parameter::PT_INT, "0:65535", "0",
      "maximum payload before alert" },

    { nullptr, Parameter::PT_MAX, nullptr, nullptr, nullptr }
};

static const RuleMap ml_nips_rules[] =
{
    { ML_NIPS_SID, "ML-based attack detection" },
    { 0, nullptr }
};

class MlNipsModule : public Module
{
public:
    MlNipsModule() : Module(s_name, s_help, ml_nips_params)
    { }

    unsigned get_gid() const override
    { return ML_NIPS_GID; }

    const RuleMap* get_rules() const override
    { return ml_nips_rules; }

    const PegInfo* get_pegs() const override
    { return simple_pegs; }

    PegCount* get_counts() const override
    { return (PegCount*)&ml_nips_stats; }

    ProfileStats* get_profile() const override
    { return &MlNipsPerfStats; }

    bool set(const char*, Value& v, SnortConfig*) override;

    Usage get_usage() const override
    { return INSPECT; }

    void set_trace(const Trace*) const override;
    const TraceOption* get_trace_options() const override;

public:
    uint16_t port = 0;
    uint16_t max = 0;
};

bool MlNipsModule::set(const char*, Value& v, SnortConfig*)
{
    if ( v.is("port") )
        port = v.get_uint16();

    else if ( v.is("max") )
        max = v.get_uint16();

    return true;
}

void MlNipsModule::set_trace(const Trace* trace) const
{ ml_nips_trace = trace; }

const TraceOption* MlNipsModule::get_trace_options() const
{
    static const TraceOption MlNips_options(nullptr, 0, nullptr);
    return &MlNips_options;
}

//-------------------------------------------------------------------------
// api stuff
//-------------------------------------------------------------------------

static Module* mod_ctor()
{ return new MlNipsModule; }

static void mod_dtor(Module* m)
{ delete m; }

static Inspector* ml_nips_ctor(Module* m)
{
    MlNipsModule* mod = (MlNipsModule*)m;
    return new MlNips(mod->port, mod->max);
}

static void ml_nips_dtor(Inspector* p)
{
    delete p;
}

static const InspectApi ml_nips_api
{
    {
        PT_INSPECTOR,
        sizeof(InspectApi),
        INSAPI_VERSION,
        0,
        API_RESERVED,
        API_OPTIONS,
        s_name,
        s_help,
        mod_ctor,
        mod_dtor
    },
    IT_NETWORK,
    PROTO_BIT__TCP | PROTO_BIT__UDP,
    nullptr, // buffers
    nullptr, // service
    nullptr, // pinit
    nullptr, // pterm
    nullptr, // tinit
    nullptr, // tterm
    ml_nips_ctor,
    ml_nips_dtor,
    nullptr, // ssn
    nullptr  // reset
};

SO_PUBLIC const BaseApi* snort_plugins[] =
{
    &ml_nips_api.base,
    nullptr
};

