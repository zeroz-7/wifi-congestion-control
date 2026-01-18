#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/yans-wifi-channel.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace ns3;

// -------- CSV helpers --------
static std::vector<std::string> SplitCsv(const std::string &s)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
  {
    item.erase(item.begin(), std::find_if(item.begin(), item.end(),
                                         [](unsigned char ch) { return !std::isspace(ch); }));
    item.erase(std::find_if(item.rbegin(), item.rend(),
                            [](unsigned char ch) { return !std::isspace(ch); }).base(),
               item.end());
    if (!item.empty())
      out.push_back(item);
  }
  return out;
}

static std::vector<uint32_t> ParseU32List(const std::string &s, uint32_t n, uint32_t defVal)
{
  auto parts = SplitCsv(s);
  std::vector<uint32_t> v(n, defVal);
  for (uint32_t i = 0; i < std::min<uint32_t>(n, (uint32_t)parts.size()); i++)
    v[i] = (uint32_t)std::stoul(parts[i]);
  return v;
}

static std::vector<double> ParseDList(const std::string &s, uint32_t n, double defVal)
{
  auto parts = SplitCsv(s);
  std::vector<double> v(n, defVal);
  for (uint32_t i = 0; i < std::min<uint32_t>(n, (uint32_t)parts.size()); i++)
    v[i] = std::stod(parts[i]);
  return v;
}

// -------- CSV output --------
static void WriteHeader(std::ofstream &out, uint32_t nAps)
{
  out << "seed,run,stepTime,nAps,nStaPerAp,nBg,";
  for (uint32_t i = 0; i < nAps; i++)
    out << "ap" << i << "_channel,";
  for (uint32_t i = 0; i < nAps; i++)
    out << "ap" << i << "_txPowerDbm,";
  for (uint32_t i = 0; i < nAps; i++)
    out << "ap" << i << "_cwMin,";
  for (uint32_t i = 0; i < nAps; i++)
    out << "ap" << i << "_cwMax,";
  for (uint32_t i = 0; i < nAps; i++)
    out << "ap" << i << "_dataRateMbps,";
  for (uint32_t i = 0; i < nAps; i++)
    out << "T" << i << "_Mbps,";
  for (uint32_t i = 0; i < nAps; i++)
    out << "Q" << i << "_ms,"; // Q = mean delay (ms) for compatibility
  out << "lossRate\n";
}

int main(int argc, char *argv[])
{
  uint32_t nAps = 3;
  uint32_t nStaPerAp = 25;
  uint32_t nBg = 15;
  double stepTime = 10.0;

  std::string channelsStr = "6,6,6";
  std::string txPowersStr = "16,16,16";
  std::string cwMinsStr = "15,15,15";
  std::string cwMaxsStr = "1023,1023,1023";
  std::string dataRatesStr = "24,24,24";

  uint32_t seed = 1;
  uint32_t run = 1;

  CommandLine cmd;
  cmd.AddValue("nAps", "Number of APs", nAps);
  cmd.AddValue("nStaPerAp", "Stations per AP", nStaPerAp);
  cmd.AddValue("nBg", "Background STAs (attach to AP0)", nBg);
  cmd.AddValue("stepTime", "Step duration (s)", stepTime);

  cmd.AddValue("channels", "CSV per-AP channels", channelsStr);
  cmd.AddValue("txPowers", "CSV per-AP txPower dBm", txPowersStr);
  cmd.AddValue("cwMins", "CSV per-AP CWmin", cwMinsStr);
  cmd.AddValue("cwMaxs", "CSV per-AP CWmax", cwMaxsStr);
  cmd.AddValue("dataRates", "CSV per-AP app rate cap (Mbps)", dataRatesStr);

  cmd.AddValue("seed", "RNG seed", seed);
  cmd.AddValue("run", "RNG run", run);
  cmd.Parse(argc, argv);

  auto channels = ParseU32List(channelsStr, nAps, 6);
  auto txPowers = ParseDList(txPowersStr, nAps, 16.0);
  auto cwMins = ParseU32List(cwMinsStr, nAps, 15);
  auto cwMaxs = ParseU32List(cwMaxsStr, nAps, 1023);
  auto dataRates = ParseDList(dataRatesStr, nAps, 24.0);

  RngSeedManager::SetSeed(seed);
  RngSeedManager::SetRun(run);

  NodeContainer aps;
  aps.Create(nAps);
  NodeContainer stas;
  stas.Create(nAps * nStaPerAp);
  NodeContainer bg;
  bg.Create(nBg);

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211n);

  YansWifiChannelHelper ychan;
  ychan.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  ychan.AddPropagationLoss("ns3::LogDistancePropagationLossModel");

  std::vector<YansWifiPhyHelper> phys(nAps);
  for (uint32_t i = 0; i < nAps; i++)
  {
    phys[i] = YansWifiPhyHelper();
    phys[i].SetChannel(ychan.Create());
    phys[i].Set("TxPowerStart", DoubleValue(txPowers[i]));
    phys[i].Set("TxPowerEnd", DoubleValue(txPowers[i]));
  }

  WifiMacHelper mac;
  std::vector<NetDeviceContainer> apDevs(nAps);
  std::vector<NetDeviceContainer> staDevsByAp(nAps);

  uint32_t staIdx = 0;
  for (uint32_t apIdx = 0; apIdx < nAps; apIdx++)
  {
    Ssid ssid = Ssid("ap-" + std::to_string(apIdx));

    mac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    apDevs[apIdx] = wifi.Install(phys[apIdx], mac, aps.Get(apIdx));

    NodeContainer staGroup;
    for (uint32_t i = 0; i < nStaPerAp; i++)
      staGroup.Add(stas.Get(staIdx++));

    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid), "ActiveProbing", BooleanValue(false));
    staDevsByAp[apIdx] = wifi.Install(phys[apIdx], mac, staGroup);
  }

  // Background attaches to AP0
  Ssid bgSsid = Ssid("ap-0");
  mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(bgSsid), "ActiveProbing", BooleanValue(false));
  NetDeviceContainer bgDevs = wifi.Install(phys[0], mac, bg);

  // ✅ Apply CWmin/CWmax using Config (version-safe)
  // Apply per-AP CW on AP nodes
  for (uint32_t i = 0; i < aps.GetN(); i++)
  {
    std::ostringstream minPath, maxPath;
    minPath << "/NodeList/" << aps.Get(i)->GetId()
            << "/DeviceList/0/$ns3::WifiNetDevice/Mac/BE_EdcaTxopN/MinCw";
    maxPath << "/NodeList/" << aps.Get(i)->GetId()
            << "/DeviceList/0/$ns3::WifiNetDevice/Mac/BE_EdcaTxopN/MaxCw";
    Config::Set(minPath.str(), UintegerValue(cwMins[i]));
    Config::Set(maxPath.str(), UintegerValue(cwMaxs[i]));
  }

  // Apply CW to all STAs (global CW for Phase-1; change later to per-AP STA grouping)
  for (uint32_t i = 0; i < stas.GetN(); i++)
  {
    std::ostringstream minPath, maxPath;
    minPath << "/NodeList/" << stas.Get(i)->GetId()
            << "/DeviceList/0/$ns3::WifiNetDevice/Mac/BE_EdcaTxopN/MinCw";
    maxPath << "/NodeList/" << stas.Get(i)->GetId()
            << "/DeviceList/0/$ns3::WifiNetDevice/Mac/BE_EdcaTxopN/MaxCw";
    Config::Set(minPath.str(), UintegerValue(cwMins[0]));
    Config::Set(maxPath.str(), UintegerValue(cwMaxs[0]));
  }

  for (uint32_t i = 0; i < bg.GetN(); i++)
  {
    std::ostringstream minPath, maxPath;
    minPath << "/NodeList/" << bg.Get(i)->GetId()
            << "/DeviceList/0/$ns3::WifiNetDevice/Mac/BE_EdcaTxopN/MinCw";
    maxPath << "/NodeList/" << bg.Get(i)->GetId()
            << "/DeviceList/0/$ns3::WifiNetDevice/Mac/BE_EdcaTxopN/MaxCw";
    Config::Set(minPath.str(), UintegerValue(cwMins[0]));
    Config::Set(maxPath.str(), UintegerValue(cwMaxs[0]));
  }

  // Mobility
  MobilityHelper apMob;
  Ptr<ListPositionAllocator> apPos = CreateObject<ListPositionAllocator>();
  for (uint32_t i = 0; i < nAps; i++)
    apPos->Add(Vector(60.0 * i, 0.0, 0.0));
  apMob.SetPositionAllocator(apPos);
  apMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  apMob.Install(aps);

  MobilityHelper staMob;
  staMob.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                          "Bounds", RectangleValue(Rectangle(-20, 60.0 * (nAps - 1) + 20, -40, 40)),
                          "Distance", DoubleValue(10.0),
                          "Speed", StringValue("ns3::ConstantRandomVariable[Constant=1.0]"));
  staMob.Install(stas);

  MobilityHelper bgMob;
  bgMob.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                         "Bounds", RectangleValue(Rectangle(-20, 60.0 * (nAps - 1) + 20, -40, 40)),
                         "Distance", DoubleValue(15.0),
                         "Speed", StringValue("ns3::ConstantRandomVariable[Constant=1.2]"));
  bgMob.Install(bg);

  // Internet
  InternetStackHelper stack;
  stack.Install(aps);
  stack.Install(stas);
  stack.Install(bg);

  Ipv4AddressHelper addr;
  addr.SetBase("10.1.0.0", "255.255.255.0");

  NetDeviceContainer allDevs;
  for (uint32_t apIdx = 0; apIdx < nAps; apIdx++)
  {
    allDevs.Add(apDevs[apIdx]);
    allDevs.Add(staDevsByAp[apIdx]);
  }
  allDevs.Add(bgDevs);

  Ipv4InterfaceContainer ifs = addr.Assign(allDevs);

  // AP IPs by device ordering
  std::vector<Ipv4Address> apIps(nAps);
  uint32_t cursor = 0;
  for (uint32_t apIdx = 0; apIdx < nAps; apIdx++)
  {
    apIps[apIdx] = ifs.GetAddress(cursor);
    cursor += 1;
    cursor += staDevsByAp[apIdx].GetN();
  }

  // UDP sinks
  uint16_t portBase = 5000;
  ApplicationContainer sinks;
  for (uint32_t apIdx = 0; apIdx < nAps; apIdx++)
  {
    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          InetSocketAddress(Ipv4Address::GetAny(), portBase + apIdx));
    sinks.Add(sink.Install(aps.Get(apIdx)));
  }
  // BG sink on AP0
  {
    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          InetSocketAddress(Ipv4Address::GetAny(), portBase + 100));
    sinks.Add(sink.Install(aps.Get(0)));
  }

  sinks.Start(Seconds(0.0));
  sinks.Stop(Seconds(stepTime));

  auto installUdpClient = [&](Ptr<Node> src, Ipv4Address dst, uint16_t port, double rateMbps) {
    UdpClientHelper client(dst, port);
    uint32_t pktSize = 1200;
    double rateBps = rateMbps * 1e6;
    double interval = (pktSize * 8.0) / rateBps;

    client.SetAttribute("PacketSize", UintegerValue(pktSize));
    client.SetAttribute("Interval", TimeValue(Seconds(interval)));
    client.SetAttribute("MaxPackets", UintegerValue(0));

    auto app = client.Install(src);
    app.Start(Seconds(1.0));
    app.Stop(Seconds(stepTime));
  };

  // STA -> AP flows
  uint32_t gSta = 0;
  for (uint32_t apIdx = 0; apIdx < nAps; apIdx++)
  {
    for (uint32_t i = 0; i < nStaPerAp; i++)
      installUdpClient(stas.Get(gSta++), apIps[apIdx], portBase + apIdx, dataRates[apIdx]);
  }
  // BG -> AP0
  for (uint32_t i = 0; i < nBg; i++)
    installUdpClient(bg.Get(i), apIps[0], portBase + 100, dataRates[0] * 1.2);

  // Flow monitor for throughput + mean delay as Q
  FlowMonitorHelper flowmon;
  Ptr<FlowMonitor> monitor = flowmon.InstallAll();

  Simulator::Stop(Seconds(stepTime));
  Simulator::Run();

  monitor->CheckForLostPackets();

  std::vector<double> T(nAps, 0.0);
  std::vector<double> delaySumS(nAps, 0.0);
  std::vector<double> rxPkts(nAps, 0.0);

  double totalLost = 0.0;
  double totalRx = 0.0;

  Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
  auto stats = monitor->GetFlowStats();

  for (const auto &kv : stats)
  {
    const FlowMonitor::FlowStats &st = kv.second;
    Ipv4FlowClassifier::FiveTuple ft = classifier->FindFlow(kv.first);

    totalLost += st.lostPackets;
    totalRx += st.rxPackets;

    uint16_t dport = ft.destinationPort;
    int apIdx = -1;
    if (dport >= portBase && dport < portBase + (int)nAps)
      apIdx = (int)(dport - portBase);

    if (apIdx >= 0 && apIdx < (int)nAps)
    {
      T[apIdx] += (st.rxBytes * 8.0) / (stepTime * 1e6);
      delaySumS[apIdx] += st.delaySum.GetSeconds();
      rxPkts[apIdx] += st.rxPackets;
    }
  }

  double lossRate = ((totalRx + totalLost) > 0.0) ? (totalLost / (totalRx + totalLost)) : 0.0;

  // Q = mean delay (ms)
  std::vector<double> Q(nAps, 0.0);
  for (uint32_t i = 0; i < nAps; i++)
    Q[i] = (rxPkts[i] > 0.0) ? (delaySumS[i] / rxPkts[i]) * 1000.0 : 0.0;

  // Write CSV
  std::string outPath = "per_ap_step_metrics.csv";
  bool headerNeeded = false;
  {
    std::ifstream chk(outPath);
    if (!chk.good())
      headerNeeded = true;
  }
  std::ofstream out(outPath, std::ios::app);
  if (headerNeeded)
    WriteHeader(out, nAps);

  out << seed << "," << run << "," << stepTime << ","
      << nAps << "," << nStaPerAp << "," << nBg << ",";

  for (uint32_t i = 0; i < nAps; i++) out << channels[i] << ",";
  for (uint32_t i = 0; i < nAps; i++) out << txPowers[i] << ",";
  for (uint32_t i = 0; i < nAps; i++) out << cwMins[i] << ",";
  for (uint32_t i = 0; i < nAps; i++) out << cwMaxs[i] << ",";
  for (uint32_t i = 0; i < nAps; i++) out << dataRates[i] << ",";

  for (uint32_t i = 0; i < nAps; i++) out << T[i] << ",";
  for (uint32_t i = 0; i < nAps; i++) out << Q[i] << ",";
  out << lossRate << "\n";
  out.close();

  Simulator::Destroy();

  std::cout << "QTYPE:MEAN_DELAY_MS ";
  std::cout << "T:";
  for (uint32_t i = 0; i < nAps; i++) std::cout << (i ? "," : "") << T[i];
  std::cout << " Q:";
  for (uint32_t i = 0; i < nAps; i++) std::cout << (i ? "," : "") << Q[i];
  std::cout << " LOSS:" << lossRate << "\n";

  return 0;
}
