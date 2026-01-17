#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/applications-module.h"

#include "control.h"

using namespace ns3;

int
main (int argc, char *argv[])
{
  uint32_t numSta = 5;
  double simTime = 30.0; // seconds

  CommandLine cmd;
  cmd.AddValue ("numSta", "Number of stations", numSta);
  cmd.Parse (argc, argv);

  /* ---------------- Nodes ---------------- */
  NodeContainer wifiStaNodes;
  wifiStaNodes.Create (numSta);
  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  /* ---------------- Wi-Fi ---------------- */
  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);

  WifiMacHelper mac;
  YansWifiPhyHelper phy;
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  phy.SetChannel (channel.Create ());

  Ssid ssid = Ssid ("rl-wifi");

  mac.SetType ("ns3::StaWifiMac",
               "Ssid", SsidValue (ssid),
               "ActiveProbing", BooleanValue (false));

  NetDeviceContainer staDevices;
  staDevices = wifi.Install (phy, mac, wifiStaNodes);

  mac.SetType ("ns3::ApWifiMac",
               "Ssid", SsidValue (ssid));

  NetDeviceContainer apDevice;
  apDevice = wifi.Install (phy, mac, wifiApNode);

  /* ---------------- Mobility ---------------- */
  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiStaNodes);
  mobility.Install (wifiApNode);

  /* ---------------- Internet ---------------- */
  InternetStackHelper stack;
  stack.Install (wifiStaNodes);
  stack.Install (wifiApNode);

  Ipv4AddressHelper address;
  address.SetBase ("10.1.1.0", "255.255.255.0");

  Ipv4InterfaceContainer staIf;
  staIf = address.Assign (staDevices);
  Ipv4InterfaceContainer apIf;
  apIf = address.Assign (apDevice);

  /* ---------------- Traffic ---------------- */
  uint16_t port = 4000;

  UdpServerHelper server (port);
  ApplicationContainer serverApp =
      server.Install (wifiApNode.Get (0));
  serverApp.Start (Seconds (1.0));
  serverApp.Stop (Seconds (simTime));

  UdpClientHelper client (apIf.GetAddress (0), port);
  client.SetAttribute ("MaxPackets", UintegerValue (1000000));
  client.SetAttribute ("Interval", TimeValue (MilliSeconds (10)));
  client.SetAttribute ("PacketSize", UintegerValue (1024));

  ApplicationContainer clientApps;
  for (uint32_t i = 0; i < numSta; ++i)
    {
      clientApps.Add (client.Install (wifiStaNodes.Get (i)));
    }

  clientApps.Start (Seconds (2.0));
  clientApps.Stop (Seconds (simTime));

  /* ---------------- Control Loop (Step 2) ---------------- */
  StartControlLoop ();

  Simulator::Stop (Seconds (simTime));
  Simulator::Run ();
  Simulator::Destroy ();

  return 0;
}
